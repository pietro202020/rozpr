#!/usr/bin/env python3
"""
"""

from mpi4py import MPI
import argparse, random, time, heapq
from enum import Enum, auto

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--Y", type=int, default=3,
               help="pojemność bramy (ile badaczy może przechodzić jednocześnie)")
p.add_argument("--iterations", type=int, default=10,
               help="ile razy każdy proces przejdzie przez bramę")
p.add_argument("--silent", action="store_true", help="wyłącz logi")
args = p.parse_args()
Y, ITERS, SILENT = args.Y, args.iterations, args.silent

# ---------- typy / narzędzia ----------
class DIR(Enum):
    A = "A"
    B = "B"

def opposite(d):
    return DIR.B if d is DIR.A else DIR.A

class State(Enum):
    RELEASED = auto()
    WANTED   = auto()
    HELD     = auto()

class MType(Enum):
    REQUEST   = 0
    ACK       = 1
    RELEASE   = 2
    TERMINATE = 3  # sygnał zakończenia

def _log(r, t, s):
    if not SILENT:
        print(f"[{r}] [t{t:06d}] {s}", flush=True)

# ---------- proces ----------
class Proc:
    def __init__(self, comm):
        self.c   = comm
        self.id  = comm.Get_rank()
        self.N   = comm.Get_size()
        self.peers = [i for i in range(self.N) if i != self.id]

        # --- zmienne pseudokodu ---
        self.clock   = 0
        self.state   = State.RELEASED
        self.wantDir = None
        self.gateDir = DIR.A           # startowy kierunek bramy (dowolnie A)
        self.Acked   = [True] * self.N
        self.Q       = []              # kopiec (ts, pid, dir)
        self.reqTS   = None            # timestamp naszego REQUEST
        self.active  = [True] * self.N # czy proces jeszcze nie dał TERMINATE
        self.should_terminate = False  # flaga, by przerwać natychmiast

    # ---- Lamport ----
    def _tick(self):
        self.clock += 1
        return self.clock

    def _upd(self, ts):
        self.clock = max(self.clock, ts) + 1

    # ---- wysyłanie komunikatów ----
    def _send(self, dst, typ, **pl):
        self._tick()
        self.c.send((typ.value, pl), dst, 0)

    def _bcast(self, typ, **pl):
        for p in self.peers:
            self._send(p, typ, **pl)

    # ---- myTurn: czy mogę wejść? ----
    def _my_turn(self):
        if not self.Q:
            return False
        sortedQ = sorted(self.Q)  # rosnąco po (ts,pid,dir)
        head_ts, head_pid, head_dir = sortedQ[0]
        # 1) jeśli czołowy nie pasuje do gateDir → nie mogę wejść
        if DIR(head_dir) != self.gateDir:
            return False
        # 2) zlicz, ile wpisów o gateDir stoi do mojego pid
        pos = 0
        for ts_i, pid_i, dir_i in sortedQ:
            if DIR(dir_i) != self.gateDir:
                break
            pos += 1
            if pid_i == self.id:
                return (pos <= Y)
        return False

    # ---- handler dla REQUEST ----
    def _h_req(self, src, ts, dir_):
        # 1) zaktualizowany clock już był w _poll()
        heapq.heappush(self.Q, (ts, src, dir_))

        # 2) jeśli tunel pusty (tj. żadnego HELD) i czołowy <normalnie> inny kolor:
        if self.state != State.HELD and DIR(self.Q[0][2]) != self.gateDir:
            # przełącz tunel na kolor czoła
            self.gateDir = DIR(self.Q[0][2])
            _log(self.id, self.clock,
                 f"Ustawiam bramę na {self.gateDir.name} (tunel pusty)")

        # 3) odeślij ACK
        self._send(src, MType.ACK, ts=self.clock)

    # ---- handler dla ACK ----
    def _h_ack(self, src):
        self.Acked[src] = True

    # ---- handler dla RELEASE ----
    def _h_rel(self, src, ts, dir_):
        # 1) usuwamy wszystkie wpisy danego procesu (pid=src), bo każdy REQUEST dubluje się po zmianach
        to_remove = [entry for entry in self.Q if entry[1] == src]
        for entry in to_remove:
            try:
                self.Q.remove(entry)
            except ValueError:
                pass
        if to_remove:
            heapq.heapify(self.Q)

        # 2) jeśli kolejka nie jest pusta i czołowy wpis ma inny kolor niż gateDir, przełącz
        if self.Q and DIR(self.Q[0][2]) != self.gateDir:
            self.gateDir = DIR(self.Q[0][2])
            _log(self.id, self.clock,
                 f"Przestawiam bramę na {self.gateDir.name}")

    # ---- handler dla TERMINATE ----
    def _h_term(self, src):
        # natychmiast przerywamy, jeśli dowolny proces wyśle TERMINATE
        self.should_terminate = True

    # ---- odbiór wiadomości non‐blocking ----
    def _poll(self):
        st = MPI.Status()
        while self.c.Iprobe(source=MPI.ANY_SOURCE, tag=0, status=st):
            src = st.Get_source()
            typ_val, pl = self.c.recv(source=src, tag=0)
            # zaktualizuj zegar Lamporta, jeśli jest ts
            if "ts" in pl:
                self._upd(pl["ts"])
            typ = MType(typ_val)
            if typ is MType.REQUEST:
                self._h_req(src, pl["ts"], pl["dir"])
            elif typ is MType.ACK:
                self._h_ack(src)
            elif typ is MType.RELEASE:
                self._h_rel(src, pl["ts"], pl["dir"])
            elif typ is MType.TERMINATE:
                self._h_term(src)

    # ---- procedura wejścia do tunelu ----
    def enter(self, d: DIR):
        self.state, self.wantDir = State.WANTED, d
        ts = self._tick()
        self.reqTS = ts

        # resetujemy ACK od wszystkich peerów
        for p in self.peers:
            self.Acked[p] = False

        # wrzucamy własny REQUEST do kolejki i rozsyłamy
        heapq.heappush(self.Q, (ts, self.id, d.value))
        self._bcast(MType.REQUEST, dir=d.value, ts=ts)
        _log(self.id, self.clock, f"Staram się o {d.name}")

        # czekamy aż: od wszystkich ACKi i myTurn == True
        while True:
            if self.should_terminate:
                return  # jeśli już TERMINATE, natychmiast kończymy

            self._poll()
            if all(self.Acked[p] or not self.active[p] for p in self.peers) \
               and self._my_turn():
                self.state = State.HELD
                _log(self.id, self.clock, "==> WCHODZĘ <==")
                break
            time.sleep(0.001)

    # ---- procedura opuszczenia tunelu ----
    def leave(self):
        # usuwamy lokalnie SWÓJ wpis (reqTS, id, wantDir)
        try:
            self.Q.remove((self.reqTS, self.id, self.wantDir.value))
            heapq.heapify(self.Q)
        except ValueError:
            pass

        self.state = State.RELEASED
        # rozsyłamy RELEASE aktualnego gateDir
        self._bcast(MType.RELEASE, dir=self.gateDir.value, ts=self._tick())
        _log(self.id, self.clock, "<== WYCHODZĘ ==>")

    # ---- główna pętla procesu ----
    def run(self):
        random.seed(self.id * 1234 + int(time.time()))

        for _ in range(ITERS):
            # jeśli już ktoś wysłał TERMINATE, przerwijmy resztę
            if self.should_terminate:
                break

            _log(self.id, self.clock, "Śpię")
            t_end = time.time() + random.uniform(0.2, 0.4)
            while time.time() < t_end:
                if self.should_terminate:
                    break
                self._poll()
                time.sleep(0.005)
            if self.should_terminate:
                break

            # próbujemy wejść
            self.enter(random.choice([DIR.A, DIR.B]))
            if self.should_terminate:
                break

            # „przejście przez tunel” – krótka pauza, w trakcie której _poll() też działa
            t_in = time.time() + random.uniform(0.15, 0.3)
            while time.time() < t_in:
                if self.should_terminate:
                    break
                self._poll()
                time.sleep(0.005)
            if self.should_terminate:
                break

            # opuszczamy tunel
            self.leave()

        # jeżeli jeszcze nie dostaliśmy TERMINATE, to wysyłamy go jako pierwsi
        if not self.should_terminate:
            self._bcast(MType.TERMINATE, ts=self._tick())
            _log(self.id, self.clock, "TERMINATE")

        # czekamy krótką chwilę (0.3 s), by inni odebrali TERMINATE
        t_fin = time.time() + 0.3
        while time.time() < t_fin:
            self._poll()
            time.sleep(0.005)

        MPI.Finalize()

# ---------- main ----------

def main():
    pr = Proc(MPI.COMM_WORLD)
    pr.run()

if __name__ == "__main__":
    main()
