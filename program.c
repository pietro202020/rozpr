#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
//#define DEBUG

#define TAG_REQUEST 1
#define TAG_ACK 2
#define TAG_RELEASE 3

#define DIR_A 0
#define DIR_B 1

#define Y 2

#ifdef DEBUG
#define DEBUG_PRINT(fmt, ...) printf("[%d] [t%d] " fmt "\n", rank, clockLamport, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...)
#endif

#define PRINT_STATE(fmt, ...) printf("[%d] [t%d] " fmt "\n", rank, clockLamport, ##__VA_ARGS__)

typedef struct {
    int ts;
    int pid;
    int dir;
} Request;

int clockLamport = 0;
int state = 0; // 0 - RELEASED, 1 - WANTED, 2 - HELD
int wantDir = DIR_A;
int gateDir = DIR_A;
int cnt = 0;
bool *acked;
int size, rank;

Request queue[100];
int queueSize = 0;

int max(int a, int b) {
    return a > b ? a : b;
}

void updateClock(int received_ts) {
    clockLamport = max(clockLamport, received_ts) + 1;
}

void broadcastRequest(int dir) {
    clockLamport++;
    PRINT_STATE("Rozpoczynam staranie o sekcję krytyczną (kierunek %d)", dir);
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        int msg[2] = {dir, clockLamport};
        MPI_Send(msg, 2, MPI_INT, i, TAG_REQUEST, MPI_COMM_WORLD);
        DEBUG_PRINT("Wysłano REQUEST do %d (kierunek %d)", i, dir);
    }
}

void sendAck(int dest) {
    clockLamport++;
    MPI_Send(&clockLamport, 1, MPI_INT, dest, TAG_ACK, MPI_COMM_WORLD);
    DEBUG_PRINT("Wysłano ACK do %d", dest);
}

void broadcastRelease() {
    clockLamport++;
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        int msg[2] = {gateDir, clockLamport};
        MPI_Send(msg, 2, MPI_INT, i, TAG_RELEASE, MPI_COMM_WORLD);
        DEBUG_PRINT("Wysłano RELEASE do %d (kierunek %d)", i, gateDir);
    }
}

bool myTurn() {
    if (queueSize == 0)
        return false;

    Request head = queue[0];
    if (head.dir != gateDir)
        return false;

    int pos = 0;
    for (int i = 0; i < queueSize; i++) {
        if (queue[i].dir == gateDir) {
            pos++;
            if (queue[i].pid == rank) break;
        }
    }
    return pos <= Y;
}

void enterCriticalSection(int dir) {
    state = 1; // WANTED
    wantDir = dir;

    for (int i = 0; i < size; i++)
        acked[i] = false;

    broadcastRequest(dir);

    while (1) {
        bool allAcked = true;
        for (int i = 0; i < size; i++) {
            if (i != rank && !acked[i]) {
                allAcked = false;
                break;
            }
        }
        if (allAcked && myTurn()) break;

        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_REQUEST) {
            int msg[2];
            MPI_Recv(msg, 2, MPI_INT, status.MPI_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            updateClock(msg[1]);
            DEBUG_PRINT("Otrzymano REQUEST od %d (kierunek %d)", status.MPI_SOURCE, msg[0]);
            Request r = {msg[1], status.MPI_SOURCE, msg[0]};
            queue[queueSize++] = r;
		
            bool sendNow = false;
            if (state == 2) { // HELD
                if (msg[0] == gateDir && cnt == 0) sendNow = true;
            } else if (state == 1) { // WANTED
                Request myReq = {clockLamport, rank, wantDir};
                if (myReq.ts < r.ts || (myReq.ts == r.ts && myReq.pid < r.pid))
                    ; // wait
                else
                    sendNow = true;
            } else {
                sendNow = true;
            }

            if (sendNow) sendAck(status.MPI_SOURCE);
        }
        else if (status.MPI_TAG == TAG_ACK) {
            int ts;
            MPI_Recv(&ts, 1, MPI_INT, status.MPI_SOURCE, TAG_ACK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            updateClock(ts);
            acked[status.MPI_SOURCE] = true;
            DEBUG_PRINT("Otrzymano ACK od %d", status.MPI_SOURCE);
        }
        else if (status.MPI_TAG == TAG_RELEASE) {
            int msg[2];
            MPI_Recv(msg, 2, MPI_INT, status.MPI_SOURCE, TAG_RELEASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            updateClock(msg[1]);
            DEBUG_PRINT("Otrzymano RELEASE od %d (kierunek %d)", status.MPI_SOURCE, msg[0]);
            // Remove from queue
            for (int i = 0; i < queueSize; i++) {
                if (queue[i].pid == status.MPI_SOURCE) {
                    for (int j = i; j < queueSize - 1; j++) queue[j] = queue[j + 1];
                    queueSize--;
                    break;
                }
            }
	    if (msg[0] == gateDir) {
       		 gateDir = (gateDir == DIR_A) ? DIR_B : DIR_A;
        	 DEBUG_PRINT("Zmieniono gateDir na %d", gateDir);
    	    }
        }
    }

    state = 2; // HELD
    PRINT_STATE("Jestem w sekcji krytycznej (kierunek %d)", dir);
}

void leaveCriticalSection() {
    PRINT_STATE("Wychodzę z sekcji krytycznej (kierunek %d)", gateDir);
    //state = 0; // RELEASED
    broadcastRelease();
    for (int i = 0; i < queueSize; i++) {
        if (queue[i].pid == rank) {
            for (int j = i; j < queueSize - 1; j++) queue[j] = queue[j + 1];
            queueSize--;
            break;
        }
    }
    state=0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    acked = malloc(size * sizeof(bool));
    srand(time(NULL) + rank);
    for (int round = 0; round < 3; round++) {
        //PRINT_STATE("Śpię");
        sleep(1);
        enterCriticalSection(rank % 2 == 0 ? DIR_A : DIR_B);
        int sleepTime = (rand() % 5) + 1; // Losuje 1–5
    	PRINT_STATE("W sekcji krytycznej przez %d sekund(y)", sleepTime);
    	sleep(sleepTime);
        leaveCriticalSection();
        sleep(1);
    }

    free(acked);
    MPI_Finalize();
    return 0;
}
