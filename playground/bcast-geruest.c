#define _POSIX_C_SOURCE 2
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

#include "mpi.h"

int np;   /* Anzahl der MPI-Prozesse */
int self; /* Nummer des eigenen Prozesses (im Bereich 0,...,np-1) */

/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

/*
 * Speicherplatz fuer `size' viele int anfordern (d.h. ein
 * int-Array der Laenge `size').
 * Der Speicherplatz ist nach Benutzung mit  free  freizugeben:
 *    int *p = allocints(100);
 *    if (p != NULL) {
 *      .... // p benutzen
 *      free(p);
 *    }
 */
static int *allocints(int size) {
    int *p;
    p = (int *)malloc(size * sizeof(int));
    return p;
}

void native_bcast(void* data, int array_size)
{
    MPI_Bcast(data, array_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void custom_bcast(void* data, int array_size)
{
    if (self == 0)
    {
        int i;
        for (i = 1; i < np; i++)
        {
            MPI_Send(data, array_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(data, array_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void custom_bcast_nonblocking(void* data, int array_size)
{
    if (self == 0)
    {
        int i;
        for (i = 1; i < np; i++)
        {
            MPI_Request request;
            MPI_Isend(data, array_size, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
        }
        /* TODO: maybe use MPI_Waitall here, but actually barrier should do the job */
    }
    else
    {
        MPI_Recv(data, array_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void tree_bcast(void* data, int array_size)
{
    int iteration = 0;
    if (self != 0)
    {
        iteration = (int)floor(log(self) / log(2));
        int sender = self % (int)pow(2, iteration);
        /*printf("%d receives in iteration %d\n", self, iteration);*/
        /* printf("%d receives from %d\n", self, sender); */
        MPI_Recv(data, array_size, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        iteration++;
    }
    
    for (; (int)pow(2, iteration) + self < np; iteration++)
    {
        int receiver = (int)pow(2, iteration) + self;
        /* printf("%d sends to %d\n", self, receiver); */
        MPI_Send(data, array_size, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    double start, end;
    int option;
    int array_size = 1048576, method = 0, num_trials = 100;

    /* MPI initialisieren und die Anzahl der Prozesse sowie
     * die eigene Prozessnummer bestimmen.
     */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    /* Beispiel fuer Kommandozeilen-Optionen mit getopt.
     * Ein nachgestellter Doppelpunkt signalisiert eine
     * Option mit Argument (in diesem Beispiel bei "-c").
     */
    while ((option = getopt(argc,argv,"s:m:t:")) != -1) {
        switch(option) {
        case 's': array_size = atoi(optarg); break;
        case 'm': method = atoi(optarg); break;
        case 't': num_trials = atoi(optarg); break;
        default:
            if (self == 0)
                fprintf(stderr, "Option error\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    if (!self)
        printf("array_size (-s): %d, method (-m): %d, num_trials (-t): %d\n",
            array_size, method, num_trials);
        
    void (*broadcastFunc)(void*, int);
    switch (method)
    {

        case 1:
            broadcastFunc = custom_bcast;
            break;
        case 2:
            broadcastFunc = custom_bcast_nonblocking;
            break;
        case 3:
            broadcastFunc = tree_bcast;
            break;
        case 0:
        default:
            broadcastFunc = native_bcast;
    }
        
    double total_time = 0.0;
    int i;
    int *data = allocints(array_size);
    for (i = 0; i < num_trials; i++)
    {
        /* hier geht's los... */
        MPI_Barrier(MPI_COMM_WORLD);
        total_time -= seconds();
        /* printf("rank = %d, size = %d\n", self, np); */
        broadcastFunc(data, array_size);
        MPI_Barrier(MPI_COMM_WORLD);
        total_time += seconds();
    }
    free(data);
    if (!self)
        printf("average time: %lf\n", total_time / num_trials);
    
    /* MPI beenden */
    MPI_Finalize();

    return 0;
}
