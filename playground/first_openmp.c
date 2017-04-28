// ErstesOpenMP.cpp : Unabhaengige Parallelitaet in zweifachem Schleifensatz
//                    Version mit Codedistribuion einer Schleife

//#include "stdafx.h"
#include "omp.h"
#include "math.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
//#include "windows.h"

#define AUSSEN 100000
#define INNEN   1000

/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

double a[AUSSEN][INNEN];

int main(int argc, char **argv)
{	int i, j;
	double start = seconds();

	omp_set_num_threads(2);

#pragma omp parallel private(i)
  	for (int i = 0; i < AUSSEN; i++)
	{
#pragma omp for nowait	
	  for (int j = 0; j < INNEN; j++)
	  { a[i][j] = sin((double)((i+j)*(i+j)));
//		printf("Faden %d: a[%d][%d] = %.1f\n",
//	           omp_get_thread_num(), i, j, a[i][j]); 
	  }
	}
	
	printf("Laufzeit = %.5f\n", seconds() - start);

	return 0;
}
