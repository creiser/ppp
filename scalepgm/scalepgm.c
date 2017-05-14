#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <omp.h>
#include <mpi.h>
#include "ppp_pnm.h"

int max(int a, int b)
{
    return a > b ? a : b;
}

int min(int a, int b)
{
    return a > b ? b : a;
}

static double seconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec) / 1000000.0;
}

void sequential(const char *file_name, int n_min, int n_max)
{
	double start, total_start;
	start = total_start = seconds();
	
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image = ppp_pnm_read(file_name, &kind,
        &rows, &columns, &maxcolor);
	printf("load: %f\n", seconds() - start);

	start = seconds();
    int i, a_min = INT_MAX, a_max = INT_MIN;
    for (i = 0; i < rows * columns; i++)
    {
		a_min = min(image[i], a_min);
		a_max = max(image[i], a_max);
    }
	printf("min-max: %f\n", seconds() - start);

	start = seconds();
	int n_diff = n_max - n_min;
	int a_diff = a_max - a_min;
	int half_a_diff = a_diff / 2;
    for (i = 0; i < rows * columns; i++)
    {
		image[i] = (((image[i] - a_min) * n_diff +
			half_a_diff) / a_diff) + n_min;
    }
	printf("scaling: %f\n", seconds() - start);
    
	start = seconds();
    if (ppp_pnm_write("out.pgm", kind, rows, columns, maxcolor, image) != 0)
		fprintf(stderr, "write error\n");
	free(image);
	printf("save: %f\n", seconds() - start);
	printf("total: %f\n", seconds() - total_start);
}

void parallel(const char *file_name, int n_min, int n_max)
{
	double start, total_start;
	start = total_start = seconds();
	
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image = ppp_pnm_read(file_name, &kind,
        &rows, &columns, &maxcolor);
	printf("load: %f\n", seconds() - start);
		
	start = seconds();
	int i, a_min, a_max;
	#pragma omp parallel for reduction(min:a_min) reduction(max:a_max)
	for (i = 0; i < rows * columns; i++)
    {
		a_min = min(image[i], a_min);
		a_max = max(image[i], a_max);
    }
    printf("a_min: %d, a_max: %d\n", a_min, a_max);
	printf("min-max: %f\n", seconds() - start);

	start = seconds();
	int n_diff = n_max - n_min;
	int a_diff = a_max - a_min;
	int half_a_diff = a_diff / 2;
	#pragma omp parallel for
    for (i = 0; i < rows * columns; i++)
    {
		image[i] = (((image[i] - a_min) * n_diff +
			half_a_diff) / a_diff) + n_min;
    }
	printf("scaling: %f\n", seconds() - start);
    
	start = seconds();
    if (ppp_pnm_write("out.pgm", kind, rows, columns, maxcolor, image) != 0)
		fprintf(stderr, "write error\n");
	free(image);
	printf("save: %f\n", seconds() - start);
	printf("total: %f\n", seconds() - total_start);
}

int self, np;
int myLength;

void get_offset_and_length(int rank, int rows, int columns,
    int *offset, int *length)
{
    /*
     * The number of pixels need not be a multiple of
     * np. Therefore, the first  (rows*columns)%np  processes get
     *    ceil((rows*columns)/np)
     * pixels, and the remaining processes get
     *    floor((rows*columns)/np)
     * pixels.
     */
    if (rank < (rows * columns) % np)
	{
	    *length = (rows * columns)/np + 1;
	    *offset = *length * rank;
    }
	else
	{
	    *length = (rows * columns) / np;
	    *offset = *length * rank + (rows * columns) % np;
    }
}

/*
 * Load a part of an image on the current processor
 */
uint8_t *partfn(enum pnm_kind kind,
                int rows, int columns,
                int *offset, int *length)
{
    if (kind != PNM_KIND_PGM)
        return NULL;

    get_offset_and_length(self, rows, columns, offset, length);
    myLength = *length;

    /*
     * Allocate space for the image part.
     * On processor 0 we allocate space for the whole
     * result image.
     */
    return (uint8_t*)malloc((self == 0 ? rows * columns : myLength)
                            * sizeof(uint8_t));
}

void distributed(const char *file_name, int n_min, int n_max)
{
	double start, total_start;
	start = total_start = seconds();
	
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *myPart = ppp_pnm_read_part(file_name, &kind, &rows,
        &columns, &maxcolor, partfn);
	if (self == 0)
		printf("load: %f\n", seconds() - start);
	
	start = seconds();
	int i, a_min, a_max;
	#pragma omp parallel for reduction(min:a_min) reduction(max:a_max)
	for (i = 0; i < myLength; i++)
    {
		a_min = min(myPart[i], a_min);
		a_max = max(myPart[i], a_max);
    }
	
	MPI_Allreduce(&a_min, &a_min, 1, MPI_UINT8_T, MPI_MIN,
				  MPI_COMM_WORLD);
    MPI_Allreduce(&a_max, &a_max, 1, MPI_UINT8_T, MPI_MAX,
				  MPI_COMM_WORLD);
	
	if (self == 0)
	{
		printf("min-max: %f\n", seconds() - start);
	}
	
	start = seconds();
	int n_diff = n_max - n_min;
	int a_diff = a_max - a_min;
	int half_a_diff = a_diff / 2;
	#pragma omp parallel for
    for (i = 0; i < myLength; i++)
    {
		myPart[i] = (((myPart[i] - a_min) * n_diff +
			half_a_diff) / a_diff) + n_min;
    }
	if (self == 0)
		printf("scaling: %f\n", seconds() - start);
	
	// Gather subarrays from all processes
	start = seconds();
	int *receive_counts = malloc(sizeof(int) * np);
	int *receive_displacements = malloc(sizeof(int) * np);
	for (i = 0; i < np; i++)
	{
        get_offset_and_length(i, rows, columns,
            &receive_displacements[i], &receive_counts[i]);
	}
	MPI_Gatherv(myPart, myLength, MPI_UINT8_T, myPart,
		receive_counts, receive_displacements, MPI_UINT8_T, 0, MPI_COMM_WORLD);
	free(receive_counts);
	free(receive_displacements);
	
	if (self == 0)
	{
		if (ppp_pnm_write("out.pgm", kind, rows, columns, maxcolor, myPart) != 0)
		    printf("write error\n");
	}
    free(myPart);
	MPI_Finalize();
	if (self == 0)
	{
		printf("save: %f\n", seconds() - start);
		printf("total: %f\n", seconds() - total_start);	
	}
}

#define SEQUENTIAL 0
#define PARALLEL 1
#define DISTRIBUTED 2

/*
 * Load a PGM (Portable Graymap) image and scale the gray values of every pixel
 * to a new interval.
 * The program is called with 4 arguments:
 *      input-image min-scale max-scale method
 * where method is an integer between 0 and 2:
 * 	    0: sequential, 1: parallel, 2: distributed
 */
int main(int argc, char **argv)
{
    int n_min = 50, n_max = 150, method = DISTRIBUTED;
    if (argc == 1)
    {
        printf("Usage: input-image min-scale max-scale method\n"
			"Method is an integer between 0 and 2\n\t"
			"0: sequential, 1: parallel, 2: distributed\n");
        exit(-1);
    }
    else if (argc >= 4)
    {
        n_min = atoi(argv[2]);
        n_max = atoi(argv[3]);
		if (argc == 5)
			method = atoi(argv[4]);
    }
    printf("n_min: %d, n_max: %d\n", n_min, n_max); 
	if (method == SEQUENTIAL)
	{
		printf("sequential\n");
		sequential(argv[1], n_min, n_max);
	}
	else if (method == PARALLEL)
	{
		printf("parallel\n");
		parallel(argv[1], n_min, n_max);
	}
    else
	{
		printf("distributed\n");
		distributed(argv[1], n_min, n_max);
	}
}
