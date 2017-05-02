#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
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

void sequential(const char *file_name, int n_min, int n_max)
{
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image = ppp_pnm_read(file_name, &kind,
        &rows, &columns, &maxcolor);

    int i;
    int a_min = INT_MAX, a_max = INT_MIN;
    for (i = 0; i < rows * columns; i++)
    {
		a_min = min(image[i], a_min);
		a_max = max(image[i], a_max);
    }
    printf("a_min: %d, a_max: %d\n", a_min, a_max);

    for (i = 0; i < rows * columns; i++)
    {
		image[i] = (((image[i] - a_min) * (n_max - n_min) +
			(a_max - a_min) / 2) / (a_max - a_min)) + n_min;
    }
    
    if (ppp_pnm_write("out.pgm", kind, rows, columns, maxcolor, image) != 0)
		fprintf(stderr, "write error\n");
}

void parallel(const char *file_name, int n_min, int n_max)
{
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image = ppp_pnm_read(file_name, &kind,
        &rows, &columns, &maxcolor);

	int i, a_min = INT_MAX, a_max = INT_MIN, local_min, local_max;
	#pragma omp parallel private(i, local_min, local_max)
    {
        local_min = INT_MAX;
        local_max = INT_MIN;
    
        #pragma omp for nowait
        for (i = 0; i < rows * columns; i++)
        {
			local_min = min(image[i], local_min);
			local_max = max(image[i], local_max);
        }
        
        printf("thread: %d, local_min: %d, local_max: %d\n",
            omp_get_thread_num(), local_min, local_max);
        
        #pragma omp critical
        {
            a_min = min(local_min, a_min);
            a_max = max(local_max, a_max);
        }
    }
    printf("a_min: %d, a_max: %d\n", a_min, a_max);

	#pragma omp parallel for
    for (i = 0; i < rows * columns; i++)
    {
		image[i] = (((image[i] - a_min) * (n_max - n_min) +
			(a_max - a_min) / 2) / (a_max - a_min)) + n_min;
    }
    
    if (ppp_pnm_write("out.pgm", kind, rows, columns, maxcolor, image) != 0)
		fprintf(stderr, "write error\n");
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
    if (rank < (rows * columns) % np) {
	    *length = (rows * columns)/np + 1;
	    *offset = *length * rank;
    } else {
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

    printf("self: %d, offset: %d, length: %d\n", self, *offset, *length);

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
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *myPart = ppp_pnm_read_part(file_name, &kind, &rows,
        &columns, &maxcolor, partfn);
	
    printf("rows: %d, columns: %d\n", rows, columns);
	printf("self: %d, myLength: %d\n", self, myLength);
	
	int i, a_min = INT_MAX, a_max = INT_MIN, local_min, local_max;
	#pragma omp parallel private(i, local_min, local_max)
    {
        local_min = INT_MAX;
        local_max = INT_MIN;
    
        #pragma omp for nowait
        for (i = 0; i < myLength; i++)
        {
			local_min = min(myPart[i], local_min);
			local_max = max(myPart[i], local_max);
        }
        
        printf("process: %d, thread: %d, local_min: %d, local_max: %d\n", self,
            omp_get_thread_num(), local_min, local_max);
        
        #pragma omp critical
        {
            a_min = min(local_min, a_min);
            a_max = max(local_max, a_max);
        }
    }
    printf("process: %d, a_min: %d, a_max: %d\n", self, a_min, a_max);
	
	MPI_Allreduce(&a_min, &a_min, 1, MPI_UINT8_T, MPI_MIN,
				  MPI_COMM_WORLD);
    MPI_Allreduce(&a_max, &a_max, 1, MPI_UINT8_T, MPI_MAX,
				  MPI_COMM_WORLD);
	
	printf("(global) process: %d, a_min: %d, a_max: %d\n", self, a_min, a_max);

	#pragma omp parallel for
    for (i = 0; i < myLength; i++)
    {
		myPart[i] = (((myPart[i] - a_min) * (n_max - n_min) +
			(a_max - a_min) / 2) / (a_max - a_min)) + n_min;
    }
	
	// Gather subarrays from all processes
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
	
	if (self == 0) {
		if (ppp_pnm_write("out.pgm", kind, rows, columns, maxcolor, myPart) != 0)
		    printf("write error\n");
	}
    free(myPart);
	
	MPI_Finalize();
}

int main(int argc, char **argv)
{
    int n_min = 50, n_max = 150;
    if (argc == 1)
    {
        printf("You have to specify a file name.\n");
        exit(-1);
    }
    else if (argc == 4)
    {
        n_min = atoi(argv[2]);
        n_max = atoi(argv[3]);
    }
    printf("n_min: %d, n_max: %d\n", n_min, n_max); 

    //sequential(argv[1], n_min, n_max);
    //parallel(argv[1], n_min, n_max);
	distributed(argv[1], n_min, n_max);
}
