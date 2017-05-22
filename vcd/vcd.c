#include <getopt.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h> 

#include "mpi.h"

#include "ppp_pnm.h"

inline double exp1(double x) {
  x = 1.0 + x / 256.0;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

double *convertByteToDouble(uint8_t *image, int rows, int columns, int maxcol)
{
	double *image_double = malloc(rows * columns * sizeof(double));
	for (int i = 0; i < rows * columns; i++)
	{
		image_double[i] = (double)image[i] / maxcol;
	}
	return image_double;
}

void convertDoubleToByte(double *image_double, uint8_t *image, int rows, int columns, int maxcol)
{
	for (int i = 0; i < rows * columns; i++)
	{
		if (image_double[i] < 0.0)
			image[i] = 0;
		else if (image_double[i] > 1.0)
			image[i] = maxcol;
		else
			image[i] = (image_double[i] * maxcol) + 0.5;
	}
}

static int N = 40;
static double epsilon = 0.005;
static double kappa = 30;
static double delta_t = 0.1;

inline static double phi(double nu)
{
	double chi = nu / kappa;
	return chi * exp1(-(chi * chi) / 2.0);
}

inline static double xi(double nu)
{
	double psi = nu / (M_SQRT2 * kappa);
	return M_SQRT1_2 * psi * exp1(-(psi * psi) / 2.0);
}

/*
 * Naive sequential VCD implementation.
 */
void vcdNaive(double *image, int rows, int columns) {
    inline double S(int c, int r)
    {
        return r >= 0 && r < rows &&
        	c >= 0 && c < columns ? image[r * columns + c] : 0;
    }
    
    inline double delta(int x, int y)
	{
		double d;
		d =  phi(S(x + 1, y) - S(x, y));
		d -= phi(S(x, y) - S(x - 1, y));
		d += phi(S(x, y + 1) - S(x, y));
		d -= phi(S(x, y) - S(x, y - 1));
		d += xi(S(x + 1, y + 1) - S(x, y));
		d -= xi(S(x, y) - S(x - 1 , y - 1));
		d += xi(S(x - 1, y + 1) - S(x, y));
		d -= xi(S(x, y) - S(x + 1, y - 1));
		return d;
	}
    
    /* Allocate a buffer to not overwrite pixels that are needed
       later again in their original version.
       
       Note that each pixel needs 4 of the surrounding pixels
       that otherwise would have been overwritten by a prior step.
       
       (left, up-left, up, up-right)
       */
	double *T = malloc(rows * columns * sizeof(double));
	
	printf("N: %d, kappa: %f, epsilon: %f, delta_t: %f\n", N, kappa, epsilon, delta_t);
	
	for (int i = 0; i < N; i++)
	{
		int epsilon_exit = 1;
		for (int y = 0; y < rows; ++y)
		{
			for (int x = 0; x < columns; ++x)
			{
				double delta_x_y = delta(x, y);
				T[y * columns + x] = S(x, y) + kappa * delta_t * delta_x_y;
				
				if (fabs(delta_x_y) > epsilon &&
						x >= 1 && x < columns - 1 &&
					    y >= 1 && y < rows - 1)
				{
					epsilon_exit = 0;
				}
			}
		}
		double *temp = T;
		T = image;
		image = temp;
		
		if (epsilon_exit)
			break;
	}
	/*free(T);*/
}

/*
 * Cache previous row and left value to reuse already calculated values.
 */
void vcdOptimized(double *image, int rows, int columns) {
    inline double S(int c, int r)
    {
        return r >= 0 && r < rows &&
        	c >= 0 && c < columns ? image[r * columns + c] : 0;
    }
    
    /* Allocate a buffer to not overwrite pixels that are needed
       later again in their original version.
       
       Note that each pixel needs 4 of the surrounding pixels
       that otherwise would have been overwritten by a prior step.
       
       (left, up-left, up, up-right)
       */
	double *T = malloc(rows * columns * sizeof(double));
	
	/* We can reuse up-left, up and up-right */
	double *up = malloc(columns * sizeof(double));
	double *up_left = malloc(columns * sizeof(double));
	double *up_right = malloc((columns + 1) * sizeof(double));
	
	printf("N: %d, kappa: %f, epsilon: %f, delta_t: %f\n", N, kappa, epsilon, delta_t);
	
	double delta_x_y;
	for (int i = 0; i < N; i++)
	{
		// Handle x = 0 as special case. Reason is explained below.
		up[0] = phi(S(0, 0));
		up_left[0] = xi(S(1, 0));
		// Fun fact: up_right[0] is never used and needn't be calculated therefore.
		for (int x = 1; x < columns; ++x)
		{
			up[x] = phi(S(x, 0)); // consider: S(x, -1) = 0
			up_left[x] = xi(S(x + 1, 0)); // consider: S(x, -1) = 0
			
			// We have
			// 	  up_left[x] = xi(S(x + 1, 0))
			// and
			// 	  up_right[x] = xi(S(x - 1, 0))
			// so
			//    up_right[x + 1] = xi(S(x - 1 + 1, 0)) = xi(S(x + 1 - 1, 0)) = up_left[x - 1]
			// We start the loop therefore at x = 1 and handle the case
			// x = 0 seperately
			up_right[x + 1] = up_left[x - 1];
		}
		// For the first row up_right[columns] will be calculated twice, but
		// thats only a tiny, tiny overhead
	
		int epsilon_exit = 1;
		for (int y = 0; y < rows; ++y)
		{
			double prev = phi(S(0, y)); // consider: S(-1, y) = 0
			double prev_up_left = xi(S(0, y)); // consider: S(-1, y - 1) = 0
			up_right[columns] = xi(S(columns - 1, y));
			for (int x = 0; x < columns; ++x)
			{
				delta_x_y = -prev;
				prev = phi(S(x + 1, y) - S(x, y));
				delta_x_y += prev;
				// Substitutes:
				// 	  delta_x_y =  phi(S(x + 1, y) - S(x, y));
				// 	  delta_x_y -= phi(S(x, y) - S(x - 1, y));
				
				delta_x_y -= up[x];
				up[x] = phi(S(x, y + 1) - S(x, y));
				delta_x_y += up[x];
				// Substitutes:
				// 	  delta_x_y += phi(S(x, y + 1) - S(x, y));
				// 	  delta_x_y -= phi(S(x, y) - S(x, y - 1));
				
				delta_x_y -= prev_up_left;
				prev_up_left = up_left[x];
				up_left[x] = xi(S(x + 1, y + 1) - S(x, y));
				delta_x_y += up_left[x];
				// Substitutes:
				// 	  delta_x_y += xi(S(x + 1, y + 1) - S(x, y));
				// 	  delta_x_y -= xi(S(x, y) - S(x - 1 , y - 1));
				
				delta_x_y -= up_right[x + 1];
				up_right[x] = xi(S(x - 1, y + 1) - S(x, y));
				delta_x_y += up_right[x];
				// Substitutes:
				// 	  delta_x_y += xi(S(x - 1, y + 1) - S(x, y));
				// 	  delta_x_y -= xi(S(x, y) - S(x + 1, y - 1));
				
				T[y * columns + x] = S(x, y) + kappa * delta_t * delta_x_y;
				
				if (fabs(delta_x_y) > epsilon &&
						x >= 1 && x < columns - 1 &&
					    y >= 1 && y < rows - 1)
				{
					epsilon_exit = 0;
				}
			}
		}
		double *temp = T;
		T = image;
		image = temp;
		
		if (epsilon_exit)
			break;
	}
	/*free(T);*/
}

/*
 * Parallel version that also uses already calculated values.
 */
void vcdOptimizedParallel(double *image, int rows, int columns) {
    inline double S(int c, int r)
    {
        return r >= 0 && r < rows &&
        	c >= 0 && c < columns ? image[r * columns + c] : 0;
    }
    
    /* Allocate a buffer to not overwrite pixels that are needed
       later again in their original version.
       
       Note that each pixel needs 4 of the surrounding pixels
       that otherwise would have been overwritten by a prior step.
       
       (left, up-left, up, up-right)
       */
	double *T = malloc(rows * columns * sizeof(double));
	
	printf("N: %d, kappa: %f, epsilon: %f, delta_t: %f\n", N, kappa, epsilon, delta_t);
	
	//printf("1\n");
	//omp_set_num_threads(4);
	
	for (int i = 0; i < N; i++)
	{
		// Share epsilon_exit among the threads
		int epsilon_exit = 1;
		
		
		//int test_order[] = {2,0,3,4,1,6,5,7};
		//for (int k = 0; k < 8; k++)
		#pragma omp parallel shared(epsilon_exit)
		{
			double delta_x_y;
			// To initalize the caching for each thread, we need to know about the assigned
			// "blocks" for each thread. Idea: manually assign subareas of the image to the
			// threads (MPI style).
			// To be safe just use the exact same code as we used for MPI.
			int num_threads = omp_get_num_threads();
			//int num_threads = 8;
			//printf("num_threads: %d\n", num_threads);
			int *counts = malloc(2 * num_threads * sizeof(int));
			int *displs = &counts[num_threads];
			displs[0] = 0;
			counts[0] = (rows / num_threads + (0 < rows % num_threads ? 1 : 0));
			for (int j = 1; j < num_threads; j++) {
				counts[j] = (rows / num_threads + (j < rows % num_threads ? 1 : 0));
				displs[j] = displs[j - 1] + counts[j - 1];
			}
		
			//int thread_num = test_order[k];
			int thread_num = omp_get_thread_num();
			int start = displs[thread_num];
			int end = start + counts[thread_num];
			
			//printf("i: %d, thread_num: %d, pixel[0]: %f\n", i, thread_num, image[0]);
			
			// We can reuse up-left, up and up-right
			// Every thread needs his own values
			double *up = malloc(columns * sizeof(double));
			double *up_left = malloc(columns * sizeof(double));
			double *up_right = malloc((columns + 1) * sizeof(double));
			
			// We cannot do the little up_right[x + 1] = up_left[x - 1]
			// trick from the sequential method here, since the second
			// term does not equal 0 in general.
			
			// Fun fact: up_right[0] is never used and needn't be calculated therefore.
			for (int x = 0; x < columns; ++x)
			{
				up[x] = phi(S(x, start) - S(x, start - 1)); // consider: S(x, -1) = 0
				up_left[x] = xi(S(x + 1, start) - S(x, start - 1)); // consider: S(x, -1) = 0
				up_right[x] = xi(S(x - 1, start) - S(x, start - 1));
			}
			
			//printf("2\n");
			//printf("start: %d, end: %d\n", start, end);
			
			// Nothing has to be changed here for the parallel version except of the
			// start and end of the loop
			for (int y = start; y < end; ++y)
			{
				double prev = phi(S(0, y)); // consider: S(-1, y) = 0
				double prev_up_left = xi(S(0, y)); // consider: S(-1, y - 1) = 0
				up_right[columns] = xi(S(columns - 1, y));
				for (int x = 0; x < columns; ++x)
				{
					delta_x_y = -prev;
					prev = phi(S(x + 1, y) - S(x, y));
					delta_x_y += prev;
					// Substitutes:
					// 	  delta_x_y =  phi(S(x + 1, y) - S(x, y));
					// 	  delta_x_y -= phi(S(x, y) - S(x - 1, y));
			
					delta_x_y -= up[x];
					up[x] = phi(S(x, y + 1) - S(x, y));
					delta_x_y += up[x];
					// Substitutes:
					// 	  delta_x_y += phi(S(x, y + 1) - S(x, y));
					// 	  delta_x_y -= phi(S(x, y) - S(x, y - 1));
			
					delta_x_y -= prev_up_left;
					prev_up_left = up_left[x];
					up_left[x] = xi(S(x + 1, y + 1) - S(x, y));
					delta_x_y += up_left[x];
					// Substitutes:
					// 	  delta_x_y += xi(S(x + 1, y + 1) - S(x, y));
					// 	  delta_x_y -= xi(S(x, y) - S(x - 1 , y - 1));
			
					delta_x_y -= up_right[x + 1];
					up_right[x] = xi(S(x - 1, y + 1) - S(x, y));
					delta_x_y += up_right[x];
					// Substitutes:
					// 	  delta_x_y += xi(S(x - 1, y + 1) - S(x, y));
					// 	  delta_x_y -= xi(S(x, y) - S(x + 1, y - 1));
			
					T[y * columns + x] = S(x, y) + kappa * delta_t * delta_x_y;
			
					if (fabs(delta_x_y) > epsilon &&
							x >= 1 && x < columns - 1 &&
							y >= 1 && y < rows - 1)
					{
						// epsilon_exit is never read inside the parallel area
						// so we don't have to introduce a critical or atomic section
						epsilon_exit = 0;
					}
				}
			}
			//printf("3\n");
			
			free(up);
			free(up_left);
			free(up_right);
		}
		double *temp = T;
		T = image;
		image = temp;
		
		printf("i: %d\n", i);
		
		if (epsilon_exit)
			break;
	}
	/*free(T);*/
}

int np, self;

/*
 * myPart (and myNewPart) has one row above and one row below the
 * actuall rows which are computed by the local process (as
 * VCD requires values from the neighbouring rows).
 */
uint8_t *myPart;
//uint8_t *myNewPart;
int myRows;

int *counts, *displs;

void vcdDistributed(double *myPartDouble, int rows, int columns) {
	// myPartDouble contains additional rows at the bottom and top
	// so let's alter the code to transparently deal with that.
	// The boundary checks for "r" are not nessecary anymore.
    inline double S(int c, int r)
    {
        return  c >= 0 && c < columns ? myPartDouble[(r + 1) * columns + c] : 0;
    }
    
    // Also reserve two additional rows for the swap buffer.
	double *T = malloc((myRows + 2) * columns * sizeof(double));
	
	printf("N: %d, kappa: %f, epsilon: %f, delta_t: %f\n", N, kappa, epsilon, delta_t);
	
	//printf("1\n");
	//omp_set_num_threads(1);
	
	int myRowOffset = displs[self] / columns;
	
	//fprintf(stderr, "self: %d, myRows: %d\n", self, myRows);
	
	MPI_Request topRequest, bottomRequest;
	MPI_Status dummyStatus;
	for (int i = 0; i < N; i++)
	{
		if (i != 0)
		{
			// Receive from top
			if (self != 0)
			{
				//fprintf(stderr, "(top) trying to receive: %d\n", self);
				MPI_Recv(myPartDouble, columns, MPI_DOUBLE, self - 1,
					0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Wait(&topRequest, &dummyStatus);
			}
				

			// Receive from bottom
			if (self != np - 1)
			{
				//fprintf(stderr, "(bot) trying to receive: %d\n", self);
				MPI_Recv(&myPartDouble[(myRows + 1) * columns], columns, MPI_DOUBLE, self + 1,
					0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Wait(&bottomRequest, &dummyStatus);
			}
		}
	
		// Share epsilon_exit among the threads
		int epsilon_exit = 1;
		
		//fprintf(stderr, "i: %d, self: %d, pixel[10000]: %f\n", i, self, myPartDouble[10000]);
		
		#pragma omp parallel shared(epsilon_exit)
		{
			//fprintf(stderr, "x1: %d\n", self);
		
			double delta_x_y;
			// To initalize the caching for each thread, we need to know about the assigned
			// "blocks" for each thread. Idea: manually assign subareas of the image to the
			// threads (MPI style).
			// To be safe just use the exact same code as we used for MPI.
			int num_threads = omp_get_num_threads();
			//printf("num_threads: %d\n", num_threads);
			int *countsPara = malloc(2 * num_threads * sizeof(int));
			int *displsPara = &countsPara[num_threads];
			displsPara[0] = 0;
			countsPara[0] = (myRows / num_threads + (0 < myRows % num_threads ? 1 : 0));
			for (int j = 1; j < num_threads; j++) {
				countsPara[j] = (myRows / num_threads + (j < myRows % num_threads ? 1 : 0));
				displsPara[j] = displsPara[j - 1] + countsPara[j - 1];
			}
			
			//fprintf(stderr, "x2: %d\n", self);
		
			int thread_num = omp_get_thread_num();
			int start = displsPara[thread_num];
			int end = start + countsPara[thread_num];
			
			fprintf(stderr, "self: %d, thread_num: %d\n", self, thread_num);
			
			//fprintf(stderr, "i: %d, thread_num: %d, pixel[0]: %f\n", i, thread_num, myPartDouble[0]);
			
			// We can reuse up-left, up and up-right
			// Every thread needs his own values
			double *up = malloc(columns * sizeof(double));
			double *up_left = malloc(columns * sizeof(double));
			double *up_right = malloc((columns + 1) * sizeof(double));
			
			// We cannot do the little up_right[x + 1] = up_left[x - 1]
			// trick from the sequential method here, since the second
			// term does not equal 0 in general.
			
			//fprintf(stderr, "x3: %d\n", self);
			
			// Fun fact: up_right[0] is never used and needn't be calculated therefore.
			for (int x = 0; x < columns; ++x)
			{
				up[x] = phi(S(x, start) - S(x, start - 1)); // consider: S(x, -1) = 0
				up_left[x] = xi(S(x + 1, start) - S(x, start - 1)); // consider: S(x, -1) = 0
				up_right[x] = xi(S(x - 1, start) - S(x, start - 1));
			}
			
			//printf("2\n");
			//printf("start: %d, end: %d\n", start, end);
			
			// Nothing has to be changed here for the parallel version except of the
			// start and end of the loop
			for (int y = start; y < end; ++y)
			{
				double prev = phi(S(0, y)); // consider: S(-1, y) = 0
				double prev_up_left = xi(S(0, y)); // consider: S(-1, y - 1) = 0
				up_right[columns] = xi(S(columns - 1, y));
				for (int x = 0; x < columns; ++x)
				{
					delta_x_y = -prev;
					prev = phi(S(x + 1, y) - S(x, y));
					delta_x_y += prev;
					// Substitutes:
					// 	  delta_x_y =  phi(S(x + 1, y) - S(x, y));
					// 	  delta_x_y -= phi(S(x, y) - S(x - 1, y));
			
					delta_x_y -= up[x];
					up[x] = phi(S(x, y + 1) - S(x, y));
					delta_x_y += up[x];
					// Substitutes:
					// 	  delta_x_y += phi(S(x, y + 1) - S(x, y));
					// 	  delta_x_y -= phi(S(x, y) - S(x, y - 1));
			
					delta_x_y -= prev_up_left;
					prev_up_left = up_left[x];
					up_left[x] = xi(S(x + 1, y + 1) - S(x, y));
					delta_x_y += up_left[x];
					// Substitutes:
					// 	  delta_x_y += xi(S(x + 1, y + 1) - S(x, y));
					// 	  delta_x_y -= xi(S(x, y) - S(x - 1 , y - 1));
			
					delta_x_y -= up_right[x + 1];
					up_right[x] = xi(S(x - 1, y + 1) - S(x, y));
					delta_x_y += up_right[x];
					// Substitutes:
					// 	  delta_x_y += xi(S(x - 1, y + 1) - S(x, y));
					// 	  delta_x_y -= xi(S(x, y) - S(x + 1, y - 1));
			
					// (y + 1) because T has also an additional row at the top
					T[(y + 1) * columns + x] = S(x, y) + kappa * delta_t * delta_x_y;
			
					// TODO: properly check for "inner" pixels
					if (fabs(delta_x_y) > epsilon &&
							x >= 1 && x < columns - 1 &&
							y + myRowOffset >= 1 && y + myRowOffset < rows - 1)
					{
						// epsilon_exit is never read inside the parallel area
						// so we don't have to introduce a critical or atomic section
						epsilon_exit = 0;
					}
				}
			}
			//printf("3\n");
			
			//fprintf(stderr, "x4: %d\n", self);
			
			free(up);
			free(up_left);
			free(up_right);
		}
		
		//fprintf(stderr, "trying to send: %d\n", self);
		
		// We send the first "real" row respectively the last "real" row,
		// i.e. rows that were calculated by this process
		// Send to top
		if (self != 0)
			MPI_Isend(&T[columns], columns, MPI_DOUBLE, self - 1,
				0, MPI_COMM_WORLD, &topRequest);
		// Send to bottom
		if (self != np - 1)
			MPI_Isend(&T[myRows * columns], columns, MPI_DOUBLE, self + 1,
				0, MPI_COMM_WORLD, &bottomRequest);
				
		//fprintf(stderr, "performed sends: %d\n", self);
		
		//fprintf(stderr, "i: %d, self: %d, epsilon_exit: %d\n", i, self, epsilon_exit);
		
		int global_epsilon_exit;
		MPI_Allreduce(&epsilon_exit, &global_epsilon_exit, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
		
		//fprintf(stderr, "i: %d, self: %d, global_epsilon_exit: %d\n", i, self, global_epsilon_exit);
		
		//fprintf(stderr, "reduced: %d\n", self); 
		
		// possibily buggy because MPI implementation doesn't use its own
		// send buffer
		double *temp = T;
		T = myPartDouble;
		myPartDouble = temp;
		
		fprintf(stderr, "i: %d\n", i);
		
		if (global_epsilon_exit)
			break;
	}
	/*free(T);*/
}

/*
 * Out of memory handler.
 */
void Oom(void) {
    fprintf(stderr, "Out of memory on processor %d\n", self);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/*
 * Put zeros in the first row in process 0 and
 * in the last row in process np-1.
 */
void prepare_myPart(int columns) {
    int x;
    if (self == 0) {
	for (x=0; x<columns; x++)
	    myPart[x] = 0;
    }
    if (self == np-1) {
	int mr1 = myRows+1;
	for (x=0; x<columns; x++)
	    myPart[mr1 * columns + x] = 0;
    }
}


/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

/*
 * Collect the pieces of the output image.
 */
void collect(int columns) {
	for (int i = 0; i < np; i++)
		printf("self: %d, counts[%d]: %d, displs[%d]: %d\n", self, i, counts[i], i, displs[i]);

    //void *sendbuf = self == 0 ? MPI_IN_PLACE : myPart;
    void *sendbuf = myPart;
    MPI_Gatherv(sendbuf, counts[self], MPI_UINT8_T,
                myPart, counts, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD);
}


/*
 * Callback function for ppp_pnm_load.
 * We determine the part of the image to be processed
 * by the local process and load one additional row
 * above and below this part as the Sobel operator needs
 * data from these two additional rows for its computations.
 */
uint8_t *partFn(enum pnm_kind kind, int rows, int columns,
                int *offset, int *length)
{
    int i;

    if (kind != PNM_KIND_PGM)
	return NULL;

    if (rows < np) {
	if (self == 0)
	    fprintf(stderr, "Cannot run with fewer rows in the image "
		    "than processors.\n");
	return NULL;
    }

    counts = malloc(2 * np * sizeof(int));
    if (counts == NULL)
	Oom();
    displs = &counts[np];

    /*
     * The number of rows need not be a multiple of
     * np. Therefore, the first  rows%np  processes get
     *    ceil(rows/np)
     * rows, and the remaining processes get
     *    floor(rows/np)
     * rows.
     */
    displs[0] = 0;
    counts[0] = (rows/np + (0 < rows%np ? 1 : 0)) * columns;
    for (i=1; i<np; i++) {
	counts[i] = (rows/np + (i < rows%np ? 1 : 0)) * columns;
	displs[i] = displs[i-1] + counts[i-1];
    }

    myRows = counts[self] / columns;

    /*
     * myPart has two additional rows, one at the top, one
     * at the bottom of the local part of image.
     */
    myPart = malloc((self == 0 ? rows : (myRows + 2)) * columns * sizeof(uint8_t));
    if (myPart == NULL) {
	free(displs);
	Oom();
    }

    /*
     * Space for the result image part without additional
     * rows at the top and bottom.
     * On processor 0, we reserve space for the whole image
     * so we can collect the parts with MPI_Gatherv into
     * this space.
     */
    /*myNewPart = malloc((self == 0 ? rows : myRows) * columns * sizeof(int));
    if (myNewPart == NULL) {
	free(displs);
	free(myPart);
	Oom();
    }*/

    /*
     * Offset and length of the part of the image to load
     * in the local process, including the additional
     * row at the top and/or the bottom.
     */
    *offset = self == 0 ? 0 : (displs[self]-columns);
    if (np == 1)
	*length = myRows * columns;
    else
	*length = (myRows + (self == 0 || self == np-1 ? 1 : 2)) * columns;

    /* Add zeros in top row in process 0 and in bottom row in process np-1. */
    prepare_myPart(columns);

    return (self == 0 ? &myPart[columns] : myPart);
}

int main(int argc, char* argv[]) {
	bool execute_vcd  = false;
	bool fast_vcd = false;
	bool execute_sobel = false;
	int option, rows, cols, maxval;
	enum pnm_kind kind;
	char* output_file = "out.pgm";
	char* input_file;
	uint8_t* picture;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    while ((option = getopt(argc,argv,"hvfso:")) != -1) {
        switch(option) {
        	case 'h':
        		printf("[-hvfs][-o name_of_output_file] name_of_input_file\nThis program takes a picture in pgm format and processes it based on the given options.\nUse \"-h\" to display this description.\nUse \"-v\" to let the picture be manipulated by the vcd algortihm.\nUse \"-f\" if you want the program to use the faster version of the vcd algortihm if \"-v\" is set, too.\nUse \"-s\" to let the picture be manipulated by the sobel algortihm.\nIf both the \"-v\" and the \"-s\" options are set, the vcd algorithm will be executed first.\nUse \"-o\" to specify the file the processed image should be saved to. The default option ist \"out.pgm\".\nThe input file has to be given as the last argument.\n");
        		return 0;
        	break;
        	case 'v':
        		execute_vcd = true;
        	break;
        	case 'f':
        		fast_vcd = true;
        	break;
        	case 's':
        		execute_sobel = true;
        	break;
        	case 'o':
        		output_file = optarg;
        	break;
        	default:
        	break;
        }
    }
    input_file = argv[argc - 1];
    
    if(!strcmp(output_file, input_file)) {
		bool abort_program = true;
		bool waiting_for_answer = true;
		char answer;
		while(waiting_for_answer) {
			printf("Warning: The input file is the same as the output file. Continuing will overwrite the input file permanently!\nContinue (y/n)? ");
			scanf(" %c", &answer);
			if(answer == 'y') {
				abort_program = false;
				waiting_for_answer = false;
			}
			if(answer == 'n') {
				waiting_for_answer = false;
			}
		}
		if(abort_program) {
			printf("Program terminated.\n");
			return 0;
		}
	}

	//picture = ppp_pnm_read(input_file, &kind, &rows, &cols, &maxval);
	picture = ppp_pnm_read_part(input_file, &kind, &rows, &cols, &maxval,
			      partFn);
	if(picture == NULL) {
		fprintf(stderr, "An error occured when trying to load the picture from the file \"%s\"! If this is not the input file you had in mind please note that it has to be specified as the last argument.\n", input_file);
		return 1;
	}
	
	double *myPartDouble = convertByteToDouble(myPart, myRows + 2, cols, maxval);
	
	double vcdStart = seconds();
	//vcdNaive(image, rows, cols);
	//vcdOptimized(image, rows, cols);
	//vcdOptimizedParallel(image, rows, cols);
	vcdDistributed(myPartDouble, rows, cols);
	fprintf(stderr, "vcd time: %f\n", seconds() - vcdStart);
	
	convertDoubleToByte(&myPartDouble[cols], myPart, myRows, cols, maxval);
	
	char debug_file[32];
	snprintf (debug_file, 32, "out_%d.pgm", self);
	ppp_pnm_write(debug_file, kind, myRows, cols, maxval, myPart);
	
	//fprintf(stderr, "z1\n");
	// free(myPartDouble);
	// free(image);
	
	collect(cols);
	
	//fprintf(stderr, "z2\n");

	if (self == 0 && ppp_pnm_write(output_file, kind, rows, cols, maxval, myPart) != 0) {
		fprintf(stderr, "An error occured when trying to write the processed picture to the output file!\n");
		return 2;
	}
	//free(picture);
	
	MPI_Finalize();
	
	return 0;
}
