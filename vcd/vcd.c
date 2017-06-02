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

/*
 * Fast approximation of the exponential function.
 * It is accurate enough for input values between 0.0 and 1.0
 * to produce actually the same image as the one obtained with
 * the libc exp() function.
 */
inline double exp1(double x)
{
	x = 1.0 + x / 256.0;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}

/*
 * For the distributed implementation
 * myPart (and myPartDouble) has one row above and one row below
 * the actual rows which are computed by the local process (as
 * VCD requires values from the neighbouring rows).
 */
uint8_t *myPart;
double *myPartDouble;

void convertByteToDouble(int rows, int columns, int maxcol)
{
	myPartDouble = malloc(rows * columns * sizeof(double));
	#pragma omp parallel for
	for (int i = 0; i < rows * columns; i++)
	{
		myPartDouble[i] = (double)myPart[i] / maxcol;
	}
}

void convertDoubleToByte(int rows, int columns, int maxcol, int offset)
{
	double *start = &myPartDouble[offset];
	#pragma omp parallel for
	for (int i = 0; i < rows * columns; i++)
	{
		if (start[i] < 0.0)
			myPart[i] = 0;
		else if (start[i] > 1.0)
			myPart[i] = maxcol;
		else
			myPart[i] = (start[i] * maxcol) + 0.5;
	}
	free(myPartDouble);
}

static int N = 40;
static double epsilon = 0.005;
static double kappa = 30;
static double delta_t = 0.1;

inline static double phi(double nu)
{
	double chi = nu / kappa;
	return chi * exp1(-(chi * chi) * 0.5);
}

inline static double xi(double nu)
{
	double psi = nu / kappa;
	return psi * 0.5 * exp1(-(psi * psi) * 0.25);
}

/*
 * Naive sequential VCD implementation.
 */
void vcdNaive(int rows, int columns) {
    inline double S(int c, int r)
    {
        return r >= 0 && r < rows &&
        	c >= 0 && c < columns ? myPartDouble[r * columns + c] : 0;
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
		T = myPartDouble;
		myPartDouble = temp;
		
		if (epsilon_exit)
			break;
	}
	free(T);
}

/*
 * Cache previous row and left value to reuse already calculated values.
 */
void vcdOptimized(int rows, int columns) {
    inline double S(int c, int r)
    {
        return r >= 0 && r < rows &&
        	c >= 0 && c < columns ? myPartDouble[r * columns + c] : 0;
    }
    
	double *T = malloc(rows * columns * sizeof(double));
	
	/* We can reuse up-left, up and up-right */
	double *up = malloc(columns * sizeof(double));
	double *up_left = malloc(columns * sizeof(double));
	double *up_right = malloc((columns + 1) * sizeof(double));
	
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
				// left
				delta_x_y = -prev;
				prev = phi(S(x + 1, y) - S(x, y));
				delta_x_y += prev;
				
				// up
				delta_x_y -= up[x];
				up[x] = phi(S(x, y + 1) - S(x, y));
				delta_x_y += up[x];
				
				// up left
				delta_x_y -= prev_up_left;
				prev_up_left = up_left[x];
				up_left[x] = xi(S(x + 1, y + 1) - S(x, y));
				delta_x_y += up_left[x];
				
				// up right
				delta_x_y -= up_right[x + 1];
				up_right[x] = xi(S(x - 1, y + 1) - S(x, y));
				delta_x_y += up_right[x];
				
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
		T = myPartDouble;
		myPartDouble = temp;
		
		if (epsilon_exit)
			break;
	}
	free(T);
}

/*
 * Parallel version that also uses already calculated values.
 */
void vcdOptimizedParallel(int rows, int columns) {
    inline double S(int c, int r)
    {
        return r >= 0 && r < rows &&
        	c >= 0 && c < columns ? myPartDouble[r * columns + c] : 0;
    }
    
	double *T = malloc(rows * columns * sizeof(double));
	
	// To initalize the caching for each thread, we need to know about the assigned
	// "blocks" for each thread. Idea: manually assign subareas of the image to the
	// threads (MPI style).
	// To be safe just use the exact same code as we used for MPI.
	int num_threads;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	int *counts = malloc(2 * num_threads * sizeof(int));
	int *displs = &counts[num_threads];
	displs[0] = 0;
	counts[0] = (rows / num_threads + (0 < rows % num_threads ? 1 : 0));
	for (int j = 1; j < num_threads; j++) {
		counts[j] = (rows / num_threads + (j < rows % num_threads ? 1 : 0));
		displs[j] = displs[j - 1] + counts[j - 1];
	}
	
	// We can reuse up-left, up and up-right
	// Every thread needs his own values.
	// Allocate memory for each thread seperately.
	double *all_up = malloc(columns * sizeof(double) * num_threads);
	double *all_up_left = malloc(columns * sizeof(double) * num_threads);
	double *all_up_right = malloc((columns + 1) * sizeof(double) * num_threads);
	
	for (int i = 0; i < N; i++)
	{
		// Share epsilon_exit among the threads
		int epsilon_exit = 1;
		#pragma omp parallel shared(epsilon_exit)
		{
			double delta_x_y;
			
			int thread_num = omp_get_thread_num();
			int start = displs[thread_num];
			int end = start + counts[thread_num];
			
			// Assign each thread its own cache.
			double *up = &all_up[thread_num * columns];
			double *up_left = &all_up_left[thread_num * columns];
			double *up_right = &all_up_right[thread_num * (columns + 1)];
			
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
			
					delta_x_y -= up[x];
					up[x] = phi(S(x, y + 1) - S(x, y));
					delta_x_y += up[x];
			
					delta_x_y -= prev_up_left;
					prev_up_left = up_left[x];
					up_left[x] = xi(S(x + 1, y + 1) - S(x, y));
					delta_x_y += up_left[x];
			
					delta_x_y -= up_right[x + 1];
					up_right[x] = xi(S(x - 1, y + 1) - S(x, y));
					delta_x_y += up_right[x];
			
					T[y * columns + x] = S(x, y) + kappa * delta_t * delta_x_y;
			
					if (fabs(delta_x_y) > epsilon &&
							x >= 1 && x < columns - 1 &&
							y >= 1 && y < rows - 1)
					{
						// epsilon_exit is never read inside the parallel area
						// so we don't have to introduce a critical or atomic section
						// It doesn't matter in which order the 0's are written to
						// the variable.
						epsilon_exit = 0;
					}
				}
			}			
		}
		double *temp = T;
		T = myPartDouble;
		myPartDouble = temp;
		
		if (epsilon_exit)
			break;
	}
	
	free(all_up);
	free(all_up_left);
	free(all_up_right);
	free(counts);
	free(T);
}


int np, self;
int myRows;
int *counts, *displs;
bool execute_sobel = false;
double sobelC = 0.9;

/*
 * New pixel value according to Sobel for given values of sx and sy.
 */
inline static double T_sobel(double sx, double sy)
{
    double v = sobelC * hypot(sx,sy);
    return v > 1.0 ? 1.0 : v;
}

/*
 * Sobel with unrolling of the first and last iteration
 * of the loop on x to avoid the case distinctions
 * in the innermost loop.
 */
void sobel(const int columns, double *T)
{
    double (*image)[columns] = (double (*)[columns]) &myPartDouble[columns];
    inline double S(int c, int r) { return image[r][c]; }

	#pragma omp parallel for
    for (int y = 0; y < myRows; ++y)
    {
        double sx, sy;

		// x == 0
		sx = 2*S(0,y-1) + S(1,y-1) - 2*S(0,y+1) - S(1,y+1);
		sy = - S(1,y-1) - 2*S(1,y) - S(1,y+1);
		T[(y + 1) * columns] = T_sobel(sx,sy);

		for (int x = 1; x < columns - 1; ++x)
		{
			sx = S(x-1,y-1) + 2*S(x,y-1) + S(x+1,y-1)
			-S(x-1,y+1) - 2*S(x,y+1) - S(x+1,y+1);
			sy = S(x-1,y-1) + 2*S(x-1,y) + S(x-1,y+1)
			    -S(x+1,y-1) - 2*S(x+1,y) - S(x+1,y+1);
			T[(y + 1) * columns + x] = T_sobel(sx,sy);
		}

		// x == columns-1
		sx = S(columns-2,y-1) + 2*S(columns-1,y-1)
			-S(columns-2,y+1) - 2*S(columns-1,y+1);
		sy = S(columns-2,y-1) + 2*S(columns-2,y) + S(columns-2,y+1);
		T[(y + 1) * columns + columns - 1] = T_sobel(sx,sy);
    }
}

/*
 * Put zeros in the first row in process 0 and
 * in the last row in process np-1.
 */
void prepare_myPart(double *buffer, int columns) {
    if (self == 0)
    {
		for (int x = 0; x < columns; x++)
			buffer[x] = 0.0;
    }
    if (self == np-1)
    {
		int mr1 = myRows+1;
		for (int x = 0; x < columns; x++)
			buffer[mr1 * columns + x] = 0.0;
    }
}

void vcdDistributed(int rows, int columns) {
	// myPartDouble contains additional rows at the bottom and top
	// so let's alter the code to transparently deal with that.
	// The boundary checks for "r" are therefore not nessecary anymore.
    inline double S(int c, int r)
    {
        return  c >= 0 && c < columns ? myPartDouble[(r + 1) * columns + c] : 0;
    }
    
    // Also reserve two additional rows for the swap buffer.
	double *T = malloc((myRows + 2) * columns * sizeof(double));
	
	// Add zeros in top row in process 0 and in bottom row in process np-1.
	// The first/last process also has to write zeros to the first row/last row
	// of T, otherwise we read from uninitalized memory after the swap.
	prepare_myPart(myPartDouble, columns);
	prepare_myPart(T, columns);
	
	int myRowOffset = displs[self] / columns;
	
	// To initalize the caching for each thread, we need to know about the assigned
	// "blocks" for each thread. Idea: manually assign subareas of the image to the
	// threads (MPI style).
	// To be safe just use the exact same code as we used for MPI.
	int num_threads;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	int *countsPara = malloc(2 * num_threads * sizeof(int));
	int *displsPara = &countsPara[num_threads];
	displsPara[0] = 0;
	countsPara[0] = (myRows / num_threads + (0 < myRows % num_threads ? 1 : 0));
	for (int j = 1; j < num_threads; j++) {
		countsPara[j] = (myRows / num_threads + (j < myRows % num_threads ? 1 : 0));
		displsPara[j] = displsPara[j - 1] + countsPara[j - 1];
	}
	double *all_up = malloc(columns * sizeof(double) * num_threads);
	double *all_up_left = malloc(columns * sizeof(double) * num_threads);
	double *all_up_right = malloc((columns + 1) * sizeof(double) * num_threads);
	
	MPI_Request topRequest, bottomRequest;
	MPI_Status dummyStatus;
	for (int i = 0; i < N; i++)
	{
		// Share epsilon_exit among the threads
		int epsilon_exit = 1;
		#pragma omp parallel shared(epsilon_exit)
		{
			double delta_x_y;
			
			int thread_num = omp_get_thread_num();
			int start = displsPara[thread_num];
			int end = start + countsPara[thread_num];
			
			if (!self && !thread_num && !i)
				fprintf(stderr, "num threads: %d, assigned rows: %d/%d\n",
					num_threads,
					end - start,
					rows);
			
			double *up = &all_up[thread_num * columns];
			double *up_left = &all_up_left[thread_num * columns];
			double *up_right = &all_up_right[thread_num * (columns + 1)];
			
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
			
			// Nothing has to be changed here for the parallel version except of the
			// start and end of the loop
			for (int y = start; y < end; ++y)
			{
				double prev = phi(S(0, y)); // consider: S(-1, y) = 0
				double prev_up_left = xi(S(0, y)); // consider: S(-1, y - 1) = 0
				up_right[columns] = xi(S(columns - 1, y));
				for (int x = 0; x < columns; ++x)
				{
					// left
					delta_x_y = -prev;
					prev = phi(S(x + 1, y) - S(x, y));
					delta_x_y += prev;
			
					// up
					delta_x_y -= up[x];
					up[x] = phi(S(x, y + 1) - S(x, y));
					delta_x_y += up[x];
			
					// up left
					delta_x_y -= prev_up_left;
					prev_up_left = up_left[x];
					up_left[x] = xi(S(x + 1, y + 1) - S(x, y));
					delta_x_y += up_left[x];
			
					// up right
					delta_x_y -= up_right[x + 1];
					up_right[x] = xi(S(x - 1, y + 1) - S(x, y));
					delta_x_y += up_right[x];
			
					// This is a novelty of the distributed version:
					// (y + 1) because T has also an additional row at the top
					T[(y + 1) * columns + x] = S(x, y) + kappa * delta_t * delta_x_y;
			
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
		}
		
		// If no process has made a change greater epsilon, i.e. epsilon_exit is 1 for all
		// processes, then we early terminate.		
		int global_epsilon_exit;
		MPI_Allreduce(&epsilon_exit, &global_epsilon_exit, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
		
		if (!global_epsilon_exit)
		{
			// We send the first "real" row respectively the last "real" row,
			// i.e. rows that were calculated by this process
			// Note that we can only receive and send simultaenously without
			// an additional buffer, because rows are send and received from/to different
			// memory loations.
			// The outmost rows (the additional rows) are received, while the
			// row below respectively above the outmost rows are sent.
		
			// Send to top
			if (self != 0)
				MPI_Isend(&T[columns], columns, MPI_DOUBLE, self - 1,
					0, MPI_COMM_WORLD, &topRequest);
			// Send to bottom
			if (self != np - 1)
				MPI_Isend(&T[myRows * columns], columns, MPI_DOUBLE, self + 1,
					0, MPI_COMM_WORLD, &bottomRequest);
		}
		
		double *temp = T;
		T = myPartDouble;
		myPartDouble = temp;
		
		if (!global_epsilon_exit)
		{
			// Receive from top
			if (self != 0)
			{
				MPI_Recv(myPartDouble, columns, MPI_DOUBLE, self - 1,
					0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Wait(&topRequest, &dummyStatus);
			}
				
			// Receive from bottom
			if (self != np - 1)
			{
				MPI_Recv(&myPartDouble[(myRows + 1) * columns], columns, MPI_DOUBLE, self + 1,
					0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Wait(&bottomRequest, &dummyStatus);
			}
		}
		else
			break;
	}
	
	if (execute_sobel)
	{
		sobel(columns, T);
		double *temp = T;
		T = myPartDouble;
		myPartDouble = temp;
	}
	
	free(all_up);
	free(all_up_left);
	free(all_up_right);
	free(countsPara);
	free(T);		
}

/*
 * Out of memory handler.
 */
void Oom(void) {
    fprintf(stderr, "Out of memory on processor %d\n", self);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

/*
 * Collect the pieces of the output image.
 */
void collect(int columns)
{
    void *sendbuf = self == 0 ? MPI_IN_PLACE : myPart;
    MPI_Gatherv(sendbuf, counts[self], MPI_UINT8_T,
                myPart, counts, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    free(counts);
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

    if (rows < np)
    {
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
    for (i = 1; i < np; i++)
    {
		counts[i] = (rows/np + (i < rows%np ? 1 : 0)) * columns;
		displs[i] = displs[i-1] + counts[i-1];
    }

    myRows = counts[self] / columns;

    /*
     * myPart has two additional rows, one at the top, one
     * at the bottom of the local part of image.
     */
    myPart = malloc((self == 0 ? rows : (myRows + 2)) * columns * sizeof(uint8_t));
    if (myPart == NULL)
    {
		free(displs);
		Oom();
    }

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

    return (self == 0 ? &myPart[columns] : myPart);
}

void usage()
{
	fprintf(stderr,
		"[-hs][-m implementation][-t number_of_omp_threads][-o name_of_output_file] name_of_input_file\n"
		"This program takes a picture in pgm format and executes the VCD algorithm on it based on the given options.\n"
		"With the \"-m\" option the implementation can be specified with an integer.\n"
		"Possible values are 0: naive, 1: optimized, 2: parallel and 3: distributed\n" 
		"Use \"-h\" to display this description.\n"
		"Use \"-s\" to let the picture be additonally manipulated by the sobel algortihm.\n"
		"Use \"-o\" to specify the file the processed image should be saved to. The default setting is \"out.pgm\".\n"
		"The input file has to be given as the last argument.\n");
}

#define VCD_NAIVE 0
#define VCD_OPTIMIZED 1
#define VCD_OPTIMIZED_PARALLEL 2
#define VCD_DISTRIBUTED 3

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &self);

	int option, rows, cols, maxval;
	enum pnm_kind kind;
	char* output_file = "out.pgm";
	char* input_file;
	uint8_t* picture;
	int implementation = VCD_OPTIMIZED_PARALLEL;
	
    while ((option = getopt(argc,argv,"hso:m:t:")) != -1) {
        switch(option) {
        	case 'h':
        		usage();
        		return 0;
        		break;
        	case 's':
        		execute_sobel = true;
        		break;
        	case 'o':
        		output_file = optarg;
    			break;
    		case 'm':
    			implementation = atoi(optarg);
    			break;
			case 't':
				omp_set_num_threads(atoi(optarg));
        		break;
        	default:
        		break;
        }
    }
    if (argv[optind] == NULL)
    {
		usage();
		return 1;
	}
    input_file = argv[argc - 1];
    
    if (execute_sobel && implementation != VCD_DISTRIBUTED)
    	fprintf(stderr, "Sobel will only be executed when the distributed implementation "
    		"with \"-m 3\" is selected.\n");
    
    if(!strcmp(output_file, input_file)) {
		bool abort_program = true;
		bool waiting_for_answer = true;
		char answer;
		while(waiting_for_answer) {
			fprintf(stderr, "Warning: The input file is the same as the output file. Continuing "
				"will overwrite the input file permanently!\nContinue (y/n)? ");
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
	
	double start = seconds();
	if (implementation == VCD_DISTRIBUTED)
	{
		if (np != 1)
			picture = ppp_pnm_read_part(input_file, &kind, &rows, &cols, &maxval,
					  partFn);
		else
		{
			fprintf(stderr, "Aborting, since there is only one process. Use the parallel version "
				"instead.\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
	}
	else
	{
		self = 0;
		picture = ppp_pnm_read(input_file, &kind, &rows, &cols, &maxval);
		myPart = picture;
	}
	if (picture == NULL)
	{
		fprintf(stderr, "An error occured when trying to load the picture from the file \"%s\"!\n"
			"If this is not the input file you had in mind please note that it has to be "
			"specified as the last argument.\n", input_file);
		return 1;
	}
	double loadTime = seconds() - start;
	
	start = seconds();
	convertByteToDouble(implementation == VCD_DISTRIBUTED ? myRows + 2 : rows, cols, maxval);
	double convertTime = seconds() - start;
	
	start = seconds();
	if (implementation == VCD_NAIVE)
		vcdNaive(rows, cols);
	else if (implementation == VCD_OPTIMIZED)
		vcdOptimized(rows, cols);
	else if (implementation == VCD_OPTIMIZED_PARALLEL)
		vcdOptimizedParallel(rows, cols);
	else if (implementation == VCD_DISTRIBUTED)
		vcdDistributed(rows, cols);
	double vcdTime = seconds() - start;
	
	start = seconds();
	int offset = implementation == VCD_DISTRIBUTED ? cols : 0;
	convertDoubleToByte(implementation == VCD_DISTRIBUTED ? myRows : rows, cols, maxval, offset);
	double backConvertTime = seconds() - start;
	
	start = seconds();
	if (implementation == VCD_DISTRIBUTED)
		collect(cols);
	double collectTime = seconds() - start;
	
	start = seconds();
	if ((implementation != VCD_DISTRIBUTED || self == 0) &&
		    ppp_pnm_write(output_file, kind, rows, cols, maxval, myPart) != 0)
    {
		fprintf(stderr, "An error occured when trying to write the processed picture to the "
			"output file!\n");
		return 2;
	}
	double saveTime = seconds() - start;
	
	if (implementation == VCD_DISTRIBUTED)
		MPI_Finalize();
	free(myPart);
	
	if (self == 0)
		fprintf(stderr,
			"load: %f\n"
			"convert: %f\n"
			"vcd: %f\n"
			"backConvert: %f\n"
			"collect: %f\n"
			"save: %f\n", loadTime, convertTime, vcdTime, backConvertTime, collectTime, saveTime);
	
	return 0;
}
