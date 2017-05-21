#include <getopt.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h> 

//#include "mpi.h"

#include "ppp_pnm.h"

inline double exp1(double x) {
  x = 1.0 + x / 256.0;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

double *convertImageToDouble(uint8_t *image, int rows, int columns, int maxcol)
{
	double *image_double = malloc(rows * columns * sizeof(double));
	for (int i = 0; i < rows * columns; i++)
	{
		image_double[i] = (double)image[i] / maxcol;
	}
	return image_double;
}

void convertDoubleToImage(double *image_double, uint8_t *image, int rows, int columns, int maxcol)
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
	free(T);
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
    
    double delta;
    //intermediate store
    double *T = malloc(rows * columns * sizeof(double));
    //caches one row of phi() below current row
    double *prev_y = malloc(columns * sizeof(double));
    for (int i = 0; i < N; i++)
	{
        int epsilon_exit = 1;
        //init prev_y
        for (int i = 0; i < columns; ++i) {
            prev_y[i] = phi(S(i, 0));
        }
        
        for (int y = 0; y < rows; ++y)
        {
            double prev_x = phi(S(0, y)); // consider: S(-1, y) = 0
            for (int x = 0; x < columns; ++x)
            {
                delta = -prev_x;
                prev_x = phi(S(x + 1, y) - S(x, y));
                delta += prev_x;
                
                delta -= prev_y[x];
                prev_y[x] = phi(S(x, y + 1) - S(x, y));
                delta += prev_y[x];
                
                delta += xi(S(x + 1, y + 1) - S(x, y));
                delta -= xi(S(x, y) - S(x - 1 , y - 1));
                delta += xi(S(x - 1, y + 1) - S(x, y));
                delta -= xi(S(x, y) - S(x + 1, y - 1));
                
                T[y * columns + x] = S(x, y) + kappa * delta_t * delta;
                
                if (fabs(delta) > epsilon && x >= 1 && x < columns - 1 &&
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
    free(prev_y);
    free(T);
}

/*
 * Cache previous row and left value to reuse already calculated values.
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


/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
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

	picture = ppp_pnm_read(input_file, &kind, &rows, &cols, &maxval);
	if(picture == NULL) {
		fprintf(stderr, "An error occured when trying to load the picture from the file \"%s\"! If this is not the input file you had in mind please note that it has to be specified as the last argument.\n", input_file);
		return 1;
	}
	
	double *image = convertImageToDouble(picture, rows, cols, maxval);

	double vcdStart = seconds();
	//vcdNaive(image, rows, cols);
	vcdOptimized(image, rows, cols);
	//vcdOptimizedParallel(image, rows, cols);
	printf("vcd time: %f\n", seconds() - vcdStart);

	if(execute_vcd && !fast_vcd) {
		// execute sequential, non-optimised vcd algorithm
	}

	if(execute_vcd && fast_vcd) {
		// execute sequential, optimised vcd algorithm
	}

	if(execute_sobel) {
		// execute sobel algorithm
	}
	
	printf("1\n");
	
	convertDoubleToImage(image, picture, rows, cols, maxval);
	// free(image);
	
	printf("2\n");

	if(ppp_pnm_write(output_file, kind, rows, cols, maxval, picture) != 0) {
		fprintf(stderr, "An error occured when trying to write the processed picture to the output file!\n");
		return 2;
	}
	free(picture);
	
	printf("3\n");
	return 0;
}
