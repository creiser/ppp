#include <getopt.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h> 
#include <mpi.h>

#include "ppp_pnm.h"

int self, np;
int myRowCount;

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

/*
 * Load a block of rows of an image on the current processor
 */
uint8_t *partfn(enum pnm_kind kind, int rows, int columns,
                int *offset, int *length) {
    if (kind != PNM_KIND_PGM)
        return NULL;

    /*
     * The number of rows need not be a multiple of np.
     * Therefore, the first rows%np  processes get
     *    ceil(rows/np) * columns
     * pixels, and the remaining processes get
     *    floor(rows/np) * columns
     * pixels.
     */
    if (self < rows%np) {
        myRowCount = rows/np + 1;
        *length = myRowCount * columns;
        *offset = *length * self;
    } else {
        myRowCount = rows/np;
        *length = myRowCount * columns;
        *offset = *length * self + rows%np*columns;
    }

    printf("self=%d, offset=%d, length=%d, rowcount=%d\n", self, *offset, *length, myRowCount);

    /*
     * Allocate space for the image part.
     * On processor 0 we allocate space for the whole result image.
     */
    return (uint8_t*)malloc((self == 0 ? rows*columns : *length) * sizeof(uint8_t));
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
 * Naive sequential VCD implementation. Exercise (a)
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
 * Exercise (c)
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
    double *upper = malloc(columns * sizeof(double));
    //caches one row of xi() for left upper corners
    double *left_upper = malloc(columns * sizeof(double));
    //caches one row of xi() for right upper corners
    double *right_upper = malloc((columns + 1) * sizeof(double));

    for (int i = 0; i < N; i++)
	{
        int epsilon_exit = 1;
        
        //init caches
        for (int i = 0; i < columns; ++i) {
            upper[i] = phi(S(i, 0));
            left_upper[i] = xi(S(i + 1, 0));
            right_upper[i] = xi(S(i - 1, 0));
        }
        
        for (int y = 0; y < rows; ++y)
        {
            double left = phi(S(0, y)); // S(-1, y) = 0
            double first_upper_left = xi(S(0, y)); // S(-1+1, y-1+1)-S(-1, y-1)
            right_upper[columns] = xi(S(columns - 1, 0));
            
            for (int x = 0; x < columns; ++x)
            {
                delta = -left;
                left = phi(S(x + 1, y) - S(x, y));
                delta += left;
                
                delta -= upper[x];
                upper[x] = phi(S(x, y + 1) - S(x, y));
                delta += upper[x];
                
                delta -= first_upper_left;
                first_upper_left = left_upper[x];
                left_upper[x] = xi(S(x + 1, y + 1) - S(x, y));
                delta += left_upper[x];
                
                delta -= right_upper[x+1];
                right_upper[x] = xi(S(x - 1, y + 1) - S(x, y));
                delta += right_upper[x];
                
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
    free(upper);
    free(T);
}

/*
 * Naive parallel VCD implementation. Exercise (e)
 */
void vcdNaiveParallel(double *image, int rows, int columns) {
	double *T = malloc(rows * columns * sizeof(double));
    double *upperRowN = malloc(columns * sizeof(double));
    double *lowerRowN = malloc(columns * sizeof(double));
    int epsilon_exit;
    MPI_Request rqS1, rqS2;
    int numThreads;
    
    inline double S(int c, int r)
    {
        return c >= 0 && c < columns ? (r < 0 ? upperRowN[c] : (r >= rows ? lowerRowN[c] : image[r * columns + c])) : 0;
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
    
    /*Init neighbouring rows which are the borders of the whole image*/
    int i, j;
    if (self == 0) {
        for (i = 0; i < columns; ++i) {
            upperRowN[i] = 0;
        }
    }
    
    if (self == np-1) {
        for (i = 0; i < columns; ++i) {
            lowerRowN[i] = 0;
        }
    }
    
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    
	int *starts = (int *) malloc(numThreads * sizeof(int));
    int *ends = (int *) malloc(numThreads * sizeof(int));
    
	starts[0] = 0;
	ends[0] = (rows/numThreads + (0 < rows % numThreads ? 1 : 0));
	for (j = 1; j < numThreads; ++j) {
		starts[j] = ends[j - 1];
		ends[j] = starts[j] + (rows/numThreads + (j < rows % numThreads ? 1 : 0));
    }
	
	for (i = 0; i < N; ++i)
	{
		int l_epsilon_exit = 1;
        
        /*Send upper row to lower neighbour*/
        if (self != 0) {
            MPI_Isend(&image[0], columns, MPI_DOUBLE, self-1, 0, MPI_COMM_WORLD,
                    &rqS1);
        }
        
        /*Send lower row to upper neighbour*/
        if (self != np-1) {
            MPI_Isend(&image[(rows-1)*columns], columns, MPI_DOUBLE, self+1, 0,
                MPI_COMM_WORLD, &rqS2);
        }
        
        /*Receive lower row from upper neighbour*/
        if (self != np-1){
            MPI_Recv(lowerRowN, columns, MPI_DOUBLE, self+1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            MPI_Wait(&rqS2, MPI_STATUS_IGNORE);
        }
        
        /*Receive upper row from lower neighbour*/
        if (self != 0){
            MPI_Recv(upperRowN, columns, MPI_DOUBLE, self-1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            MPI_Wait(&rqS1, MPI_STATUS_IGNORE);
        }
        
        #pragma omp parallel shared(l_epsilon_exit)
        {
            double delta_x_y;
            int threadNr = omp_get_thread_num();
        
            for (int y = starts[threadNr]; y < ends[threadNr]; ++y) {
                for (int x = 0; x < columns; ++x) {
                    delta_x_y = delta(x, y);
                    T[y * columns + x] = S(x, y) + kappa * delta_t * delta_x_y;
                    
                    if (fabs(delta_x_y) > epsilon &&
                            x >= 1 && x < columns - 1 &&
                            (self == 0 && y >= 1 || y >= 0) &&
                            (self == np - 1 && y < rows - 1 || y < rows)) {
                        l_epsilon_exit = 0;
                    }
                }
            }
        }
        
		double *temp = T;
		T = image;
		image = temp;
        
        MPI_Allreduce(&l_epsilon_exit, &epsilon_exit, 1, MPI_INT, MPI_LAND,
                MPI_COMM_WORLD);
        
		if (epsilon_exit)
			break;
	}
    
    free(lowerRowN);
    free(upperRowN);
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
    bool parallel_vcd = false;
	bool execute_sobel = false;
	int option, rows, cols, maxval;
	enum pnm_kind kind;
	char* output_file = "out.pgm";
	char* input_file;
	uint8_t* picture;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    
    while ((option = getopt(argc,argv,"hvfpso:")) != -1) {
        switch(option) {
        	case 'h':
        		printf("[-hvfs][-o name_of_output_file] name_of_input_file\nThis program takes a picture in pgm format and processes it based on the given options.\nUse \"-h\" to display this description.\nUse \"-v\" to let the picture be manipulated by the vcd algorithm.\nUse \"-f\" if you want the program to use the faster version of the vcd algorithm if \"-v\" is set, too.\nUse \"-p\" to execute the parallelized vcd version.\nCombine \"-f\" and \"-p\" as needed.\nUse \"-s\" to let the picture be manipulated by the sobel algorithm.\nIf both the \"-v\" and the \"-s\" options are set, the vcd algorithm will be executed first.\nUse \"-o\" to specify the file the processed image should be saved to. The default option ist \"out.pgm\".\nThe input file has to be given as the last argument.\n");
        		return 0;
        	break;
        	case 'v':
        		execute_vcd = true;
        	break;
        	case 'f':
        		fast_vcd = true;
        	break;
            case 'p':
                parallel_vcd = true;
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
	
    double *image;
	double vcdStart;
    
    if (!parallel_vcd) {
        picture = ppp_pnm_read(input_file, &kind, &rows, &cols, &maxval);
            
        if(picture == NULL) {
            fprintf(stderr, "An error occured when trying to load the picture from the file \"%s\"! If this is not the input file you had in mind please note that it has to be specified as the last argument.\n", input_file);
            return 1;
        }
        
        image = convertImageToDouble(picture, rows, cols, maxval);
    } else{
        picture = ppp_pnm_read_part(input_file, &kind, &rows, &cols,
                    &maxval, partfn);
                    
        if(picture == NULL) {
            fprintf(stderr, "An error occured when trying to load the picture from the file \"%s\"! If this is not the input file you had in mind please note that it has to be specified as the last argument.\n", input_file);
            return 1;
        }
                
        image = convertImageToDouble(picture, myRowCount, cols, maxval);
    }
    
    /*execute different variants of the vcd algorithm*/
	if(execute_vcd) {
        if(!parallel_vcd) {
            if(!fast_vcd){
                // execute sequential, non-optimised vcd algorithm
                vcdStart = seconds();
                vcdNaive(image, rows, cols);
                printf("vcd time: %f\n", seconds() - vcdStart);
            } else {
                // execute sequential, optimised vcd algorithm
                vcdStart = seconds();
                vcdOptimized(image, rows, cols);
                printf("vcd time: %f\n", seconds() - vcdStart);
            }
        } else if (parallel_vcd) {
            if (!fast_vcd) {
                //execute parallel, non-optimised vcd algorithm
                vcdStart = seconds();
                vcdNaiveParallel(image, myRowCount, cols);
                printf("vcd time: %f\n", seconds() - vcdStart);
            } else {
                // execute parallel, optimised vcd algorithm
                vcdStart = seconds();
                //vcdOptimizedParallel(image, rows, cols);
                printf("vcd time: %f\n", seconds() - vcdStart);
            }
        }
	}

    /*execute the sobel algorithm*/
	if(execute_sobel) {
        if(!parallel_vcd) {
            
        } else {
            
        }
	}
    
    if(!parallel_vcd) {
        convertDoubleToImage(image, picture, rows, cols, maxval);
        free(image);
    } else {
        convertDoubleToImage(image, picture, myRowCount, cols, maxval);
        free(image);
        
        int *displs = (int *) malloc(np*sizeof(int));
        int *rcvCounts = (int *) malloc(np*sizeof(int));
        int rCount, offset = 0;
        
        for (int i = 0; i < np; i++) {
            displs[i] = offset;
            
            rCount = rows/np;
            if (i < rows%np) {
                rCount++;
            }
            
            rcvCounts[i] = rCount * cols;
            offset = offset + rCount * cols;
        }
        
        MPI_Gatherv(picture, myRowCount*cols, MPI_UINT8_T, picture,
            rcvCounts, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    }
    
    if(!parallel_vcd || (parallel_vcd && self == 0)) {
        if(ppp_pnm_write(output_file, kind, rows, cols, maxval, picture) != 0) {
            fprintf(stderr, "An error occured when trying to write the processed picture to the output file!\n");
            return 2;
        }
    }
    
	free(picture);
	MPI_Finalize();
	return 0;
}
