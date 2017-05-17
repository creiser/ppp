#include <getopt.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

//#include "mpi.h"

#include "ppp_pnm.h"

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
		image[i] = image_double[i] * maxcol;
	}
}

static int N = 40;
static double epsilon = 0.005;
static double kappa = 30;
static double delta_t = 0.1;

inline static double phi(double nu)
{
	double chi = nu / kappa;
	return chi * exp(-(chi * chi) / 2);
}

inline static double xi(double nu)
{
	double psi = nu / (M_SQRT2 * kappa);
	return M_SQRT1_2 * psi * exp(-(psi * psi) / 2);
}

/*
 * Naive sequential VCD implementation.
 */
void vcdNaive(double *image, int rows, int columns) {
    inline double S(int c, int r)
    {
        return r >= 0 && r <= rows &&
        	c >= 0 && c < columns ? image[r * columns + c] : 0;
    }

	double delta;
    for (int y = 0; y < rows; ++y)
    {
		for (int x = 0; x < columns; ++x)
		{
			delta =  phi(S(x + 1, y) - S(x, y));
			delta -= phi(S(x, y) - S(x - 1, y));
			delta += phi(S(x, y + 1) - S(x, y));
			delta -= phi(S(x, y) - S(x, y - 1));
			delta += xi(S(x + 1, y + 1) - S(x, y));
			delta -= xi(S(x, y) - S(x - 1 , y - 1));
			delta += xi(S(x - 1, y + 1) - S(x, y));
			delta -= xi(S(x, y) - S(x + 1, y - 1));
			image[y * columns + x] = S(x, y) + kappa * delta_t * delta;
		}
    }    
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
	vcdNaive(image, rows, cols);

	if(execute_vcd && !fast_vcd) {
		// execute sequential, non-optimised vcd algorithm
	}

	if(execute_vcd && fast_vcd) {
		// execute sequential, optimised vcd algorithm
	}

	if(execute_sobel) {
		// execute sobel algorithm
	}
	
	convertDoubleToImage(image, picture, rows, cols, maxval);

	if(ppp_pnm_write(output_file, kind, rows, cols, maxval, picture) != 0) {
		fprintf(stderr, "An error occured when trying to write the processed picture to the output file!\n");
		return 2;
	}
	free(picture);
	return 0;
}
