#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "ppp_pnm.h"

/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

int main(int argc, char* argv[]) {
	bool execute_vcd  = false;
	bool execute_sobel = false;
	int option, rows, cols, maxval;
	enum pnm_kind kind;
	char* output_file = "out.pgm";
	char* input_file;
	uint8_t* picture;
    while ((option = getopt(argc,argv,"hvso:")) != -1) {
        switch(option) {
        	case 'h':
        		printf("[-hvs][-o name_of_output_file] name_of_input_file\nThis program takes a picture in pgm format and processes it based on the given options.\nUse \"-h\" to display this description.\nUse \"-v\" to let the picture be manipulated by the vcd algortihm.\nUse \"-s\" to let the picture be manipulated by the sobel algortihm.\nIf both the \"-v\" and the \"-s\" options are set, the vcd algorithm will be executed first.\nUse \"-o\" to specify the file the processed image should be saved to. The default option ist \"out.pgm\".\nThe input file has to be given as the last argument.\n");
        		return 0;
        	break;
        	case 'v':
        		execute_vcd = true;
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

	if(execute_vcd) {
		// execute vcd algorithm
	}

	if(execute_sobel) {
		// execute sobel algorithm
	}

	if(ppp_pnm_write(output_file, kind, rows, cols, maxval, picture) != 0) {
		fprintf(stderr, "An error occured when trying to write the processed picture to the output file!\n");
		return 2;
	}
	free(picture);
	return 0;
}