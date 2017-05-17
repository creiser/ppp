#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>

/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

int main(int argc, char* argv[]) {
	bool execute_vcd  = false;
	bool execute_sobel = false;
	char* output_file = "out.pgm";
	char* input_file;
	uint8_t picture;
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
}