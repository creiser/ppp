#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "ppp_pnm.h"
#include <omp.h>
#include <stdbool.h>

int main(int argc, char* argv[]) {
	if(argc < 3) {
		fprintf(stderr, "Too few arguments, at least two expected!");
	}
	uint8_t* pictures[argc - 1];
	boolean is_same_as_suc[argc - 2];
	int maxval, rows, cols;
	enum pnm_kind kind;

#pragma omp parallel for
	for(int i = 1; i < argc; i++) {
		pictures[i - 1] = ppp_pnm_read(argv[i], &kind, &rows, &cols, &maxval);
	}
	for(int i = 0; i < argc - 2; i++) {
		is_same_as_suc[i] = true;
#pragma omp parallel for
		for(int j = 0; j < rows * cols; j++) {
			is_same_as_suc[i] = is_same_as_suc[i] && pictures[i][j] == pictures[i + 1][j];
			if(!is_same_as_suc[i]) {
				break;
			}
		}
	}
	for(int i = 0; i < argc - 2; i++) {
		is_same_as_suc[i] ? printf("Bild %d und %d stimmen überein!\n", i + 1, i + 2) : printf("Bild %d und %d stimmen nicht überein!\n", i + 1, i + 2);
		free(pictures[i]);
	}
	free(pictures[argc - 2]);
	return 0;
}