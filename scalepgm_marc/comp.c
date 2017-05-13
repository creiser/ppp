#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "ppp_pnm.h"
#include <omp.h>
#include "mpi.h"
#include <stdbool.h>

int np, self;
uint8_t* mpi_part(enum pnm_kind kind, int rows, int cols, int* offset, int* length) {
	int size = rows * cols;
	int rest = size % np;
	int rounded_length = size / np;
	uint8_t* pointer;

	if (self < rest) {
		*length = rounded_length + 1;
		*offset = self * *length;
		pointer = (uint8_t*) calloc(rounded_length + 1, sizeof(uint8_t));

	}
	else {
		*length = rounded_length;
		*offset = rest * (rounded_length + 1) + (self - rest) * rounded_length;
		pointer = (uint8_t*) calloc(rounded_length, sizeof(uint8_t));
	}
	return pointer;
}

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

	if(argc < 3) {
		fprintf(stderr, "Too few arguments, at least two expected!");
	}

	for(int i = 0; i < argc - 1; i++) {
		
	}
}