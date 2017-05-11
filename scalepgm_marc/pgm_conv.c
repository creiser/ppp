#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "ppp_pnm.h"
#include <omp.h>
#include "mpi.h"

int np, self;

/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

/* Sucht sequentiell das Minimum und Maximum der size Werte von picture und speichert sie in min und max. */ 
void find_min_and_max(uint8_t* min, uint8_t* max, uint8_t* picture, int size) {
	for(int i = 0; i < size; i++) {
		if(*max < picture[i]) {
			*max = picture[i];
		}
		if(*min > picture[i]) {
			*min = picture[i];
		}
	}
}

/* Sucht parallelisiert das Minimum und Maximum der size Werte von picture und speichert sie in min und max. */ 
void omp_find_min_and_max(uint8_t* min_ptr, uint8_t* max_ptr, uint8_t* picture, int size) {
	uint8_t min = *min_ptr;
	uint8_t max = *max_ptr;
#pragma omp parallel for reduction(min:min), reduction(max:max)
	for(int i = 0; i < size; i++) {
		if(max < picture[i]) {
			max = picture[i];
		}
		if(min > picture[i]) {
			min = picture[i];
		}
	}
	*min_ptr = min;
	*max_ptr = max;
}

uint8_t* mpi_part(enum pnm_kind kind, int rows, int cols, int* offset, int* length) {
	int size = rows * cols;
	int rest = size % np;
	int rounded_length = size / np;
	uint8_t* pointer;

	if (self < rest) {
		*length = rounded_length + 1;
		*offset = self * *length;
		if (self == 0) {
			pointer = (uint8_t*) calloc(size, sizeof(uint8_t));
		}
		else {
						pointer = (uint8_t*) calloc(rounded_length + 1, sizeof(uint8_t));
		}

	}
	else {
		*length = rounded_length;
		*offset = rest * (rounded_length + 1) + (self - rest) * rounded_length;
		pointer = (uint8_t*) calloc(rounded_length, sizeof(uint8_t));
	}
	return pointer;
}

int seq_grey_scaling (char* picture_name, int new_min, int new_max, double* duration) {

	uint8_t* picture;
	uint8_t min, max;
	int maxval, rows, cols, size;
	enum pnm_kind kind;
	double start, end;

	start = seconds();
	picture = ppp_pnm_read(picture_name, &kind, &rows, &cols, &maxval);
	if(picture == NULL) {
		fprintf(stderr, "An error occured while trying to read the picture\n");
		return 1;
	}

	max = 0;
	min = (uint8_t) maxval;
	size = rows * cols;
	find_min_and_max(&min, &max, picture, size);

	for(int i = 0; i < size; i++) {
		picture[i] = ((picture[i] - min) * (new_max - new_min) + (max - min) / 2) / (max - min) + new_min;
	}

	ppp_pnm_write("scaled_seq.pgm", kind, rows, cols, maxval, picture);
	end = seconds();
	free(picture);
	*duration = end - start;
	return 0;
}

int omp_grey_scaling (char* picture_name, int new_min, int new_max, double* duration) {

	uint8_t* picture;
	uint8_t min, max;
	int maxval, rows, cols, size;
	enum pnm_kind kind;
	double start, end;

	start = seconds();
	picture = ppp_pnm_read(picture_name, &kind, &rows, &cols, &maxval);
	if(picture == NULL) {
		fprintf(stderr, "An error occured while trying to read the picture\n");
		return 1;
	}

	max = 0;
	min = (uint8_t) maxval;
	size = rows * cols;
	printf("size: %u\n", size);
	omp_find_min_and_max(&min, &max, picture, size);

#pragma omp parallel for
	for(int i = 0; i < size; i++) {
		picture[i] = ((picture[i] - min) * (new_max - new_min) + (max - min) / 2) / (max - min) + new_min;
	}

	ppp_pnm_write("scaled_omp.pgm", kind, rows, cols, maxval, picture);
	end = seconds();
	free(picture);
	*duration = end - start;
	return 0;
}

int mpi_grey_scaling (char* picture_name, int new_min, int new_max, double* duration) {

	uint8_t* picture_part, picture;
	uint8_t min, max;
	int maxval, rows, cols, size, lengths[np], offsets[np];
	enum pnm_kind kind;
	double start, end;

	start = seconds();
	picture_part = ppp_pnm_read_part(picture_name, &kind, &rows, &cols, &maxval, &mpi_part);
	if(picture_part == NULL) {
		fprintf(stderr, "An error occured while trying to read the picture\n");
		return 1;
	}
	printf("Read durch!\n");

	max = 0;
	min = (uint8_t) maxval;
	size = rows * cols;
	printf("size: %u\n", size);
	lengths[self] = self < size % np ? size / np + 1 : size / np;
	offsets[self] = self < size % np ? self * lengths[self] : size % np * (size / np + 1) + (self - size % np) * size / np;
	printf("%d %u\n", self, lengths[self]);
	printf("%d %u\n", self, offsets[self]);
	omp_find_min_and_max(&min, &max, picture_part, lengths[self]);
	MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_UINT8_T, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_UINT8_T, MPI_MAX, MPI_COMM_WORLD);

#pragma omp parallel for
	for(int i = 0; i < lengths[self]; i++) {
		picture_part[i] = ((picture_part[i] - min) * (new_max - new_min) + (max - min) / 2) / (max - min) + new_min;
	}
MPI_Gatherv(picture_part, lengths[self], MPI_UINT8_T, picture_part, lengths, offsets, MPI_UINT8_T, 0, MPI_COMM_WORLD);
	if(self == 0) {
		//picture = (uint8_t*) calloc(size, sizeof(uint8_t));
		
		ppp_pnm_write("scaled_mpi.pgm", kind, rows, cols, maxval, picture);
	}
/*	else {
		MPI_Gatherv(picture_part, lengths[self], MPI_UINT8_T, NULL, NULL, NULL, MPI_UINT8_T, 0, MPI_COMM_WORLD);
	}*/
	printf("Gatherv durch!\n");
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Barrier durch!\n");
	end = seconds();
	free(picture_part);
	if(self == 0) {
	//	free(picture);
	}
	*duration = end - start;
	return 0;
}


int main(int argc, char* argv[]) {
    int min, max;
    double duration = 0.0;
	char* picture_name;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
	
	if(argc != 4) {
		fprintf(stderr, "Wrong number of arguments, those three arguments are expected: The path to the picture, the lower bound of the interval and the higher bound of the new range of the colors.\n");
		return -1;
	}
	picture_name = argv[1];
	min = atoi(argv[2]);
	max = atoi(argv[3]);

	seq_grey_scaling(picture_name, min, max, &duration);
	//MPI_Reduce(MPI_IN_PLACE, &duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (self == 0) {
		printf("Sequentielle Ausführungszeit: %lf\n", duration / np);
	}

	omp_grey_scaling(picture_name, min, max, &duration);
	//MPI_Reduce(MPI_IN_PLACE, &duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (self == 0) {
		printf("OpenMP-parallele Ausführungszeit: %lf\n", duration / np);
	}

	mpi_grey_scaling(picture_name, min, max, &duration);
	//MPI_Reduce(MPI_IN_PLACE, &duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (self == 0) {
		printf("MPI-OpenMP-parallele Ausführungszeit: %lf\n", duration / np);
	}

	MPI_Finalize();
}