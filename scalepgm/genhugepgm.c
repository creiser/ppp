#include "ppp_pnm.h"
#include <stdlib.h>
#include <time.h>

#define SIZE 40000
#define SIZE_SQ (size_t)SIZE * SIZE
#define MAXVAL 255

int main(int argc, char **argv)
{
	srand(time(NULL));
	uint8_t *image = malloc(SIZE_SQ * sizeof(uint8_t));
	for (size_t i = 0; i < SIZE_SQ; i++)
		image[i] = rand() % (MAXVAL + 1);
	ppp_pnm_write("huge.pgm", PNM_KIND_PGM, SIZE, SIZE, MAXVAL, image);
}