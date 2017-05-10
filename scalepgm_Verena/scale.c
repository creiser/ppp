/*
 * Beispiel fuer die Benutzung der ppp_pnm Bibliothek.
 * Kompilieren mit:
 *
 *     gcc -std=c99 -Wall -o scale scale.c
 *          -I/home/ppp2017/ppp_pnm
 *          -L/home/ppp2017/ppp_pnm
 *          -lppp_pnm
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "ppp_pnm.h"

static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + ((double) tv.tv_usec)/1000000.0;
}

/*
 * Load a PGM (Portable Graymap) image and scale its values to a new intervall.
 * The program is called with 4 arguments:
 *      Input-image Output-image min-scale max-scale
 */
int main(int argc, char *argv[]) {
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image;
    int x, y, nmin, nmax, omin, omax, current, ndiff, odiff, error;
    double start, end;

    if (argc != 5) {
        fprintf(stderr, "USAGE: %s IN OUT NEW-MIN NEW-MAX\n", argv[0]);
        return 1;
    }

    start = seconds();

    image = ppp_pnm_read(argv[1], &kind, &rows, &columns, &maxcolor);
    end = seconds();    
    printf("Loading image: %fs\n", (end-start));
    
    if (image != NULL) {
        if (kind == PNM_KIND_PGM) {
            nmin = atoi(argv[3]);
            nmax = atoi(argv[4]);
            ndiff = nmax - nmin;
            omin = maxcolor;
            omax = 0;

            start = seconds();

            for (y = 0; y < rows; y++) {
                for (x = 0; x < columns; x++) {
                    current = image[y*columns+x];
                    if (current < omin) {
                        omin = current;
                    }
                    if (omax < current) {
                        omax = current;
                    }
                }
            }

            end = seconds();
            printf("Calculate min/max: %fs\n", (end-start));
            start = seconds();

            odiff = omax - omin;
            error = odiff/2;

            for (y = 0; y < rows; y++) {
                for (x = 0; x < columns; x++) {
                    image[y*columns+x] = ((image[y*columns+x] - omin)*ndiff + error)/odiff + nmin;
                }
            }

            end = seconds();
            printf("Scaling image: %fs\n", (end-start));
            
            if (ppp_pnm_write(argv[2], kind, rows, columns, maxcolor, image) != 0)
            fprintf(stderr, "write error\n");
        } else
            fprintf(stderr, "not a PGM image\n");

        free(image);
    } else
    fprintf(stderr, "could not load image\n");

    return 0;
}
