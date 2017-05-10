#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "mpi.h"
#include "ppp_pnm.h"

int self, np;
int myLength;

static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + ((double) tv.tv_usec)/1000000.0;
}

/*
 * Load a part of an image on the current processor
 */
uint8_t *partfn(enum pnm_kind kind, int rows, int columns,
                int *offset, int *length) {
    if (kind != PNM_KIND_PGM)
        return NULL;

    /*
     * The number of pixels need not be a multiple of np.
     * Therefore, the first (rows*columns)%np  processes get
     *    ceil((rows*columns)/np)
     * pixels, and the remaining processes get
     *    floor((rows*columns)/np)
     * pixels.
     */
    if (self < (rows*columns)%np) {
        *length = (rows*columns)/np + 1;
        *offset = *length * self;
    } else {
        *length = (rows*columns)/np;
        *offset = *length * self  +  (rows*columns)%np;
    }

    myLength = *length;
    printf("self=%d, offset=%d, length=%d\n", self, *offset, *length);

    /*
     * Allocate space for the image part.
     * On processor 0 we allocate space for the whole result image.
     */
    return (uint8_t*)malloc((self == 0 ? rows*columns : myLength) * sizeof(uint8_t));
}

/*
 * Load a PGM (Portable Graymap) image and scale the gray values of every pixel
 * to a now intervall.
 * The program is called with 4 arguments:
 *      Input-image Output-image min-scale max-scale
 */
int main(int argc, char *argv[]) {
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *myPart;
    int iSize, nmin, nmax, omin, omax, ndiff, odiff, error;
    int *displs, *rcvCounts;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (argc != 5) {
        printf("USAGE: %s IN OUT NEW-MIN NEW-MAX\n", argv[0]);
        return 1;
    }

    start = seconds();

    myPart = ppp_pnm_read_part(argv[1], &kind, &rows, &columns, &maxcolor, partfn);

    end = seconds();
    printf("Loading image: %fs\n", (end - start));

    iSize = rows*columns;
    nmin = atoi(argv[3]);
    nmax = atoi(argv[4]);
    ndiff = nmax - nmin;
    
    if (myPart != NULL) {
        int i, current;
        int lomin = maxcolor;
        int lomax = 0;
        int offset = 0;

        start = seconds();

        /*Calculates local min and max gray value*/
        for (i = 0; i < myLength; i++) {
            current = myPart[i];
            
            if (current < lomin) {
                lomin = current;
            }
            if (lomax < current) {
                lomax = current;
            }
        }
        
        /*
         * Gathers global min and max gray value from calculated local values of
         * each process.
         */
        MPI_Allreduce(&lomin, &omin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&lomax, &omax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        end = seconds();
        printf("Reducing min/max: %fs\n", (end - start));
        start = seconds();

        odiff = omax - omin;
        error = odiff/2;
        
        /*Sets new gray value for each pixel of local image part.*/
        for (i = 0; i < myLength; i++) {
            myPart[i] = ((myPart[i] - omin)*ndiff + error)/odiff + nmin;
        }

        end = seconds();
        printf("Scaling image: %fs\n", (end - start));
        start = seconds();

        displs = (int *) malloc(np*sizeof(int));
        rcvCounts = (int *) malloc(np*sizeof(int));
        int len;
        
        for (i = 0; i < np; i++) {
            displs[i] = offset;
            
            len = iSize/np;
            if (i < iSize%np) {
                len++;
            }
            
            rcvCounts[i] = len;
            offset = offset + len;
        }
        
        MPI_Gatherv(myPart, myLength, MPI_UINT8_T, myPart, rcvCounts, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD);
        
        end = seconds();
        printf("Gather image parts: %fs\n", (end - start));

        if (self == 0) {
            if (ppp_pnm_write(argv[2], kind, rows, columns, maxcolor, myPart) != 0)
            printf("write error\n");
        }

        free(myPart);
    } else {
        printf("could not load image\n");
    }

    MPI_Finalize();

    return 0;
}
