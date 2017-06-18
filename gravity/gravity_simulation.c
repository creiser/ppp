#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h> 

#include "mpi.h"
#include "ppp_pnm.h"


/*
 * Die Gravitationskonstante in m^3/(kg*s^2).
 */
static const long double G = 6.674e-11;

int np, self;

/*
 * Datentyp zur Beschreibung eines Koerpers.
 * (Die Repraesentation der Koerper kann natuerlich bei Bedarf so
 * geaendert werden, wie es fuer die angestrebte Loesung
 * der Aufgabe erforderlich ist.)
 */
typedef struct {
    long double mass;    /* Masse in kg */
    long double x, y;    /* x- und y-Position in m */
    long double vx, vy;  /* x- und y-Geschwindigkeit in m/s */  
} body;

/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

/*
 * Kommentare (mit "# ...") in ".dat" Dateien ueberspringen.
 */
static void skipComments(FILE *f) {
    int n;
    int dummy; /* um "unused result" Warnungen zu unterdruecken */
    do {
	n=0;
        dummy = fscanf(f, " #%n", &n);
        if (n > 0) {
            dummy += fscanf(f, "%*[^\n]"); 
            dummy += fscanf(f, "\n");
        }
    } while (n>0);
}

/*
 * Eine ".dat" Datei mit Beschreibungen der Koerper einlesen.
 * (Format siehe Uebungsblatt).
 *    f: Dateihandle, aus dem gelesen wird
 *    n: Output-Parameter fuer die Anzahl der gelesenen Koerper
 * Die Koerper werden in einem Array von body-Strukturen
 * zurueckgeliefert. Im Fehlerfall wird NULL zurueckgegeben.
 */
body* readBodies(FILE *f, int *n) {
    int i, conv;
    body *bodies;

    skipComments(f);
    if (fscanf(f, " %d", n) != 1)
        return NULL;
    bodies = (body *) malloc(sizeof(body) * *n);
    if (bodies == NULL)
	return NULL;

    for (i=0; i<*n; i++) {
	skipComments(f);
	conv = fscanf(f, " %Lf %Lf %Lf %Lf %Lf",
		      &(bodies[i].mass),
		      &(bodies[i].x), &(bodies[i].y),
		      &(bodies[i].vx), &(bodies[i].vy));
	if (conv != 5) {
	    free(bodies);
	    return NULL;
	}
    }
    return bodies;
}

/*
 * Schreibe 'n' Koerper aus dem Array 'bodies' in die
 * durch das Dateihandle 'f' bezeichnete Datei im ".dat" Format.
 */
void writeBodies(FILE *f, const body *bodies, int n) {
    int i;
    fprintf(f, "%d\n", n);
    for (i=0; i<n; i++) {
	fprintf(f, "% 10.4Lg % 10.4Lg % 10.4Lg % 10.4Lg % 10.4Lg\n",
		bodies[i].mass, bodies[i].x, bodies[i].y,
		bodies[i].vx, bodies[i].vy);
    }
}

/*
 * Berechne den Gesamtimpuls des Systems.
 *   bodies:  Array der Koerper
 *   nBodies: Anzahl der Koerper
 *   (px,py): Output-Parameter fuer den Gesamtimpuls
 */
void totalImpulse(const body *bodies, int nBodies,
                  long double *px, long double *py) {
    long double px_=0, py_=0;
    int i;

    for (i=0; i<nBodies; i++) {
        px_ += bodies[i].mass * bodies[i].vx;
        py_ += bodies[i].mass * bodies[i].vy;
    }
    *px = px_;
    *py = py_;
}

double interactionRate(const int nBodies, const int steps, const double time) {
    return nBodies * (nBodies - 1) * ((double) steps / time);
}

/*
 * Parameter fuer saveImage().
 *   width, height: Breite und Hoehe des Raumausschnitts in Metern
 *         der im Bild abgespeichert wird. (0,0) liegt im Zentrum.
 *   imgWidth, imgHeight: Breite und Hoehe des Bilds in Pixel, in dem
 *         der Raumausschnitt abgespeichert wird.
 *   imgFilePrefix: Praefix des Dateinamens fuer die Bilder. An
 *         das Praefix wird 00000.pbm, 00001.pbm, 00002.pbm, etc.
 *         angehaengt.
 */
struct ImgParams {
    long double width, height;
    int imgWidth, imgHeight;
    char *imgFilePrefix;
};

/*
 * Einfache Routine zur Ausgabe der Koerper als Bild.
 * Legt ein PBM (portable bitmap) Bild mit einem weissen
 * Pixel fuer jeden Koerper an.
 *   imgNum:  Nummer des Bildes (geht in den Dateinamen ein)
 *   bodies:  Array der Koerper
 *   nBodies: Anzahl der Koerper
 *   params:  Parameter fuer das Bild
 */
void saveImage(int imgNum, const body *bodies, int nBodies,
               const struct ImgParams *params) {
    int i, x, y;
    const int pixels = params->imgWidth * params->imgHeight;
    char name[strlen(params->imgFilePrefix)+10];
    uint8_t *img = (uint8_t *) malloc(sizeof(uint8_t) * pixels);

    if (img == NULL) {
        fprintf(stderr, "Oops: could not allocate memory for image\n");
        return;
    }

    sprintf(name, "%s%05d.pbm", params->imgFilePrefix, imgNum);
    for (i=0; i<pixels; i++) {
        img[i] = 0;
    }

    for (i=0; i<nBodies; i++) {
        x = params->imgWidth/2  + bodies[i].x*params->imgWidth/params->width;
        y = params->imgHeight/2 - bodies[i].y*params->imgHeight/params->height;

        if (x >= 0 && x < params->imgWidth && y >= 0 && y < params->imgHeight) {
            img[y*params->imgWidth + x] = 1;            
        }
    }

    if (ppp_pnm_write(name, PNM_KIND_PBM, params->imgHeight, params->imgWidth,
                      1, img) != 0) {
        fprintf(stderr, "Error writing image\n");
    }
    free(img);
}

/*
 * Implementation of a naive sequential gravity simulation. (a)
 */
void gravity_naive(body **bodies, const int n,
                   const int steps, const int tDelta) {
    long double aTotalX;
    long double aTotalY;
    long double vLength;
    body *T = (body *) malloc(sizeof(body) * n);
    
    for (int s = 0 ; s < steps; ++s) {
        for (int i = 0; i < n; ++i) {
            aTotalX = 0;
            aTotalY = 0;
    
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    vLength = hypot((*bodies)[j].x - (*bodies)[i].x,
                                    (*bodies)[j].y - (*bodies)[i].y);
                    
                    if (vLength != 0) {
                        aTotalX += G * (*bodies)[j].mass
                                   * ((*bodies)[j].x - (*bodies)[i].x)
                                   / pow(vLength, 3);
                    }
                    
                    if (vLength != 0) {
                        aTotalY += G * (*bodies)[j].mass
                                   * ((*bodies)[j].y - (*bodies)[i].y)
                                   / pow(vLength, 3);
                    }
                }
            }
            
            T[i].mass = (*bodies)[i].mass;
            T[i].vx = (*bodies)[i].vx + aTotalX * tDelta;
            T[i].vy = (*bodies)[i].vy + aTotalY * tDelta;
            T[i].x = (*bodies)[i].x + (*bodies)[i].vx * tDelta
                        + 0.5 * aTotalX * tDelta * tDelta;
            T[i].y = (*bodies)[i].y + (*bodies)[i].vy * tDelta
                        + 0.5 * aTotalY * tDelta * tDelta;
        }
        
        body *temp = T;
		T = *bodies;
		*bodies = temp;
    }
        
}

void usage() {
	fprintf(stderr,
		"[-m implementation][-S number_of_steps][-t time_delta][-h][-o name_of_output_file] name_of_input_file\n"
		"This program takes a .dat file with defined bodies and executes a gravity simulation on it based on the given options.\n"
		"With the \"-m\" option the implementation can be specified with an integer.\n"
		"Possible values are 0: naive, 1: optimized, 2: distributed and 3: distributed optimized\n"
        "Use \"-S\" to specify the amount of simulation steps.\n"
        "Use \"-t\" to specify the duration of one simulation step.\n"
		"Use \"-h\" to display this description.\n"
		"Use \"-o\" to specify the file the simulation result should be saved to. The default setting is \"out.dat\".\n"
		"The input file has to be given as the last argument.\n");
}

#define GRAVITY_NAIVE 0
#define GRAVITY_OPTIMIZED 1
#define GRAVITY_DISTRIBUTED 2
#define GRAVITY_OPTIMIZED_DISTRIBUTED 3

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &self);
    
    FILE *f;
    char *output_file = "out.dat";
    body *bodies;
    int option, nrBodies;
    int steps = 366;
    int timeDelta = 86400;
    int implementation = GRAVITY_OPTIMIZED_DISTRIBUTED;
    double start;
    long double px, py;
    struct ImgParams params;
    
    while ((option = getopt(argc,argv,"ho:m:S:t:")) != -1) {
        switch(option) {
        	case 'h':
        		usage();
        		return 0;
        		break;
        	case 'o':
        		output_file = optarg;
    			break;
    		case 'm':
    			implementation = atoi(optarg);
    			break;
            case 'S':
                steps = atoi(optarg);
                break;
			case 't':
				timeDelta = atoi(optarg);
        		break;
        	default:
                fprintf(stderr, "'%c' is not a defined parameter.", option);
        		break;
        }
    }

    f = fopen(argv[argc - 1], "r");
    if (f == NULL) {
        fprintf(stderr, "Could not open file '%s'.\n", argv[1]);
        MPI_Finalize();
        return 1;
    }

    bodies = readBodies(f, &nrBodies);
    if (bodies == NULL) {
        fprintf(stderr, "Error reading .dat file\n");
        fclose(f);
        MPI_Finalize();
        return 1;
    }
    
    totalImpulse(bodies, nrBodies, &px, &py);
    printf("Total impulse before simulation: px=%Lg, py=%Lg\n", px, py);

    start = seconds();
    if (implementation == 0) {
        gravity_naive(&bodies, nrBodies, steps, timeDelta);
    }
    double time = seconds() - start;
    double rate = interactionRate(nrBodies, steps, time);
    printf("Interaction rate of simulation: %f\n", rate);
    
    f = NULL;
    f = fopen(output_file, "w");
    if (f == NULL) {
        fprintf(stderr, "Could not open or create file '%s'.\n", output_file);
        fclose(f);
        MPI_Finalize();
        return 1;
    }
    
    writeBodies(f, bodies, nrBodies);

    totalImpulse(bodies, nrBodies, &px, &py);
    printf("Total impulse after simulation: px=%Lg, py=%Lg\n", px, py);

    /*
     * Gelesene Koerper als PBM-Bild abspeichern.
     * Parameter passen zu "twogalaxies.dat".
     */
    // params.imgFilePrefix = argv[2];
    // params.imgWidth = params.imgHeight = 200;
    // params.width = params.height = 2.0e21;
    // saveImage(0, bodies, nrBodies, &params);
    fclose(f);
    MPI_Finalize();

    return 0;
}
