#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h> 
#include "ppp_pnm.h"
#include "mpi.h"


/*
 * Die Gravitationskonstante in m^3/(kg*s^2).
 */
static const long double G = 6.674e-11;

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

int readBodiesFromFile(char* filename, body **bodies, int *n)
{
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
	fprintf(stderr, "Could not open file '%s'.\n", filename);
	return 1;
    }
    *bodies = readBodies(f, n);
    if (bodies == NULL) {
	fprintf(stderr, "Error reading .dat file\n");
	return 1;
    }
    fclose(f);
    return 0;
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

int writeBodiesToFile(char* filename, body *bodies, int numBodies)
{
	FILE *f = fopen(filename, "w");
    if (f == NULL) {
	fprintf(stderr, "Could not open file '%s'.\n", filename);
	return 1;
    }
    writeBodies(f, bodies, numBodies);
    fclose(f);
    return 0;
}

/*
 * Berechne den Gesamtimpuls des Systems.
 *   bodies:  Array der Koerper
 *   nBodies: Anzahl der Koerper
 *   (px,py): Output-Parameter fuer den Gesamtimpuls
 */
void totalImpulse(const body *bodies, int nBodies,
                  long double *px, long double *py)
{
    long double px_=0, py_=0;
    int i;

    for (i=0; i<nBodies; i++) {
	px_ += bodies[i].mass * bodies[i].vx;
	py_ += bodies[i].mass * bodies[i].vy;
    }
    *px = px_;
    *py = py_;
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
               const struct ImgParams *params)
{
    int i, x, y;
    const int pixels = params->imgWidth * params->imgHeight;
    char name[strlen(params->imgFilePrefix)+10];
    uint8_t *img = (uint8_t *) malloc(sizeof(uint8_t) * pixels);

    if (img == NULL) {
        fprintf(stderr, "Oops: could not allocate memory for image\n");
	return;
    }

    sprintf(name, "%s%05d.pbm", params->imgFilePrefix, imgNum);
    for (i=0; i<pixels; i++)
	img[i] = 0;

    for (i=0; i<nBodies; i++) {
	x = params->imgWidth/2  + bodies[i].x*params->imgWidth/params->width;
	y = params->imgHeight/2 - bodies[i].y*params->imgHeight/params->height;

	if (x >= 0 && x < params->imgWidth && y >= 0 && y < params->imgHeight)
	    img[y*params->imgWidth + x] = 1;
    }

    if (ppp_pnm_write(name, PNM_KIND_PBM, params->imgHeight, params->imgWidth,
                      1, img) != 0) {
        fprintf(stderr, "Error writing image\n");
    }
    free(img);
}

/* liefert die Sekunden seit dem 01.01.1970 */
static double seconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

static inline long double max(long double a, long double b) 
{
	return a > b ? a : b;
}

static inline long double min(long double a, long double b) 
{
	return a < b ? a : b;
}

static const int num_steps = 100;
static const long double delta_t = 3.16e10;

int self;
int np;

void simulateDistributed(body *bodies, int nBodies)
{
	long double *accel =
		malloc(2 * nBodies * sizeof(long double));
		
	long double delta_t_squared = delta_t * delta_t;
	
	// Calculate the 'i' share as usual, but every process
	// will do all the 'j's and therefore receive all bodies
	int *counts = malloc(2 * np * sizeof(int));
    int *displs = &counts[np];
	displs[0] = 0;
    counts[0] = (nBodies/np + (0 < nBodies%np ? 1 : 0)) * 5;
    for (int i = 1; i < np; i++)
    {
		counts[i] = (nBodies/np + (i < nBodies%np ? 1 : 0)) * 5;
		displs[i] = displs[i-1] + counts[i-1];
    }
    int myStart = displs[self] / 5;
    int myEnd = myStart + counts[self] / 5;
    int myLength = myEnd - myStart;
    printf("self: %d, myStart: %d, myEnd: %d, myLength: %d\n",
    	self, myStart, myEnd, myLength);
    
	// Use hardcoded path for testing.
	struct ImgParams params;
	params.imgFilePrefix = "iter";
    params.imgWidth = params.imgHeight = 200;
    params.width = params.height = 2.0e21;
	
	double gather_time = 0.0;
	double calc_time = 0.0;
	
	MPI_Request *requests = malloc((np - 1) * sizeof(MPI_Request));
	MPI_Request *receive_requests = malloc((np - 1) * sizeof(MPI_Request));
	for (int iteration = 0; iteration < num_steps; iteration++)
	{
		if (iteration != 0) {
			for (int z = 1; z < np; z++) {
				int sender = (self - z + np) % np;
				int remoteStart = displs[sender] / 5;
				//int remoteEnd = remoteStart + counts[sender] / 5;
				MPI_Irecv(&bodies[remoteStart], counts[sender], MPI_LONG_DOUBLE, sender, 0,
						MPI_COMM_WORLD, &receive_requests[z - 1]);
				//Enqueue(sender);
			}
		}
	
		double start = seconds();
		#pragma omp parallel for
		for (int i = myStart; i < myEnd; i++)
		{
			int x = 2 * i, y = 2 * i + 1;
			accel[x] = accel[y] = 0.0;
			for (int j = myStart; j < myEnd; j++)
			{
				if (i != j)
				{
					long double grav_mass = G * bodies[j].mass;
					long double x_diff = bodies[j].x - bodies[i].x;
					long double y_diff = bodies[j].y - bodies[i].y;
					long double dist = hypotl(x_diff, y_diff);
					dist = dist * dist * dist;
					accel[x] += grav_mass * x_diff / dist;
					accel[y] += grav_mass * y_diff / dist;
				}
			}
		}

		for (int z = 1; z < np; z++) {
			int indx = z;
			if (iteration != 0) {
				MPI_Waitany(np - 1, receive_requests, &indx, MPI_STATUSES_IGNORE);
				indx++;
			}
		
			//int sender = Front();
			//int sender = (self - z + np) % np;
			int sender = (self - indx + np) % np;
			int remoteStart = displs[sender] / 5;
			int remoteEnd = remoteStart + counts[sender] / 5;
			/*if (iteration != 0) {
				MPI_Wait(&receive_requests[z - 1], MPI_STATUS_IGNORE);
				
				MPI_Recv(&bodies[remoteStart], counts[sender], MPI_LONG_DOUBLE, sender, 0,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}*/
				
			for (int i = myStart; i < myEnd; i++)
			{
				int x = 2 * i, y = 2 * i + 1;
				//accel[x] = accel[y] = 0.0;
				for (int j = remoteStart; j < remoteEnd; j++)
				{
					if (i != j)
					{
						long double grav_mass = G * bodies[j].mass;
						long double x_diff = bodies[j].x - bodies[i].x;
						long double y_diff = bodies[j].y - bodies[i].y;
						long double dist = hypotl(x_diff, y_diff);
						dist = dist * dist * dist;
						accel[x] += grav_mass * x_diff / dist;
						accel[y] += grav_mass * y_diff / dist;
					}
				}
			}
		}
		
		if (iteration != 0) {
			MPI_Waitall(np - 1, requests, MPI_STATUSES_IGNORE);
		}
		
		for (int i = myStart; i < myEnd; i++)
		{
			int x = 2 * i, y = 2 * i + 1;
			
			bodies[i].x += bodies[i].vx * delta_t + 0.5 * accel[x] * delta_t_squared;
			bodies[i].y += bodies[i].vy * delta_t + 0.5 * accel[y] * delta_t_squared;
			
			bodies[i].vx += accel[x] * delta_t;
			bodies[i].vy += accel[y] * delta_t;
		}
		calc_time += seconds() - start;
		
		if (iteration != num_steps - 1)
		{
			for (int i = 1; i < np; i++) {
				int receiver = (self + i) % np;
				MPI_Isend(&bodies[myStart], counts[self], MPI_LONG_DOUBLE, receiver, 0, MPI_COMM_WORLD,
					&requests[i - 1]);
			}
		} else {
			MPI_Allgatherv(MPI_IN_PLACE, counts[self], MPI_LONG_DOUBLE, bodies,
			counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
		}
		
		//saveImage(iteration, bodies, nBodies, &params);
	}
	
	
	fprintf(stderr, "gather share: %f\n", gather_time / (calc_time + gather_time));
	
	free(receive_requests);
	free(requests);
	free(counts);
	free(accel);
}

void simulate(body *bodies, int nBodies)
{
	long double *accel =
		malloc(2 * nBodies * sizeof(long double));
		
	long double delta_t_squared = delta_t * delta_t;
	
	// Use hardcoded path for testing.
	struct ImgParams params;
	params.imgFilePrefix = "iter";
    params.imgWidth = params.imgHeight = 200;
    params.width = params.height = 2.0e21;
	
	for (int iteration = 0; iteration < num_steps; iteration++)
	{
		for (int i = 0; i < 2 * nBodies; i++) {
			accel[i] = 0.0;
		}
	
		for (int i = 0, x = 0; i < nBodies; i++, x += 2)
		{
			int y = x + 1;
			for (int j = i + 1, x_t = 2 * j; j < nBodies; j++, x_t += 2)
			{
				int y_t = x_t + 1;
				long double x_diff = bodies[j].x - bodies[i].x;
				long double y_diff = bodies[j].y - bodies[i].y;
				long double dist = hypotl(x_diff, y_diff);
				dist *= dist * dist;
				long double without_mass_x = G * x_diff / dist;
				accel[x]   += without_mass_x * bodies[j].mass;
				accel[x_t] -= without_mass_x * bodies[i].mass;
				long double without_mass_y = G * y_diff / dist;
				accel[y]   += without_mass_y * bodies[j].mass;
				accel[y_t] -= without_mass_y * bodies[i].mass;
			}
		}
		
		for (int i = 0; i < nBodies; i++)
		{
			int x = 2 * i, y = x + 1;
			
			bodies[i].x += bodies[i].vx * delta_t + 0.5 * accel[x] * delta_t_squared;
			bodies[i].y += bodies[i].vy * delta_t + 0.5 * accel[y] * delta_t_squared;
			
			bodies[i].vx += accel[x] * delta_t;
			bodies[i].vy += accel[y] * delta_t;
		}
		
		//saveImage(iteration, bodies, nBodies, &params);
	}
	
	free(accel);
}

static inline long double relative_error(long double a, long double b)
{
	return fabsl((a - b) / a);
}

/*
 * Testprogramm fuer readBodies.
 */
int main(int argc, char *argv[])
{
    //struct ImgParams params;

    /*if (argc != 3) {
	fprintf(stderr, "Need exactly two arguments: "
                "a .dat file and an image file\n");
	return 1;
    }*/
    
    int verbose = 0;
    
    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &self);
    
	int numBodies;
	body *bodies;
    if (readBodiesFromFile("twogalaxies.dat", &bodies, &numBodies)) {
    	fprintf(stderr, "Could not read input file\n");
    	return 1;
    }

	double start = seconds();
	//simulate(bodies, numBodies);
    simulateDistributed(bodies, numBodies);
    fprintf(stderr, "time: %f\n", seconds() - start);
    
    
    long double px, py;
    if (verbose) {
		printf("Calculated results:\n");
		for (int i=0; i<numBodies && i < 5; i++) {
		printf("Body %d: mass = %Lg, x = %Lg, y = %Lg, vx = %Lg, vy = %Lg\n",
			   i, bodies[i].mass, bodies[i].x, bodies[i].y,
			   bodies[i].vx, bodies[i].vy);
		}
		totalImpulse(bodies, numBodies, &px, &py);
    	printf("Calculated impulse: px=%Lg, py=%Lg\n", px, py);
	}
    
	body *reference_bodies;
    if (readBodiesFromFile("twogalaxies_nach_100_Schritten.dat", &reference_bodies, &numBodies)) {
    	fprintf(stderr, "Could not read input file\n");
    	return 1;
    }
    
    if (verbose) {
		printf("Reference results:\n");
		for (int i=0; i<numBodies && i < 5; i++) {
		printf("Body %d: mass = %Lg, x = %Lg, y = %Lg, vx = %Lg, vy = %Lg\n",
			   i, reference_bodies[i].mass, reference_bodies[i].x, reference_bodies[i].y,
			   reference_bodies[i].vx, reference_bodies[i].vy);
		}
		totalImpulse(reference_bodies, numBodies, &px, &py);
    	printf("Reference impulse: px=%Lg, py=%Lg\n", px, py);
	}
    
    // Write to output file and read in again to get the same precision
    // as the reference file was written with.
    writeBodiesToFile("out.dat", bodies, numBodies);
    readBodiesFromFile("out.dat", &bodies, &numBodies); 

    // Calculate the maximum relative error between calculated and reference values.
    long double max_diff_x = 0;
    long double max_diff_y = 0;
    long double max_diff_vx = 0;
    long double max_diff_vy = 0;
    int num_different_vals = 0;
    for (int i=0; i<numBodies; i++) {
    	long double diff_x = relative_error(bodies[i].x, reference_bodies[i].x);
    	long double diff_y = relative_error(bodies[i].y, reference_bodies[i].y);
    	long double diff_vx = relative_error(bodies[i].vx, reference_bodies[i].vx);
    	long double diff_vy = relative_error(bodies[i].vy, reference_bodies[i].vy);
    	max_diff_x = max(diff_x, max_diff_x);
    	max_diff_y = max(diff_y, max_diff_y);
    	max_diff_vx = max(diff_vx, max_diff_vx);
    	max_diff_vy = max(diff_vy, max_diff_vy);
    	num_different_vals += (bodies[i].x != reference_bodies[i].x) + 
    		(bodies[i].y != reference_bodies[i].y) +
    		(bodies[i].vx != reference_bodies[i].vx) +
    		(bodies[i].vy != reference_bodies[i].vy);
    }
    printf("max_diff_x: %Lg, max_diff_y: %Lg, max_diff_vx: %Lg, max_diff_vy: %Lg\n",
    	max_diff_x, max_diff_y, max_diff_vx, max_diff_vy);
	printf("Number of different values: %d\n", num_different_vals);
    	
	MPI_Finalize();

    return 0;
}
