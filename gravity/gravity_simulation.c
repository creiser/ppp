#include <getopt.h>
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

static int num_steps;
static long double delta_t;
static long double delta_t_squared;
static struct ImgParams imgParams;
static int image_save_interval;

/* Returns true if an image should be saved in the passed iteration. */
static inline int shouldSaveImage(int iteration)
{
	return iteration != num_steps - 1 &&
		image_save_interval != -1 && iteration % image_save_interval == 0;
}

static int self;
static int np;

static inline void calculateAccel(const body *bodies, long double *accel, 
	int x, int y, int i, int j)
{
	// This loop could be unrolled, but since we get rid of this problem
	// with the Newton optimization anyway, we do not address it here.
	if (i != j)
	{
		long double grav_mass = G * bodies[j].mass;
		long double x_diff = bodies[j].x - bodies[i].x;
		long double y_diff = bodies[j].y - bodies[i].y;
		long double dist = hypotl(x_diff, y_diff);
		dist *= dist * dist;
		accel[x] += grav_mass * x_diff / dist;
		accel[y] += grav_mass * y_diff / dist;
	}
}

void simulateSequential(body *bodies, int nBodies)
{
	long double *accel =
		malloc(2 * nBodies * sizeof(long double));
	
	for (int iteration = 0; iteration < num_steps; iteration++)
	{
		for (int i = 0; i < nBodies; i++)
		{
			int x = 2 * i, y = 2 * i + 1;
			accel[x] = accel[y] = 0.0;
			for (int j = 0; j < nBodies; j++)
				calculateAccel(bodies, accel, x, y, i, j);
		}
		
		// It is important to update the positions before the velocities.
		for (int i = 0; i < nBodies; i++)
		{
			int x = 2 * i, y = 2 * i + 1;
			bodies[i].x += bodies[i].vx * delta_t + 0.5 * accel[x] * delta_t_squared;
			bodies[i].y += bodies[i].vy * delta_t + 0.5 * accel[y] * delta_t_squared;
			bodies[i].vx += accel[x] * delta_t;
			bodies[i].vy += accel[y] * delta_t;
		}
		
		if (self == 0 && shouldSaveImage(iteration))
			saveImage(iteration, bodies, nBodies, &imgParams);
	}
	
	free(accel);
}

void simulateDistributed(body *bodies, int nBodies)
{
	long double *accel =
		malloc(2 * nBodies * sizeof(long double));
	
	// Calculate the share of each process as usual.
	const int body_struct_size = 5;
	int *counts = malloc(2 * np * sizeof(int));
    int *displs = &counts[np];
	displs[0] = 0;
    counts[0] = (nBodies / np + (0 < nBodies % np ? 1 : 0)) * body_struct_size;
    for (int i = 1; i < np; i++)
    {
		counts[i] = (nBodies / np + (i < nBodies % np ? 1 : 0)) * body_struct_size;
		displs[i] = displs[i-1] + counts[i-1];
    }
    int myStart = displs[self] / body_struct_size;
    int myEnd = myStart + counts[self] / body_struct_size;
    
	for (int iteration = 0; iteration < num_steps; iteration++)
	{
		// Since different processes will not write to the same position
		// in the accel array this is very straightforward to parallelize.
		#pragma omp parallel for
		for (int i = myStart; i < myEnd; i++)
		{
			int x = 2 * i, y = 2 * i + 1;
			accel[x] = accel[y] = 0.0;
			for (int j = 0; j < nBodies; j++)
				calculateAccel(bodies, accel, x, y, i, j);
		}
	
		// This could be parallized with an if clause, but for the provided input files
		// there is no need to parallelize this.
		for (int i = myStart; i < myEnd; i++)
		{
			int x = 2 * i, y = 2 * i + 1;
			bodies[i].x += bodies[i].vx * delta_t + 0.5 * accel[x] * delta_t_squared;
			bodies[i].y += bodies[i].vy * delta_t + 0.5 * accel[y] * delta_t_squared;
			bodies[i].vx += accel[x] * delta_t;
			bodies[i].vy += accel[y] * delta_t;
		}

		// Body mass is also sent here, which is an overhead, since it does not change over
		// time. In the optimized version this issue is addressed as a side effect
		// by communicating the acceleration values instead.
		MPI_Allgatherv(MPI_IN_PLACE, counts[self], MPI_LONG_DOUBLE, bodies,
			counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

		if (self == 0 && shouldSaveImage(iteration))
			saveImage(iteration, bodies, nBodies, &imgParams);
	}
	free(counts);
	free(accel);
}

static inline void calculateAccelOptimized(const body *bodies, long double *accel, 
	int x, int y, int i, int j)
{
	int x_t = 2 * j, y_t = x_t + 1;
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

void simulateDistributedOptimized(body *bodies, int nBodies)
{
	int num_threads;
	#pragma omp parallel
	num_threads = omp_get_num_threads();
	if (self == 0)
		fprintf(stderr, "Number of threads: %d\n", num_threads);

	// Every thread gets its own acceleration array to avoid
	// update conflicts.
	long double *allAccel =
		malloc(num_threads * 2 * nBodies * sizeof(long double));
	
	// Calculate the processes share, only we do not need 
	// all the counts and offsets because no Gather is performed.
	int myStart = 0;
	int myEnd = (nBodies / np + (0 < nBodies % np ? 1 : 0));
	for (int i = 1; i <= self; i++) {
		myStart = myEnd;
		myEnd = myStart + (nBodies / np + (i < nBodies % np ? 1 : 0));
	}

	for (int iteration = 0; iteration < num_steps; iteration++)
	{
		#pragma omp parallel
		{
			long double *accel = &allAccel[omp_get_thread_num() * 2 * nBodies];
			for (int i = 0; i < 2 * nBodies; i++) {
				accel[i] = 0.0;
			}

			#pragma omp for
			for (int i = myStart; i < myEnd; i++)
			{
				// columnDist specifies how many acceleration values are
				// calculated locally. Only about half of the acceleration
				// values need to be calculated locally due to Newton's
				// optimization. The other half will be calculated by other processes.
				int columnDist = nBodies / 2 -
    				((nBodies % 2 == 0 && i >= nBodies / 2) ? 1 : 0);
				int x = 2 * i, y = x + 1;
				for (int j = i + 1; j < i + 1 + columnDist; j++)
				{
					calculateAccelOptimized(bodies, accel, x, y, i, j % nBodies);
				}
			}
			
			// Iterave over j in the the outer loop so continous pieces of memory
			// will be accessed in the inner loop
			for (int j = 1; j < num_threads; j++)
			{
				long double *accel = &allAccel[j * 2 * nBodies];
				#pragma omp for
				for (int i = 0; i < 2 * nBodies; i++)
				{
					allAccel[i] += accel[i];
				}
			}
		}

		MPI_Allreduce(MPI_IN_PLACE, allAccel, 2 * nBodies, MPI_LONG_DOUBLE,
			MPI_SUM, MPI_COMM_WORLD);

		for (int i = 0; i < nBodies; i++)
		{
			int x = 2 * i, y = 2 * i + 1;
			bodies[i].x += bodies[i].vx * delta_t + 0.5 * allAccel[x] * delta_t_squared;
			bodies[i].y += bodies[i].vy * delta_t + 0.5 * allAccel[y] * delta_t_squared;
			bodies[i].vx += allAccel[x] * delta_t;
			bodies[i].vy += allAccel[y] * delta_t;
		}
		
		if (self == 0 && shouldSaveImage(iteration))
			saveImage(iteration, bodies, nBodies, &imgParams);
	}

	free(allAccel);
}

double interactionRate(const int nBodies, const int steps, const double time) {
    return nBodies * (nBodies - 1) * ((double) steps / time);
}

void usage() {
	fprintf(stderr,
		"[-m implementation][-S number_of_steps][-t time_delta][-h][-o name_of_output_file] "
		"[-I image_prefix] [-s image_size] [-i image_interval] [-r reference_file] "
		"name_of_input_file\n"
		"This program takes a .dat file with defined bodies and executes a gravity simulation on "
		"it based on the given options.\n"
		"With the \"-m\" option the implementation can be specified with an integer.\n"
		"Possible values are\n\t0: sequential\n\t1: distributed\n\t2: distributed with Newton\'s "
		"3rd law optimization. \n"
        "Use \"-S\" to specify the amount of simulation steps. Default: 100\n"
        "Use \"-t\" to specify the duration of one simulation step. Default: 3.16e10\n"
		"Use \"-h\" to display this description.\n"
		"Use \"-o\" to specify the file the simulation result should be saved to. "
		"Default: \"out.dat\".\n"
		"Use \"-I\" to specify the filename prefix of the output image. Default: \"out\".\n"
		"Use \"-s\" to specify the size (width and height) of the output image in meters. "
		"Default: 2.0e21\n"
		"Use \"-i\" to specify the interval at which output images will be saved. "
		"Default: single image generated at the end of the simulation\n"
		"Use \"-r\" to specify a reference .dat file the output will be compared to.\n"
		"The input file has to be given as the last argument.\n");
}

#define GRAVITY_SEQUENTIAL 0
#define GRAVITY_DISTRIBUTED 1
#define GRAVITY_DISTRIBUTED_OPTIMIZED 2

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &self);
    
    char *output_file = "out.dat";
    char *reference_file = NULL;
    int option;
    int implementation = GRAVITY_DISTRIBUTED_OPTIMIZED;
    num_steps = 100;
	delta_t = 3.16e10;
	imgParams.imgFilePrefix = "out";
    imgParams.imgWidth = imgParams.imgHeight = 200;
    imgParams.width = imgParams.height = 2.0e21;
    image_save_interval = -1;
	
    while ((option = getopt(argc,argv,"ho:m:S:t:I:s:i:r:")) != -1) {
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
                num_steps = atoi(optarg);
                break;
			case 't':
				delta_t = atoi(optarg);
        		break;
    		case 'I':
    			imgParams.imgFilePrefix = optarg;
    			break;
			case 's':
				imgParams.width = imgParams.height = atoi(optarg);
				break;
			case 'i':
				image_save_interval = atoi(optarg);
				break;
			case 'r':
				reference_file = optarg;
				break;
        	default:
                fprintf(stderr, "'%c' is not a defined parameter.", option);
        		break;
        }
    }
    delta_t_squared = delta_t * delta_t;
    
	int numBodies;
	body *bodies;
    if (readBodiesFromFile(argv[argc - 1], &bodies, &numBodies)) {
    	fprintf(stderr, "Could not read input file\n");
    	return 1;
    }
    
    long double px, py;
    if (self == 0)
    {
    	totalImpulse(bodies, numBodies, &px, &py);
		printf("Impulse before simulation: px=%Lg, py=%Lg\n", px, py);
    }

	double start = seconds();
	if (implementation == GRAVITY_SEQUENTIAL)
		simulateSequential(bodies, numBodies);
	else if (implementation == GRAVITY_DISTRIBUTED)
		simulateDistributed(bodies, numBodies);
	else if (implementation == GRAVITY_DISTRIBUTED_OPTIMIZED)
		simulateDistributedOptimized(bodies, numBodies);
    
    if (self == 0)
    {
    	double calculationTime = seconds() - start;
    	printf("Time: %f\n", calculationTime);
    	
    	saveImage(num_steps - 1, bodies, numBodies, &imgParams);
    
    	totalImpulse(bodies, numBodies, &px, &py);
		printf("Impulse after simulation: px=%Lg, py=%Lg\n", px, py);
		
		printf("Interaction rate of the simulation: %f\n",
			interactionRate(numBodies, num_steps, calculationTime));
		
    	// Write to output file and read in again to get the same precision
    	// as the reference file was written with.
		writeBodiesToFile(output_file, bodies, numBodies);
		
		// Check if all values match the specified reference file.
		if (reference_file != NULL)
		{
			readBodiesFromFile(output_file, &bodies, &numBodies);
			body *reference_bodies;
			if (readBodiesFromFile(reference_file,
					&reference_bodies, &numBodies)) {
				fprintf(stderr, "Could not read input file\n");
				return 1;
			}
			int num_different_vals = 0;
			for (int i = 0; i < numBodies; i++) {
				num_different_vals += (bodies[i].x != reference_bodies[i].x) + 
					(bodies[i].y != reference_bodies[i].y) +
					(bodies[i].vx != reference_bodies[i].vx) +
					(bodies[i].vy != reference_bodies[i].vy);
			}
			printf("Number of different values: %d\n", num_different_vals);
		}
	}
    	
	MPI_Finalize();
}
