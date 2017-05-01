#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include <mpi.h>

/* PGM read & write code from https://ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/ */

typedef struct _PGMData {
	int size;
	int col;
	int row;
    int max_gray;
    int *pixels;
} PGMData;

/*int **allocate_dynamic_matrix(int row, int col)
{
    int **ret_val;
    int i;
 
    ret_val = (int **)malloc(sizeof(int *) * row);
    if (ret_val == NULL) {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }
 
    for (i = 0; i < row; ++i) {
        ret_val[i] = (int *)malloc(sizeof(int) * col);
        if (ret_val[i] == NULL) {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }
 
    return ret_val;
}*/
 
/*void deallocate_dynamic_matrix(int **matrix, int row)
{
    int i;
 
    for (i = 0; i < row; ++i)
        free(matrix[i]);
    free(matrix);
}*/

#define HI(num) (((num) & 0x0000FF00) >> 8)
#define LO(num) ((num) & 0x000000FF)

void SkipComments(FILE *fp)
{
    int ch;
    char line[100];
 
    while ((ch = fgetc(fp)) != EOF && isspace(ch))
        ;
    if (ch == '#') {
        fgets(line, sizeof(line), fp);
        SkipComments(fp);
    } else
        fseek(fp, -1, SEEK_CUR);
}

/*for reading:*/
PGMData* readPGM(const char *file_name, PGMData *data)
{
    FILE *pgmFile;
    char version[3];
    int i, j;
    int lo, hi;
 
    pgmFile = fopen(file_name, "rb");
    if (pgmFile == NULL) {
        perror("cannot open file to read");
        exit(EXIT_FAILURE);
    }
 
    fgets(version, sizeof(version), pgmFile);
    if (strcmp(version, "P5")) {
        fprintf(stderr, "Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
 
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->row);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->col);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->max_gray);
    fgetc(pgmFile);
	data->size = data->row * data->col;
 
	data->pixels = malloc(sizeof(int) * data->size);
    if (data->max_gray > 255)
        for (i = 0; i < data->size; ++i) {
			hi = fgetc(pgmFile);
			lo = fgetc(pgmFile);
			data->pixels[i] = (hi << 8) + lo;
        }
    else
        for (i = 0; i < data->size; ++i) {
			lo = fgetc(pgmFile);
			data->pixels[i] = lo;
        }
 
    fclose(pgmFile);
    return data;
}

/*and for writing*/
 
void writePGM(const char *filename, const PGMData *data)
{
    FILE *pgmFile;
    int i, j;
    int hi, lo;
 
    pgmFile = fopen(filename, "wb");
    if (pgmFile == NULL) {
        perror("cannot open file to write");
        exit(EXIT_FAILURE);
    }
 
    fprintf(pgmFile, "P5 ");
    fprintf(pgmFile, "%d %d ", data->col, data->row);
    fprintf(pgmFile, "%d ", data->max_gray);
 
    if (data->max_gray > 255) {
        for (i = 0; i < data->size; ++i) {
			hi = HI(data->pixels[i]);
			lo = LO(data->pixels[i]);
			fputc(hi, pgmFile);
			fputc(lo, pgmFile);
        }
    } else {
        for (i = 0; i < data->size; ++i) {
			lo = LO(data->pixels[i]);
			fputc(lo, pgmFile);
		}
    }
 
    fclose(pgmFile);
	free(data->pixels);
}

int max(int a, int b)
{
    return a > b ? a : b;
}

int min(int a, int b)
{
    return a > b ? b : a;
}

void sequential(PGMData *data, int n_min, int n_max)
{
    int i, j;
    int a_min = INT_MAX, a_max = INT_MIN;
    for (i = 0; i < data->size; i++)
    {
		a_min = min(data->pixels[i], a_min);
		a_max = max(data->pixels[i], a_max);
    }
    printf("a_min: %d, a_max: %d\n", a_min, a_max);

    for (i = 0; i < data->size; i++)
    {
		data->pixels[i] = (((data->pixels[i] - a_min) * (n_max - n_min) +
			(a_max - a_min) / 2) / (a_max - a_min)) + n_min;
    }
}

/*void parallel(PGMData *data, int n_min, int n_max)
{
    int i, j;
    int a_min = INT_MAX, a_max = INT_MIN;
    int local_min, local_max;
    
    #pragma omp parallel private(i, j, local_min, local_max)
    {
        local_min = INT_MAX;
        local_max = INT_MIN;
    
        #pragma omp for nowait
        for (i = 0; i < data->row; i++)
        {
            for (j = 0; j < data->col; j++)
            {
                local_min = min(data->pixels[i], local_min);
                local_max = max(data->pixels[i], local_max);
            }
        }
        
        printf("tid: %d, local_min: %d, local_max: %d\n",
            omp_get_thread_num(), local_min, local_max);
        
        #pragma omp critical
        {
            a_min = min(local_min, a_min);
            a_max = max(local_max, a_max);
        }
    }
    printf("a_min: %d, a_max: %d\n", a_min, a_max);

    #pragma omp parallel for private(j)
    for (i = 0; i < data->row; i++)
    {
        for (j = 0; j < data->col; j++)
        {
            data->pixels[i] = (((data->pixels[i] - a_min) * (n_max - n_min) +
                (a_max - a_min) / 2) / (a_max - a_min)) + n_min;
        }
    }
}*/

int round_up_div(int x, int y)
{
	return (x + y - 1) / y;
}

void distributed(PGMData *data, int n_min, int n_max, int rank, int num_processes)
{
	int num_pixels_per_process = round_up_div(data->size, num_processes);
	int start = rank * num_pixels_per_process;
	int end =  min((rank + 1) * num_pixels_per_process, data->size);
	
	printf("Process %d is assigned pixels %d to %d\n", rank, start, end);
	
    int i;
    int process_min = INT_MAX, process_max = INT_MIN;
    for (i = start; i < end; i++)
    {
		process_min = min(data->pixels[i], process_min);
		process_max = max(data->pixels[i], process_max);
    }
    printf("process: %d, process_min: %d, process_max: %d\n", rank, process_min, process_max);
	
	int a_min, a_max;
	MPI_Allreduce(&process_min, &a_min, 1, MPI_INT, MPI_MIN,
				  MPI_COMM_WORLD);
    MPI_Allreduce(&process_max, &a_max, 1, MPI_INT, MPI_MAX,
				  MPI_COMM_WORLD);
	
	printf("process: %d, a_min: %d, a_max: %d\n", rank, a_min, a_max);

    for (i = start; i < end; i++)
    {
		data->pixels[i] = (((data->pixels[i] - a_min) * (n_max - n_min) +
			(a_max - a_min) / 2) / (a_max - a_min)) + n_min;
    }
	
	int *receive_counts = malloc(sizeof(int) * num_processes);
	int *receive_displacements = malloc(sizeof(int) * num_processes);
	for (i = 0; i < num_processes; i++)
	{
		receive_counts[i] = num_pixels_per_process;
		receive_displacements[i] = num_pixels_per_process * i;
	}
	receive_counts[i] = data->size - (num_processes + 1) * num_pixels_per_process;
	
	int send_length = end - start;
	MPI_Gatherv(&data->pixels[start], send_length, MPI_INT, data->pixels,
		receive_counts, receive_displacements, MPI_INT, 0, MPI_COMM_WORLD);
	
	free(receive_counts);
	free(receive_displacements);
}

int main(int argc, char **argv)
{
    int n_min = 50, n_max = 150;
    if (argc == 1)
    {
        printf("You have to specify a file name.\n");
        exit(-1);
    }
    else if (argc == 4)
    {
        n_min = atoi(argv[2]);
        n_max = atoi(argv[3]);
    }
    printf("n_min: %d, n_max: %d\n", n_min, n_max); 
	
	MPI_Init(NULL, NULL);
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int num_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    PGMData *data = malloc(sizeof(PGMData));
    data = readPGM(argv[1],  data);
    printf("width: %d, height: %d\n", data->row, data->col);

    //sequential(data, n_min, n_max);
    //parallel(data, n_min, n_max);
	distributed(data, n_min, n_max, rank, num_processes);
    
	if (rank == 0) {
		writePGM("out.pgm", data);
	}
    free(data);
	
	MPI_Finalize();
}
