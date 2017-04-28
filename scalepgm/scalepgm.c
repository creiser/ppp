#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

/* PGM read & write code from https://ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/ */

typedef struct _PGMData {
    int row;
    int col;
    int max_gray;
    int **matrix;
} PGMData;

int **allocate_dynamic_matrix(int row, int col)
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
}
 
void deallocate_dynamic_matrix(int **matrix, int row)
{
    int i;
 
    for (i = 0; i < row; ++i)
        free(matrix[i]);
    free(matrix);
}

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
    fscanf(pgmFile, "%d", &data->col);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->row);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->max_gray);
    fgetc(pgmFile);
 
    data->matrix = allocate_dynamic_matrix(data->row, data->col);
    if (data->max_gray > 255)
        for (i = 0; i < data->row; ++i)
            for (j = 0; j < data->col; ++j) {
                hi = fgetc(pgmFile);
                lo = fgetc(pgmFile);
                data->matrix[i][j] = (hi << 8) + lo;
            }
    else
        for (i = 0; i < data->row; ++i)
            for (j = 0; j < data->col; ++j) {
                lo = fgetc(pgmFile);
                data->matrix[i][j] = lo;
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
        for (i = 0; i < data->row; ++i) {
            for (j = 0; j < data->col; ++j) {
                hi = HI(data->matrix[i][j]);
                lo = LO(data->matrix[i][j]);
                fputc(hi, pgmFile);
                fputc(lo, pgmFile);
            }
 
        }
    } else {
        for (i = 0; i < data->row; ++i)
            for (j = 0; j < data->col; ++j) {
                lo = LO(data->matrix[i][j]);
                fputc(lo, pgmFile);
            }
    }
 
    fclose(pgmFile);
    deallocate_dynamic_matrix(data->matrix, data->row);
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
    for (i = 0; i < data->row; i++)
    {
        for (j = 0; j < data->col; j++)
        {
            a_min = min(data->matrix[i][j], a_min);
            a_max = max(data->matrix[i][j], a_max);
        }
    }
    printf("a_min: %d, a_max: %d\n", a_min, a_max);

    for (i = 0; i < data->row; i++)
    {
        for (j = 0; j < data->col; j++)
        {
            int old = data->matrix[i][j];
            data->matrix[i][j] = (((data->matrix[i][j] - a_min) * (n_max - n_min) +
                (a_max - a_min) / 2) / (a_max - a_min)) + n_min;
        }
    }
}

void parallel(PGMData *data, int n_min, int n_max)
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
                local_min = min(data->matrix[i][j], local_min);
                local_max = max(data->matrix[i][j], local_max);
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
            data->matrix[i][j] = (((data->matrix[i][j] - a_min) * (n_max - n_min) +
                (a_max - a_min) / 2) / (a_max - a_min)) + n_min;
        }
    }
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

    PGMData *data = malloc(sizeof(PGMData));
    data = readPGM(argv[1],  data);
    printf("width: %d, height: %d\n", data->row, data->col);

    //sequential(data, n_min, n_max);
    parallel(data, n_min, n_max);
    
    writePGM("out.pgm", data);
    free(data);
}
