#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
    /*int i, j;
    int n = 5;
    int x[4];
    int sum_of_powers[100];
    #pragma omp parallel for private(j) lastprivate(x)
    for (i = 0; i < n; i++)
    {
        x[0] = 1;
        for (j = 1; j < 4; j++)
            x[j] = x[j-1] * (i+1);
        sum_of_powers[i] = x[0] + x[1] + x[2] + x[3];
    }
    int n_cubed = x[3];
    printf("n_cubed: %d\n", n_cubed);*/
    
    double area, pi, x;
    int i, n = 10000;
    
    area = 0.0;
    
    #pragma omp parallel for private(x)
    for (i = 0; i < n; i++)
    {
        x = (i + 0.5) / n;
        #pragma omp critical
        area += 4.0 / (1.0 + x * x);
    }
    pi = area / n;
    printf("pi: %lf\n", pi);
}
