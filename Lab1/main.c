#include "stdio.h"
#include "stdlib.h"
#include <time.h>

void createMatrix(double **a, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            a[i][j] = rand() % 9 + 1;
    }
}

double det(double **a, int n) {
    double mult = 1;
    for (int k = 0; k < n - 1; k++) {
        for (int i = k; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                a[i][j] /= a[i][k];
            }
            mult *= a[i][k];
            a[i][k] = 1;
        }
        for (int i = k + 1; i < n; i++) {
            for (int j = k; j < n; j++) {
                a[i][j] -= a[k][j];
            }
        }
    }
    return mult * a[n - 1][n - 1];
}

double timeOfDet(int n, double **a) {
    clock_t bg = clock();
    det(a, n);
    double time_go = clock() - bg;
    time_go /= 1000;
    return time_go;
}

void speedTest(int start, int finish, int step) {
    int k = 0;
    double *timeToFile = (double *) malloc((((finish - start) / step) * sizeof(double)));
    for (int i = start; i < finish; i += step) {
        double **a = (double **) malloc(sizeof(double *) * (i + 1));
        for (int j = 0; j < i + 1; j++) {
            a[j] = (double *) malloc(sizeof(double) * (i + 1));
        }
        createMatrix(a, i + 1);
        double t = timeOfDet(i + 1, a);
        timeToFile[k++] = t;
        //printf("%.6f\n", t);
        for (int j = 0; j < i + 1; ++j) {
            free(a[j]);
        }
        free(a);
    }
    FILE *fp;
    if ((fp = fopen("write.txt", "w")) != NULL) {
        for (int i = 0; i < (finish - start) / step; ++i) {
            fprintf(fp, "%.6f\n", timeToFile[i]);
        }
    }
    fclose(fp);
    free(timeToFile);
}

int main() {
    int start = 0;
    int finish = 500;
    int step = 1;
    printf("%d %d %d", start, finish, step);
    speedTest(start, finish, step);
    printf("\nFinish");
    return 0;
}