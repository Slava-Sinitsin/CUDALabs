#include "cuda_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

void filling(double *a, int n) {
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 9 + 1;
    }
}

__global__ void div(double *a, const int *N, const int *shift, const int *level) {
    int i = threadIdx.x;
    int j = *N - threadIdx.y - 1;
    a[i * (*N) + j + (*shift) + i * (*level)] /= a[i * (*N) + (*shift) + i * (*level)];
}

__global__ void sub(double *a, const int *N, const int *shift, const int *level) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    a[(i + 1) * (*N) + (*shift) + (i + 1) * (*level) + j] -= a[(*shift) + j];
}

__global__ void mult(double *a, int *N, const int *shift, const int *level, double *answer) {
    for (int i = 0; i < *N; i++) {
        (*answer) *= a[i * (*N) + (*shift) + i * (*level)];
    }
}

double detCalc(double *a, int N) {
    double answer = 1;
    int shift = 0;
    int dailySize = 0;
    double *a_d;
    int *N_d;
    int *shift_d;
    int *level_d;
    double *answer_d;
    cudaMalloc((void **) &a_d, N * N * sizeof(double));
    cudaMalloc((void **) &N_d, sizeof(int));
    cudaMalloc((void **) &shift_d, sizeof(int));
    cudaMalloc((void **) &level_d, sizeof(int));
    cudaMalloc((void **) &answer_d, sizeof(double));
    cudaMemcpy(a_d, a, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(answer_d, &answer, sizeof(double), cudaMemcpyHostToDevice);
    for (int level = 0; level < N; level++) {
        dailySize = N - level;
        shift = level * (N + 1);
        cudaMemcpy(N_d, &dailySize, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(shift_d, &shift, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(level_d, &level, sizeof(int), cudaMemcpyHostToDevice);
        mult <<<1, dim3(1, 1, 1) >>>(a_d, N_d, shift_d, level_d, answer_d);
        cudaDeviceSynchronize();
        div <<<8, dim3(dailySize, dailySize, 1) >>>(a_d, N_d, shift_d, level_d);
        cudaDeviceSynchronize();
        sub <<<8, dim3(dailySize - 1, dailySize, 1) >>>(a_d, N_d, shift_d, level_d);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(&answer, answer_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(N_d);
    cudaFree(shift_d);
    cudaFree(answer_d);
    return answer;
}

double getTime(double *a, int n) {
    clock_t bg = clock();
    detCalc(a, n);
    double time_go = clock() - bg;
    time_go /= 1000;
    return time_go;
}

void getInfo() {
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device name: %s\n", deviceProp.name);
    printf("Total global memory: %ull\n", deviceProp.totalGlobalMem);
    printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
    printf("Registers per block: %d\n", deviceProp.regsPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("Memory pitch: %d\n", deviceProp.memPitch);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("Max grid size: x = %d, y = %d, z = %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("Clock rate: %d\n", deviceProp.clockRate);
    printf("Total constant memory: %d\n", deviceProp.totalConstMem);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Texture alignment: %d\n", deviceProp.textureAlignment);
    printf("Device overlap: %d\n", deviceProp.deviceOverlap);
    printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n", deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
}

void speedTest(int start, int finish, int step) {
    auto *timeToFile = (double *) malloc((((finish - start) / step) * sizeof(double)));
    int k = 0;
    for (int i = start; i < (finish - start) / step; i += step) {
        auto *a = (double *) malloc((i + 1) * (i + 1) * sizeof(double));
        filling(a, i + 1);
        timeToFile[k++] = getTime(a, i + 1);
        free(a);
        //printf("%f\n", timeToFile[i]);
    }
    FILE *fp;
    timeToFile[0] = 0;
    if ((fp = fopen("write.txt", "w")) != nullptr) {
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
    getInfo();
    printf("%d %d %d", start, finish, step);
    speedTest(start, finish, step);
    printf("\nFinish");
    return 0;
}