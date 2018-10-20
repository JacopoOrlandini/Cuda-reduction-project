
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define subCOL 16
#define COL 5248
#define ROW 358
#define subMatDim subCOL*ROW
#define targetMatDim ROW * COL

__global__ void kernel(float *g_idata, float *g_odata);

float convesionFloat(char *str);
char* mystrsep(char **stringp, const char *delim);
void stampa_mat(float*a);


int main(int argc, char *argv[])	{
	/*Declaration*/
	fprintf(stdin, "Inizio main\n");
	float *dev_input, *dev_output;
	float *subMatrix, *host_output;
	float **A = new float*[ROW];

		
	printf("Main initialitation\n");

	FILE *file;

	/*Memory allocation*/
	printf("GPU and CPU: Allocating memory\n");

	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	printf("Set device 0\n");

	// Allocate 2 GPU buffers and 1 CPU buffer
	cudaStatus = cudaMallocHost((void**)&subMatrix, subMatDim * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed!");
		//goto Error;
	}
	printf("Cuda malloc host on subMatrix\n");
	
	cudaStatus = cudaMallocHost((void**)&host_output, subCOL * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed!");
	//	goto Error;
	}		
	printf("Cuda malloc host on host_result\n");

	cudaStatus = cudaMalloc((void**)&dev_input, subMatDim * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	printf("CudaMalloc on dev_input\n");
	cudaStatus = cudaMalloc((void **)&dev_output, subCOL * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	printf("CudaMalloc on dev_output\n");
	
	printf("Malloc on host_output\n");
	//Opening file from folder
	printf("Opening file from folder\n");
	file = fopen("s00_target.txt", "r");
	if (file == NULL)
	{
		printf("Error opening file\n");
		//goto Error;
	}
	printf("Initialitation of Matrix A\n");

	//Instantiating matrix A

	A[0] = new float[ROW * COL];
	for (int i = 1; i < ROW; ++i)
		A[i] = A[i - 1] + COL;
	for (int i = 0; i < ROW; i++) {
		for (int j = 0; j < COL; j++) {
			char *string = (char *)malloc(sizeof(char) * 100);
			if (fscanf(file, "%s ", string) <= 0)
				perror("Error reading from file\n");
			A[i][j] = convesionFloat(string);
		}
	}
	//Instatiating subMatrix
	printf("Initialitatio of subMatrix\n");
	for (int i = 0; i < subCOL; ++i) {
		for (int j = 0; j < ROW; ++j) {
			subMatrix[i * ROW + j ] = A[j][i];
			}
	}
	printf("\n");
	
	cudaStatus = cudaMemcpy(dev_input, subMatrix, sizeof(subMatrix), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemCpy failed input!");
	}
	printf("Copied subMatrix memory to dev_a\n");
	dim3 blocksize(ROW, 1, 1);//numero di thread corrispondenti alle righe
	printf("Created blocksize\n");
	dim3 gridsize(subCOL, 1, 1); //numero di blocchi corrispondenti alle colonne
	printf("Created gridSize\n");
	kernel <<<gridsize, blocksize>>> (dev_input, dev_output);
	printf("Kernel completed\n");
	cudaStatus = cudaMemcpy(host_output, dev_output, sizeof(host_output), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemCpy failed output!");
	}
	printf("Transfer dev_output to b\n");
	for (int i = 0; i < 1; i++)	printf("mean value = %f\n", host_output[0]);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	printf("All right, close program\n");
	return 1;
Error:
	cudaFree(subMatrix);
	cudaFree(dev_input);
	cudaFree(dev_output);
	printf("Deallocated memory\n");
	return 0;
}


float convesionFloat(char *str) {
	float result = -99;
	int exponent = +99;
	char *s_mantissa = mystrsep(&str, "e");
	exponent = atoi(str);
	result = strtof(s_mantissa, NULL)*pow(10, exponent);
	return result;
}
char* mystrsep(char **stringp, const char *delim)
{
	char *start = *stringp, *p = start ? strpbrk(start, delim) : NULL;

	if (!p) {
		*stringp = NULL;
	}
	else {
		*p = 0;
		*stringp = p + 1;
	}

	return start;
}

void stampa_mat(float*a) {
	int i, j;

	for (i = 0; i < ROW; ++i) {
		for (j = 0; j < subCOL; ++j) {
			if( i == 0)
				printf("%f, ", a[j + i * ROW]);
		}
		printf("\n");
	}
}

__global__ void kernel(float *g_idata, float *g_odata) {

	__shared__ float sdata[ROW];

	int i = blockIdx.x *gridDim.x*threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	for (int s = 1; s < blockDim.x; s *= 2)
	{
		int index = 2 * s * threadIdx.x;
		if (index < blockDim.x) {
			if ((index + s) >= blockDim.x)
				sdata[index + s] = 0;
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0]/ROW;
		__syncthreads();
	}
}