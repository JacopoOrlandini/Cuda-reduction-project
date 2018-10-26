
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "device_launch_parameters.h"

//Standard C library
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define subCOL 5248
#define COL 5248
#define ROW 358
#define WARPABLEROW 512 
#define blocksize 256
#define subMatDim subCOL*WARPABLEROW
#define targetMatDim ROW * COL
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
template <unsigned int blockSize> __global__ void reduce(float *g_idata, float *g_odata);
template <unsigned int blockSize> __device__ void warpReduce(volatile float* sdata, int tid);
float convesionFloat(char *str);
char* mystrsep(char **stringp, const char *delim);
void stampa_mat(float*a);


int main(int argc, char *argv[])	{
	printf("Running app: %s\nINFO\n\n", argv[0]);
	printf("--------------------------\n");
	printf("COL = %d\nROW = %d\nsubCOL = %d\nWarpable row = %d\n blocksize = %d\n", COL, ROW, subCOL,WARPABLEROW,blocksize);
	printf("Executing on s00_Target.txt\n");
	printf("--------------------------\n\n");

	/*Declaration*/
	float *dev_input, *dev_output;
	float *subMatrix;
	float **A = new float*[ROW];
	float *h_b;		

	FILE *file,*output;

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
	/*Versione pinned*/
	cudaStatus = cudaMallocHost(&subMatrix, subMatDim * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed!");
		goto Error;
	}/**/
	/*Version unpinned
	subMatrix = (float*)malloc(sizeof(float)*subMatDim);
	/**/
	printf("Cuda malloc host on subMatrix\n");
	
	cudaStatus = cudaMallocHost(&h_b, subCOL * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed h_b!");
		goto Error;
	}
	printf("Cuda malloc host on h_b\n");

	cudaStatus = cudaMalloc(&dev_input, subMatDim * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	printf("CudaMalloc on dev_input\n");
	cudaStatus = cudaMalloc(&dev_output, subCOL * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	printf("CudaMalloc on dev_output\n");
	
	printf("Malloc on host_output\n");
	//Opening file from folder
	printf("Opening file from folder\n");
	file = fopen("s00_target.txt", "r");
	if (file == NULL)
	{
		printf("Error opening file\n");
		goto Error;
	}
	printf("Initialitation of Matrix A\n");

	//Instantiating matrix : target 358*5248
	A[0] = new float[ROW * COL];
	for (int i = 1; i < ROW; ++i)
		A[i] = A[i - 1] + COL;
	for (int i = 0; i < ROW; i++) {
		for (int j = 0; j < COL; j++) {
			char *string = (char *)malloc(sizeof(char) * 40);
			if (fscanf(file, "%s ", string) <= 0)
				perror("Error reading from file\n");
			A[i][j] = convesionFloat(string);
		}
	}
	//Instatiating subMatrix
	for (int i = 0; i < subCOL; i++) {
		for (int j = 0; j < WARPABLEROW; j++) {
			if(j > 357)//359-esimo elemento
				subMatrix[j + i * WARPABLEROW] = 0;
			else
			subMatrix[j + i* WARPABLEROW] = A[j][i];
		}
	}
	printf("Instantiation subMatrix");
	gpuErrchk(cudaMemcpy(dev_input, subMatrix, subMatDim * sizeof(float) , cudaMemcpyHostToDevice));
	printf("Copied subMatrix memory to dev_input\n");
	dim3 blockSize(blocksize,1,1);//numero di thread corrispondenti alle righe
	printf("Created blocksize\n");
	dim3 gridsize(subCOL,1,1); //numero di blocchi corrispondenti alle colonne
	printf("Created gridSize\n");
	reduce<blocksize><<<gridsize, blockSize >>> (dev_input, dev_output);
	cudaDeviceSynchronize();
	printf("\nKernel completed\n");
	cudaMemcpy(h_b, dev_output, sizeof(float)*subCOL , cudaMemcpyDeviceToHost);
	printf("Transfer dev_output to b\n");
	output = fopen("target.csv","w");
	for (int i = 0; i < subCOL; i++)	
		fprintf(output,"%f,",h_b[i]);
	printf("[SUCCESS]Written file output");
	
	fclose(output);
	fclose(file);
	cudaFreeHost(subMatrix);
	cudaFreeHost(h_b);
	cudaFree(dev_input);
	cudaFree(dev_output);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}
	printf("All right, close program\n");
	return 1;
Error:
	printf("Error on program, deallocating memory\n");
	return 0;
}



template <unsigned int blockSize> __global__ void reduce(float *g_idata, float *g_odata) {
	__shared__ float sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize *2 + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockSize];
	__syncthreads();
	//if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0]/ROW;

}
template <unsigned int blockSize> __device__ void warpReduce(volatile float* sdata, int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
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
	int i;

	for (i = 0; i < ROW; ++i) {
		printf("%f\t ", a[i]);
		if (i % 10 == 0 && i > 0)
			printf("\n");
	}
}

/*Unused Version of reduce
*g_size not used for this project: erasable
*/

__global__ void reduce1(int *g_idata, int *g_odata, int g_size)
{
	__shared__ int sdata[blocksize];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(int *g_idata, int *g_odata, int g_size)
{
	__shared__ int sdata[blocksize];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		int index = 2 * s*tid;

		if (index < blockDim.x)
		{
			sdata[index] += sdata[index + s];
		}

		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce3(int *g_idata, int *g_odata, int g_size)
{
	__shared__ int sdata[blocksize];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce4(int *g_idata, int *g_odata, int g_size)
{
	__shared__ int sdata[blocksize];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce5(int *g_idata, int *g_odata, int g_size)
{
	__shared__ int sdata[blocksize];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}