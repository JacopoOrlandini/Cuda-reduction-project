#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "helper_cuda.h"
 #include <sys/stat.h>
 #include <sys/types.h>


/*
 * Info
 * matrice s00_target.txt (row,col) 358*5248
 * (DRAM), tends to have long access latencies (hundreds of clock cycles) and finite access band-width.
 * Max dimension size of a thread block (x,y,z): (1024, 1024, 64)	(GeForce 940MX nel mio caso)
 */
int *pArgc = NULL;
char **pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
		int device) {
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if (CUDA_SUCCESS != error) {
		fprintf(
				stderr,
				"cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
				error, __FILE__, __LINE__);

		exit(EXIT_FAILURE);
	}
}

#endif /* CUDART_VERSION < 5000 */

/* Info matrixMean tiling matrix
 * max n_thread= (1024,1024,64)
 *Uso la grid di piu blocchi per ottenere tutti i risultati della matrice.
 *
 *TODO Fare tutto in double
 */
__global__ void matrixMeanFLOAT(int rows, int cols, float *matrix, float *dest) {

	int iter = blockDim.x*blockIdx.x + threadIdx.x;
	//	int iter = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	for (int  i = 0;  i < rows; ++i ) {
		float iterCol = matrix[cols*i + iter]; // scorro i valori della colonna per riga
		dest[iter] += iterCol; //sommo nella destinazione in device memory.
	}
	dest[iter] = dest[iter]/rows;	//media della somma
}

__global__ void matrixMeanDouble(int rows, int cols, double *matrix, double *dest) {

	int iter = blockDim.x*blockIdx.x + threadIdx.x;
	for (int  i = 0;  i < rows; ++i ) {
		double iterCol = matrix[cols*i + iter]; // scorro i valori della colonna per riga
		dest[iter] += iterCol; //sommo nella destinazione in device memory.
	}
	dest[iter] = dest[iter]/rows;	//media della somma
}



// A function to print all prime factors of a given number n
int* primeFactors(int n)
{
	static int temp[10];
	int cycle = 0;
	int a=0;
	while (n%2 == 0)
	{
		cycle++;
		n = n/2;
	}
	temp[0]= pow(2,cycle);

	for (int i = 3; i <= sqrt(n); i = i+2)
	{
		a++;
		while (n%i == 0)
		{
			if(temp[a] == 0)
				temp[a] = i;
			else
				temp[a] = temp[a]*i;
			n = n/i;
		}
		if (temp[a] == 0) {
			a--;
		}
	}
	a++;
	if (n > 2)
		temp[a]=n;
//	int i = 0;
//	while(temp[i]!=0){
//		printf("val : %d\n",temp[i]);
//		i++;}
	return temp;
}
float convesionFloat(char *str){
	float result = -99;
	int esponente = +99;
	char *s_mantissa = strsep(&str, "e");
	esponente = atoi(str);
	result = strtof(s_mantissa,NULL)*pow(10,esponente);
	//printf("exp = %s, esponente = %d , result = %lf\n",str,esponente, result);

	return result;
}

// * Purpose: conversion Double
// * Input: Stringa da file testuale in formato
// * Output:numero in formato double
//

double convesionDouble(char *str){
	double result = -99;
	int esponente = +99;

	const char* from = str;
	char *stringExp = (char*) malloc(4);
	strncpy(stringExp, from+22,4);
	esponente = atoi(stringExp);

	char *s_mantissa = strsep(&str, "e");
	result = strtof(s_mantissa,NULL)*pow(10,esponente);
	return result;
}


int main(int argc, char **argv)
{

	pArgc = &argc;
	pArgv = argv;

	printf("%s Starting...\n\n", argv[0]);
	printf(
			" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
				static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	} else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
				driverVersion / 1000, (driverVersion % 100) / 10,
				runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
				deviceProp.major, deviceProp.minor);

		char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(msg, sizeof(msg),
				"  Total amount of global memory:                 %.0f MBytes "
				"(%llu bytes)\n",
				static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
				(unsigned long long)deviceProp.totalGlobalMem);
#else
		snprintf(msg, sizeof(msg),
				"  Total amount of global memory:                 %.0f MBytes "
				"(%llu bytes)\n",
				static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
				(unsigned long long)deviceProp.totalGlobalMem);
#endif
		printf("%s", msg);

		printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
				deviceProp.multiProcessorCount,
				_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
				_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
				deviceProp.multiProcessorCount);
		printf(
				"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
				"GHz)\n",
				deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n",
				deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",
				deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
					deviceProp.l2CacheSize);
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the
		// CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
				dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n",
				memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth,
				CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n",
				memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
					L2CacheSize);
		}

#endif

		printf(
				"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
				"%d), 3D=(%d, %d, %d)\n",
				deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
				deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
				deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf(
				"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
				deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf(
				"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
				"layers\n",
				deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
				deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %lu bytes\n",
				deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n",
				deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n",
				deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
				deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
				deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n",
				deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n",
				deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n",
				deviceProp.textureAlignment);
		printf(
				"  Concurrent copy and kernel execution:          %s with %d copy "
				"engine(s)\n",
				(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n",
				deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n",
				deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n",
				deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n",
				deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n",
				deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
				deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
						: "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n",
				deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device supports Compute Preemption:            %s\n",
				deviceProp.computePreemptionSupported ? "Yes" : "No");
		printf("  Supports Cooperative Kernel Launch:            %s\n",
				deviceProp.cooperativeLaunch ? "Yes" : "No");
		printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
				deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
				deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char *sComputeMode[] = {
				"Default (multiple host threads can use ::cudaSetDevice() with device "
				"simultaneously)",
				"Exclusive (only one host thread in one process is able to use "
				"::cudaSetDevice() with this device)",
				"Prohibited (no host thread can use ::cudaSetDevice() with this "
				"device)",
				"Exclusive Process (many threads in one process is able to use "
				"::cudaSetDevice() with this device)",
				"Unknown",
				NULL};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}

	// If there are 2 or more GPUs, query to determine whether RDMA is supported
	if (deviceCount >= 2) {
		cudaDeviceProp prop[64];
		int gpuid[64];  // we want to find the first two GPUs that can support P2P
		int gpu_p2p_count = 0;

		for (int i = 0; i < deviceCount; i++) {
			checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

			// Only boards based on Fermi or later can support P2P
			if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
					// on Windows (64-bit), the Tesla Compute Cluster driver for windows
					// must be enabled to support this
					&& prop[i].tccDriver
#endif
			) {
				// This is an array of P2P capable GPUs
				gpuid[gpu_p2p_count++] = i;
			}
		}

		// Show all the combinations of support P2P GPUs
		int can_access_peer;

		if (gpu_p2p_count >= 2) {
			for (int i = 0; i < gpu_p2p_count; i++) {
				for (int j = 0; j < gpu_p2p_count; j++) {
					if (gpuid[i] == gpuid[j]) {
						continue;
					}
					checkCudaErrors(
							cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
					printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
							prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
							can_access_peer ? "Yes" : "No");
				}
			}
		}
	}

	// csv masterlog info
	// *****************************
	// exe and CUDA driver name
	printf("\n");
	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
	char cTemp[16];

	// driver version
	sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
			(driverVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Runtime version
	sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
			(runtimeVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Device count
	sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d", deviceCount);
#else
	snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
	sProfileString += cTemp;
	sProfileString += "\n";
	printf("%s", sProfileString.c_str());

	printf("Result = PASS\n");

	/*
	 * CORE APPLICATION
	 */
	clock_t t, t_cuda;
	FILE *file;
	float f_tot;
	int offset = 7;
	static  int rows = 358;
	static const int cols = 5248;
	t = clock();	// starting clock
	FILE *output;
	printf("\n1)Target.txt\n2)NonTarget.txt\nInserire codice file:  ");
	int cod = 0;
	scanf("%d",&cod);
	if (cod == 1) {
		file=fopen("./s00_target.txt", "r");
		offset = 1;
	}else if (cod == 2){
		file=fopen("./s00_nontarget.txt", "r");
		offset = 7;
	}else{
		perror("Invalid number\n");
	}
	if(!file)
	{
		perror("Error opening file. Is file in this folder?\n");
		exit(2);
	}

	/*	ARRAY implementation Float
	 *  The basic concept is easy: flatten out the 2D array into a single dimensional array.
	 *  The matrix is a 2D dynamic matrix on the host.The memory is contiguous on the host.
	 *  Now we have a dynamic array that is contiguous in memory! We can use memcpy and/or cudaMemcpy by referencing A[0], and we can also use.
	 */
	float** A = new float*[rows*offset];
	A[0] = new float[rows *offset* cols];
	for (int i = 1; i < rows*offset; ++i) A[i] = A[i-1] + cols;

	printf("Filling table...\n");
	for (int i = 0; i < rows*offset; ++i) {
		for (int j = 0; j < cols; ++j) {
			f_tot = 0;
			char *string = (char*)malloc(sizeof(char)*100);
			if (!fscanf(file, "%s ", string))
			{
				perror("Errore in lettura della tabella\n");
			}
			else
			{
				f_tot = convesionFloat(string);
				A[i][j] = f_tot;
			}
		}
	}
	t = clock() - t;
	double time_taken = ((double)t)/CLOCKS_PER_SEC;
	printf("Elapsed time to fill table : %fs (second) \n", time_taken);

	int *p = primeFactors(cols);
	if (p[2]!=0) {
		printf("[WARNING] Selezionare suddivisione blocchi CUDA manualmente ( valore < 1024)\n");
		int a = 0;
		while(p[a] != 0){
			printf("valore disponibile : %d\n",p[a]);
			a++;
		}
		printf("Inserisci il valore del 1° blocco: ");
		scanf("%d", &p[0]);
		printf("Inserisci il valore del 2° blocco: ");
		scanf("%d", &p[1]);
	}
	for(int offset_x = 0; offset_x < offset; offset_x++){
		float** tempA = new float*[rows];
		tempA[0] = new float[rows * cols];
		for (int i = 1; i < rows; ++i){
			tempA[i] = tempA[i-1] + cols;}
		for (int c = 0; c < rows; ++c) {
			for (int d = 0; d < cols; ++d) {
				tempA[c][d] = A[c*offset+offset_x][d];
			}
		}
		const int m_size = rows*cols*sizeof(float);
		float *d_A;
		float* d_B = (float *)malloc(sizeof(float) * cols);
		float *h_C = (float *)malloc(sizeof(float) * cols);
		t_cuda = clock();
		cudaMalloc((void **)&d_A, m_size);
		cudaMalloc((void **)&d_B, cols*sizeof(float));
		cudaMemcpy(d_A, tempA[0], sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
		dim3 dimBlock( p[0], 1);	//41*128 = 5248
		dim3 dimGrid( p[1], 1);
		matrixMeanFLOAT<<<dimGrid,dimBlock>>>(rows, cols, d_A, d_B);
		cudaMemcpy(h_C, d_B, sizeof(float)*cols, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		/* Writing to file meanMatrix
		 * Sono suddivisi nei 64 sensori per 82 elementi di media.
		 */
		printf("Scrivo i file ...\n");

		char dir[50] = "./Sensore";
		char indice[10] ;
		sprintf(indice,"%d",offset_x);
		strcat(dir,indice);
		int result = mkdir(dir, 0777);


		for (int i = 0;  i < 64; ++ i)
		{
			char str[12];
			char filename[50] = "/sensore";
			char path[100] = "";
			strcat(path,dir);
			sprintf(str, "%d.csv",i);
			strcat(path,filename);
			strcat(path,str);
			output = fopen(path, "w");
			if (output == NULL){
				printf("path = %s\n",path);
				perror("Error opening file!\n");
				exit(1);
			}
			for (int l = 0;  l < 82; ++ l) {
				if(l==81)
					fprintf(output,"%lf",h_C[i*82+l]);
				else
					fprintf(output,"%lf,",h_C[i*82+l]);
			}

		}
		t_cuda = clock() - t_cuda;
		time_taken = ((double)t_cuda)/CLOCKS_PER_SEC;
		printf("File written correctly\nCuda execution time : %fs (second) \n", time_taken);
		fclose(output);

		// Clean up memory
		free(h_C);
		checkCudaErrors(cudaFree(d_A));
		checkCudaErrors(cudaFree(d_B));
	}
	//close file
	free(A);
	fclose(file);

	return 1;
}
