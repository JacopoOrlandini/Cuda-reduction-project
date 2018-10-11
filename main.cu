#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#define h 358	//column
#define w  32  //row
#define multi 2
#define N w*h

__global__ void reduce(float *g_idata, float *g_odata);
void fill_array (float* matrix, int dim);
void stampa_mat(float* matrix);
float *computeHost(float *input);


int main( void ) {
    float a[N];
    float *resultHost;
    float *b, *dev_a, *dev_b;
    int size2D;
    int colSizeMem;
    int countError;
    int dimFloat = sizeof(float);

    colSizeMem = dimFloat * h;
    size2D = N * dimFloat;
    countError = 0;
    resultHost = (float*) malloc(colSizeMem);
    cudaMallocHost( (void **) &b, colSizeMem);
    cudaMalloc( (void**)&dev_a, size2D );
    cudaMalloc( (void**)&dev_b, colSizeMem);

    fill_array( a, N );
    printf("+-------------------------+\n");
    stampa_mat(a);
    printf("+-------------------------+\n\n");

    resultHost = computeHost(a);

    cudaMemcpy( dev_a, a, size2D , cudaMemcpyHostToDevice );
    dim3 blocksize(w);//numero di thread corrispondenti alle righe
    dim3 gridsize(h); //numero di blocchi corrispondenti alle colonne
    reduce<<<gridsize, blocksize>>>(dev_a, dev_b);
    cudaMemcpy( b, dev_b, colSizeMem, cudaMemcpyDeviceToHost);

    /*
     * Checking error from cuda to cpu execution
     */

    for (int i = 0; i < h; ++i) {
    	if(resultHost[i] != b[i])
    		countError++;
		printf("resultHost = %f, b = %f\n",resultHost[i],b[i]);
	}

    if(countError!=0){
    	printf("trovati errori nell'elaborazione\n");
    	printf("Error count: %d \n", countError);
    }
    else
    	printf("Nessun errore trovato, tutto corretto\n");
    /*
     * Clening memory Host and Device
     */
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree(b);
    return 0;
}

__global__ void reduce(float *g_idata, float *g_odata) {

	__shared__ float sdata[w];

    int i = blockIdx.x + gridDim.x*threadIdx.x; //gridDIm = h, blockDIm = w
    int tid = threadIdx.x;
    sdata[tid] = g_idata[i];
	//printf("[loaded]	sdata[%f]=%f, blckIdx.x:%f\n",i,sdata[threadIdx.x],blockIdx.x);
    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x ){
        	if( (index+s) >= blockDim.x )
        		sdata[index+s] = 0;
        	sdata[index] += sdata[index + s];
        }
    	__syncthreads();
    }

    if (tid == 0){
    	g_odata[blockIdx.x] = sdata[0]/w;
    	__syncthreads();
    }
    //prende tutti i valori dei vari blocchi e li somma
        /*
         * atomicAdd(g_odata[blockIdx.x],sdata[tid]);
         * in questo caso quel che succede è:
         * sdata[0]=256; //blc0
         * sdata[0]=256; //blc1
         * sdata[0]=256; //blc2
         * sdata[0]=256; //blc3
         * L'atomic add restituisce 1024
         */
}

// CPU function to generate a vector of random integers
void fill_array (float*a, int n)
{
    for (int i = 0; i < n; i++){
        a[i] = i;
    }
}
float *computeHost(float *input){
  float *output;
  for (int i = 0; i < h; ++i) {
	  output[i] = 0;
    for (int j = 0;  j < w; j++)
    	output[i] += input[j*h+i];
    output[i] = output[i]/w;
}
  return output;
}
void stampa_mat(float*a){
	int i, j;
	for (i = 0; i < w; ++i) {
		for (j = 0; j < h; ++j) {
			printf("%f\t",a[j+i*h]);
		}
		printf("\n");
	}
}
