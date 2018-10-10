#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define w  358
#define h 20	//colonne
#define COl 5248
#define ROW 358
#define N w*h

__global__ void reduce(float *g_idata, float *g_odata);
void fill_array (float*a, int n);
void stampa_mat(float*a);

int main( void ) {
    float a[N];
    float *b, *dev_a, *dev_b;
    int size = N * sizeof( float);
    cudaMallocHost( (void **) &b, sizeof(float)*h);
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, sizeof(float)*h );

    fill_array( a, N );
    //printf("Valori che voglio ottenere dalla riduzione\n");
    printf("+-------------------------+\n");
    stampa_mat(a);
    printf("+-------------------------+\n\n");
	float tot[h];

    for (int i = 0; i < h; ++i) {//colonna
    	tot[i] = 0;
    	for (int j = 0;  j < w; j++) {//riga
    		tot[i] += a[j*h+i];
		}
	}

    cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
    //cudaMemcpy( dev_b, b, sizeof(float)*h, cudaMemcpyHostToDevice );
    dim3 blocksize(w);//numero di thread corrispondenti alle righe
    dim3 gridsize(h); //numero di blocchi corrispondenti alle colonne
    reduce<<<gridsize, blocksize>>>(dev_a, dev_b);
    cudaMemcpy( b, dev_b, h * sizeof(float), cudaMemcpyDeviceToHost);
    /*
     * Checking error from cuda to cpu execution
     */
    int countError = 0;
    for (int i = 0; i < h; ++i) {
		printf("tot = %f, b = %f\n",tot[i],b[i]);
    	if(tot[i] != b[i]){
    		countError++;
    	}
	}
    printf("Error count: %d \n", countError);
    cudaFree( dev_a );
    cudaFree( dev_b );

    return 0;
}

__global__ void reduce(float *g_idata, float *g_odata) {
    __shared__ float sdata[w];

    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
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
        	if (blockIdx.x == 0) {
        		//printf("%lf + %f = %f\n",sdata[index],sdata[index+s],sdata[index]+sdata[index+s]);
			}
        	sdata[index] += sdata[index + s];
        }
    	__syncthreads();

    }

    if (tid == 0){
    	g_odata[blockIdx.x] = sdata[0];
    	__syncthreads();
    }


//
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
    for (int i = 0; i < n; i++)
        a[i] = i;
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
