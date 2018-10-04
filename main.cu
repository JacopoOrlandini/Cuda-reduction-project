#include <stdio.h>

#define w 4
#define h 4
#define N w*h

__global__ void reduce(int *g_idata, int *g_odata);
void fill_array (int *a, int n);
void stampa_mat(int *a);

int main( void ) {
    int a[N], b[N];
    int *dev_a, *dev_b;
    int size = N * sizeof( int ); // we need space for N integers

    // allocate device copies of a, b, c
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, size );

    fill_array( a, N );
    int i;
    printf("Valori che voglio ottenere dalla riduzione\n");
    printf("+-------------------------+\n");
    stampa_mat(a);
    printf("+-------------------------+\n\n");
    for (i = 0; i < 2; ++i) {
    	int tot = 0;
    	for (int j = 0;  j < w; ++ j) {
    		tot += a[j+i*4];
		}
    	printf("tot[%d] = %d\n",i,tot);
	}
    b[0] = 0;  //initialize the first value of b to zero
    // copy inputs to device
    cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

    dim3 blocksize(w); // create 1D threadblock
    dim3 gridsize(w/2);  //create 1D grid

    reduce<<<gridsize, blocksize>>>(dev_a, dev_b);

    // copy device result back to host copy of c
    cudaMemcpy( b, dev_b, sizeof( int ) , cudaMemcpyDeviceToHost );

    printf("Reduced sum of Array elements = %d \n", b[0]);
    printf("Value should be: %d \n", 28);
    cudaFree( dev_a );
    cudaFree( dev_b );

    return 0;
}

__global__ void reduce(int *g_idata, int *g_odata) {

    __shared__ int sdata[w]; // per block

    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
    int i = blockIdx.x + blockDim.x*threadIdx.x;
    //blockDim = 256;
    //blockIdx.x = 0,1,2,3;
    sdata[threadIdx.x] = g_idata[i]; //sdata filled with g_idata per block
	printf("[loaded]	sdata[%d]=%d, blckIdx.x:%d\n",i,sdata[threadIdx.x],blockIdx.x);
    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadIdx.x == 0){
    	for (int i  = 0;  i  < w*1; ++i ) {
    		printf("[reduction]	sdata[%d]=%d\n",i,sdata[i]);
    		__syncthreads();
		}
        atomicAdd(g_odata,sdata[0]); //prende tutti i valori dei vari blocchi e li somma
        /*
         * in questo caso quel che succede è:
         * sdata[0]=256; //blc0
         * sdata[0]=256; //blc1
         * sdata[0]=256; //blc2
         * sdata[0]=256; //blc3
         * L'atomic add restituisce 1024
         */
    	}
}

// CPU function to generate a vector of random integers
void fill_array (int *a, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = i;
}
void stampa_mat(int *a){
	int i, j;
	for (i = 0; i < w; ++i) {
		for (j = 0; j < h; ++j) {
			printf("%d\t",a[j+i*w]);
		}
		printf("\n");
	}
}
