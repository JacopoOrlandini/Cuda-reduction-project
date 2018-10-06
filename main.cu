#include <stdio.h>
#include <cuda.h>
#define w  128//righe
#define h 5248	//colonne
#define N w*h

__global__ void reduce(float*g_idata, float*g_odata);
void fill_array (float*a, int n);
void stampa_mat(float*a);

int main( void ) {
    float a[N], b[N];
    float*dev_a, *dev_b;
    float size = N * sizeof( float); // we need space for N integers

    // allocate device copies of a, b, c
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, size );

    fill_array( a, N );
//    printf("Valori che voglio ottenere dalla riduzione\n");
    printf("+-------------------------+\n");
   // stampa_mat(a);
    printf("+-------------------------+\n\n");
    for (int i = 5221; i < 5225; ++i) {//colonna
    	float tot = 0;
    	for (int j = 0;  j < w; ++ j) {//riga
    		tot += a[j*h+i];
		}
    	printf("tot[%d] = %lf\n",i,tot);
	}
    b[0] = 0;  //initialize the first value of b to zero
    // copy inputs to device
    cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

    dim3 blocksize(w); // create 1D threadblock
    dim3 gridsize(h);  //create 1D grid

    //8*358
    reduce<<<gridsize, blocksize>>>(dev_a, dev_b);

    // copy device result back to host copy of c
    cudaMemcpy( b, dev_b, sizeof( float) , cudaMemcpyDeviceToHost );

    printf("Reduced sum of Array elements = %lf \n", b[0]);
    printf("Value should be: %d \n", 28);
    cudaFree( dev_a );
    cudaFree( dev_b );

    return 0;
}

__global__ void reduce(float*g_idata, float*g_odata) {

    __shared__ float sdata[w]; // w=16

    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
    int i = blockIdx.x + gridDim.x*threadIdx.x;//non va bene per piu righe che colonne
    unsigned int tid = threadIdx.x;
    //blockDim = 256;
    //blockIdx.x = 0,1,2,3;
    sdata[tid] = g_idata[i]; //sdata filled with g_idata per block
	//printf("[loaded]	sdata[%f]=%f, blckIdx.x:%f\n",i,sdata[threadIdx.x],blockIdx.x);
    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x )
        {
      //      printf("index:%f,threadIdx:%f,blockIdx.x:%f, sdata[%f]=%f\n",index,threadIdx.x,blockIdx.x,index,sdata[index]);
        	if(threadIdx.x == 0 && blockIdx.x == 0){
        		//printf("sdata[index+s] = %f,index=%f,s=%f\n",sdata[index+s],index,s);
        	}
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0 && blockIdx.x >5220){
    	printf("[reduction]	sdata[%f]=%f\n",blockIdx.x,sdata[tid]);
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
