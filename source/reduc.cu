/*
 * reduc.cu
 *
 *  Created on: Jun 12, 2017
 *      Author: Munesh Singh
 */
#include "utils.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#define ARRSZ (1 << 25)
#define BLOCK_SIZE 1024

using namespace std;

__global__
void reduce0(int *g_idata,int *g_odata)
{
	extern __shared__ int sdata[];

	//each thread loads one element from global to shared memory
	unsigned int tid=threadIdx.x;
	unsigned int i=threadIdx.x+blockIdx.x*blockDim.x;

	sdata[tid]=g_idata[i];
	__syncthreads();


	//do reduction in shared memory
	for(unsigned int s=1;s<blockDim.x;s*=2) {
		if((tid%(2*s))==0) {
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	//write the result for this block to global mem
	if(tid==0)
		g_odata[blockIdx.x]=sdata[0];

}

int main() {
	srand(time(0));
	//host allocations
	int *h_idata, *h_odata;

	//allocating and initializing host memory
	h_idata=(int*)malloc(ARRSZ * sizeof(int));
	for(int i=0;i<ARRSZ;i++)
		h_idata[i]=randomNumber();

	for(int i=0;i<10;i++)
			cout << h_idata[i] << endl;

	//allocating device memories
	int *g_idata, *g_odata;
	gpuErrchk(cudaMalloc((void**)&g_idata,sizeof(int)*ARRSZ));


	int GRID_SIZE=divup(ARRSZ,BLOCK_SIZE);

	cout << "GRID_SIZE= " << GRID_SIZE << endl;
	gpuErrchk(cudaMalloc((void**)&g_odata,sizeof(int)*GRID_SIZE));
	h_odata=(int*)malloc(GRID_SIZE * sizeof(int));
	//gpuErrchk(cudaMemset(g_odata,0,sizeof(int)*GRID_SIZE));

	gpuErrchk(cudaMemcpy(g_idata,h_idata,sizeof(int)*ARRSZ,cudaMemcpyHostToDevice));

	reduce0<<<GRID_SIZE,BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(g_idata,g_odata);

	gpuErrchk(cudaMemcpy(h_odata,g_odata,sizeof(int)*GRID_SIZE,cudaMemcpyDeviceToHost));

	for(int i=0;i<GRID_SIZE;i++)
		cout << h_odata[i] << endl;

	//Serial reduce
	int sum=0;
	for(int i=0;i<ARRSZ;i++)
		sum+=h_idata[i];

	cout << "Serial sum = " << sum << endl;

	sum=0;
	for(int i=0;i<GRID_SIZE;i++)
			sum+=h_odata[i];

	cout << "Parallel sum = " << sum << endl;


	free(h_idata);
	free(h_odata);
	cudaFree(g_idata);
	cudaFree(g_odata);

	return 0;
}
