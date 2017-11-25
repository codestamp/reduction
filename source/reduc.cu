/*
 * reduc.cu
 *
 *  Created on: Jun 12, 2017
 *      Author: Munesh Singh
 */
#include "utils.cuh"
#include "gputimer.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#define ARRSZ (1 << 22)
#define BLOCK_SIZE 128

using namespace std;

__global__
void reduce3(int *g_idata,int *g_odata)
{
	extern __shared__ int sdata[];

	//each thread loads one element from global to shared memory
	unsigned int tid=threadIdx.x;
	unsigned int i=threadIdx.x+blockIdx.x*blockDim.x;

	sdata[tid]=g_idata[i];
	__syncthreads();


	//do reduction in shared memory
	for(unsigned int s=blockDim.x/2;s>0;s>0,s>>=1) {
		if(tid<s) {
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	//write the result for this block to global mem
	if(tid==0)
		g_odata[blockIdx.x]=sdata[0];

}

float reduc3func(int *g_idata,int *g_odata,int GRID_SIZE) {

	GpuTimer gp;
	gp.Start();

	int *h_odata;
	h_odata=(int*)malloc(sizeof(int)*GRID_SIZE);

	reduce3<<<GRID_SIZE,BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(g_idata,g_odata);

	gp.Stop();

	gpuErrchk(cudaMemcpy(h_odata,g_odata,sizeof(int)*GRID_SIZE,cudaMemcpyDeviceToHost));

//	for(int i=0;i<GRID_SIZE;i++)
//		cout << h_odata[i] << endl;


	int sum=0;
	for(int i=0;i<GRID_SIZE;i++)
			sum+=h_odata[i];

	cout << "Parallel Reduction3 sum = " << sum << endl;

	//cout << gp.Elapsed() << " milli secs (reduction-1)" << endl;

	free(h_odata);

	return gp.Elapsed();
}
	

__global__
void reduce2(int *g_idata,int *g_odata)
{
	extern __shared__ int sdata[];

	//each thread loads one element from global to shared memory
	unsigned int tid=threadIdx.x;
	unsigned int i=threadIdx.x+blockIdx.x*blockDim.x;

	sdata[tid]=g_idata[i];
	__syncthreads();


	//do reduction in shared memory
	for(unsigned int s=1;s<blockDim.x;s*=2) {
		int index = 2*s*tid;

		if(index<blockDim.x) {
			sdata[index] += sdata[index+s];
		}
		__syncthreads();
	}

	//write the result for this block to global mem
	if(tid==0)
		g_odata[blockIdx.x]=sdata[0];

}

float reduc2func(int *g_idata,int *g_odata,int GRID_SIZE) {

	GpuTimer gp;
	gp.Start();

	int *h_odata;
	h_odata=(int*)malloc(sizeof(int)*GRID_SIZE);

	reduce2<<<GRID_SIZE,BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(g_idata,g_odata);

	gp.Stop();

	gpuErrchk(cudaMemcpy(h_odata,g_odata,sizeof(int)*GRID_SIZE,cudaMemcpyDeviceToHost));

//	for(int i=0;i<GRID_SIZE;i++)
//		cout << h_odata[i] << endl;


	int sum=0;
	for(int i=0;i<GRID_SIZE;i++)
			sum+=h_odata[i];

	cout << "Parallel Reduction2 sum = " << sum << endl;

	//cout << gp.Elapsed() << " milli secs (reduction-1)" << endl;

	free(h_odata);

	return gp.Elapsed();
}
	
__global__
void reduce1(int *g_idata,int *g_odata)
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

float reduc1func(int *g_idata,int *g_odata,int GRID_SIZE) {

	GpuTimer gp;
	gp.Start();

	int *h_odata;
	h_odata=(int*)malloc(sizeof(int)*GRID_SIZE);

	reduce1<<<GRID_SIZE,BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(g_idata,g_odata);

	gp.Stop();

	gpuErrchk(cudaMemcpy(h_odata,g_odata,sizeof(int)*GRID_SIZE,cudaMemcpyDeviceToHost));

//	for(int i=0;i<GRID_SIZE;i++)
//		cout << h_odata[i] << endl;


	int sum=0;
	for(int i=0;i<GRID_SIZE;i++)
			sum+=h_odata[i];

	cout << "Parallel Reduction1 sum = " << sum << endl;

	//cout << gp.Elapsed() << " milli secs (reduction-1)" << endl;

	free(h_odata);

	return gp.Elapsed();
}
	

int main() {
	srand(time(0));
	//host allocations
	int *h_idata;

	//allocating and initializing host memory
	h_idata=(int*)malloc(ARRSZ * sizeof(int));
	for(int i=0;i<ARRSZ;i++)
		h_idata[i]=randomNumber();

//	for(int i=0;i<10;i++)
//			cout << h_idata[i] << endl;

	//allocating device memories
	int *g_idata, *g_odata;
	gpuErrchk(cudaMalloc((void**)&g_idata,sizeof(int)*ARRSZ));


	int GRID_SIZE=divup(ARRSZ,BLOCK_SIZE);

	cout << "GRID_SIZE= " << GRID_SIZE << endl;
	gpuErrchk(cudaMalloc((void**)&g_odata,sizeof(int)*GRID_SIZE));
	//gpuErrchk(cudaMemset(g_odata,0,sizeof(int)*GRID_SIZE));

	gpuErrchk(cudaMemcpy(g_idata,h_idata,sizeof(int)*ARRSZ,cudaMemcpyHostToDevice));

	float elapsedReduc1=reduc1func(g_idata,g_odata,GRID_SIZE);
	float elapsedReduc2=reduc2func(g_idata,g_odata,GRID_SIZE);
	float elapsedReduc3=reduc3func(g_idata,g_odata,GRID_SIZE);
	

	//Serial reduce
	int sum=0;
	for(int i=0;i<ARRSZ;i++)
		sum+=h_idata[i];

	cout << "Serial sum = " << sum << endl;

	cout << "Reduction 1  elapsed time (milli seconds): " << elapsedReduc1 << endl;
	cout << "Reduction 2  elapsed time (milli seconds): " << elapsedReduc2 << endl;
	cout << "Reduction 3  elapsed time (milli seconds): " << elapsedReduc3 << endl;

	cout << "Reduction 1  Effective Bandwidth (GB/s): " << ARRSZ*4*2/elapsedReduc1/1e6 << endl;
	cout << "Reduction 2  Effective Bandwidth (GB/s): " << ARRSZ*4*2/elapsedReduc2/1e6 << endl;
	cout << "Reduction 3  Effective Bandwidth (GB/s): " << ARRSZ*4*2/elapsedReduc3/1e6 << endl;

	free(h_idata);
	cudaFree(g_idata);
	cudaFree(g_odata);

	return 0;
}
