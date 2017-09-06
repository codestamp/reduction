#include "timinggpu.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTimingGPU {
	cudaEvent_t start;
	cudaEvent_t stop; 
};

//default constructor
TimingGPU::TimingGPU() { privateTimingGPU = new PrivateTimingGPU(); }

//default destructor
TimingGPU::~TimingGPU() {}

void TimingGPU::StartCounter() {
	cudaEventCreate(&((*privateTimingGPU).start));
	cudaEventCreate(&((*privateTimingGPU).stop));
}

void TimingGPU::StartCounterFlags() {
	int eventFlags=cudaEventBlockingSync;

	cudaEventCreateWithFlags(&((*privateTimingGPU).start),eventFlags);
	cudaEventCreateWithFlags(&((*privateTimingGPU).stop),eventFlags);
	cudaEventRecord((*privateTimingGPU).start,0);
}

//gets the counter in milliseconds
float TimingGPU::GetCounter() {
	float time;
	cudaEventRecord((*privateTimingGPU).stop,0);
	cudaEventSynchronize((*privateTimingGPU).stop);
	cudaEventElapsedTime(&time,(*privateTimingGPU).start,(*privateTimingGPU).stop);
	return time;
}



