#ifndef __TIMING_CUH_
#define __TIMING_CUH_

/**************/
/* Timing GPU */
/**************/

//Events are part of CUDA API and provide a system independent way to measure 
//execution times on CUDA devices to the precision of 0.5 micro seconds

struct PrivateTimingGPU;

class TimingGPU
{
	private:
		privateTimingGPU *privateTimingGPU;

	public:
		TimingGPU();
		~TimingGPU();

		void StartCounter();
		void StartCounterFlags();

		float GetCounter();
};	//TimingGPU class
#endif

