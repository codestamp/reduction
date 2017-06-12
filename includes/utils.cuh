/*
 * utils.cuh
 *
 *  Created on: Jun 12, 2017
 *      Author: buddy
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_


#include <iostream>
#include <ctime>


#define gpuErrchk(ans) { gpuAssert((ans),__FILE__,__LINE__); }

inline void gpuAssert(cudaError_t code,const char *file,int line,bool abort=true)
{
	if(code!=cudaSuccess)
	{
		std::cerr << "GPUAssert: " << cudaGetErrorString(code)
				<< " " << file << " " << line << std::endl;
		if(abort) {exit(code);}
	}
}

inline int divup(int a,int b) { return (a%b)==0 ? a/b : a/b+1; }

int randomNumber()
{
	/* use srand(time(0)) in the code */
	int min=1,max=5;
	int output = min + (rand() % static_cast<int>(max - min + 1));
	return output;
}


#endif /* UTILS_CUH_ */
