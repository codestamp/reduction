# CMakeLists.txt to build reduc.cu
project(reduction-test)

# required cmake version
	cmake_minimum_required(VERSION 2.8)

# packages
	find_package(CUDA)

# nvcc flags

	set(
		CUDA_NVCC_FLAGS 
		${CUDA_NVCC_FLAGS};
		-Wno-deprecated-gpu-targets
		-gencode arch=compute_20,code=sm_20
	   )

# include directories
	include_directories(
			includes
			)

cuda_compile(REDUC_O source/reduc.cu)	
# Specify binary name and source file to build it from
	cuda_add_executable(
			reduc ${REDUC_O} includes/utils.cuh
			)

# For linking ...
# Specify target and libraries to link with
# target_link_libraries(reduc)
