# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/buddy/Documents/cprograms/reduction

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/buddy/Documents/cprograms/reduction/build

# Include any dependencies generated for this target.
include CMakeFiles/reduc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reduc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reduc.dir/flags.make

CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o: CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o.depend
CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o: CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o.cmake
CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o: ../source/reduc.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/buddy/Documents/cprograms/reduction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o"
	cd /home/buddy/Documents/cprograms/reduction/build/CMakeFiles/cuda_compile.dir/source && /usr/local/bin/cmake -E make_directory /home/buddy/Documents/cprograms/reduction/build/CMakeFiles/cuda_compile.dir/source/.
	cd /home/buddy/Documents/cprograms/reduction/build/CMakeFiles/cuda_compile.dir/source && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/buddy/Documents/cprograms/reduction/build/CMakeFiles/cuda_compile.dir/source/./cuda_compile_generated_reduc.cu.o -D generated_cubin_file:STRING=/home/buddy/Documents/cprograms/reduction/build/CMakeFiles/cuda_compile.dir/source/./cuda_compile_generated_reduc.cu.o.cubin.txt -P /home/buddy/Documents/cprograms/reduction/build/CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o.cmake

# Object files for target reduc
reduc_OBJECTS =

# External object files for target reduc
reduc_EXTERNAL_OBJECTS = \
"/home/buddy/Documents/cprograms/reduction/build/CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o"

reduc: CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o
reduc: CMakeFiles/reduc.dir/build.make
reduc: /usr/local/cuda/lib64/libcudart_static.a
reduc: /usr/lib/x86_64-linux-gnu/librt.so
reduc: CMakeFiles/reduc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/buddy/Documents/cprograms/reduction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reduc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reduc.dir/build: reduc

.PHONY : CMakeFiles/reduc.dir/build

CMakeFiles/reduc.dir/requires:

.PHONY : CMakeFiles/reduc.dir/requires

CMakeFiles/reduc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reduc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reduc.dir/clean

CMakeFiles/reduc.dir/depend: CMakeFiles/cuda_compile.dir/source/cuda_compile_generated_reduc.cu.o
	cd /home/buddy/Documents/cprograms/reduction/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/buddy/Documents/cprograms/reduction /home/buddy/Documents/cprograms/reduction /home/buddy/Documents/cprograms/reduction/build /home/buddy/Documents/cprograms/reduction/build /home/buddy/Documents/cprograms/reduction/build/CMakeFiles/reduc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reduc.dir/depend

