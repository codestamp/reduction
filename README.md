# Reduction

The project aims to implement various versions of the reduction sum algorithm.
The main reference being the Reduction.pdf by Mark Harris.

The slaient features that are explored in this project are as under:

1.  Understand and implement different types of reduction sum techniques.
2.  Implement reduction using various features of CUDA such as shared memory, etc.
3.  Study memory coalescing, bank conflicts, etc.

Calculating Bandwidth. They are of 2 types - Theoretical Bandwidth and Effective Bandwidth

Theoretical bandwidth is hardware dependent and can be calculated for the hardware inofrmation as under:

Bandwidth(TH) = (BusWidth * MemorySpeed *2)/1000*8

Effective Bandwidth is calculated as under: 
(Ref: https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/)

Bandwidth(Eff) = N*4*p/elapsedtime(ms)/1e6

where N*4 - number of bytes of data
      p   - read and write of the data errors 





Further additions shall be added in due course ..
