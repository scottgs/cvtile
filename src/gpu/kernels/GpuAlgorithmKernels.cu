/*
*******************************************************************************

Copyright (c) 2015, The Curators of the University of Missouri
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************
*/

#include "GpuAlgorithmKernels.cuh"

namespace cvt {
namespace gpu {


/*Explicit instantiations for data copy kernels */

template void launch_simpleDataCopy<signed char, signed char>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, signed char * in_data,
						signed char * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount,
						bool useTexture);

template void launch_simpleDataCopy<unsigned char, unsigned char>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, unsigned char * in_data,
						unsigned char * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount,
						bool useTexture);

template void launch_simpleDataCopy<short, short>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, short * in_data,
						short * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount,
						bool useTexture);

template void launch_simpleDataCopy<unsigned short, unsigned short>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, unsigned short * in_data,
						unsigned short * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount,
						bool useTexture);

template void launch_simpleDataCopy<int, int>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, int * in_data,
						int * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount,
						bool useTexture);
/*
* Float instaniations of functions
*
**/
template void launch_simpleDataCopy<float, float>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, float * in_data,
						float * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount,
						bool useTexture);

template void launchConvolution<short,short,short>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, short* inputData,
						short* gpuOutputData, int2* relativeOffsets, short* const filterWeights, const unsigned int filterSize,
						unsigned int outputWidth, unsigned int outputHeight, unsigned int bandCount,
						bool usingTexture);

template void launch_window_histogram_statistics<unsigned short, float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_window_histogram_statistics<short, float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_window_histogram_statistics<float, float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

/*template void launch_window_histogram_statistics<unsigned short, float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight,
			 const unsigned int numElements, const unsigned int buffer);

template void launch_window_histogram_statistics<short, float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight,
			 const unsigned int numElements, const unsigned int buffer);

template void launch_window_histogram_statistics<short, short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  short * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight,
			 const unsigned int numElements, const unsigned int buffer);

template void launch_window_histogram_statistics<float, float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight,
			 const unsigned int numElements, const unsigned int buffer);*/



template void launch_dilate<unsigned char,unsigned char>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  unsigned char * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);


template void launch_dilate<short,short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  short * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_dilate<unsigned short,unsigned short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  unsigned short * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_dilate<float,float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_dilate<short,float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_erode<unsigned char,unsigned char>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  unsigned char * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_erode<short,short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  short * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_erode<unsigned short,unsigned short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  unsigned short * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_erode<float,float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);

template void launch_erode<short,float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer);


template void launch_absDifference<unsigned char,unsigned char>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
							const cudaStream_t stream, unsigned char * const outputData, const unsigned int roiWidth,
						  const unsigned int roiHeight);

template void launch_absDifference<short,short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
							const cudaStream_t stream, short * const outputData, const unsigned int roiWidth,
						  const unsigned int roiHeight);

template void launch_absDifference<unsigned short,unsigned short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
							const cudaStream_t stream, unsigned short * const outputData, const unsigned int roiWidth,
						  const unsigned int roiHeight);

template void launch_absDifference<float,float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
							const cudaStream_t stream, float * const outputData, const unsigned int roiWidth,
						  const unsigned int roiHeight);

template void launch_absDifference<short,float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
							const cudaStream_t stream, float * const outputData, const unsigned int roiWidth,
						  const unsigned int roiHeight);

template void launch_local_binary_pattern<unsigned char, unsigned char>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream,
						unsigned char* gpuOutputData, int2* relativeOffsets, unsigned int relativeOffsetsSize,
						const unsigned int roiWidth, const unsigned int roiHeight, const unsigned int buffer);

template void launch_local_binary_pattern<short, short>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream,
						short* gpuOutputData, int2* relativeOffsets, unsigned int relativeOffsetsSize,
						const unsigned int roiWidth, const unsigned int roiHeight, const unsigned int buffer);

}; //end gpu namespace
}; //end cvt namespace
