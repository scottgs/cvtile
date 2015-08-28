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

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#ifndef _GPU_Algorithm_Kernels_HPP_
#define _GPU_Algorithm_Kernels_HPP_

namespace cvt {

namespace gpu {

template <typename TextureType, int Texture>
cudaError_t bind_texture(cudaArray* gpuInputData);

template <typename TextureType, int Texture>
cudaError_t unbind_texture(cudaArray* gpuInputData);

template <typename InputPixelType, typename OutputPixelType>
void launch_window_histogram_statistics(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  OutputPixelType * const outputData,
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets,
		   const unsigned int numElements);

template <typename InputPixelType, typename OutputPixelType>
void launch_dilate(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize, 
		   const cudaStream_t stream,  OutputPixelType * const outputData, 
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets, 
		   const unsigned int numElements);

template <typename InputPixelType, typename OutputPixelType>
void launch_erode(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize, 
		   const cudaStream_t stream,  OutputPixelType * const outputData, 
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets, 
		   const unsigned int numElements);

template <typename InputPixelType, typename OutputPixelType>
void launch_absDifference(const dim3 dimGrid, const dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream,
						  OutputPixelType * outputData, const unsigned int width,
						  const unsigned int height);

} // end of gpu namespace
} // end of cvt namespace

#endif
