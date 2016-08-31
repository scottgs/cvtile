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

#ifndef _GPU_ERODE_ALGORITHM_
#define _GPU_ERODE_ALGORITHM_

#include "GpuWindowFilterAlgorithm.hpp"

namespace cvt {

namespace gpu {

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class GpuErode  : public GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>
{

public:

	 explicit GpuErode(unsigned int cudaDeviceId, size_t unbufferedDataWidth,
							 size_t unbufferedDataHeight, ssize_t windowRadius);

	~GpuErode();

protected:
	ErrorCode launchKernel(unsigned blockWidth, unsigned blockHeight);

};


template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuErode<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::GpuErode(
	unsigned int cudaDeviceId, size_t unbufferedDataWidth,
	size_t unbufferedDataHeight, ssize_t windowRadius) :
	cvt::gpu::GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>(
	cudaDeviceId, unbufferedDataWidth,unbufferedDataHeight, windowRadius)
{
	;
}



template< typename inputpixeltype, int inputbandcount, typename outputpixeltype, int outputbandcount >
GpuErode<inputpixeltype, inputbandcount, outputpixeltype, outputbandcount>::~GpuErode()
{
	;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuErode<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::launchKernel(unsigned blockWidth, unsigned blockHeight)
{
	dim3 dimBlock(blockWidth,blockHeight);

	size_t gridWidth = this->dataSize.width / dimBlock.x + (((this->dataSize.width % dimBlock.x)==0) ? 0 :1 );
	size_t gridHeight = this->dataSize.height / dimBlock.y + (((this->dataSize.height % dimBlock.y)==0) ? 0 :1 );
	dim3 dimGrid(gridWidth, gridHeight);

	// Bind the texture to the array and setup the access parameters
	cvt::gpu::bind_texture<InputPixelType,0>(this->gpuInputDataArray);
	cudaError cuer = cudaGetLastError();
	if (cudaSuccess != cuer)
	{
		return CudaError; // needs to be changed
	}

	// ====================================================
	// Really launch, after one last error check!
	// ====================================================
	cuer = cudaGetLastError();
	if (cudaSuccess != cuer)
	{
		return CudaError; // needs to be changed
	}
	//TODO: Use this line when updating to use shared memory
	 //const unsigned int shmem_bytes = neighbor_coordinates_.size() * sizeof(double) * blockWidth * blockHeight;
	 cvt::gpu::launch_erode<InputPixelType, OutputPixelType>(dimGrid, dimBlock, 0, this->stream,(OutputPixelType *)this->gpuOutputData,
	   this->roiSize_.width,this->roiSize_.height, this->relativeOffsetsGpu_,
	   this->relativeOffsets_.size(),this->bufferWidth_);

	cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		std::cout << "CUDA ERROR = " << cuer << std::endl;
		throw std::runtime_error("KERNEL LAUNCH FAILURE");
	}
	return CudaError; // needs to be changed

};

} //END OF GPU NAMESPACE
} // END OF CVT NAMESPACE

#endif
