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

#ifndef _GPU_ABSOLUTE_DIFFERENCE_ALGORITHM_
#define _GPU_ABSOLUTE_DIFFERENCE_ALGORITHM_

#include "GpuBinaryImageAlgorithm.hpp"

namespace cvt {

namespace gpu {

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class GpuAbsoluteDifference  : public GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>
{

public:

	 explicit GpuAbsoluteDifference(unsigned int cudaDeviceId, size_t unbufferedDataWidth,
							 size_t unbufferedDataHeight);

	~GpuAbsoluteDifference();

protected:
	ErrorCode launchKernel(unsigned blockWidth, unsigned blockHeight);

};


template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuAbsoluteDifference<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::GpuAbsoluteDifference(
	unsigned int cudaDeviceId, size_t unbufferedDataWidth,
	size_t unbufferedDataHeight) :
	cvt::gpu::GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>(
	cudaDeviceId, unbufferedDataWidth,unbufferedDataHeight)
{
	;
}



template< typename inputpixeltype, int inputbandcount, typename outputpixeltype, int outputbandcount >
GpuAbsoluteDifference<inputpixeltype, inputbandcount, outputpixeltype, outputbandcount>::~GpuAbsoluteDifference()
{
	;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuAbsoluteDifference<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::launchKernel(unsigned blockWidth, unsigned blockHeight)
{
	dim3 dimBlock(blockWidth,blockHeight);
	size_t gridWidth = this->dataSize.width / dimBlock.x + (((this->dataSize.width % dimBlock.x)==0) ? 0 :1 );
	size_t gridHeight = this->dataSize.height / dimBlock.y + (((this->dataSize.height % dimBlock.y)==0) ? 0 :1 );
	dim3 dimGrid(gridWidth, gridHeight);

	// Bind the texture to the array and setup the access parameters
	bind_texture<InputPixelType, 0>(this->gpuInputDataArray);
	bind_texture<InputPixelType, 1>(this->gpuInputDataArrayTwo_);

 	cvt::gpu::launch_absDifference<InputPixelType,OutputPixelType>(dimGrid, dimBlock, 0,
	   this->stream, (OutputPixelType *)this->gpuOutputData,
	   this->dataSize.width, this->dataSize.height);

	cudaError cuer;
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
