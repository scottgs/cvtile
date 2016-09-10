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

#ifndef H_GPU_CONVOLUTION_H
#define H_GPU_CONVOLUTION_H

#include "GpuWindowFilterAlgorithm.hpp"

namespace cvt {
namespace gpu {

template<typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount, typename ConvolutionType>
class GpuConvolution : public GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>
{
	public:
		explicit GpuConvolution(unsigned int cudaDeviceId, size_t unbufferedDataWidth, size_t unbufferedDataHeight, ssize_t filterRadius, cv::Mat weights);
		~GpuConvolution();
		typename std::vector<ConvolutionType>::size_type getOffsetsSize();

	protected:
		virtual ErrorCode launchKernel(unsigned blockWidth, unsigned blockHeight);
		virtual ErrorCode allocateAdditionalGpuMemory();

		/* Protected Attributes */
		cv::Mat weights;
		ConvolutionType* weightsGpu_;

	private:
		ErrorCode transferWeights();
};

template<typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount, typename ConvolutionType>
GpuConvolution<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount, ConvolutionType>::GpuConvolution(
		unsigned int cudaDeviceId, size_t unbufferedDataWidth, size_t unbufferedDataHeight, ssize_t filterRadius,
		cv::Mat weight) :
		cvt::gpu::GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>
		(cudaDeviceId, unbufferedDataWidth, unbufferedDataHeight, filterRadius),
		weights(weight)
{
	;
}

template<typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount, typename ConvolutionType>
GpuConvolution<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount, ConvolutionType>::~GpuConvolution()
{
	cudaFree(this->weightsGpu_);
	cudaError cuer = cudaGetLastError();
	if(cuer == cudaErrorInvalidValue)
		this->lastError = DestructFailcuOutArraycudaErrorInvalidValue;
	else if(cuer == cudaErrorInitializationError)
		this->lastError = DestructFailcuOutArraycudaErrorInitializationError;
	else if(cuer != cudaSuccess)
		this->lastError = CudaError;
}

template<typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount, typename ConvolutionType>
ErrorCode GpuConvolution<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount, ConvolutionType>::allocateAdditionalGpuMemory()
{
	GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::allocateAdditionalGpuMemory();
	cudaMalloc((void**)&this->weightsGpu_, sizeof(ConvolutionType) * this->relativeOffsets_.size());
	cudaError cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		throw std::runtime_error("GPU CONVOLUTION () FAILURE TO ALLOCATE MEMORY FOR WEIGHTS");
	}
	if (cuer == cudaErrorMemoryAllocation)
	{
		this->lastError = InitFailcuOutArrayMemErrorcudaErrorMemoryAllocation;
	}
	return (this->lastError = transferWeights());
}

template<typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount, typename ConvolutionType>
ErrorCode GpuConvolution<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount, ConvolutionType>::transferWeights()
{
	std::vector<ConvolutionType> weightsVector;
	weightsVector.assign((ConvolutionType*)this->weights.datastart, (ConvolutionType*)this->weights.dataend);
	cudaMemcpyAsync(
		this->weightsGpu_,
		weightsVector.data(),
		weightsVector.size() * sizeof(ConvolutionType),
		cudaMemcpyHostToDevice,
		this->stream
	);
	cudaError cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		throw std::runtime_error("GPU CONVOLUTION FAILED TO MEMCPY COORDS TO DEVICE");
	}
	return this->lastError;
}

template<typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount, typename ConvolutionType>
ErrorCode GpuConvolution<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount, ConvolutionType>::launchKernel(unsigned blockWidth, unsigned blockHeight)
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
	 cvt::gpu::launchConvolution<InputPixelType, OutputPixelType, ConvolutionType>(dimGrid, dimBlock, 0,
	   this->stream, (InputPixelType*)this->gpuInput,  (OutputPixelType*)this->gpuOutputData, this->relativeOffsetsGpu_,
	   this->weightsGpu_, this->relativeOffsets_.size(), this->dataSize.width, this->dataSize.height, InputBandCount,
	   this->usingTexture);

	cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		std::cout << "CUDA ERROR = " << cuer << std::endl;
		throw std::runtime_error("KERNEL LAUNCH FAILURE");
	}
	return CudaError; // needs to be changed

};


} // end namespace gpu
} // end namespace cvt

#endif
