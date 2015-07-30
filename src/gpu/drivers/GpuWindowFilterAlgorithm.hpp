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

#ifndef GPU_WINDOW_FITLER_ALGORITHM_
#define GPU_WINDOW_FITLER_ALGORITHM_

#include "../../Cuda4or5.h"
#include "GpuAlgorithm.hpp"
#include "../kernels/GpuAlgorithmKernels.hpp"
#include <vector>
#include <sstream>

namespace cvt {

namespace gpu {


enum windowRadiusType{
	SQUARE,
	CIRCLE
};


template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class GpuWindowFilterAlgorithm : public GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>
{

	public:
	
	 explicit GpuWindowFilterAlgorithm(unsigned int cudaDeviceId, size_t unbufferedDataWidth,
							 size_t unbufferedDataHeight, ssize_t windowRadius);

	 virtual ~GpuWindowFilterAlgorithm();
	 ErrorCode initializeDevice(enum windowRadiusType type);

	virtual ErrorCode operator()(const cvt::cvTile<InputPixelType>& tile,
													  const cvt::cvTile<OutputPixelType> ** outTile);

	protected:


	ErrorCode allocateAdditionalGpuMemory();
	virtual ErrorCode launchKernel(unsigned bw, unsigned bh);
	void computeRelativeOffsets();
	ErrorCode transferRelativeOffsetsToDevice(); 


	/**
	 * PROTECTED ATTRIBUTES
	 * */
	ssize_t windowRadius_;
	std::vector<int2> relativeOffsets_;
	int2 *relativeOffsetsGpu_;
	enum windowRadiusType type;	
	

}; // END of GpuWindowFilterAlgorithm

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::GpuWindowFilterAlgorithm(
	unsigned int cudaDeviceId, size_t unbufferedDataWidth, 
	size_t unbufferedDataHeight, ssize_t windowRadius) 	
	: cvt::gpu::GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>(
	cudaDeviceId, unbufferedDataWidth,unbufferedDataHeight) 
{
	windowRadius_ = windowRadius;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::~GpuWindowFilterAlgorithm() {
	////////////////////////////////////////
	// FREE CUDA ARRAYS USED FOR GPU INPUT //
	////////////////////////////////////////
	
	cudaFree(relativeOffsetsGpu_);

	cudaError cuer3 = cudaGetLastError();
	if(cuer3 == cudaErrorInvalidValue)
		this->lastError = DestructFailcuOutArraycudaErrorInvalidValue;
	else if(cuer3 == cudaErrorInitializationError)
		this->lastError = DestructFailcuOutArraycudaErrorInitializationError;
	else if(cuer3 != cudaSuccess)
		this->lastError = CudaError;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::operator()(const cvt::cvTile<InputPixelType>& tile,
													  const cvt::cvTile<OutputPixelType> ** outTile)
{
		//TO-DO Error Check Template Params for Type/Bounds

	const cv::Size2i tileSize = tile.getSize();
	
	if (tileSize != this->dataSize)
	{
		std::stringstream ss;
		ss << tileSize << " expected of " << this->dataSize << std::endl; 
		throw std::runtime_error(ss.str());
	}

	/*
	 *  Copy data down for tile using the parents implementation
	 */
	this->lastError = copyTileToDevice(tile);
	if (this->lastError != cvt::Ok)
	{
		throw std::runtime_error("Failed to copy tile to device");
	}

	// Invoke kernel with empirically chosen block size
	unsigned short bW = 16;
	unsigned short bH = 16;

	launchKernel(bW, bH);

	this->lastError = copyTileFromDevice(outTile);
	if(this->lastError != cvt::Ok) {
		std::runtime_error("Failed copy off tile from device");
	}
	return Ok;
}	

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::initializeDevice(enum windowRadiusType type)
{
	/*
	 * Attempts to check the GPU and begin warm up
	 *
	 */

	this->type = type;

	this->lastError = this->setGpuDevice();
	if(this->lastError)
		return this->lastError;
	
	if (this->properties.getMajorCompute() < 1)
	{
		this->lastError = InitFailNoCUDA;
		return this->lastError;
	}
	/*
	 * Verfies the properities of GPU
	 *
	 */
	cudaStreamCreate(&this->stream);
	cudaError cuer = cudaGetLastError();
	if(cuer != cudaSuccess){
		this->lastError = InitFailcuStreamCreateErrorcudaErrorInvalidValue;
		return this->lastError;
	}

	//Set descriptor of input data before allocation
	// Sets is at single channel, 16-bit unsigned integer
	std::string inTypeIdentifier(typeid(this->tempForTypeTesting).name());
	size_t bitDepth = 0;
	cudaChannelFormatDesc inputDescriptor;

	if(inTypeIdentifier == "a" || 
	   inTypeIdentifier == "s" || 
	   inTypeIdentifier == "i" ||
	   inTypeIdentifier == "l")
	{
		this->channelType = cudaChannelFormatKindSigned;
	}
	else if(inTypeIdentifier == "h" || 
			inTypeIdentifier == "t" || 
			inTypeIdentifier == "j" || 
			inTypeIdentifier == "m")
	{
		this->channelType = cudaChannelFormatKindUnsigned;
	}
	else if(inTypeIdentifier == "f" || 
			inTypeIdentifier == "d") 
	{
		this->channelType = cudaChannelFormatKindFloat;
	}
	else
	{
		this->lastError = InitFailUnsupportedInputType;
		return this->lastError;
	}

	bitDepth = sizeof(this->tempForTypeTesting) * 8;

	inputDescriptor = cudaCreateChannelDesc(bitDepth, 0, 0, 0, this->channelType);
	cuer = cudaGetLastError();
	
	if (cuer != cudaSuccess) {
		this->lastError = CudaError;
		std::cout << "CUDA ERR = " << cuer << std::endl;
		throw std::runtime_error("GPU WHS INIT FAILED TO CREATE CHANNEL");
	}	


	if(cuer != cudaSuccess){
		this->lastError = CudaError;
		return this->lastError;
	}	
	
	//////////////////////////////////////////////////////////
	// ALLOCATE MEMORY FOR GPU INPUT AND OUTPUT DATA (TILE) //
	/////////////////////////////////////////////////////////
	
	cuer = cudaGetLastError();
	/*Gpu Input Data*/
	cudaMallocArray(
					(cudaArray**)&this->gpuInputDataArray,   
					 &inputDescriptor, 
					 this->dataSize.width,  
					 this->dataSize.height
					);
	this->gpuInput = this->gpuInputDataArray;
	this->usingTexture = true;	
	cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		std::cout << "CUDA ERR = " << cuer << std::endl;
		throw std::runtime_error("GPU WHS INIT FAILED TO ALLOCATE MEMORY");
	}

	//Gpu Output Data 
	const size_t bytes = this->dataSize.width * this->dataSize.height * OutputBandCount * sizeof(OutputPixelType);
	this->outputDataSize = bytes;
	cudaMalloc((void**) &this->gpuOutputData, bytes);
	cuer = cudaGetLastError();

	if (cuer != cudaSuccess) {
		throw new std::runtime_error("GPU WHS INIT FAILED TO ALLOCATE OUTPUT MEMORY");
	}
	if (cuer == cudaErrorMemoryAllocation)
	{
		this->lastError = InitFailcuOutArrayMemErrorcudaErrorMemoryAllocation;
		return this->lastError;
	}
	//////////////////////////////////////////////////////////////////////////////////////
	// CALL FUNCTION TO ALLOCATE ADDITIONAL GPU STORAGE - DOES NOTHING IF NOT OVERRIDEN //
	/////////////////////////////////////////////////////////////////////////////////////
	/* Initialize the neighborhood coordinates */
	/*uses two ints to store width and height coords by the windowRadius_*/	
	computeRelativeOffsets();
	/*
	 * Allocates the memory needed for the results coming back from the GPU
	 *
	 */
	this->lastError = allocateAdditionalGpuMemory();
	transferRelativeOffsetsToDevice();	


	return this->lastError;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::launchKernel(unsigned bw, unsigned bh) {
	return Ok; // NEED TO ADD DEFAULT KERNEL FOR FILTER
}



template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::allocateAdditionalGpuMemory()
{
	cudaMalloc((void**)&relativeOffsetsGpu_, sizeof(int2) * relativeOffsets_.size());
	cudaError cuer = cudaGetLastError(); 
	if (cuer != cudaSuccess) {
		throw std::runtime_error("GPU WHS () FAILURE TO ALLOCATE MEMORY FOR RELATIVE COORDS");
	}
	if (cuer == cudaErrorMemoryAllocation)
	{
		this->lastError = InitFailcuOutArrayMemErrorcudaErrorMemoryAllocation;
	}
	return this->lastError;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
void GpuWindowFilterAlgorithm<InputPixelType,InputBandCount, OutputPixelType, OutputBandCount>::computeRelativeOffsets() 
{
	if (type == CIRCLE) {
		const size_t radius_squared = windowRadius_ * windowRadius_;
		for(ssize_t i = 0 - windowRadius_; i <= windowRadius_; ++i){
			size_t i_squared = i * i;
			for(ssize_t j = 0 - windowRadius_; j <= windowRadius_; ++j){
				if(i_squared + j * j <= radius_squared)
					relativeOffsets_.push_back(make_int2(i,j));
			}
		}
	}
	else {
		for (ssize_t i = 0 - windowRadius_; i <= windowRadius_; ++i) {
			for (ssize_t j = 0 - windowRadius_; j <= windowRadius_; ++j) {
				relativeOffsets_.push_back(make_int2(i,j));	
			}
		}

	}
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::transferRelativeOffsetsToDevice() 
{
	cudaMemcpyAsync (
	relativeOffsetsGpu_,
	relativeOffsets_.data(),
	relativeOffsets_.size() * sizeof(int2),
	cudaMemcpyHostToDevice,
	this->stream
	);
	cudaError cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		std::cout << "CUDA ERR = " << cuer << std::endl;
		throw std::runtime_error("GPU WHS () FAILED TO MEMCPY RELATIVE COORDS ON TO DEVICE");
	}
	
	return this->lastError;
}

} // END of cvt namespace
} // END of gpu namespace

#endif
