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

#ifndef GPU_BINARY_IMAGE_ALGORITHM_
#define GPU_BINARY_IMAGE_ALGORITHM_

#include "GpuAlgorithm.hpp"
#include "../kernels/GpuAlgorithmKernels.hpp"

namespace cvt {
namespace gpu {

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class GpuBinaryImageAlgorithm : public GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>
{

	public:

	 explicit GpuBinaryImageAlgorithm(unsigned int cudaDeviceId, size_t unbufferedDataWidth,
							 size_t unbufferedDataHeight);

	 virtual ~GpuBinaryImageAlgorithm();
	 ErrorCode initializeDevice();

	virtual ErrorCode operator()(const cvt::cvTile<InputPixelType>& tile,
													  const cvt::cvTile<OutputPixelType> ** outTile);

	virtual ErrorCode operator()(const cvt::cvTile<InputPixelType>& tile, const cvt::cvTile<InputPixelType> &tile2,
													  const cvt::cvTile<OutputPixelType> ** outTile);

	protected:

	virtual ErrorCode launchKernel(unsigned bw, unsigned bh);

	/**
	 * PROTECTED ATTRIBUTES
	 * */
	cudaArray * gpuInputDataArrayTwo_;
	size_t bufferWidth_;

}; // END of GpuBinaryImageAlgorithm

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::GpuBinaryImageAlgorithm(
	unsigned int cudaDeviceId, size_t unbufferedDataWidth,
	size_t unbufferedDataHeight)
	: cvt::gpu::GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>(
	cudaDeviceId, unbufferedDataWidth,unbufferedDataHeight)
{
	bufferWidth_ = 0;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::~GpuBinaryImageAlgorithm() {
	////////////////////////////////////////
	// FREE CUDA ARRAYS USED FOR GPU INPUT //
	////////////////////////////////////////

	cudaFree(gpuInputDataArrayTwo_);

	cudaError cuer = cudaGetLastError();
	if(cuer == cudaErrorInvalidValue)
		this->lastError = DestructFailcuOutArraycudaErrorInvalidValue;
	else if(cuer == cudaErrorInitializationError)
		this->lastError = DestructFailcuOutArraycudaErrorInitializationError;
	else if(cuer != cudaSuccess)
		this->lastError = CudaError;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::operator()(
					__attribute__((unused)) const cvt::cvTile<InputPixelType>& tile,
					__attribute__((unused)) const cvt::cvTile<OutputPixelType> ** outTile) {

	return Ok;
}



template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::operator()(const cvt::cvTile<InputPixelType>& tile,
					const cvt::cvTile<InputPixelType> &tile2,const cvt::cvTile<OutputPixelType> ** outTile)
{
	//TO-DO Error Check Template Params for Type/Bounds

	const cv::Size2i tileSize = tile.getSize();

	if (tileSize != this->dataSize)
	{
		std::stringstream ss;
		ss << tileSize << " expected of " << this->dataSize << std::endl;
		throw std::runtime_error(ss.str());
	}

	if (tileSize != tile2.getSize()) {
		throw std::runtime_error("Both the incoming tiles must have the same size.");
	}

	/*
	 *  Copy data down for tile using the parents implementation
	 */
	this->lastError = this->copyTileToDevice(tile);
	if (this->lastError != cvt::Ok)
	{
		throw std::runtime_error("Failed to copy tile to device");
	}

	cudaError cuer;
	cudaChannelFormatDesc inputDescriptor;
	inputDescriptor = this->template setupCudaChannelDescriptor<InputPixelType>();
	cuer = cudaGetLastError();

	if (cuer != cudaSuccess) {
		throw std::runtime_error("GPU BINARY IMAGE RUN FAILED TO CREATE CHANNEL");
	}

	cudaMallocArray((cudaArray**)&gpuInputDataArrayTwo_, &inputDescriptor, this->dataSize.width, this->dataSize.height);
	const unsigned int offsetX = 0;
	const unsigned int offsetY = 0;
	const unsigned char* tile_data_ptr = &(tile2[0].data[0]);
	const unsigned int tileArea = tile.getSize().area();

	cudaMemcpyToArrayAsync(gpuInputDataArrayTwo_,	// the device | destination address
			offsetX , offsetY,  			// the destination offsets
			tile_data_ptr,					// the host | source address
			sizeof(InputPixelType) * tileArea,		// the size of the copy in bytes
			cudaMemcpyHostToDevice,			// the type of the copy
			this->stream);						// the device command stream

	cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		std::cout << "CUDA ERR = " << cuer << std::endl;
		throw std::runtime_error("GPU BINARY IMAGE RUN FAILED TO ALLOCATE MEMORY");
	}

	// Invoke kernel with empirically chosen block size
	unsigned short bW = 16;
	unsigned short bH = 16;

	launchKernel(bW, bH);

	this->lastError = this->copyTileFromDevice(outTile);
	if(this->lastError != cvt::Ok) {
		std::runtime_error("Failed copy off tile from device");
	}
	return Ok;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::initializeDevice()
{
	static_assert(InputBandCount == 1, "Binary Image Algorithms only support 1 band images");

	/*
	 * Attempts to check the GPU and begin warm up
	 *
	 */
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

	
	cudaChannelFormatDesc inputDescriptor;
	inputDescriptor = this->template setupCudaChannelDescriptor<InputPixelType>();
	
	cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		this->lastError = CudaError;
		throw std::runtime_error("GPU WHS INIT FAILED TO CREATE CHANNEL");
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

	cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		std::cout << "CUDA ERR = " << cuer << std::endl;
		throw std::runtime_error("GPU WHS INIT FAILED TO ALLOCATE MEMORY");
	}

	//TO-DO fix hardcoded
	this->usingTexture = true;	
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
	/*
	 * Allocates the memory needed for the results coming back from the GPU
	 *
	 */
	this->lastError = this->allocateAdditionalGpuMemory();
	return this->lastError;
}


template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::launchKernel(
	__attribute__((unused)) unsigned bw,
	__attribute__((unused)) unsigned bh) {
	return Ok; // NEED TO ADD DEFAULT KERNEL FOR FILTER
}

} // END of cvt namespace
} // END of gpu namespace

#endif
