#ifndef GPU_BINARY_IMAGE_ALGORITHM_
#define GPU_BINARY_IMAGE_ALGORITHM_

#include "../../Cuda4or5.h"
#include "GpuAlgorithm.hpp"
#include "../kernels/GpuAlgorithmKernels.hpp"
#include <vector>
#include <sstream>

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

}; // END of GpuBinaryImageAlgorithm

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::GpuBinaryImageAlgorithm(
	unsigned int cudaDeviceId, size_t unbufferedDataWidth, 
	size_t unbufferedDataHeight) 	
	: cvt::gpu::GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>(
	cudaDeviceId, unbufferedDataWidth,unbufferedDataHeight) 
{
	;
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
ErrorCode GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::operator()(const cvt::cvTile<InputPixelType>& tile,
					const cvt::cvTile<OutputPixelType> ** outTile) {
					
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

	/*
	 *  Copy data down for tile using the parents implementation
	 */
	this->lastError = this->copyTileToDevice(tile);
	if (this->lastError != cvt::Ok)
	{
		throw std::runtime_error("Failed to copy tile to device");
	}

	cudaChannelFormatDesc input_descriptor = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMallocArray((cudaArray**)&gpuInputDataArrayTwo_, &input_descriptor, this->dataSize.width, this->dataSize.height);
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

	cudaError cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		std::cout << "CUDA ERR = " << cuer << std::endl;
		throw std::runtime_error("GPU WHS INIT FAILED TO ALLOCATE MEMORY");
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

	cuer = cudaGetLastError();
	if (cuer != cudaSuccess) {
		std::cout << "CUDA ERR = " << cuer << std::endl;
		throw std::runtime_error("GPU WHS INIT FAILED TO ALLOCATE MEMORY");
	}
	this->usingTexture = true;	
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
	/*
	 * Allocates the memory needed for the results coming back from the GPU
	 *
	 */
	this->lastError = this->allocateAdditionalGpuMemory();
	return this->lastError;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuBinaryImageAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::launchKernel(unsigned bw, unsigned bh) {
	return Ok; // NEED TO ADD DEFAULT KERNEL FOR FILTER
}

} // END of cvt namespace
} // END of gpu namespace

#endif
