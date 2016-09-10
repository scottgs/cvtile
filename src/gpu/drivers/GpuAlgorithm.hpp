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

#ifndef CGI_GPU_DRIVERS_GPU_ALGORITHM
#define CGI_GPU_DRIVERS_GPU_ALGORITHM


/////////////////////
// LOCAL INCLUDES  //
/////////////////////

#include "../../base/cvTile.hpp"
#include "GPUProperties.hpp"
//#include "GpuPixelCutter.hpp"

// Remove warnings from boost / opencv
// This is clang / gcc safe -- see clang docs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"

/////////////////////
// OpenCV INCLUDES //
/////////////////////

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

/////////////////////
// Boost INCLUDES //
/////////////////////

#include <boost/mpl/set.hpp>
#include <boost/mpl/assert.hpp>

#pragma GCC diagnostic pop

namespace cvt {

/** @brief Return status values
 *
 *   The first position indicates the class of error while
 *   all subsequent portions specify the precise errors that occurred
 *
 *   Note that the cudaError enum names are used as error specifers where necessary
 *   and that they are a 1 to 1 mapping with their counterparts in CUDA.
 */
enum ErrorCode{
	/** @brief No Error. */
	Ok = 0,
	/** @brief Object Construction Failure */
	GpuAlgoNoConstructBadInputValue,
	/** @brief Initialize Method Not Called */
	CudaNotInit,
	/** @brief Template for InputPixel Not Supported */
	InitFailUnsupportedInputType,
	/** @brief Cuda Init Failure; device in use */
	InitFailcudaErrorDeviceAlreadyInUse,
	/** @brief Cuda Init Failure; invalid device */
	InitFailcudaErrorInvalidDevice,
	/** @brief Device not CUDA capable */
	InitFailNoCUDA,
	/** @brief Not enough global memory on Card to hold Input Data */
	InitFailInsufficientMemoryForInputData,
	/** @brief Failed to create CUDA stream in device Init */
	InitFailcuStreamCreateErrorcudaErrorInvalidValue,
	/** @brief Failed to crete the gpu array for the input data */
	InitFailcuInputArrayMemErrorcudaErrorMemoryAllocation,
	/** @brief Failed to create the gpu array for its output */
	InitFailcuOutArrayMemErrorcudaErrorMemoryAllocation,
	/** @brief Failed to free gpuInputDataArray cudaArray */
	DestructFailcuInArraycudaErrorInvalidValue,
	DestructFailcuInArraycudaErrorInitializationError,
	/** @brief Failed to free gpuOutData cuda linear memory */
	DestructFailcuOutArraycudaErrorInvalidValue,
	DestructFailcuOutArraycudaErrorInitializationError,
	/** @brief Destructor encounterd error freeing in and out arrays */
	DestructFailcuBothArrayNotFreed,
	/** @brief Tile not copied to device due to **/
	TileToDevicecudaErrorInvalidDevicePointer,
	/** @brief Tile not copied to device due to **/
	TileToDevicecudaErrorInvalidMemcpyDirection,
	/** @brief Tile not copied to device due to **/
	TileToDevicecudaErrorInvalidValue,
	/** @brief Tile not copied to device due to **/
	TileFromDevicecudaErrorInvalidDevicePointer,
	/** @brief Tile not copied to device due to **/
	TileFromDevicecudaErrorInvalidMemcpyDirection,
	/** @brief Tile not copied to device due to **/
	TileFromDevicecudaErrorInvalidValue,

	/** @brief unspecified CUDA error */

	//TO-DO Specify all the CUDA errors
	CudaError
};

namespace gpu {

/** @brief Generic GPU algorithm parent class that initializes the card for CV tiles
		 *    Returns an error if the creation fails.
		 *
		 *  @templateparam
		 *  @templateparam driverName GDAL driver name to use for file creation.
		 *  @templateparam rasterSize Size of the image to create.
		 *  @templateparam nBands  Number of bands to create.
*/

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class GpuAlgorithm {

	public:

	///////////////////////
	// MEMBER FUNCTIONS //
	//////////////////////

		explicit GpuAlgorithm(unsigned int cudaDeviceId, unsigned int dataWidth,
							 unsigned int dataHeight );

		virtual ~GpuAlgorithm();

		virtual ErrorCode initializeDevice();

		ErrorCode getLastError() const;

		virtual ErrorCode operator()(const cvt::cvTile<InputPixelType>& tile,
													  const cvt::cvTile<OutputPixelType> ** outTile) = 0;

		virtual ErrorCode operator()(const cvt::cvTile<InputPixelType>& tile,
													  const cvt::cvTile<InputPixelType>& tileTwo,
													  const cvt::cvTile<OutputPixelType> ** outTile);


		const cv::Size2i getDataSize() const;

		std::string errToString(const ErrorCode err) const;

		const GPUProperties getProperties() const;

		size_t getInputBytesToTransfer() const;

		bool getUsingTexture() const;

		cudaChannelFormatKind getChannelType() const;

		size_t getOutputDataSize() const;

	protected:

		virtual ErrorCode allocateAdditionalGpuMemory();

		virtual ErrorCode launchKernel(unsigned bw, unsigned bh) = 0;

		ErrorCode copyTileToDevice(const cvt::cvTile<InputPixelType> &tile);

		virtual ErrorCode copyTileFromDevice(const cvt::cvTile<OutputPixelType> ** tilePtr);

		ErrorCode setGpuDevice();

		template< typename InputChannelType >
		cudaChannelFormatDesc setupCudaChannelDescriptor();

	private:

		//ErrorCode setGpuDevice();


	//////////////////////
	// MEMBER VARIABLES //
	//////////////////////

	public:

		////////////////////
		// TILE META DATA //
		////////////////////

		cv::Size2i dataSize;

		///////////////////////
		// CUDA DEVICE INFO  //
		//////////////////////

		cudaStream_t stream;

		unsigned int deviceID;

		GPUProperties properties;

		cudaChannelFormatKind channelType;


		////////////////////////
		// GPU DATA POINTERS //
		///////////////////////

		/* Used for Band Counts 1-4 */
		cudaArray * gpuInputDataArray;

		/* Used for Band Counts > 4 */
		InputPixelType * gpuInputDataGlobal;

		/* Mask that will point to correct Input Data */
		OutputPixelType * gpuOutputData;

		void * gpuInput;

		size_t bytesToTransfer;

		size_t outputDataSize;

		//////////////////////////////
		// ERROR STATE/ LOGGING		//
		//////////////////////////////

		ErrorCode lastError;

		bool usingTexture;

	private:

};

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::GpuAlgorithm(unsigned int cudaDeviceId, unsigned int dataWidth,
						   unsigned int dataHeight )
						   : dataSize(dataWidth, dataHeight),
						   	 deviceID(cudaDeviceId), properties(cudaDeviceId),
						   	 gpuInputDataArray(NULL), gpuInputDataGlobal(NULL), gpuOutputData(NULL),
						   	 gpuInput(NULL), lastError(Ok), usingTexture(false)
{
	namespace mpl = boost::mpl;
	using allowed_types = mpl::set< float, short ,int, long, unsigned int, unsigned short, unsigned char, signed char >;

	BOOST_MPL_ASSERT((mpl::has_key<allowed_types, InputPixelType>));
	BOOST_MPL_ASSERT((mpl::has_key<allowed_types, OutputPixelType>));

	bytesToTransfer = dataHeight * dataWidth * InputBandCount * sizeof(InputPixelType);
	if(bytesToTransfer == 0){
		lastError = GpuAlgoNoConstructBadInputValue;
		return;
	}

}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::~GpuAlgorithm()
{
	////////////////////////////////////////
	// FREE CUDA ARRAY USED FOR GPU INPUT //
	////////////////////////////////////////

	cudaFreeArray(gpuInputDataArray);

	cudaError cuer = cudaGetLastError();
	if(cuer == cudaErrorInvalidValue)
		lastError = DestructFailcuInArraycudaErrorInvalidValue;
	else if(cuer == cudaErrorInitializationError)
		lastError = DestructFailcuInArraycudaErrorInitializationError;
	else if(cuer != cudaSuccess)
		lastError = CudaError;

	cudaFree(gpuOutputData);

	cudaError cuer2 = cudaGetLastError();
	if(cuer2 == cudaErrorInvalidValue)
		lastError = DestructFailcuOutArraycudaErrorInvalidValue;
	else if(cuer2 == cudaErrorInitializationError)
		lastError = DestructFailcuOutArraycudaErrorInitializationError;
	else if(cuer2 != cudaSuccess)
		lastError = CudaError;

	if(cuer != cudaSuccess && cuer2 != cudaSuccess)
		lastError = DestructFailcuBothArrayNotFreed;

}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::initializeDevice()
{

	lastError = setGpuDevice();
	if(lastError)
		return lastError;

	if (properties.getMajorCompute() < 1)
	{
		lastError = InitFailNoCUDA;
		return lastError;
	}

	///////////////////////////////////////////
	// VERIFY THAT GPU HAS SUFFICIENT MEMORY //
   	///////////////////////////////////////////

	if(properties.getTotalGlobalMemoryBytes() < bytesToTransfer)
	{
		lastError = InitFailInsufficientMemoryForInputData;
		return lastError;
	}

	//TO-DO Check for sufficient texture memory

	cudaStreamCreate(&stream);
	cudaError cuer = cudaGetLastError();
	if(cuer != cudaSuccess){
		lastError = InitFailcuStreamCreateErrorcudaErrorInvalidValue;
		return lastError;
	}

	cudaChannelFormatDesc inputDescriptor;
	inputDescriptor = setupCudaChannelDescriptor< InputPixelType >();

	cuer = cudaGetLastError();

	if(cuer != cudaSuccess){
		lastError = CudaError;
		return lastError;
	}

	//////////////////////////////////////////////////////////
	// ALLOCATE MEMORY FOR GPU INPUT AND OUTPUT DATA (TILE) //
	/////////////////////////////////////////////////////////

	if(InputBandCount <= 1 && InputBandCount != 0){
	/* Gpu Input Data */
		cudaMallocArray(
						(cudaArray**)&gpuInputDataArray,
						 &inputDescriptor,
						 dataSize.width,
						 dataSize.height
						);
		gpuInput = gpuInputDataArray;
		usingTexture = true;
	}
	else if(InputBandCount >= 2){
		cudaMalloc((void **)&gpuInputDataGlobal, bytesToTransfer);
		gpuInput = gpuInputDataGlobal;
	}

	if (cuer == cudaErrorMemoryAllocation){
		lastError = InitFailcuInputArrayMemErrorcudaErrorMemoryAllocation;
		return lastError;
	}
	else if(cuer != cudaSuccess){
		lastError = CudaError;
		return lastError;
	}

	/* Gpu Output Data */
	const size_t bytes = dataSize.width * dataSize.height * OutputBandCount * sizeof(OutputPixelType);
	outputDataSize = bytes;
	cudaMalloc((void**) &gpuOutputData, bytes);
	cuer = cudaGetLastError();
	if (cuer == cudaErrorMemoryAllocation)
	{
		lastError = InitFailcuOutArrayMemErrorcudaErrorMemoryAllocation;
		return lastError;
	}

	//////////////////////////////////////////////////////////////////////////////////////
	// CALL FUNCTION TO ALLOCATE ADDITIONAL GPU STORAGE - DOES NOTHING IF NOT OVERRIDEN //
	/////////////////////////////////////////////////////////////////////////////////////
	lastError = allocateAdditionalGpuMemory();

	return lastError;

}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::setGpuDevice()
{
	cudaSetDevice(deviceID);
	cudaError cuer = cudaGetLastError();

	if(cuer == cudaErrorInvalidDevice)
		lastError = InitFailcudaErrorInvalidDevice;
	else if( cuer == cudaErrorDeviceAlreadyInUse)
		lastError = InitFailcudaErrorDeviceAlreadyInUse;
	else if( cuer != cudaSuccess )
		lastError = CudaError;

	return lastError;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::allocateAdditionalGpuMemory()
{
	return Ok;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::getLastError() const
{
	return lastError;
}


template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::copyTileToDevice(const cvt::cvTile<InputPixelType> &tile)
{

	const unsigned int tileArea = tile.getSize().area();
	unsigned char* tileDataPtr = NULL;
	int offsetY = 0;

	cudaError cuer;// = cudaGetLastError();

	if(usingTexture){
		tileDataPtr = tile[0].data;
		cudaMemcpyToArrayAsync(gpuInputDataArray,0, 0,
			tileDataPtr,
			sizeof(InputPixelType) * tileArea,
			cudaMemcpyHostToDevice,
			stream
		);
		//cuer = cudaGetLastError();
	}
	else
	{
		for(int currentBand = 0; currentBand < InputBandCount; ++currentBand)
		{
			tileDataPtr = tile[currentBand].data;
			cudaMemcpyAsync(
				(void *)(((unsigned char*)gpuInputDataGlobal) + offsetY),
				(void*) tileDataPtr,
				(size_t)tileArea * sizeof(InputPixelType),
				cudaMemcpyHostToDevice
			);
			offsetY += tileArea * sizeof(InputPixelType);
			cuer = cudaGetLastError();
			if(cuer == cudaErrorInvalidValue)
				lastError = TileToDevicecudaErrorInvalidValue;
			else if(cuer == cudaErrorInvalidDevicePointer)
				lastError = TileToDevicecudaErrorInvalidDevicePointer;
			else if(cuer == cudaErrorInvalidMemcpyDirection)
				lastError = TileToDevicecudaErrorInvalidMemcpyDirection;
			else if(cuer != cudaSuccess)
				lastError = CudaError;
		}
	}

	cuer = cudaGetLastError();
	if(cuer == cudaErrorInvalidValue)
		lastError = TileToDevicecudaErrorInvalidValue;
	else if(cuer == cudaErrorInvalidDevicePointer)
		lastError = TileToDevicecudaErrorInvalidDevicePointer;
	else if(cuer == cudaErrorInvalidMemcpyDirection)
		lastError = TileToDevicecudaErrorInvalidMemcpyDirection;
	else if(cuer != cudaSuccess)
		lastError = CudaError;

	return lastError;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
template< typename InputChannelType >
cudaChannelFormatDesc GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::setupCudaChannelDescriptor()
{
	namespace mpl = boost::mpl;
	using allowed_types = mpl::set< float, short, int, long, unsigned int, unsigned short, unsigned char, signed char >;

	BOOST_MPL_ASSERT((mpl::has_key<allowed_types, InputChannelType>));

	size_t bitDepth = 0;
	if(std::is_floating_point<InputChannelType>::value)
	{
		channelType = cudaChannelFormatKindFloat;
	}
	else if(std::is_unsigned<InputChannelType>::value)
	{
		channelType = cudaChannelFormatKindUnsigned;
	}
	else if(std::is_integral<InputChannelType>::value
		|| std::is_signed<InputChannelType>::value )
	{
		channelType = cudaChannelFormatKindSigned;
	}

	bitDepth = sizeof(InputPixelType) * 8;
	cudaChannelFormatDesc inputDescriptor;
	if(InputBandCount == 1 ){
		inputDescriptor = cudaCreateChannelDesc(bitDepth, 0, 0, 0, channelType);
	}
	else if(InputBandCount == 2){
		inputDescriptor = cudaCreateChannelDesc(bitDepth, bitDepth, 0, 0, channelType);
	}
	else if(InputBandCount == 3){
		inputDescriptor = cudaCreateChannelDesc(bitDepth, bitDepth, bitDepth, 0, channelType);
	}
	else if(InputBandCount == 4){
		inputDescriptor = cudaCreateChannelDesc(bitDepth, bitDepth, bitDepth, bitDepth, channelType);
	}

	cudaError cuer;
	cuer = cudaGetLastError();
	if(cuer != cudaSuccess){
		lastError = CudaError;
		throw std::runtime_error("Error constructing channel descriptor");
	}

	return inputDescriptor;
}


template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::copyTileFromDevice(const cvt::cvTile<OutputPixelType> ** tilePtr)
{

	const size_t bytes = dataSize.area() * OutputBandCount * sizeof(OutputPixelType);
	std::vector<OutputPixelType> data;
	data.resize(bytes/sizeof(OutputPixelType));

	cudaMemcpyAsync(
			&(data[0]),
			gpuOutputData,
			bytes,
			cudaMemcpyDeviceToHost,
			this->stream
			);

	cudaError cuer = cudaGetLastError();
	if(cuer == cudaErrorInvalidValue)
		lastError = TileFromDevicecudaErrorInvalidValue;
	else if(cuer == cudaErrorInvalidDevicePointer)
		lastError = TileFromDevicecudaErrorInvalidDevicePointer;
	else if(cuer == cudaErrorInvalidMemcpyDirection)
		lastError = TileFromDevicecudaErrorInvalidMemcpyDirection;
	else if(cuer != cudaSuccess)
		lastError = CudaError;

	/*for(const auto& ele: data)
	{
		std::cout << "data: " << ele << std::endl;

	}*/
	if(cuer == cudaSuccess) {
		(*tilePtr) = new cvTile<OutputPixelType>(&(data[0]), dataSize, OutputBandCount);
	}
	else
		(*tilePtr) = NULL;

	return lastError;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
const cv::Size2i GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::getDataSize() const
{
	return dataSize;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
const GPUProperties GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::getProperties() const
{
	return properties;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
size_t GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::getInputBytesToTransfer() const
{
	return bytesToTransfer;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
bool GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::getUsingTexture() const
{
	return usingTexture;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
size_t GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::getOutputDataSize() const
{
	return outputDataSize;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
cudaChannelFormatKind GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::getChannelType() const
{
	return channelType;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::operator()(
														__attribute__((unused)) const cvt::cvTile<InputPixelType>& tile,
	 													__attribute__((unused)) const cvt::cvTile<InputPixelType>& tileTwo,
													  __attribute__((unused)) const cvt::cvTile<OutputPixelType> ** outTile)
{
	return Ok;
}


} /* end gpu namespace */
} /* end cvt namespace */

#endif
