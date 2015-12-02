#ifndef _GPU_ABSOLUTE_DIFFERENCE_ALGORITHM_
#define _GPU_ABSOLUTE_DIFFERENCE_ALGORITHM_

#include "../../Cuda4or5.h"
#include "../kernels/GpuAlgorithmKernels.hpp"
#include "GpuBinaryImageAlgorithm.hpp"
#include <vector>

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
