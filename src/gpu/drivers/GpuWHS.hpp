#ifndef _GPU_WINDOW_HISTOGRAM_STATISTICS_ALGORITHM_
#define _GPU_WINDOW_HISTOGRAM_STATISTICS_ALGORITHM_

#include "../../Cuda4or5.h"
#include "../kernels/GpuAlgorithmKernels.hpp"
//#include "../kernels/GpuTileAlgorithmKernels.hpp"
#include "GpuWindowFilterAlgorithm.hpp"
#include <vector>

namespace cvt {

namespace gpu {

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class GpuWHS  : public GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>
{

public:

	 explicit GpuWHS(unsigned int cudaDeviceId, size_t unbufferedDataWidth,
							 size_t unbufferedDataHeight, ssize_t windowRadius);

	~GpuWHS();

protected:
	ErrorCode launchKernel(unsigned blockWidth, unsigned blockHeight);

};


template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuWHS<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::GpuWHS(
	unsigned int cudaDeviceId, size_t unbufferedDataWidth, 
	size_t unbufferedDataHeight, ssize_t windowRadius) : 
	cvt::gpu::GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>(
	cudaDeviceId, unbufferedDataWidth,unbufferedDataHeight, windowRadius)
{
	;
}



template< typename inputpixeltype, int inputbandcount, typename outputpixeltype, int outputbandcount >
GpuWHS<inputpixeltype, inputbandcount, outputpixeltype, outputbandcount>::~GpuWHS() 
{	
	;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuWHS<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::launchKernel(unsigned blockWidth, unsigned blockHeight)
{
	dim3 dimBlock(blockWidth,blockHeight);
	size_t gridWidth = this->dataSize.width / dimBlock.x + (((this->dataSize.width % dimBlock.x)==0) ? 0 :1 );
	size_t gridHeight = this->dataSize.height / dimBlock.y + (((this->dataSize.height % dimBlock.y)==0) ? 0 :1 );
	dim3 dimGrid(gridWidth, gridHeight);

	// Bind the texture to the array and setup the access parameters
	cvt::gpu::bindTexture_sdsk_ushortTwoD(this->gpuInputDataArray);
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
	 cvt::gpu::launch_window_histogram_statistics<InputPixelType, OutputPixelType>(dimGrid, dimBlock, 0,
	   this->stream, (OutputPixelType *)this->gpuOutputData,
	   this->dataSize.width, this->dataSize.height, this->relativeOffsetsGpu_,
	   this->relativeOffsets_.size());
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
