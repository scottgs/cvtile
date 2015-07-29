#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#ifndef _GPU_Algorithm_Kernels_HPP_
#define _GPU_Algorithm_Kernels_HPP_

namespace cvt {

namespace gpu {


cudaError bindTexture_sdsk_floatHueSaturation(cudaArray * gpu_input_data);

cudaError bindTexture_sdsk_shortTwoDNormalized(cudaArray * gpu_input_data);

cudaError unbindTexture_sdsk_shortTwoDNormalized();

cudaError bindTexture_sdsk_shortTwoD(cudaArray * gpu_input_data);

cudaError unbindTexture_sdsk_shortTwoD();

cudaError bindTexture_sdsk_ushortTwoD(cudaArray * gpu_input_data);

cudaError unbindTexture_sdsk_ushortTwoD();

cudaError bindTexture_sdsk_floatTileOne(cudaArray * gpu_input_data);

cudaError unbindTexture_sdsk_floatTileOne();

cudaError bindTexture_sdsk_floatTileTWo(cudaArray * gpu_input_data);

cudaError unbindTexture_sdsk_floatTileTwo();

cudaError bindTexture_sdsk_shortTileOne(cudaArray * gpu_input_data);

cudaError bindTexture_sdsk_shortTileTwo(cudaArray * gpu_input_data);

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

}
}

#endif
