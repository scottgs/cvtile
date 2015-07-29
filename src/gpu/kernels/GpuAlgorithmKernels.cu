#include "GpuAlgorithmKernels.cuh"
/*#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>*/

namespace cvt { 
namespace gpu {

// ==========================================================
//Interpolate the texture at point x, y 
// ==========================================================
__forceinline__ __device__ float texture_interpolateShort(const texture<short, cudaTextureType2D, cudaReadModeElementType> texref, float x, float y)
{
	int ix = (int)x; 
	int iy = (int)y;
	float dx = x - (float)ix;
	float dy = y - (float)iy;
	
	return(((float)tex2D(texref, ix, iy)*(1.f-dx)+(float)tex2D(texref, ix+1, iy)*(dx))*(1.f-dy)+((float)tex2D(texref, ix+1, iy+1)*(dx)+(float)tex2D(texref, ix, iy+1)*(1.f-dx))*(dy));
}

// ==========================================================
//	texture reference bind / un-bind
// ==========================================================

cudaError bindTexture_sdsk_shortTileOne(cudaArray * gpu_input_data)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_shortTileOne, gpu_input_data);

	return cudaGetLastError();
}

cudaError unbindTexture_sdsk_shortTileOne()
{
	cudaUnbindTexture(sdsk_shortTileOne);

	return cudaGetLastError();
}

cudaError bindTexture_sdsk_shortTileTwo(cudaArray * gpu_input_data)
{
	sdsk_shortTileTwo.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileTwo.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileTwo.filterMode = cudaFilterModePoint;
	sdsk_shortTileTwo.normalized = false;
	cudaBindTextureToArray(sdsk_shortTileTwo, gpu_input_data);

	return cudaGetLastError();
}

cudaError unbindTexture_sdsk_shortTileTwo()
{
	cudaUnbindTexture(sdsk_shortTileTwo);

	return cudaGetLastError();
}


cudaError bindTexture_sdsk_floatHueSaturation(cudaArray * gpu_input_data)
{
	sdsk_floatHueSaturation.addressMode[0] = cudaAddressModeClamp;
	sdsk_floatHueSaturation.addressMode[1] = cudaAddressModeClamp;
	sdsk_floatHueSaturation.filterMode = cudaFilterModePoint;
	sdsk_floatHueSaturation.normalized = false;
	cudaBindTextureToArray(sdsk_floatHueSaturation, gpu_input_data);

	return cudaGetLastError();
}

cudaError bindTexture_sdsk_shortTwoDNormalized(cudaArray * gpu_input_data)
{
	// ====================================================
	//	Bind the cuda array input to the texture
	// 		clamp indexing to border pixels of texture
	// ====================================================
	sdsk_shortTwoDNormalized.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTwoDNormalized.addressMode[1] = cudaAddressModeClamp;
	// 		Perform hardware linear interpolation
	sdsk_shortTwoDNormalized.filterMode = cudaFilterModeLinear;
	// 		NOTE: normalized = true makes indexing into texture [0,1) instead of [0,N)
	sdsk_shortTwoDNormalized.normalized = false;

	// Bind Texture
	// The change Description from the
	cudaBindTextureToArray(sdsk_floatHueSaturation, gpu_input_data);

	return cudaGetLastError();
}

cudaError unbindTexture_sdsk_shortTwoDNormalized()
{
	cudaUnbindTexture(sdsk_shortTwoDNormalized);

	return cudaGetLastError();
}

cudaError bindTexture_sdsk_shortTwoD(cudaArray * gpu_input_data)
{
	// ====================================================
	//	Bind the cuda array input to the texture
	// 		clamp indexing to border pixels of texture
	// ====================================================
	sdsk_shortTwoD.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTwoD.addressMode[1] = cudaAddressModeClamp;
	// 		Perform hardware linear interpolation
	sdsk_shortTwoD.filterMode = cudaFilterModePoint;
	// 		NOTE: normalized = true makes indexing into texture [0,1) instead of [0,N)
	sdsk_shortTwoD.normalized = false;

	// Bind Texture
	// The change Description from the
	cudaBindTextureToArray(sdsk_shortTwoD, gpu_input_data);

	return cudaGetLastError();
}

cudaError unbindTexture_sdsk_shortTwoD()
{
	cudaUnbindTexture(sdsk_shortTwoD);

	return cudaGetLastError();
}

cudaError bindTexture_sdsk_ushortTwoD(cudaArray * gpu_input_data)
{
	// ====================================================
	//	Bind the cuda array input to the texture
	// 		clamp indexing to border pixels of texture
	// ====================================================
	sdsk_ushortTwoD.addressMode[0] = cudaAddressModeClamp;
	sdsk_ushortTwoD.addressMode[1] = cudaAddressModeClamp;
	// 		Perform hardware linear interpolation
	sdsk_ushortTwoD.filterMode = cudaFilterModePoint;
	// 		NOTE: normalized = true makes indexing into texture [0,1) instead of [0,N)
	sdsk_ushortTwoD.normalized = false;

	// Bind Texture
	// The change Description from the
	cudaBindTextureToArray(sdsk_ushortTwoD, gpu_input_data);

	return cudaGetLastError();
}

cudaError unbindTexture_sdsk_ushortTwoD()
{
	cudaUnbindTexture(sdsk_ushortTwoD);

	return cudaGetLastError();
}


/*Explicit instantiations for data copy kernels */

template void launch_simpleDataCopy<int, int>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, int * in_data, 
						int * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<unsigned int, unsigned int>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, unsigned int * in_data, 
						unsigned int * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<unsigned char, unsigned char>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, unsigned char * in_data, 
						unsigned char * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<long, long>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, long * in_data, 
						long * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<unsigned short, unsigned short>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, unsigned short * in_data, 
						unsigned short * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<char, char>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, char * in_data, 
						char * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<float, float>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, float * in_data, 
						float * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<double, double>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, double * in_data, 
						double * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<short, short>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, short * in_data, 
						short * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<unsigned long, unsigned long>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, unsigned long * in_data, 
						unsigned long * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_simpleDataCopy<signed char, signed char>(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, signed char * in_data, 
						signed char * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount);

template void launch_window_histogram_statistics<short, float>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  float * const outputData,
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets,
		   const unsigned int numElements);


template void launch_dilate<short,short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize, 
		   const cudaStream_t stream,  short * const outputData, 
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets, 
		   const unsigned int numElements);

template void launch_erode<short,short>(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize, 
		   const cudaStream_t stream,  short * const outputData, 
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets, 
		   const unsigned int numElements);


}; //end gpu namespace
}; //end cvt namespace
