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

namespace cvt {
namespace gpu {

/*
 * Texture cache:
 * Due to limitations with CUDA arch < 3.0, the textures must be declared at compile
 * time, globally. In order to encapsulate the internal use from user's wishing to
 * implement tile algorithms, an implict texture cache is created that is used by
 * the GpuAlgorithm parent classes. Each texture of related type is numbered one,
 * or two, and the bind_texture template is overloaded to inherently choose the correct
 * tile to bind based on its numerical position. The caller of bind texture is aware
 * of this implicit data structure and correctly binds/unbinds the textures that it requires
 * by passing in an index (0,1) corresponding to tiles one and two respectively.
 */
texture<signed char, cudaTextureType2D, cudaReadModeElementType> sdsk_scharTileOne;
texture<signed char, cudaTextureType2D, cudaReadModeElementType> sdsk_scharTileTwo;
texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> sdsk_ucharTileOne;
texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> sdsk_ucharTileTwo;
texture<short, cudaTextureType2D, cudaReadModeElementType> sdsk_shortTileOne;
texture<short, cudaTextureType2D, cudaReadModeElementType> sdsk_shortTileTwo;
texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> sdsk_ushortTileOne;
texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> sdsk_ushortTileTwo;
texture<int, cudaTextureType2D, cudaReadModeElementType> sdsk_intTileOne;
texture<int, cudaTextureType2D, cudaReadModeElementType> sdsk_intTileTwo;
texture<float, cudaTextureType2D, cudaReadModeElementType> sdsk_floatTileOne;
texture<float, cudaTextureType2D, cudaReadModeElementType> sdsk_floatTileTwo;
texture<double, cudaTextureType2D, cudaReadModeElementType> sdsk_doubleTileOne;
texture<double, cudaTextureType2D, cudaReadModeElementType> sdsk_doubleTileTwo;

/* Bind template and specializations */
template<typename T, int n>
cudaError_t bind_texture(cudaArray* gpuInputData);

template<typename T, int n>
cudaError_t unbind_texture();

template <>
cudaError_t bind_texture<signed char, 0>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_scharTileOne, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<signed char, 0>()
{
	cudaUnbindTexture(sdsk_scharTileOne);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<signed char, 1>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_scharTileTwo, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<signed char, 1>()
{
	cudaUnbindTexture(sdsk_scharTileTwo);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<unsigned char, 0>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_ucharTileOne, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<unsigned char, 0>()
{
	cudaUnbindTexture(sdsk_ucharTileOne);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<unsigned char, 1>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_ucharTileTwo, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<unsigned char, 1>()
{
	cudaUnbindTexture(sdsk_ucharTileOne);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<short, 0>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_shortTileOne, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<short, 0>()
{
	cudaUnbindTexture(sdsk_shortTileOne);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<short, 1>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_shortTileTwo, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<short, 1>()
{
	cudaUnbindTexture(sdsk_shortTileTwo);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<unsigned short, 0>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_ushortTileOne, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<unsigned short, 0>()
{
	cudaUnbindTexture(sdsk_ushortTileOne);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<unsigned short, 1>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_ushortTileTwo, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<unsigned short, 1>()
{
	cudaUnbindTexture(sdsk_ushortTileTwo);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<int, 0>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_intTileOne, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<int, 0>()
{
	cudaUnbindTexture(sdsk_intTileOne);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<int, 1>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_intTileTwo, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<int, 1>()
{
	cudaUnbindTexture(sdsk_intTileTwo);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<float, 0>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_floatTileOne, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<float, 0>()
{
	cudaUnbindTexture(sdsk_floatTileOne);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<float, 1>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_floatTileTwo, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<float, 1>()
{
	cudaUnbindTexture(sdsk_floatTileTwo);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<double, 0>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_doubleTileOne, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<double, 0>()
{
	cudaUnbindTexture(sdsk_doubleTileOne);
	return cudaGetLastError();
}

template <>
cudaError_t bind_texture<double, 1>(cudaArray* gpuInputData)
{
	sdsk_shortTileOne.addressMode[0] = cudaAddressModeClamp;
	sdsk_shortTileOne.addressMode[1] = cudaAddressModeClamp;
	sdsk_shortTileOne.filterMode = cudaFilterModePoint;
	sdsk_shortTileOne.normalized = false;
	cudaBindTextureToArray(sdsk_doubleTileTwo, gpuInputData);
	return cudaGetLastError();
}

template<>
cudaError_t unbind_texture<double, 1>()
{
	cudaUnbindTexture(sdsk_doubleTileTwo);
	return cudaGetLastError();
}

/* Collection of specializations for texture fetches */
template< typename InputPixelType, int TextureNumber >
__device__ __forceinline__ InputPixelType fetchTexture(int x, int y);

template <>
__device__ __forceinline__ unsigned char fetchTexture<unsigned char, 0>(int x, int y)
{
	return tex2D(sdsk_ucharTileOne, x, y);
}

template <>
__device__ __forceinline__ unsigned char fetchTexture<unsigned char, 1>(int x, int y)
{
	return tex2D(sdsk_ucharTileTwo, x, y);
}

template <>
__device__ __forceinline__ signed char fetchTexture<signed char, 0>(int x, int y)
{
	return tex2D(sdsk_scharTileOne, x, y);
}

template <>
__device__ __forceinline__ signed char fetchTexture<signed char, 1>(int x, int y)
{
	return tex2D(sdsk_scharTileTwo, x, y);
}

template <>
__device__ __forceinline__ short fetchTexture<short, 0>(int x, int y)
{
	return tex2D(sdsk_shortTileOne, x, y);
}

template <>
__device__ __forceinline__ short fetchTexture<short, 1>(int x, int y)
{
	return tex2D(sdsk_shortTileTwo, x, y);
}

template <>
__device__ __forceinline__ unsigned short fetchTexture<unsigned short, 0>(int x, int y)
{
	return tex2D(sdsk_ushortTileOne, x, y);
}

template <>
__device__ __forceinline__ unsigned short fetchTexture<unsigned short, 1>(int x, int y)
{
	return tex2D(sdsk_ushortTileTwo, x, y);
}

template <>
__device__ __forceinline__ int fetchTexture<int, 0>(int x, int y)
{
	return tex2D(sdsk_intTileOne, x, y);
}

template <>
__device__ __forceinline__ int fetchTexture<int, 1>(int x, int y)
{
	return tex2D(sdsk_intTileTwo, x, y);
}

template <>
__device__ __forceinline__ float fetchTexture<float, 0>(int x, int y)
{
	return tex2D(sdsk_floatTileOne, x, y);
}

template <>
__device__ __forceinline__ float fetchTexture<float, 1>(int x, int y)
{
	return tex2D(sdsk_floatTileTwo, x, y);
}

//////////////////////////////////
// Kernels and Launch Functions //
/////////////////////////////////

template< typename InputPixelType, typename OutputPixelType>
__global__ static void simpleDataCopyGlobal(InputPixelType* inputData, OutputPixelType* outputData, unsigned int width, unsigned int height, unsigned int bandCount)
{
	width *= bandCount; height *= bandCount;
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int pixel_one_d = xIndex + yIndex * width;

	if (xIndex < width && yIndex < height)
		outputData[pixel_one_d] = inputData[pixel_one_d];
}

template< typename InputPixelType, typename OutputPixelType>
__global__ static void simpleDataCopyTexture(OutputPixelType* outputData, unsigned int width, unsigned int height, unsigned int bandCount)
{
	width *= bandCount; height *= bandCount;
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int pixel_one_d = xIndex + yIndex * width;

	if (xIndex < width && yIndex < height)
		outputData[pixel_one_d] = fetchTexture<InputPixelType, 0>(xIndex, yIndex);
}

template< typename InputPixelType, typename OutputPixelType >
void launch_simpleDataCopy(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, InputPixelType* inputData,
						OutputPixelType * gpuOutputData, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount,
						bool usingTexture)
{
	if (!usingTexture)
		simpleDataCopyGlobal<InputPixelType, OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(inputData, gpuOutputData, outputWidth,  outputHeight, bandCount);
	else
	{
		simpleDataCopyTexture<InputPixelType, OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(gpuOutputData, outputWidth,  outputHeight, bandCount);
	}
}

template< typename InputPixelType, typename OutputPixelType, typename FilterType>
__global__ static
void convolutionTexture(OutputPixelType* const outputData, const unsigned int width,
						const unsigned int height, const int2* relativeOffsets,
						FilterType* const filterWeights, const unsigned int filterSize,
						const unsigned int bandCount)
{
	const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if(yIndex < height && xIndex < width){
		const unsigned int pixel_one_d = xIndex + yIndex * width;
		OutputPixelType outputValue;

		for(unsigned int i = 0; i < filterSize; ++i){
			const int cur_x_index = xIndex + relativeOffsets[i].x;
			const int cur_y_index = yIndex + relativeOffsets[i].y;
			if( cur_y_index < height && cur_y_index >= 0 && cur_x_index < width && cur_x_index >= 0 ){
				outputValue += fetchTexture<InputPixelType, 0>(cur_x_index, cur_y_index) * filterWeights[i];
			}
		}
		outputData[pixel_one_d] = outputValue;
	}
}

template< typename InputPixelType, typename OutputPixelType, typename FilterType>
__global__ static
void convolutionGlobal(InputPixelType* inputData, OutputPixelType* const outputData, const unsigned int width,
						const unsigned int height, const int2* relativeOffsets,
						FilterType* const filterWeights, const unsigned int filterSize, const unsigned int bandCount)
{
	const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int pixel_one_d = xIndex + yIndex * width;

	if(yIndex < height && xIndex < width){
		OutputPixelType outputValue;
		for(unsigned int i = 0; i < filterSize; ++i){
			const int cur_x_index = xIndex + relativeOffsets[i].x;
			const int cur_y_index = yIndex + relativeOffsets[i].y;
			const int cur_pixel_one_d = cur_x_index + cur_y_index * width;
			if( cur_y_index < height && cur_y_index >= 0 && cur_x_index < width && cur_x_index >= 0 ){
				outputValue += inputData[cur_pixel_one_d] * filterWeights[i];
			}
		}
		outputData[pixel_one_d] = outputValue;
	}
}

template<typename InputPixelType, typename OutputPixelType, typename FilterType>
void launchConvolution(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, InputPixelType* inputData,
						OutputPixelType * gpuOutputData, int2* relativeOffsets, FilterType* const filterWeights, const unsigned int filterSize,
						unsigned int outputWidth, unsigned int outputHeight, unsigned int bandCount,
						bool usingTexture)
{
	if (!usingTexture)
		convolutionGlobal<InputPixelType, OutputPixelType, FilterType><<<dimGrid, dimBlock, shmemSize, stream>>>(inputData, gpuOutputData, outputWidth,  outputHeight, relativeOffsets, filterWeights, filterSize, bandCount);
	else
		convolutionTexture<InputPixelType, OutputPixelType, FilterType><<<dimGrid, dimBlock, shmemSize, stream>>>(gpuOutputData, outputWidth,  outputHeight, relativeOffsets, filterWeights, filterSize, bandCount);
}

template<typename InputPixelType, typename OutputPixelType>
__global__ static
void localBinaryPatternTexture(OutputPixelType * outputData, const unsigned int roiWidth, const unsigned int roiHeight,
				    const unsigned int buffer, const int2* relativeOffsets, const unsigned int relOffsetSize)
{

    //LBP is a child of WindowFilter and is ALWAYS buffered.
    const int width = roiWidth + buffer + buffer;
    const int height = roiHeight + buffer + buffer;

    const unsigned int roiXidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int roiYidx = blockIdx.y * blockDim.y + threadIdx.y;

    // Block indexing is within the ROI, add the left and top buffer size to get global data index position
    const unsigned int xIndex = roiXidx + buffer;
    const unsigned int yIndex = roiYidx + buffer;

   if(roiYidx < roiHeight && roiXidx < roiWidth && xIndex < width && yIndex < height)
   {
       OutputPixelType outputValue = 0;
       OutputPixelType maskBit = 1;
       const int currentPixelOneD = roiXidx + roiYidx * roiWidth;

       //InputPixelType curValue = inputData[curIdx];
       InputPixelType curValue = fetchTexture<InputPixelType, 0>(xIndex, yIndex);

       InputPixelType nghValue = 0;
       for (unsigned int i = 0; i<relOffsetSize; ++i) {
		   const int nghX = xIndex + relativeOffsets[i].x;
		   const int nghY = yIndex + relativeOffsets[i].y;
		   //const int nghPixelOneD = nghX + nghY * width; //Global memory
		   nghValue = fetchTexture<InputPixelType, 0>(nghX, nghY);
		   if (nghValue >= curValue)
		   {
		       outputValue |= maskBit;
		   }
		   maskBit <<= 1;
       }
       outputData[currentPixelOneD] = outputValue;
   }

}

template<typename InputPixelType, typename OutputPixelType>
void launch_local_binary_pattern(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream,
						OutputPixelType* gpuOutputData, int2* relativeOffsets, unsigned int relativeOffsetsSize,
						const unsigned int roiWidth, const unsigned int roiHeight, const unsigned int buffer)
{
    localBinaryPatternTexture<InputPixelType, OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(gpuOutputData, roiWidth, roiHeight,
														buffer, relativeOffsets, relativeOffsetsSize);
}

template< typename InputPixelType, typename OutputPixelType>
__global__ static
void absDiffernceTexture(OutputPixelType * const outputData, const unsigned int roiWidth, const unsigned int roiHeight)
{
   // assuming not bufferred
   const int width = roiWidth;
   const int height = roiHeight;

   const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	InputPixelType t_one;
	InputPixelType t_two;

   if(xIndex < width && yIndex < height)
   {

		// Output size is the ROI size
		const unsigned int pixel_one_d = xIndex + yIndex * width; // xIndex + yIndex
		t_one = fetchTexture<InputPixelType, 0>(xIndex, yIndex);
		t_two = fetchTexture<InputPixelType, 1>(xIndex, yIndex);
		const OutputPixelType diff = t_one - t_two;
		outputData[pixel_one_d] = diff > 0 ? diff : -diff;
	}
}

template< typename InputPixelType, typename OutputPixelType>
void launch_absDifference(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize, const cudaStream_t stream,
						  OutputPixelType * const outputData, const unsigned int roiWidth,
						  const unsigned int roiHeight)
{
	absDiffernceTexture<InputPixelType,OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(outputData, roiWidth, roiHeight);
}

/*template< typename InputPixelType, typename OutputPixelType>
__global__ static
void window_histogram_statistics(OutputPixelType * const  outputData, const unsigned int roiWidth,
	    const unsigned int roiHeight,
	    const unsigned int numElements, const unsigned int buffer)*/



template< typename InputPixelType, typename OutputPixelType>
__global__ static
void window_histogram_statistics(OutputPixelType * const  outputData, const unsigned int roiWidth,
	    const unsigned int roiHeight, const int2 * relativeOffsets,
	    const unsigned int numElements, const unsigned int buffer)
{

   // Data index now buffered
   const int width = roiWidth + buffer + buffer;
   const int height = roiHeight + buffer + buffer;

   const unsigned int roiXidx = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int roiYidx = blockIdx.y * blockDim.y + threadIdx.y;

   // Block indexing is within the ROI, add the left and top buffer size to get data index position
   const unsigned int xIndex = roiXidx + buffer;
   const unsigned int yIndex = roiYidx + buffer;

   if(roiYidx < roiHeight && roiXidx < roiWidth && xIndex < width && yIndex < height)
   {
		// Output size is the ROI size
		const unsigned int pixel_one_d = roiXidx + roiYidx * roiWidth; // xIndex + yIndex
		const unsigned int outputBandSize = roiHeight * roiWidth;
		//init texture support; work in progress

		int cur_y_index;
		int cur_x_index;
		InputPixelType min = fetchTexture<InputPixelType, 0>(xIndex, yIndex);
		InputPixelType max = fetchTexture<InputPixelType, 0>(xIndex, yIndex);

		double values[1024];

		for(unsigned int i = 0; i < numElements; ++i)
		{
			cur_x_index = xIndex + relativeOffsets[i].x;
			cur_y_index = yIndex + relativeOffsets[i].y;
			//if( cur_y_index < roiHeight && cur_y_index >= 0 && cur_x_index < roiWidth && cur_x_index >= 0){
			values[i] = fetchTexture<InputPixelType, 0>(cur_x_index, cur_y_index);

			if (values[i] > max) {
				max = values[i];
			}
			if (values[i] < min) {
				min = values[i];
			}
		}

		/*
		 * Find the sum
		 */
		double sum = 0;
		double mean = 0;

		for (unsigned int i = 0; i < numElements; ++i) {
			sum = sum + values[i];
		}

		mean = (double) sum/ numElements;

		short num_bins = 128;
		short histogram[128];
		float pdf[128];

		for (unsigned int i = 0; i < num_bins; ++i) {
			histogram[i] = 0;
		}

		/*
		* Create histogram
		*/
		short bin_width = (max - min) / num_bins;
		if (bin_width < 1) {
			bin_width = 1;
		}

		short bin_idx = 0;
		for (unsigned int i  = 0; i < numElements; ++i) {

			bin_idx = (short) ((values[i] - min) / bin_width);

			if (bin_idx >= 0 && bin_idx < num_bins) {
				histogram[bin_idx]++;
			}
			else
				histogram[127]++;

		}

		/*
		 * Calculate the PDF array
		 */
		for (unsigned int i = 0; i < num_bins; ++i) {
			pdf[i] = ((float) histogram[i]) / numElements;
		}

		 /*
		  * Find Entropy
		  */
		 double entropy = 0;
		 for (short i = 0; i < num_bins; ++i) {
			if (pdf[i] != 0) {
				entropy += (pdf[i] * log2(pdf[i]));
			}
		 }

		// Normalize data with the mean
		for (unsigned int i = 0; i < numElements; ++i) {
			values[i] = values[i] - mean;
		}

		/*
		 * Find the variance
		 */
		double variance = 0;
		double std = 0;
		for (unsigned int i = 0; i < numElements; ++i) {
				 variance = variance + (values[i] * values[i]);

		}

		variance = (double) variance / (numElements);
		std = sqrtf(variance);



		if (std == 0 || variance == 0) {
			//band 0 = entropy
			outputData[pixel_one_d] = 0;

			outputData[pixel_one_d + outputBandSize] = (float )mean;
			//printf("mean %d\n", mean);

			//band 2 = variance
			outputData[pixel_one_d + (outputBandSize * 2)] = 0;

			//band 3 = skewness
			outputData[pixel_one_d + (outputBandSize * 3)] = 0;

			//band 4 = kurtosis
			outputData[pixel_one_d + (outputBandSize * 4)] = 0;

			return;

	}
	// ELSE

	/*
	 * Find Skewness
	 **/
	double skewness = 0;
	double kurtosis = 0;

	for (int i = 0; i < numElements; ++i) {
		skewness = skewness + (values[i] * values[i] * values[i]);
		kurtosis = kurtosis + (values[i] * values[i] * values[i] * values[i]);
	}
	skewness = (double)skewness/(numElements * variance * std);
	kurtosis = (double) kurtosis/(numElements * variance * variance);

		//band 0 = entropy
	outputData[pixel_one_d] = (OutputPixelType) (entropy * -1);
		//outputData[pixel_one_d] = 1;
		//band 1 = mean
	outputData[pixel_one_d + outputBandSize] = mean;

	outputData[pixel_one_d + outputBandSize] = ( OutputPixelType)mean;
		//printf("mean %d\n", mean);
		//band 2 = variance
	outputData[pixel_one_d + (outputBandSize * 2)] = (OutputPixelType) variance;

		//band 3 = skewness
	outputData[pixel_one_d + (outputBandSize * 3)] = (OutputPixelType) skewness;

		//band 4 = kurtosis
	outputData[pixel_one_d + (outputBandSize * 4)] = (OutputPixelType) kurtosis;


	} // END OF A VALID PIXEL POSITION

}

template< typename InputPixelType, typename OutputPixelType >
void launch_window_histogram_statistics (const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  OutputPixelType * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer) {
	window_histogram_statistics<InputPixelType, OutputPixelType><<<dimGrid, dimBlock, shmemSize,stream>>>(outputData, roiHeight, roiWidth, relativeOffsets, numElements, buffer);
}

/*template< typename InputPixelType, typename OutputPixelType >
void launch_window_histogram_statistics (const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  OutputPixelType * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight,
		   const unsigned int numElements, const unsigned int buffer) {
	window_histogram_statistics<InputPixelType, OutputPixelType><<<dimGrid, dimBlock, shmemSize,stream>>>(outputData, roiHeight, roiWidth, numElements, buffer);
}*/


/*cudaError_t load_relative_offsets(const cudaStream_t stream, int2* host_offsets, size_t num_offsets) {
	int2* cpnt;
	cudaError_t cuer;
	cudaGetSymbolAddress((void **)&cpnt,relativeOffsets);

	cudaMemcpyAsync(
		cpnt,
		host_offsets,
		num_offsets * sizeof(int2),
		cudaMemcpyHostToDevice,
		stream
	);
	cuer = cudaGetLastError();
	return cuer;
}*/


/* Assumes 2-D Grid, 2-D Block Config, 1 to 1 Mapping */
template< typename InputPixelType, typename OutputPixelType>
__global__ static
void erode(OutputPixelType* const  outputData, const unsigned int roiHeight,
	    const unsigned int roiWidth, const int2 * relativeOffsets,
	    const unsigned int numElements, const unsigned int buffer)
{

	// Data index now buffered
   const int width = roiWidth + buffer + buffer;
   const int height = roiHeight + buffer + buffer;

   const unsigned int roiXidx = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int roiYidx = blockIdx.y * blockDim.y + threadIdx.y;

   // Block indexing is within the ROI, add the left and top buffer size to get data index position
   const unsigned int xIndex = roiXidx + buffer;
   const unsigned int yIndex = roiYidx + buffer;

   if(roiYidx < roiHeight && roiXidx < roiWidth && xIndex < width && yIndex < height)
   {

		// Output size is the ROI size
		const unsigned int pixel_one_d = roiXidx + roiYidx * roiWidth; // xIndex + yIndex
		int cur_y_index;
		int cur_x_index;

		InputPixelType min = fetchTexture<InputPixelType, 0>(xIndex, yIndex);

		OutputPixelType values[1024];

		for(unsigned int i = 0; i < numElements; ++i)
		{
			cur_x_index = xIndex + relativeOffsets[i].x;
			cur_y_index = yIndex + relativeOffsets[i].y;

			values[i] = fetchTexture<InputPixelType, 0>(cur_x_index, cur_y_index);
			if (values[i] < min) {
				min = values[i];
			}


		}
		outputData[pixel_one_d] = min;
	}
}

template< typename InputPixelType, typename OutputPixelType>
void launch_erode(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  OutputPixelType * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer)
{
	erode<InputPixelType,OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(outputData, roiHeight, roiWidth, relativeOffsets, numElements, buffer);
}

template< typename InputPixelType, typename OutputPixelType>
__global__ static
void dilate(OutputPixelType* const  outputData, const unsigned int roiHeight,
	    const unsigned int roiWidth, const int2 * relativeOffsets,
	    const unsigned int numElements, const unsigned int buffer)
{

	// Data index now buffered
   const int width = roiWidth + buffer + buffer;
   const int height = roiHeight + buffer + buffer;

   const unsigned int roiXidx = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int roiYidx = blockIdx.y * blockDim.y + threadIdx.y;

   // Block indexing is within the ROI, add the left and top buffer size to get data index position
   const unsigned int xIndex = roiXidx + buffer;
   const unsigned int yIndex = roiYidx + buffer;

   if(roiYidx < roiHeight && roiXidx < roiWidth && xIndex < width && yIndex < height)
   {

		// Output size is the ROI size
		const unsigned int pixel_one_d = roiXidx + roiYidx * roiWidth; // xIndex + yIndex
		InputPixelType max = fetchTexture<InputPixelType, 0>(xIndex, yIndex);


		OutputPixelType values[1024];
		size_t cur_x_index;
		size_t cur_y_index;

		for(unsigned int i = 0; i < numElements; ++i)
		{
			cur_x_index = xIndex + relativeOffsets[i].x;
			cur_y_index = yIndex + relativeOffsets[i].y;


			values[i] = fetchTexture<InputPixelType, 0>(cur_x_index, cur_y_index);
			if (values[i] > max) {
				max = values[i];
			}

		}
		outputData[pixel_one_d] = max;
	}
}

template< typename InputPixelType, typename OutputPixelType>
void launch_dilate(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  OutputPixelType * const outputData,
		   const unsigned int roiWidth,  const unsigned int roiHeight, int2 * const relativeOffsets,
		   const unsigned int numElements, const unsigned int buffer)
{
	dilate<InputPixelType,OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(outputData, roiHeight, roiWidth, relativeOffsets, numElements, buffer);
}

}; //end gpu namespace
}; //end cvt namespace
