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
// constant memory for linear structuring elements
/////////////////////////////////

__constant__ int2 relativeOffsets [800];
__constant__ unsigned int relativeOffsetCount;

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

template< typename InputPixelType, typename OutputPixelType>
__global__ static
void window_histogram_statistics(OutputPixelType * const  outputData, const unsigned int roiWidth,
	    const unsigned int roiHeight, const unsigned int buffer)
{

   const unsigned int roiXidx = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int roiYidx = blockIdx.y * blockDim.y + threadIdx.y;

   // Block indexing is within the ROI, add the left and top buffer size to get data index position
   const unsigned int xIndex = roiXidx + buffer;
   const unsigned int yIndex = roiYidx + buffer;
	
   if(roiYidx < roiHeight && roiXidx < roiWidth && xIndex < (roiWidth + 2*buffer) && yIndex < (roiHeight + 2*buffer))
   {
		// Output size is the ROI size
		const unsigned int pixel_one_d = roiXidx + roiYidx * roiWidth; // xIndex + yIndex
		const unsigned int outputBandSize = roiHeight * roiWidth;

		InputPixelType pixel_temp = fetchTexture<InputPixelType, 0>(xIndex, yIndex);
		InputPixelType min = pixel_temp;
		InputPixelType max = pixel_temp;

		double sum = 0.0;
		double mean = 0.0;
		double variance = 0.0;
		double std = 0.0;
		double pixel_difference = 0.0;
		
		for(unsigned int i = 0; i < relativeOffsetCount; ++i)
		{
			pixel_temp = fetchTexture<InputPixelType, 0>(xIndex + relativeOffsets[i].x, yIndex + relativeOffsets[i].y);

			// Roll sum calculation into min/max search for obvious reasons
			sum += pixel_temp;

			max = pixel_temp > max ? pixel_temp : max;
			min = pixel_temp < min ? pixel_temp : min;
		}

		mean = sum / relativeOffsetCount;

		// Create histogram
		const short num_bins = 128;
		short histogram[num_bins];
		//float pdf[num_bins];

		// This loop is the worst and hurts my soul
		for (unsigned int i = 0; i < num_bins; ++i) {
			histogram[i] = 0;
		}

		// This would go crazy if min and max were negative
		// Simple fix here is to change the detection to bin_width being zero
		// BUT, should it be corrected to -1 or 1?
		// 90% sure if the max is negative, then so is the min, which means we should be -1
		// Gotta think about it more. if min and max are both neg, then width should be neg.
		// But what if it's only one that's negative?

		// Either way, bin_width should probably be a float.
		// Then round the result for bin_idx
		short bin_width = (max - min) / num_bins;
		if (bin_width < 1) {
			bin_width = 1;
		}
		
		for (unsigned int i  = 0; i < relativeOffsetCount; ++i) {
		
			pixel_temp = fetchTexture<InputPixelType, 0>(xIndex + relativeOffsets[i].x, yIndex + relativeOffsets[i].y);
			// folding in variance calculation since we're already iterating through the data
			// and there's no other data requirements
			pixel_difference = pixel_temp - mean;
			variance += pixel_difference * pixel_difference;
			
			short bin_idx =  (pixel_temp - min) / bin_width;

			// I'm thinking these checks are not needed since we've already calibrated min/max/width properly
			// But once we look into negatives more, we may need something. IDK.
			if (bin_idx >= 0 && bin_idx < num_bins) {
				histogram[bin_idx]++;
			}
			else{
				histogram[127]++;
			}
		}
		
		variance /= relativeOffsetCount;

		// No point in continuting if variance is zero
		if (variance == 0.0) {
			//band 0 = entropy
			outputData[pixel_one_d] = 0;

			//band 1 = mean
			outputData[pixel_one_d + outputBandSize] = (OutputPixelType) mean;

			//band 2 = variance
			outputData[pixel_one_d + (outputBandSize * 2)] = 0;

			//band 3 = skewness
			outputData[pixel_one_d + (outputBandSize * 3)] = 0;

			//band 4 = kurtosis
			outputData[pixel_one_d + (outputBandSize * 4)] = 0;

			return;
		}
		
		std = sqrt(variance);

		// Calculate the PDF array
		//for (unsigned int i = 0; i < num_bins; ++i) {
		//	pdf[i] = ((float) histogram[i]) / relativeOffsetCount;
		//}
		// PDF isn't used outside entropy, calculating it when needed.
		 
		// Find Entropy
		double entropy = 0.0;
		for (short i = 0; i < num_bins; ++i) {
			// double?
			float pdf = ((float) histogram[i]) / relativeOffsetCount;
			entropy += pdf == 0.0f ? 0.0 : (pdf * log2(pdf));
		}
	
		// Find Skewness & Kurtosis
		
		double skewness = 0;
		double kurtosis = 0;

		for (int i = 0; i < relativeOffsetCount; ++i) {
			pixel_difference = fetchTexture<InputPixelType, 0>(xIndex + relativeOffsets[i].x, yIndex + relativeOffsets[i].y) - mean;
			double diff_cubed = pixel_difference * pixel_difference * pixel_difference;
			skewness += diff_cubed;
			kurtosis += diff_cubed * pixel_difference;
		}
		skewness /= relativeOffsetCount * variance * std;
		kurtosis /= relativeOffsetCount * variance * variance;

		//band 0 = entropy
		outputData[pixel_one_d] = (OutputPixelType) (entropy * -1);	

		//band 1 = mean		
		outputData[pixel_one_d + outputBandSize] = (OutputPixelType) mean;

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
		   const unsigned int roiWidth,  const unsigned int roiHeight, const unsigned int buffer) {
    window_histogram_statistics<InputPixelType, OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(
        outputData, roiHeight, roiWidth, buffer);
}


cudaError_t load_relative_offsets(const cudaStream_t stream, const int2* host_offsets, const unsigned int num_offsets) {
	int2* cpnt;
	unsigned int *deviceOffsetCount;
	cudaError_t cuer;
	cudaGetSymbolAddress((void **)&cpnt,relativeOffsets);
	cudaGetSymbolAddress((void **)&deviceOffsetCount,relativeOffsetCount);
	
	cudaMemcpyAsync(
		cpnt,
		host_offsets,
		num_offsets * sizeof(int2),
		cudaMemcpyHostToDevice,
		stream
	);
	cudaMemcpyAsync(
		deviceOffsetCount,
		&num_offsets,
		sizeof(unsigned int),
		cudaMemcpyHostToDevice,
		stream
	);
	cuer = cudaGetLastError();
	return cuer;
}


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
