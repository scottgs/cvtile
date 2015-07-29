
namespace cvt { 
namespace gpu {

const float normalizedFloatToShort_scaling_factor = 32767.f;

//Texture to bind input array
//cudaReadModeNormalizedFloat - Preferred, but can't do linear
// NOTE: cudaReadModeNormalizedFloat is required to get the hardware interpolation later
texture<short, cudaTextureType2D, cudaReadModeNormalizedFloat> sdsk_shortTwoDNormalized;
texture<short, cudaTextureType2D, cudaReadModeElementType> sdsk_shortTwoD;
texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> sdsk_ushortTwoD;
texture<float2, cudaTextureType2D, cudaReadModeElementType> sdsk_floatHueSaturation;
texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> sdsk_shortTileOne;
texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> sdsk_shortTileTwo;


template< typename InputPixelType, typename OutputPixelType> 
__global__ static void simpleDataCopy( InputPixelType * inputData, OutputPixelType *outputData, unsigned int width, unsigned int height, unsigned int bandCount)
{
	width *= bandCount; height *= bandCount;
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//index into our image
	int pixel_one_d = xIndex + yIndex * width;

	if (xIndex < width && yIndex < height)
		outputData[pixel_one_d] = inputData[pixel_one_d];
}

template< typename InputPixelType, typename OutputPixelType >
void launch_simpleDataCopy(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream, InputPixelType * in_data, 
						OutputPixelType * gpu_output_data, unsigned int outputWidth,  unsigned int outputHeight, unsigned int bandCount)
{
	simpleDataCopy<InputPixelType, OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(in_data, gpu_output_data, outputWidth,  outputHeight, bandCount);
}


template< typename InputPixelType, typename OutputPixelType>
__global__ static
void window_histogram_statistics(OutputPixelType * const  outputData, const unsigned int height,
	    const unsigned int width, const int2 * relativeOffsets,
	    const unsigned int numElements)
{

	const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if(yIndex < height && xIndex < width){

	const unsigned int pixel_one_d = xIndex + yIndex * width;
	const unsigned int area = height * width;
	const unsigned int outputBandSize = area;

	int cur_y_index;
	int cur_x_index;
	InputPixelType min = tex2D(sdsk_ushortTwoD, xIndex, yIndex);
	InputPixelType max = tex2D(sdsk_ushortTwoD, xIndex, yIndex);
	
	//extern __shared__ double values[]; uncomment when wanting to use dyanmic shared memory
	double values[1024];
	//const unsigned int pos = (threadIdx.x + threadIdx.y * blockDim.y) * numElements;

	for(unsigned int i = 0; i < numElements; ++i)
	{
		cur_x_index = xIndex + relativeOffsets[i].x;
		cur_y_index = yIndex + relativeOffsets[i].y;	
		
		if( cur_y_index < height && cur_y_index >= 0 && cur_x_index < width && cur_x_index >= 0){
			values[i] = tex2D(sdsk_ushortTwoD, cur_x_index, cur_y_index);
			//printf("GOOD COUNDS(%d, %d)\n", cur_x_index, cur_y_index);
		}
		else 
			values[i] = 0;

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
		//outputData[pixel_one_d + outputBandSize] = mean;
			
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
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets,
		   const unsigned int numElements) {
	window_histogram_statistics<InputPixelType, OutputPixelType><<<dimGrid, dimBlock, shmemSize,stream>>>(outputData, height, width, relativeOffsets, numElements);
}






/* Assumes 2-D Grid, 2-D Block Config, 1 to 1 Mapping */
template< typename InputPixelType, typename OutputPixelType>
__global__ static
void erode(OutputPixelType * const  outputData, const unsigned int height, 
	    const unsigned int width, const int2 * relativeOffsets, 
	    const unsigned int numElements)
{
	const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(yIndex < height && xIndex < width){

		const unsigned int pixel_one_d = xIndex + yIndex * width; 
		int cur_y_index;
		int cur_x_index;

		InputPixelType min = tex2D(sdsk_ushortTwoD, xIndex, yIndex);
	
		//extern __shared__ int2 offSets[]; 
		OutputPixelType values[1024];

		for(unsigned int i = 0; i < numElements; ++i)
		{
			cur_x_index = xIndex + relativeOffsets[i].x;
			cur_y_index = yIndex + relativeOffsets[i].y;	
		
			if( cur_y_index < height && cur_y_index >= 0 && cur_x_index < width && cur_x_index >= 0 ){
				values[i] = tex2D(sdsk_ushortTwoD, cur_x_index, cur_y_index);
				if (values[i] < min) {
					min = values[i];
				}

			}
		}
		outputData[pixel_one_d] = min;
	
	}

}


template< typename InputPixelType, typename OutputPixelType>
void launch_erode(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize, 
		   const cudaStream_t stream,  OutputPixelType * const outputData, 
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets, 
		   const unsigned int numElements)
{
	erode<InputPixelType,OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(outputData, height, width, relativeOffsets, numElements);
}

template< typename InputPixelType, typename OutputPixelType>
__global__ static
void dilate(OutputPixelType * const  outputData, const unsigned int height, 
	    const unsigned int width, const int2 * relativeOffsets, 
	    const unsigned int numElements)
{
	const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(yIndex < height && xIndex < width){

		const unsigned int pixel_one_d = xIndex + yIndex * width; 
		int cur_y_index;
		int cur_x_index;

		InputPixelType max = tex2D(sdsk_ushortTwoD, xIndex, yIndex);
	
		//extern __shared__ int2 offSets[]; 
		OutputPixelType values[1024];

		for(unsigned int i = 0; i < numElements; ++i)
		{
			cur_x_index = xIndex + relativeOffsets[i].x;
			cur_y_index = yIndex + relativeOffsets[i].y;	
		
			if( cur_y_index < height && cur_y_index >= 0 && cur_x_index < width && cur_x_index >= 0 ){
				values[i] = tex2D(sdsk_ushortTwoD, cur_x_index, cur_y_index);
				if (values[i] > max) {
					max = values[i];
				}

			}
		}
		outputData[pixel_one_d] = max;
	
	}

}

template< typename InputPixelType, typename OutputPixelType>
void launch_dilate(const dim3 dimGrid, const dim3 dimBlock, const unsigned int shmemSize,
		   const cudaStream_t stream,  OutputPixelType * const outputData, 
		   const unsigned int width,  const unsigned int height, int2 * const relativeOffsets, 
		   const unsigned int numElements)
{
	dilate<InputPixelType,OutputPixelType><<<dimGrid, dimBlock, shmemSize, stream>>>(outputData, height, width, relativeOffsets, numElements);
}



}; //end gpu namespace
}; //end cvt namespace
