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

// TODO: Check into what this is for, and if it's still needed.

// ==========================================================
//      Test Kernel For Mike
// ==========================================================

__global__
void test_kernel(unsigned char *outputData, unsigned int outputWidth,  unsigned int outputHeight)
{

	//Get the current xIndex and yIndex of the image
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//index into our image
	int pixel_one_d = xIndex + yIndex * outputWidth;
	//int pixel_one_d_2 = (xIndex + yIndex * outputWidth) + (outputWidth * outputHeight);

	//float sampleX = ((float)xIndex) / sampleRate + 0.5;
	//float sampleY = ((float)yIndex) / sampleRate + 0.5;

	// CHECK VALID OUTPUT PIXEL Location
	if (xIndex < outputWidth && yIndex < outputHeight)
	{
		outputData[pixel_one_d] = 1;
		outputData[pixel_one_d + (outputWidth * outputHeight)] = 2;
		outputData[pixel_one_d + (outputWidth * outputHeight * 2)] = 3;
		outputData[pixel_one_d + (outputWidth * outputHeight * 3)] = 4;
	}
}


void launch_test(dim3 dimGrid, dim3 dimBlock, unsigned int shmemSize, cudaStream_t stream,
						unsigned char *outputData, unsigned int outputWidth,  unsigned int outputHeight)
{
	test_kernel<<<dimGrid, dimBlock, shmemSize, stream>>>(outputData,outputWidth,outputHeight);
}
