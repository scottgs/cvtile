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
#ifndef H_GPU_LBP_ALGORITHM_H
#define H_GPU_LBP_ALGORITHM_H

#include "GpuWindowFilterAlgorithm.hpp"
#include <vector>

namespace cvt {

namespace gpu {

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class GpuLBP : public GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>
{

    public:
        using GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::operator();

        explicit GpuLBP(unsigned int cudaDeviceId, size_t unbufferedDataWidth,
                                 size_t unbufferedDataHeight, ssize_t windowRadius);
        ErrorCode operator()(const cvt::cvTile<InputPixelType>& tile,
                             const cvt::cvTile<OutputPixelType> ** outTile,
                             unsigned short blockWidth, unsigned short blockHeight);
        ~GpuLBP();

    protected:
        ErrorCode launchKernel(unsigned blockWidth, unsigned blockHeight);

};

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
GpuLBP<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::GpuLBP(
    unsigned int cudaDeviceId, size_t unbufferedDataWidth,
    size_t unbufferedDataHeight, ssize_t windowRadius) :
    cvt::gpu::GpuWindowFilterAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>(
    cudaDeviceId, unbufferedDataWidth,unbufferedDataHeight, windowRadius)
{
    ;
}


template< typename inputpixeltype, int inputbandcount, typename outputpixeltype, int outputbandcount >
GpuLBP<inputpixeltype, inputbandcount, outputpixeltype, outputbandcount>::~GpuLBP()
{
    ;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
ErrorCode GpuLBP<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::operator()(const cvt::cvTile<InputPixelType>& tile,
                                                      const cvt::cvTile<OutputPixelType> ** outTile, unsigned short blockWidth, unsigned short blockHeight)
{
    //TODO: TEST BOTH DEFAULT IMPL AND THIS.
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

    // Invoke kernel with empirically chosen block size
    //unsigned short bW = 16; // 16
    //unsigned short bH = 16; // 16


    if ((unsigned int) tile.getROI().x != this->bufferWidth_) {
        throw std::runtime_error("Buffer width of incoming tile is not equal to the window radius");
    }

    launchKernel(blockWidth, blockHeight);

    this->lastError = this->copyTileFromDevice(outTile);
    if(this->lastError != cvt::Ok) {
        std::runtime_error("Failed copy off tile from device");
    }
    return Ok;
}

template<typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount>
ErrorCode GpuLBP<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::launchKernel(unsigned blockWidth, unsigned blockHeight)
{

    dim3 dimBlock(blockWidth,blockHeight);

    size_t gridWidth = this->dataSize.width / dimBlock.x + (((this->dataSize.width % dimBlock.x)==0) ? 0 :1 );
    size_t gridHeight = this->dataSize.height / dimBlock.y + (((this->dataSize.height % dimBlock.y)==0) ? 0 :1 );
    dim3 dimGrid(gridWidth, gridHeight);

    // Look into the texture stuff soon...
    // Bind the texture to the array and setup the access parameters
    cvt::gpu::bind_texture<InputPixelType,0>(this->gpuInputDataArray);
    cudaError cuer = cudaGetLastError();
    if (cudaSuccess != cuer)
    {
        return CudaError; // needs to be changed
    }

    //TODO: Use this line when updating to use shared memory
    //const unsigned int shmem_bytes = neighbor_coordinates_.size() * sizeof(double) * blockWidth * blockHeight;
    cvt::gpu::launch_local_binary_pattern<InputPixelType, OutputPixelType>(dimGrid, dimBlock, 0,
            this->stream, (OutputPixelType *)this->gpuOutputData,
            this->relativeOffsetsGpu_, this->relativeOffsets_.size(),
            this->roiSize_.width, this->roiSize_.height,
            this->bufferWidth_);

    cuer = cudaGetLastError();
    if (cuer != cudaSuccess) {
        std::cout << "CUDA ERROR = " << cuer << std::endl;
        throw std::runtime_error("KERNEL LAUNCH FAILURE");
    }
    return CudaError; // needs to be changed

};

} //END OF GPU NAMESPACE
} // END OF CVT NAMESPACE

#endif //H_GPU_LBP_ALGORITHM_H
