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

#ifndef _GPU_DRIVERS_PROPERTIES_
#define _GPU_DRIVERS_PROPERTIES_

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>

// CUDA Headers
#include <cuda_runtime.h>
// TODO: It's questionable if these extra CUDA includes are even necessary.
// The CUDART API sets MACRO CUDART_VERSION
#if CUDART_VERSION >= 7000
	//#pragma message("Using CUDA Toolkit 7 or higher")
	#include <cuda.h>
	// #include <help_cuda.h>
#elif CUDART_VERSION >= 5000
	#pragma message("Using CUDA Toolkit 5")
	#include <helper_functions.h>
// CUDA Toolkit 4.2 is minimum supported for compute_30 compatibility.
#elif CUDART_VERSION >= 4200
	#pragma message("Using CUDA Toolkit 4")
	#include <sdkHelper.h>
	#include <shrQATest.h>
	#include <shrUtils.h>
#endif


namespace cvt {
namespace gpu {

/// CUDA Device Properties
/// Wraps away the cudaDeviceProp structure
class GPUProperties {

public:

	///Constructor/Destructor
	///\param gpu_id The GPU ID to represent
	GPUProperties(int gpu_id);

	/// Deconstructor
	virtual ~GPUProperties();


	// ------------------------------------
	// -- Memory Information
	// ------------------------------------

	/// \returns true if Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
	bool getCanMapHostMemory() const
	{
		return (_props.canMapHostMemory > 0);
	}

	/// \returns true if Device has ECC support enabled
	inline bool isECCEnabled() const
	{
		return (_props.ECCEnabled > 0);
	}

	/// \returns Size of L2 cache in bytes
	inline int getL2CacheSizeBytes() const
	{
		return _props.l2CacheSize;
	}

	/// \returns Global memory bus width in bits
	inline int getMemoryBusWidthBits() const
	{
		return _props.memoryBusWidth;
	}

	/// \returns Peak memory clock frequency in kilohertz
	inline int getMemoryClockRate() const
	{
		return _props.memoryClockRate;
	}

	/// \returns Maximum pitch in bytes allowed by memory copies
	inline int getMaxMemoryPitch() const
	{
		return _props.memPitch;
	}

	/// \returns 32-bit registers available per block
	inline int getRegistersPerBlock() const
	{
		return _props.regsPerBlock;
	}

	/// \returns Shared memory available per block in bytes
	inline size_t getSharedMemPerBlockBytes() const
	{
		return _props.sharedMemPerBlock;
	}

	/// \returns Constant memory available on device in bytes
	inline size_t getTotalConstantMemoryBytes() const
	{
		return _props.totalConstMem;
	}

	/// \returns Global memory available on device in bytes
	inline size_t getTotalGlobalMemoryBytes() const
	{
		return _props.totalGlobalMem;
	}

	/// \returns true if Device shares a unified address space with the host
	inline bool hasUnifiedAddressing() const
	{
		return (_props.unifiedAddressing > 0);
	}

	// ------------------------------------
	// -- Compute Information
	// ------------------------------------


	/// \returns Number of asynchronous engines
	inline int getAsyncEngineCount() const
	{
		return _props.asyncEngineCount;
	}

	/// \returns Clock frequency in kilohertz
	inline int getClockRate() const
	{
		return _props.clockRate;
	}

	/// \brief Gets the compute mode of the device
	/// This int equates to a Enumerator in the CUDA SDK
	/// cudaComputeModeDefault Default compute mode (Multiple threads can use cudaSetDevice() with this device);
	///
	/// cudaComputeModeExclusive Compute-exclusive-thread mode (Only one thread in one process will be able to
	/// use cudaSetDevice() with this device);
	///
	/// cudaComputeModeProhibited Compute-prohibited mode (No threads can use cudaSetDevice() with this device);
	///
	/// cudaComputeModeExclusiveProcess Compute-exclusive-process mode (Many threads in one process will be
	/// able to use cudaSetDevice() with this device)
	///
	/// \returns Compute Mode
	inline int getComputeMode() const
	{
		return _props.computeMode;
	}

	/// \returns true if Device can possibly execute multiple kernels concurrently
	inline bool supportsConcurrentKernels() const
	{
		return (_props.concurrentKernels > 0);
	}

	/// \returns true if Device is integrated, false if Device is discrete
	inline bool isIntegrated() const
	{
		return (_props.integrated > 0);
	}

	/// \returns true if there is a run time limit on kernels
	inline bool isKernelExecTimeoutEnabled() const
	{
		return (_props.kernelExecTimeoutEnabled > 0);
	}

	/// \returns Major & Minor Compute as double
	inline double getCompute() const
	{
		return(compute);
	}

	/// \returns Major compute capability
	inline int getMajorCompute() const
	{
		return _props.major;
	}

	/// \returns Minor compute capability
	inline int getMinorCompute() const
	{
		return _props.minor;
	}

	/// \returns Number of multiprocessors on device
	inline int getMultiProcessorCount() const
	{
		return _props.multiProcessorCount;
	}

	/// \returns ASCII string identifying device
	inline std::string getDeviceName() const
	{
		return std::string(_props.name);
	}

	/// \returns true if device is a Tesla device using TCC driver, false otherwise
	inline bool supportsTeslaComputeClusterDriver() const
	{
		return (1 == _props.tccDriver);
	}

	/// \returns Warp size in threads
	inline int getWarpSize() const
	{
		return _props.warpSize;
	}


	// ------------------------------------
	// -- PCI Information
	// ------------------------------------

	/// \returns PCI bus ID of the device
	inline int getPciBusID() const
	{
		return _props.pciBusID;
	}

	/// \returns PCI device ID of the device
	inline int getPciDeviceID() const
	{
		return _props.pciDeviceID;
	}

	/// \returns PCI domain ID of the device
	inline int getPciDomainID() const
	{
		return _props.pciDomainID;
	}


	// ------------------------------------
	// -- GRID Dimensions Limits
	// ------------------------------------

	/// \returns a vector(3) of the x,y,z maximum size of each dimension of a grid
	std::vector<int> getMaxGridDims() const;

	/// \returns maximum X dimension of a grid
	inline int getMaxGridDim_X() const
	{
		return _props.maxGridSize[0];
	}

	/// \returns maximum Y dimension of a grid
	inline int getMaxGridDim_Y() const
	{
		return _props.maxGridSize[1];
	}

	/// \returns maximum Z dimension of a grid
	inline int getMaxGridDim_Z() const
	{
		return _props.maxGridSize[2];
	}

	// ------------------------------------
	// -- Thread / Block Dimensions Limits
	// ------------------------------------

	/// \returns Maximum size of each dimension of a block
	std::vector<int> getMaxThreadsDims() const;

	/// \returns Maximum size of X dimension of a block
	inline int getMaxThreadsPerBlock_X() const
	{
		return _props.maxThreadsDim[0];
	}

	/// \returns Maximum size of Y dimension of a block
	inline int getMaxThreadsPerBlock_Y() const
	{
		return _props.maxThreadsDim[1];
	}

	/// \returns Maximum size of Z dimension of a block
	inline int getMaxThreadsPerBlock_Z() const
	{
		return _props.maxThreadsDim[2];
	}

	/// \returns Maximum number of threads per block
	inline int getMaxThreadsPerBlock() const
	{
		return _props.maxThreadsPerBlock;
	}

	/// \returns Maximum resident threads per multiprocessor
	inline int getMaxThreadsPerMultiProcessor() const
	{
		return _props.maxThreadsPerMultiProcessor;
	}



	// ------------------------------------
	// -- Surface Dimension Limits
	// ------------------------------------

	/// \returns Maximum 1D surface size
	inline int getMaxSurface1DSizeBytes() const
	{
		return _props.maxSurface1D;
	}

	/// \returns Maximum 1D layered surface dimensions
	std::vector<int> getMaxSurface1DLayeredDims() const;

	/// \returns Maximum 2D surface dimensions
	std::vector<int> getMaxSurface2DDims() const;

	/// \returns Maximum X dimension of 2D surface
	inline int getMaxSurface2D_X() const
	{
		return _props.maxSurface2D[0];
	}

	/// \returns Maximum Y dimension of 2D surface
	inline int getMaxSurface2D_Y() const
	{
		return _props.maxSurface2D[1];
	}

	/// \returns Maximum 2D layered surface dimensions
	std::vector<int> getMaxSurface2DLayeredDims() const;

	/// \returns Maximum 3D surface dimensions
	std::vector<int> getMaxSurface3DDims() const;

	/// \returns Maximum X dimension of 3D surface
	inline int getMaxSurface3D_X() const
	{
		return _props.maxSurface3D[0];
	}

	/// \returns Maximum Y dimension of 3D surface
	inline int getMaxSurface3D_Y() const
	{
		return _props.maxSurface3D[1];
	}

	/// \returns Maximum Z dimension of 3D surface
	inline int getMaxSurface3D_Z() const
	{
		return _props.maxSurface3D[2];
	}

	/// \returns Maximum Cubemap surface dimensions
	inline int getMaxSurfaceCubemap() const
	{
		return _props.maxSurfaceCubemap;
	}

	/// \returns Maximum Cubemap layered surface dimensions
	std::vector<int> getMaxSurfaceCubemapLayered() const;

	/// \returns Alignment requirements for surfaces
	inline size_t getSurfaceAlignment() const
	{
		return _props.surfaceAlignment;
	}

	// ------------------------------------
	// -- Texture Dimension Limits
	// ------------------------------------

	/// \returns Maximum 1D texture size
	inline int getMaxTexture1DSizeBytes() const
	{
		return _props.maxTexture1D;
	}

	/// \returns Maximum 1D layered texture dimensions
	std::vector<int> getMaxTexture1DLayered() const;

	/// \returns Maximum size for 1D textures bound to linear memory
	inline int getMaxTexture1DLinearSizeBytes() const
	{
		return _props.maxTexture1DLinear;
	}

	/// \returns Maximum 2D texture dimensions
	std::vector<int> getMaxTexture2DDims() const;

	/// \returns Maximum X dimension of 2D texture
	inline int getMaxTexture2D_X() const
	{
		return _props.maxTexture2D[0];
	}

	/// \returns Maximum Y dimension of 2D texture
	inline int getMaxTexture2D_Y() const
	{
		return _props.maxTexture2D[1];
	}

	/// \returns Maximum 2D texture dimensions if texture gather operations have to be performed
	std::vector<int> getMaxTexture2DGatherDims() const;

	/// \returns Maximum X dimension of 2D texture if texture gather operations have to be performed
	inline int getMaxTexture2DGather_X() const
	{
		return _props.maxTexture2DGather[0];
	}

	/// \returns Maximum Y dimension of 2D texture if texture gather operations have to be performed
	inline int getMaxTexture2DGather_Y() const
	{
		return _props.maxTexture2DGather[1];
	}

	/// \returns Maximum 2D layered texture dimensions
	std::vector<int> getMaxTexture2DLayered() const;

	/// \returns Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
	std::vector<int> getMaxTexture2DLinearDims() const;

	/// \returns Maximum X dimension of 2D texture if texture gather operations have to be performed
	inline int getMaxTexture2DLinear_X() const
	{
		return _props.maxTexture2DLinear[0];
	}

	/// \returns Maximum Y dimension of 2D texture if texture gather operations have to be performed
	inline int getMaxTexture2DLinear_Y() const
	{
		return _props.maxTexture2DLinear[1];
	}

	/// \returns Maximum Y dimension of 2D texture if texture gather operations have to be performed
	inline int getMaxTexture2DLinear_Pitch() const
	{
		return _props.maxTexture2DLinear[2];
	}

	/// \returns Maximum 3D texture dimensions
	std::vector<int> getMaxTexture3DDims() const;

	/// \returns Maximum X dimension of 3D surface
	inline int getMaxTexture3D_X() const
	{
		return _props.maxTexture3D[0];
	}

	/// \returns Maximum Y dimension of 3D surface
	inline int getMaxTexture3D_Y() const
	{
		return _props.maxTexture3D[1];
	}

	/// \returns Maximum Z dimension of 3D surface
	inline int getMaxTexture3D_Z() const
	{
		return _props.maxTexture3D[2];
	}

	/// \returns Maximum Cubemap surface dimensions
	inline int getMaxTextureCubemap() const
	{
		return _props.maxTextureCubemap;
	}

	/// \returns Maximum Cubemap layered surface dimensions
	std::vector<int> getMaxTextureCubemapLayered() const;


	/// \returns Alignment requirement for textures
	inline size_t getTextureAlignment() const
	{
		return _props.textureAlignment;
	}

	/// \returns Pitch alignment requirement for texture references bound to pitched memory
	inline size_t getTexturePitchAlignment() const
	{
		return _props.texturePitchAlignment;
	}


friend std::ostream& operator<<(std::ostream& os, const GPUProperties& p);

protected:
	cudaDeviceProp _props;
	double compute;

};


std::vector<int> getGpuDeviceIds(int min_major = 1, int min_minor = 0);

}; // END cgi::gpu namespace
}; // END cgi namespace

//std::ostream& operator<<(std::ostream& os, const vmr::gpu::GPUProperties& p);

#endif /* CGI_GPU_DRIVERS_PROPERTIES */
