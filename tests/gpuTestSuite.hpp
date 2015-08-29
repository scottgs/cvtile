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

#ifndef  gpu_Test_Suite_h
#define  gpu_Test_Suite_h

#include <cxxtest/TestSuite.h> 
#include "gpuTestHelpers.hpp"
#include "../src/base/cvTile.hpp"
#include "../src/base/Tiler.hpp"

#include <boost/filesystem.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>

/* Gpu Depends */
//#include "../src/gpu/drivers/GpuTileAlgorithm.hpp"
#include "../src/gpu/drivers/GPUProperties.hpp"
//#include "../src/gpu/drivers/GpuErodeDilate.hpp"
#include "../src/gpu/drivers/GpuAlgorithm.hpp"
#include "../src/gpu/kernels/GpuAlgorithmKernels.hpp"

#define SHOW_OUTPUT 0

using namespace std;
using namespace cvt;

template < typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
	class gpuAlgoImpl : public cvt::gpu::GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>

	{
		public:
		gpuAlgoImpl(unsigned int deviceId, unsigned int dataHeight, unsigned int dataWidth)
			: cvt::gpu::GpuAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>(deviceId, dataHeight, dataWidth)
		{
		
		}

		virtual ErrorCode operator()(const cvt::cvTile<InputPixelType> &tile, const cvt::cvTile<OutputPixelType> ** outTile);

		protected:
		virtual ErrorCode launchKernel(unsigned bw, unsigned bh);
	
	};	

		
template < typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
		ErrorCode gpuAlgoImpl<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::launchKernel(unsigned bw, unsigned bh)
		{

			dim3 blockDim(bw, bh);
			const unsigned int gridWidth = this->dataSize.width * OutputBandCount / blockDim.x + (((this->dataSize.width % blockDim.x) == 0) ? 0 : 1); 
			const unsigned int gridHeight = this->dataSize.height * OutputBandCount / blockDim.y + (((this->dataSize.height % blockDim.y) == 0) ? 0 : 1); 
			dim3 gridDim(gridWidth, gridHeight);
		
			cvt::gpu::launch_simpleDataCopy<InputPixelType, OutputPixelType>(gridDim, blockDim, 0u, this->stream, (InputPixelType  *)this->gpuInput, this->gpuOutputData, this->dataSize.width, this->dataSize.height, (unsigned int)OutputBandCount, this->usingTexture); 

			return Ok;
	}
		
template < typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
		ErrorCode gpuAlgoImpl<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::operator()(const cvt::cvTile<InputPixelType>& tile,
			const cvt::cvTile<OutputPixelType> ** outTile)
		{
	
			this->lastError = this->copyTileToDevice(tile);
			if(this->lastError != cvt::Ok){
				return this->lastError;
			}
		
			if(this->usingTexture)
				cvt::gpu::bind_texture<InputPixelType, 0>((cudaArray*)this->gpuInput);	
			launchKernel(this->dataSize.width, this->dataSize.height);
			
			this->lastError = this->copyTileFromDevice(outTile);

			if(this->usingTexture)	
				cvt::gpu::unbind_texture<InputPixelType, 0>();	
			return Ok;
		}


class gpuTestSuite : public CxxTest::TestSuite{

	public:
		void setUp(){}
		void tearDown(){}

		void testInvalidConsturctBadWidth(){

			gpuAlgoImpl<signed char, 1, signed char, 1> gpuAlgo(0, 0, 100);
			ErrorCode lastError = gpuAlgo.getLastError();

			TS_ASSERT_EQUALS(cvt::GpuAlgoNoConstructBadInputValue, lastError);

		}

		void testInvalidConsturctBadHeight(){

			gpuAlgoImpl<signed char, 1, signed char, 1> gpuAlgo(0, 100, 0);
			ErrorCode lastError = gpuAlgo.getLastError();
 
			TS_ASSERT_EQUALS(cvt::GpuAlgoNoConstructBadInputValue, lastError);

		}

		void testValidConstruct(){

			gpuAlgoImpl<unsigned char, 1, unsigned char, 1> gpuAlgo(0, 100, 150);

			cv::Size2i dSize = gpuAlgo.getDataSize();
			cvt::gpu::GPUProperties prop = gpuAlgo.getProperties();
			size_t inBytes = gpuAlgo.getInputBytesToTransfer();
			
			ErrorCode lastError = gpuAlgo.initializeDevice();
			TS_ASSERT_EQUALS(cvt::Ok, lastError);

			TS_ASSERT_EQUALS(dSize.width, 100);
			TS_ASSERT_EQUALS(dSize.height, 150);
			TS_ASSERT_EQUALS(inBytes, dSize.width * dSize.height * sizeof(char));

			//TO-DO Ammend Properties class to test for valid ID then test here too

		}

		void testInitBadDevice(){

			/* Construct with a device ID one too large */
			std::vector<int> devices = cvt::gpu::getGpuDeviceIds();
			gpuAlgoImpl<unsigned char, 1, unsigned char, 1> gpuAlgo(devices.size(),100,100);
			ErrorCode lastError = gpuAlgo.getLastError();


			/* Initialize the card */
			lastError = gpuAlgo.initializeDevice();
			TS_ASSERT_EQUALS(cvt::InitFailcudaErrorInvalidDevice, lastError);

		}
	
		//Uint_Max not big enough - may need to update to long - To-do: Decide
		/*void (){
	
			gpuAlgoImpl<unsigned char, 1, unsigned char, 1> gpuAlgo(0,UINT_MAX, 1);
			std::cout << "-----------" << std::endl;
			ErrorCode lastError = gpuAlgo.initializeDevice();
			std::cout << "-----------" << std::endl;
			TS_ASSERT_EQUALS(cvt::InitFailInsufficientMemoryForInputData, lastError);
		
		}*/

		//TO-DO test for cuda stream create

		void testFloatChannelType(){

			gpuAlgoImpl<float, 1, float, 1> gpuAlgo(0,100,100);
			ErrorCode lastError = gpuAlgo.getLastError();

			TS_ASSERT_EQUALS(cvt::Ok, lastError);

			lastError = gpuAlgo.initializeDevice();
			TS_ASSERT_EQUALS(cvt::Ok, lastError);
			TS_ASSERT_EQUALS(gpuAlgo.getChannelType(), cudaChannelFormatKindFloat);

		}

		void testUnsignedCharChannelType(){

			gpuAlgoImpl<unsigned char, 1, unsigned char, 1> gpuAlgo(0,100,100);
			ErrorCode lastError = gpuAlgo.getLastError();

			TS_ASSERT_EQUALS(cvt::Ok, lastError);

			lastError = gpuAlgo.initializeDevice();
			TS_ASSERT_EQUALS(cvt::Ok, lastError);
			TS_ASSERT_EQUALS(gpuAlgo.getChannelType(), cudaChannelFormatKindUnsigned);
		}

		void testUnsignedShortChannelType(){

			gpuAlgoImpl<unsigned short, 1, unsigned short, 1> gpuAlgo(0,100,100);
			ErrorCode lastError = gpuAlgo.getLastError();

			TS_ASSERT_EQUALS(cvt::Ok, lastError);

			lastError = gpuAlgo.initializeDevice();
			TS_ASSERT_EQUALS(cvt::Ok, lastError);
			TS_ASSERT_EQUALS(gpuAlgo.getChannelType(), cudaChannelFormatKindUnsigned);

		}

		void testSignedCharChannelType(){

			gpuAlgoImpl<signed char, 1, signed char, 1> gpuAlgo(0,100,100);
			ErrorCode lastError = gpuAlgo.getLastError();

			TS_ASSERT_EQUALS(cvt::Ok, lastError);

			lastError = gpuAlgo.initializeDevice();
			TS_ASSERT_EQUALS(cvt::Ok, lastError);
			TS_ASSERT_EQUALS(gpuAlgo.getChannelType(), cudaChannelFormatKindSigned);

		}

		void testShortChannelType(){

			gpuAlgoImpl<short, 1, short, 1> gpuAlgo(0,100,100);
			ErrorCode lastError = gpuAlgo.getLastError();

			TS_ASSERT_EQUALS(cvt::Ok, lastError);

			lastError = gpuAlgo.initializeDevice();
			TS_ASSERT_EQUALS(cvt::Ok, lastError);
			TS_ASSERT_EQUALS(gpuAlgo.getChannelType(), cudaChannelFormatKindSigned);

		}

		void testIntChannelType(){

			gpuAlgoImpl<int, 1, int, 1> gpuAlgo(0,100,100);
			ErrorCode lastError = gpuAlgo.getLastError();

			TS_ASSERT_EQUALS(cvt::Ok, lastError);

			lastError = gpuAlgo.initializeDevice();
			TS_ASSERT_EQUALS(cvt::Ok, lastError);
			TS_ASSERT_EQUALS(gpuAlgo.getChannelType(), cudaChannelFormatKindSigned);

		}

		//TO-DO Impl this behavior then test for it
		void testExceptionThrownForZeroBands(){
			
		}

		//Eventually - cheat for now and test for only 1
		void testTextureUsedForUnderFourBands(){
			
			gpuAlgoImpl<int, 1, int, 1> gpuAlgo1(0,100,100);
			gpuAlgoImpl<int, 2, int, 2> gpuAlgo2(0,100,100);
			gpuAlgoImpl<int, 3, int, 3> gpuAlgo3(0,100,100);
			gpuAlgoImpl<int, 4, int, 4> gpuAlgo4(0,100,100);		

			TS_ASSERT_EQUALS(cvt::Ok, gpuAlgo1.initializeDevice());
			TS_ASSERT_EQUALS(cvt::Ok, gpuAlgo2.initializeDevice());
			TS_ASSERT_EQUALS(cvt::Ok, gpuAlgo3.initializeDevice());
			TS_ASSERT_EQUALS(cvt::Ok, gpuAlgo4.initializeDevice());

			TS_ASSERT(gpuAlgo1.getUsingTexture());	
			TS_ASSERT(!gpuAlgo2.getUsingTexture());				
		    TS_ASSERT(!gpuAlgo3.getUsingTexture());
			TS_ASSERT(!gpuAlgo4.getUsingTexture());
		
		}

		void testGlobalUsedForFivePlusBands(){
			
			gpuAlgoImpl<int, 5, int, 5> gpuAlgo(0,100,100);
			
			TS_ASSERT_EQUALS(cvt::Ok, gpuAlgo.initializeDevice());
			TS_ASSERT_EQUALS(false, gpuAlgo.getUsingTexture());
		}

		void testOutputDataSize(){
			
			gpuAlgoImpl<int, 5, int, 5> gpuAlgo(0,100,100);
			TS_ASSERT_EQUALS(cvt::Ok, gpuAlgo.initializeDevice());
			TS_ASSERT_EQUALS(gpuAlgo.getOutputDataSize(), 100 * 100 * sizeof(int) * 5);
		}

		template<typename T>
		void OneBandCopyToDevice()
		{
			cv::Size2i dSize(3,3);
			gpuAlgoImpl<T, 1, T, 1> gpuAlgo(0, dSize.width, dSize.height);
			TS_ASSERT_EQUALS(cvt::Ok, gpuAlgo.initializeDevice());

			vector<T> data;
			data.resize(dSize.area());

			for (unsigned int i = 0; i < 9; ++i) {
				data[i] = i;
			}

			cvt::cvTile<T> inTile(data.data(), dSize, 1);	
			cvt::cvTile<T>* outTile;

			gpuAlgo(inTile, (const cvt::cvTile<T> **)(&outTile));
			TS_ASSERT_EQUALS(0, (outTile == NULL));
			
			for(int i = 0; i < 3; ++i)
			{
				cv::Mat& a = inTile[0]; 
				cv::Mat& b = (*outTile)[0];
				for(int j = 0; j < 3; ++j)
				{
					TS_ASSERT_EQUALS(a.at<T>(i,j), b.at<T>(i,j));
				}
			}
		}

		void test1BandCopyToDevice()
		{
			TEST_ALL_TYPES(OneBandCopyToDevice);
		}

		/* The default operator of the test class performs a simple data copy */
		void test3BandCopyToDevice(){

			cv::Size2i dSize(3,3);
			gpuAlgoImpl<short, 3, short, 3> gpuAlgo(0,dSize.width,dSize.height);
			TS_ASSERT_EQUALS(cvt::Ok, gpuAlgo.initializeDevice());

			vector<short> data;
			data.resize(dSize.area() * 3);
			srand(time(NULL));
			
			for (unsigned int i = 0; i < 27; ++i) {
				data[i] = i + 1;
			}

			cvt::cvTile<short> inTile(data.data(), dSize, 3);	
			cvt::cvTile<short>*  outTile;

			gpuAlgo(inTile, (const cvt::cvTile<short> **)(&outTile));
		
			TS_ASSERT_EQUALS(0, (outTile == NULL));
			auto err = gpuAlgo.getLastError();

			for(int k = 0; k < 3; ++k)
			{
				for(int i = 0; i < 3; ++i)
				{
					cv::Mat& a = inTile[k]; 
					cv::Mat& b = (*outTile)[k];
					cv::Mat C(a);
					for(int j = 0; j < 3; ++j)
					{
#if SHOW_OUTPUT
						std::cout << "a: " <<  a.at<short>(i,j) << std::endl;	
						std::cout << "b: " <<  b.at<short>(i,j) << std::endl;	
						std::cout << "c: " <<  C.at<short>(i,j) << std::endl;
#endif
						TS_ASSERT_EQUALS(a.at<short>(i,j), b.at<short>(i,j));
					}
				}
			}

		}

//		void testErode(){
//
//			cvt::Tiler read_tiler;
//			cvt::Tiler write_tiler;
//
//			std::string sourceFile("test2-4.tif");
//			std::string outFile("test2-4-eroded.tif");
//
//			/* Ensure file opens correctly */
//			TS_ASSERT_EQUALS(cvt::NoError, read_tiler.open(sourceFile));
//
//			cv::Size2i rSize = read_tiler.getRasterSize();
//			unsigned long numPixels = rSize.area();
//
//			cv::Size2i tSize(256,256);
//
//			read_tiler.setCvTileSize(tSize);
//			write_tiler.setCvTileSize(tSize);
//
//			if(boost::filesystem::exists(outFile)){
//				boost::filesystem::remove(outFile);
//			}
//
//			/* Ensure tiler opens outfile correctly */
//			TS_ASSERT_EQUALS(cvt::NoError, write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), 1, cvt::Depth8U))
//
//			std::chrono::high_resolution_clock::time_point startI = std::chrono::high_resolution_clock::now();
//
//			int deviceID = 0;
//			int iH = 256;
//			int iW = 256;
//
//			/* Initialize a square structing element with a radius of 2 */
//			cvt::gpu::GpuErodeDilate gpuErode(deviceID, iW, iH, cvt::gpu::SQUARE, 2);
//
//
//			/* Initialize CUDA device */
//			gpuErode.initializeDevice();
//			TS_ASSERT_EQUALS(gpuErode.verifyInitialization(), true);
//
//			std::cout << read_tiler.getCvTileCount() << std::endl;
//
//			/* Loop through all the tiles in the image */
//			for(int i = 0; i < read_tiler.getCvTileCount(); ++i){
//
//				std::chrono::high_resolution_clock::time_point startT = std::chrono::high_resolution_clock::now();
//
//				/* Retrieve a tile, with 126 pixel edge buffer */
//				const cvt::cvTile<short> tile = read_tiler.getCvTile<short>(i, 0); //126 pixel buffer
//
//				/* Erode the tile */
//				gpuErode(tile, cvt::gpu::ERODE);
//
//				cvt::cvTile<short> * outTile;
//				gpuErode.copyTileFromDevice(&outTile);
//
//				write_tiler.putCvTile( *outTile  , i);
//
//				std::chrono::high_resolution_clock::time_point stopT = std::chrono::high_resolution_clock::now();
//				std::chrono::duration<double> time_spanT = std::chrono::duration_cast<std::chrono::duration<double> >(stopT - startT);
//
//				std::cout << "Tile [" << i << "]    : " << time_spanT.count() <<  std::endl;
//
//			}
//
//			std::chrono::high_resolution_clock::time_point stopI = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double> time_spanI = std::chrono::duration_cast<std::chrono::duration<double> >(stopI - startI);
//
//			double pixels_per_second = numPixels / time_spanI.count();
//			double seconds_per_pixel = time_spanI.count() / numPixels;
//
//			std::cout << "It took me    :" << time_spanI.count() <<  std::endl;
//			std::cout << "Pixels/Second :" << pixels_per_second <<  std::endl;
//			std::cout << "Seconds/Pixel :" << seconds_per_pixel <<  std::endl;
//
//			write_tiler.close();
//			read_tiler.close();
//
//			
//		}
//
//		void testDilate(){
//
//			cvt::Tiler read_tiler;
//			cvt::Tiler write_tiler;
//
//			std::string sourceFile("test2-4.tif");
//			std::string outFile("test2-4-dilated.tif");
//
//			/* Ensure file opens correctly */
//			TS_ASSERT_EQUALS(cvt::NoError, read_tiler.open(sourceFile));
//
//			cv::Size2i rSize = read_tiler.getRasterSize();
//			unsigned long numPixels = rSize.area();
//
//			cv::Size2i tSize(256,256);
//
//			read_tiler.setCvTileSize(tSize);
//			write_tiler.setCvTileSize(tSize);
//
//			if(boost::filesystem::exists(outFile)){
//				boost::filesystem::remove(outFile);
//			}
//
//			/* Ensure tiler opens outfile correctly */
//			TS_ASSERT_EQUALS(cvt::NoError, write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), 1, cvt::Depth8U))
//
//			std::chrono::high_resolution_clock::time_point startI = std::chrono::high_resolution_clock::now();
//
//			int deviceID = 0;
//			int iH = 256;
//			int iW = 256;
//
//			/* Initialize a square structing element with a radius of 4 */
//			cvt::gpu::GpuErodeDilate gpuDilate(deviceID, iW, iH, cvt::gpu::SQUARE, 2);
//
//
//			/* Initialize CUDA device */
//			gpuDilate.initializeDevice();
//			TS_ASSERT_EQUALS(gpuDilate.verifyInitialization(), true);
//
//			std::cout << read_tiler.getCvTileCount() << std::endl;
//
//			/* Loop through all the tiles in the image */
//			for(int i = 0; i < read_tiler.getCvTileCount(); ++i){
//
//				std::chrono::high_resolution_clock::time_point startT = std::chrono::high_resolution_clock::now();
//
//				/* Retrieve a tile, with 126 pixel edge buffer */
//				const cvt::cvTile<short> tile = read_tiler.getCvTile<short>(i, 0); //126 pixel buffer
//
//				/* Dilate the tile */
//				gpuDilate(tile, cvt::gpu::DILATE);
//
//				cvt::cvTile<short> * outTile;
//				gpuDilate.copyTileFromDevice(&outTile);
//
//				write_tiler.putCvTile( *outTile  , i);
//
//				std::chrono::high_resolution_clock::time_point stopT = std::chrono::high_resolution_clock::now();
//				std::chrono::duration<double> time_spanT = std::chrono::duration_cast<std::chrono::duration<double> >(stopT - startT);
//
//				std::cout << "Tile [" << i << "]    : " << time_spanT.count() <<  std::endl;
//
//			}
//
//			std::chrono::high_resolution_clock::time_point stopI = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double> time_spanI = std::chrono::duration_cast<std::chrono::duration<double> >(stopI - startI);
//
//			double pixels_per_second = numPixels / time_spanI.count();
//			double seconds_per_pixel = time_spanI.count() / numPixels;
//
//			std::cout << "It took me    :" << time_spanI.count() <<  std::endl;
//			std::cout << "Pixels/Second :" << pixels_per_second <<  std::endl;
//			std::cout << "Seconds/Pixel :" << seconds_per_pixel <<  std::endl;
//
//			write_tiler.close();
//			read_tiler.close();
//
//			
//		}

		//		void testErode(){
//
//			cvt::Tiler read_tiler;
//			cvt::Tiler write_tiler;
//
//			std::string sourceFile("test2-4.tif");
//			std::string outFile("test2-4-eroded.tif");
//
//			/* Ensure file opens correctly */
//			TS_ASSERT_EQUALS(cvt::NoError, read_tiler.open(sourceFile));
//
//			cv::Size2i rSize = read_tiler.getRasterSize();
//			unsigned long numPixels = rSize.area();
//
//			cv::Size2i tSize(256,256);
//
//			read_tiler.setCvTileSize(tSize);
//			write_tiler.setCvTileSize(tSize);
//
//			if(boost::filesystem::exists(outFile)){
//				boost::filesystem::remove(outFile);
//			}
//
//			/* Ensure tiler opens outfile correctly */
//			TS_ASSERT_EQUALS(cvt::NoError, write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), 1, cvt::Depth8U))
//
//			std::chrono::high_resolution_clock::time_point startI = std::chrono::high_resolution_clock::now();
//
//			int deviceID = 0;
//			int iH = 256;
//			int iW = 256;
//
//			/* Initialize a square structing element with a radius of 2 */
//			cvt::gpu::GpuErodeDilate gpuErode(deviceID, iW, iH, cvt::gpu::SQUARE, 2);
//
//
//			/* Initialize CUDA device */
//			gpuErode.initializeDevice();
//			TS_ASSERT_EQUALS(gpuErode.verifyInitialization(), true);
//
//			std::cout << read_tiler.getCvTileCount() << std::endl;
//
//			/* Loop through all the tiles in the image */
//			for(int i = 0; i < read_tiler.getCvTileCount(); ++i){
//
//				std::chrono::high_resolution_clock::time_point startT = std::chrono::high_resolution_clock::now();
//
//				/* Retrieve a tile, with 126 pixel edge buffer */
//				const cvt::cvTile<short> tile = read_tiler.getCvTile<short>(i, 0); //126 pixel buffer
//
//				/* Erode the tile */
//				gpuErode(tile, cvt::gpu::ERODE);
//
//				cvt::cvTile<short> * outTile;
//				gpuErode.copyTileFromDevice(&outTile);
//
//				write_tiler.putCvTile( *outTile  , i);
//
//				std::chrono::high_resolution_clock::time_point stopT = std::chrono::high_resolution_clock::now();
//				std::chrono::duration<double> time_spanT = std::chrono::duration_cast<std::chrono::duration<double> >(stopT - startT);
//
//				std::cout << "Tile [" << i << "]    : " << time_spanT.count() <<  std::endl;
//
//			}
//
//			std::chrono::high_resolution_clock::time_point stopI = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double> time_spanI = std::chrono::duration_cast<std::chrono::duration<double> >(stopI - startI);
//
//			double pixels_per_second = numPixels / time_spanI.count();
//			double seconds_per_pixel = time_spanI.count() / numPixels;
//
//			std::cout << "It took me    :" << time_spanI.count() <<  std::endl;
//			std::cout << "Pixels/Second :" << pixels_per_second <<  std::endl;
//			std::cout << "Seconds/Pixel :" << seconds_per_pixel <<  std::endl;
//
//			write_tiler.close();
//			read_tiler.close();
//
//			
//		}
//
//		void testAbsD(){
//
//			cvt::Tiler read_tiler;
//			cvt::Tiler write_tiler;
//
//			std::string sourceFile("test2-4.tif");
//			std::string outFile("test2-4-diffedWSelf.tif");
//
//			/* Ensure file opens correctly */
//			TS_ASSERT_EQUALS(cvt::NoError, read_tiler.open(sourceFile));
//
//			cv::Size2i rSize = read_tiler.getRasterSize();
//			unsigned long numPixels = rSize.area();
//
//			cv::Size2i tSize(256,256);
//
//			read_tiler.setCvTileSize(tSize);
//			write_tiler.setCvTileSize(tSize);
//
//			if(boost::filesystem::exists(outFile)){
//				boost::filesystem::remove(outFile);
//			}
//
//			/* Ensure tiler opens outfile correctly */
//			TS_ASSERT_EQUALS(cvt::NoError, write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), 1, cvt::Depth8U))
//
//			std::chrono::high_resolution_clock::time_point startI = std::chrono::high_resolution_clock::now();
//
//			int deviceID = 0;
//			int iH = 256;
//			int iW = 256;
//
//			/* Initialize a square structing element with a radius of 4 */
//			cvt::gpu::GpuErodeDilate gpuDilate(deviceID, iW, iH, cvt::gpu::SQUARE, 2);
//
//
//			/* Initialize CUDA device */
//			gpuDilate.initializeDevice();
//			TS_ASSERT_EQUALS(gpuDilate.verifyInitialization(), true);
//
//			std::cout << read_tiler.getCvTileCount() << std::endl;
//
//			/* Loop through all the tiles in the image */
//			for(int i = 0; i < read_tiler.getCvTileCount(); ++i){
//
//				std::chrono::high_resolution_clock::time_point startT = std::chrono::high_resolution_clock::now();
//
//				/* Retrieve a tile, with 126 pixel edge buffer */
//				const cvt::cvTile<short> tile = read_tiler.getCvTile<short>(i, 0); //126 pixel buffer
//
//				/* Dilate the tile */
//				gpuDilate(tile, tile, cvt::gpu::ABSDIFF);
//
//				cvt::cvTile<short> * outTile;
//				gpuDilate.copyTileFromDevice(&outTile);
//
//				write_tiler.putCvTile( *outTile  , i);
//
//				std::chrono::high_resolution_clock::time_point stopT = std::chrono::high_resolution_clock::now();
//				std::chrono::duration<double> time_spanT = std::chrono::duration_cast<std::chrono::duration<double> >(stopT - startT);
//
//				std::cout << "Tile [" << i << "]    : " << time_spanT.count() <<  std::endl;
//
//			}
//
//			std::chrono::high_resolution_clock::time_point stopI = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double> time_spanI = std::chrono::duration_cast<std::chrono::duration<double> >(stopI - startI);
//
//			double pixels_per_second = numPixels / time_spanI.count();
//			double seconds_per_pixel = time_spanI.count() / numPixels;
//
//			std::cout << "It took me    :" << time_spanI.count() <<  std::endl;
//			std::cout << "Pixels/Second :" << pixels_per_second <<  std::endl;
//			std::cout << "Seconds/Pixel :" << seconds_per_pixel <<  std::endl;
//
//			write_tiler.close();
//			read_tiler.close();
//
//			
//		}

};

#endif
