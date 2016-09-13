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

#ifndef _gpu_ERODE_TEST_SUITE_H_
#define _gpu_ERODE_TEST_SUITE_H_

#include <boost/filesystem.hpp>
#include <cxxtest/TestSuite.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <numeric>
#include <cstring>
#include <cstdio>
#include "../src/base/cvTile.hpp"
#include "../src/base/Tiler.hpp"
#include "../src/gpu/drivers/GpuErode.hpp"
#include "../src/gpu/drivers/GpuWindowFilterAlgorithm.hpp"

#ifdef HAVE_CGI
	#include "../src/base/cvTileConversion.hpp"
#endif

#define SHOW_OUTPUT 0
// Set device # in src/gpu/drivers/GPUProperties.hpp
// #define CUDA_DEVICE 0

class gpuErodeTestSuite : public CxxTest::TestSuite
{
	public:

		void setUp()
		{;}

		void tearDown()
		{;}

		short erode_window_data (std::vector<short>& data) {
			short min = data[0];
			for (size_t i = 0; i < data.size(); ++i) {
				if (min > data[i]) {
					min = data[i];
				}
			}
			return min;
		}

		unsigned short erode_window_data (std::vector<unsigned short>& data) {
			short min = data[0];
			for (size_t i = 0; i < data.size(); ++i) {
				if (min > data[i]) {
					min = data[i];
				}
			}
			return min;
		}

		void testErodeFullPixelVerification () {
			std::cout << std::endl << "GPU ERODE VERIFICATION TEST" << std::endl;
			cvt::Tiler read_tiler;

			cv::Size2i tSize(256,256);
			read_tiler.setCvTileSize(tSize);

			read_tiler.open("test1-1.tif");

			cvt::cvTile<short> inputTile;

			/* Loop through all the tiles in the image */
			for (int window = 1; window <= 11; window++) {
					inputTile = read_tiler.getCvTile<short>(4, window);
					cvt::gpu::GpuErode<short,1,short,1> erode(CUDA_DEVICE,
					inputTile.getROI().width,inputTile.getROI().height,window);
					erode.initializeDevice(cvt::gpu::SQUARE);

					cvt::cvTile<short> *outputTile = NULL;
					erode(inputTile,(const cvt::cvTile<short> **)&outputTile);
					if (!outputTile) {
						std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
						std::cout << "HERE" <<std::endl;
						exit(1);
					}
					TS_ASSERT_EQUALS(outputTile->getBandCount(),1);
					/*Calculate Window Histogram Statistics for each pixel*/
					//TODO remove dims if it is not used.
					//cv::Size2i dims = inputTile.getSize();
					cv::Rect roiDims = inputTile.getROI();

					const int outArea = roiDims.width * roiDims.height;
					std::vector<short> results;
					results.resize(outArea);
					int r = 0;

					for (r = 0; r < outArea; ++r){
						std::vector<short> data;

						for (int x = 0 - window; x <= window; ++x) {
							for (int y = 0 - window; y <= window; ++y) {

								const int X = (r/roiDims.width) + x + window;
								const int Y = (r%roiDims.height) + y + window;
								data.push_back(inputTile[0].at<short>(X,Y));
							}

						}
						results[r] = erode_window_data(data);

					}

					for (size_t s = 0; s < results.size(); ++s) {
						const size_t row = s / roiDims.height;
						const size_t col = s % roiDims.width;

						TS_ASSERT_EQUALS(results[s],(*outputTile)[0].at<short>(row,col));
					}
					//return;
					delete outputTile;


			}
			read_tiler.close();

		}

		void testUnsignedErodeFullPixelVerification () {
			std::cout << std::endl << "GPU ERODE UNSIGNED VERIFICATION TEST" << std::endl;
			cvt::Tiler read_tiler;

			cv::Size2i tSize(256,256);
			read_tiler.setCvTileSize(tSize);

			read_tiler.open("test1-1.tif");

			cvt::cvTile<unsigned char> inputTile;

			/* Loop through all the tiles in the image */
			for (int window = 0; window <= 11; window++) {
					inputTile = read_tiler.getCvTile<unsigned char>(4, window);
					cvt::gpu::GpuErode<unsigned char,1,unsigned char,1> erode(CUDA_DEVICE,
					inputTile.getROI().width,inputTile.getROI().height,window);
					erode.initializeDevice(cvt::gpu::SQUARE);

					cvt::cvTile<unsigned char> *outputTile = NULL;
					erode(inputTile,(const cvt::cvTile<unsigned char> **)&outputTile);
					if (!outputTile) {
						std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
						std::cout << "HERE" <<std::endl;
						exit(1);
					}
					TS_ASSERT_EQUALS(outputTile->getBandCount(),1);
					/*Calculate Window Histogram Statistics for each pixel*/

					//TODO remove dims if it is not used.
					//cv::Size2i dims = inputTile.getSize();
					cv::Rect roiDims = inputTile.getROI();

					const int outArea = roiDims.width * roiDims.height;
					std::vector<unsigned char> results;
					results.resize(outArea);
					int r = 0;

					for (r = 0; r < outArea; ++r){
						std::vector<short> data;

						for (int x = 0 - window; x <= window; ++x) {
							for (int y = 0 - window; y <= window; ++y) {

								const int X = (r/roiDims.width) + x + window;
								const int Y = (r%roiDims.height) + y + window;
								data.push_back(inputTile[0].at<unsigned char>(X,Y));
							}

						}
						results[r] = erode_window_data(data);

					}

					for (size_t s = 0; s < results.size(); ++s) {
						const size_t row = s / roiDims.height;
						const size_t col = s % roiDims.width;

						TS_ASSERT_EQUALS(results[s],(*outputTile)[0].at<unsigned char>(row,col));
					}
					//return;
					delete outputTile;


			}
			read_tiler.close();

		}

};

#endif
