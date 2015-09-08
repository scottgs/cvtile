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

#include <cxxtest/TestSuite.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <chrono>
#include <numeric>


#ifdef HAVE_CGI
	#include "../src/base/cvTileConversion.hpp"
#endif

#include "../src/base/cvTile.hpp"
#include "../src/base/Tiler.hpp"
#include <boost/filesystem.hpp>


#include "../src/gpu/drivers/GpuDilate.hpp"
#include "../src/gpu/drivers/GpuWindowFilterAlgorithm.hpp"


#define SHOW_OUTPUT 0

class gpuDilateTestSuite : public CxxTest::TestSuite
{
	public:

		void setUp()
		{;}

		void tearDown()
		{;}

		short dilate_window_data (std::vector<short>& data) {
			short max = data[0]; 
			for (size_t i = 0; i < data.size(); ++i) {
				if (max < data[i]) {
					max = data[i];
				}
			}
			return max;
		}

		void testDilateFullPixelVerification () {
			std::cout << std::endl << "GPU DILATE VERIFICATION TEST" << std::endl;
			int cuda_device_id = 0;
			cvt::Tiler read_tiler;

			cv::Size2i tSize(256,256);
			read_tiler.setCvTileSize(tSize);

			read_tiler.open("test1-1.tif");

			cvt::cvTile<short> inputTile;

			inputTile = read_tiler.getCvTile<short>(4);

			/* Loop through all the tiles in the image */
			for (int window = 1; window <= 11; window++) {	

					cvt::gpu::GpuDilate<short,1,short,1> dilate(cuda_device_id,
					tSize.width,tSize.height,window);
					dilate.initializeDevice(cvt::gpu::SQUARE);

					cvt::cvTile<short> *outputTile;
					dilate(inputTile,(const cvt::cvTile<short> **)&outputTile);
					if (!outputTile) {
						std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
						return;
					}
					TS_ASSERT_EQUALS(outputTile->getBandCount(),1);	
					/*Calculate Window Histogram Statistics for each pixel*/
					cv::Size2i dims = inputTile.getSize();
					const int area = dims.width * dims.height;
					std::vector<short> results;
					results.resize(area);

					for (int i = 0; i < area; ++i) {
						std::vector<short> data;
						for (int x = 0 - window; x <= window; ++x) {
							for (int y = 0 - window; y <= window; ++y) {
								const int X = (i / 256) + x;
								const int Y = (i % 256) + y;
								if (X >= 0 && X < dims.width && Y >= 0 && Y < dims.height) {	
									data.push_back(inputTile[0].at<short>(X,Y));
								}
							}
						}
						results[i] = dilate_window_data(data);

					}
					
					for (size_t s = 0; s < results.size(); ++s) {
						const size_t row = s / 256;
						const size_t col = s % 256;
	
						TS_ASSERT_DELTA(results[s],(*outputTile)[0].at<short>(row,col),1e-5);
					}
					delete outputTile;
					

			}
			read_tiler.close();
		
		}

};


#endif
