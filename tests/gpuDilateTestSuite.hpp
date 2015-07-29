#ifndef _gpu_ERODE_TEST_SUITE_H_
#define _gpu_ERODE_TEST_SUITE_H_

#include <cxxtest/TestSuite.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <chrono>


#ifdef HAVE_CGI
	#include "../src/base/cvTileConversion.hpp"
	#include <ext/numeric>
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
