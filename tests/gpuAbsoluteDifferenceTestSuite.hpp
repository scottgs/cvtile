#ifndef _gpu_ABSOLUTE_DIFF_TEST_SUITE_H_
#define _gpu_ABSOLUTE_DIFF_TEST_SUITE_H_

#include <cxxtest/TestSuite.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <chrono>


#ifdef HAVE_CGI
	#include "../src/base/cvTileConversion.hpp"
#endif

#include <numeric>
#include "../src/base/cvTile.hpp"
#include "../src/base/Tiler.hpp"
#include <boost/filesystem.hpp>


#include "../src/gpu/drivers/GpuAbsoluteDifference.hpp"

#define SHOW_OUTPUT 0

class gpuAbsoluteDifferenceTestSuite : public CxxTest::TestSuite
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

		void testAbsDiffFullPixelVerification () {
			std::cout << std::endl << "GPU ABS DIFF VERIFICATION TEST" << std::endl;
			int cuda_device_id = 0;
			cvt::Tiler read_tiler;
			cvt::Tiler read_tiler2;
			cv::Size2i tSize(256,256);
			read_tiler.setCvTileSize(tSize);
			read_tiler2.setCvTileSize(tSize);
			read_tiler.open("test1-1.tif");
			read_tiler2.open("test1.tif");

			cvt::cvTile<short> inputTile;
			cvt::cvTile<short> inputTile2;

			inputTile = read_tiler.getCvTile<short>(4);
			inputTile2 = read_tiler2.getCvTile<short>(4);

			cvt::gpu::GpuAbsoluteDifference<short,1,short,1> absDiff(cuda_device_id,
				tSize.width,tSize.height);
			absDiff.initializeDevice();

			cvt::cvTile<short> *outputTile;
			absDiff(inputTile,inputTile2,(const cvt::cvTile<short> **)&outputTile);
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
				const size_t row = i / tSize.height;
				const size_t col = i % tSize.width;
				results[i] = abs((inputTile[0].at<short>(row,col) - inputTile2[0].at<short>(row,col)));	
			}
					
			for (size_t s = 0; s < results.size(); ++s) {
					const size_t row = s / tSize.height;
					const size_t col = s % tSize.width;
	
					TS_ASSERT_EQUALS(results[s],(*outputTile)[0].at<short>(row,col));
			}
			delete outputTile;
				
			read_tiler.close();
			read_tiler2.close();
		
		}

};

#endif
