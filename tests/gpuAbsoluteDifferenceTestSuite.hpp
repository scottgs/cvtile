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

		void testAbsSimplePixelVerification() {	
			std::cout << std::endl << "GPU ABS DIFF VERIFICATION TEST" << std::endl;
			int cuda_device_id = 0;
			cvt::Tiler read_tiler;

			cv::Size2i tSize(256,256);
			read_tiler.setCvTileSize(tSize);
			read_tiler.open("test1.tif");
			cvt::cvTile<unsigned short> inputTile;
			inputTile = read_tiler.getCvTile<unsigned short>(4);

			std::vector<unsigned short> data(tSize.area(),0);
			cvt::cvTile<unsigned short> zeroedTile(data.data(),tSize,1);

			cvt::gpu::GpuAbsoluteDifference<unsigned short,1,unsigned short,1> absDiff(cuda_device_id,
				tSize.width,tSize.height);
			absDiff.initializeDevice();

			cvt::cvTile<unsigned short> *outputTile;
			absDiff(inputTile,zeroedTile,(const cvt::cvTile<unsigned short> **)&outputTile);
			if (!outputTile) {
				std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
				return;
			}
			TS_ASSERT_EQUALS(outputTile->getBandCount(),1);	
			for (int i = 0; i < tSize.area(); ++i) {
				const size_t row = i / tSize.height;
				const size_t col = i % tSize.width;
				TS_ASSERT_EQUALS(inputTile[0].at<unsigned short>(row,col),(*outputTile)[0].at<unsigned short>(row,col));
			}

			read_tiler.close();
		} 

		void testAbsDiffFullPixelVerification () {
			std::cout << std::endl << "GPU ABS DIFF VERIFICATION TEST" << std::endl;
			int cuda_device_id = 0;
			cvt::Tiler read_tiler;
			cvt::Tiler read_tiler2;
			cv::Size2i tSize(256,256);
			read_tiler.setCvTileSize(tSize);
			read_tiler2.setCvTileSize(tSize);
			read_tiler.open("test1.tif");
			read_tiler2.open("test1-1.tif");

			cvt::cvTile<unsigned short> inputTile;
			cvt::cvTile<unsigned short> inputTile2;

			inputTile = read_tiler.getCvTile<unsigned short>(4);
			inputTile2 = read_tiler2.getCvTile<unsigned short>(4);

			cvt::gpu::GpuAbsoluteDifference<unsigned short,1,unsigned short,1> absDiff(cuda_device_id,
				tSize.width,tSize.height);
			absDiff.initializeDevice();

			cvt::cvTile<unsigned short> *outputTile;
			absDiff(inputTile,inputTile2,(const cvt::cvTile<unsigned short> **)&outputTile);
			if (!outputTile) {
				std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
				return;
			}
			TS_ASSERT_EQUALS(outputTile->getBandCount(),1);	
			/*Calculate Window Histogram Statistics for each pixel*/
			cv::Size2i dims = inputTile.getSize();
			const int area = dims.width * dims.height;
			std::vector<int> results;
			results.resize(area);

			for (int i = 0; i < area; ++i) {
				const size_t row = i / tSize.height;
				const size_t col = i % tSize.width;
				const short diff = abs(inputTile[0].at<unsigned short>(row,col) - inputTile2[0].at<unsigned short>(row,col));
				results[i] = diff > 0 ? diff : -diff;
			}
			for (size_t s = 0; s < results.size(); ++s) {
					const size_t row = s / tSize.height;
					const size_t col = s % tSize.width;
	
					TS_ASSERT_EQUALS((unsigned short)results[s],(*outputTile)[0].at<unsigned short>(row,col));
			}
			delete outputTile;
				
			read_tiler.close();
			read_tiler2.close();
		
		}

};

#endif
