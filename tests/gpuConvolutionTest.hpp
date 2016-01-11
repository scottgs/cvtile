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

#ifndef H_GPU_CONVOLUTION_TEST_SUITE_H
#define H_GPU_CONVOLUTION_TEST_SUITE_H

#include <cxxtest/TestSuite.h>

#include "../src/base/cvTile.hpp"
#include "../src/base/Tiler.hpp"

#include "../src/gpu/drivers/GpuConvolution.hpp"

class gpuConvolutionTestSuite : public CxxTest::TestSuite
{
	public:

		void setUp()
		{;}

		void tearDown()
		{;}

		//TO-DO test for all types once instantiated
		void testCovolutionFullPixelOneBand() {
			ssize_t filterRadius = 1;
			cv::Size2i roi(5,5);
			cv::Size2i dSize(roi.width + filterRadius * 2,roi.height + filterRadius * 2);
			vector<short> data;
			data.resize(dSize.area());

			for(unsigned int i = 0; i < dSize.area(); ++i) {
				data[i] = i;
			}

			cvt::cvTile<short> inTile(data.data(), dSize, 1);
			cvt::cvTile<short>* outTile;

			cv::Mat weightsMat = cv::Mat::zeros(3,3,CV_16UC1);
			for(int i = 0; i < 3; ++i) {
				for(int j = 0; j < 3; ++j) {
					weightsMat.at<short>(i,j) = 2;
				}
			}
			
			
			cvt::gpu::GpuConvolution<short,1,short,1,short> conv(0, roi.width, roi.height,
									    filterRadius, weightsMat);

			TS_ASSERT_EQUALS(cvt::Ok, conv.initializeDevice(cvt::gpu::SQUARE));
			
			conv(inTile, (const cvt::cvTile<short>**)&outTile);
			TS_ASSERT_EQUALS(0, (outTile == NULL));

			cv::Mat& a = inTile[0];
			cv::Mat& b = (*outTile)[0];

			for(int i = 0; i < dSize.width; ++i) {
				for(int j = 0; j < dSize.height; ++j) {
					std::cout << "b[" << i << "," << j << "] = " << b.at<short>(i,j) << std::endl;
				}
			}
		}
		
};

#endif
