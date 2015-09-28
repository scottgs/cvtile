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

#ifndef _gpu_WINDOW_HISTOGRAM_SUITE_H_
#define _gpu_WINDOW_HISTOGRAM_SUITE_H_

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


#include "../src/gpu/drivers/GpuWHS.hpp"
#include "../src/gpu/drivers/GpuWindowFilterAlgorithm.hpp"


#define TIMING_ON 0

struct Stats {
	float mean;
	float entropy;
	float variance;
	float skewness;
	float kurtosis;
};


class gpuWHSTestSuite : public CxxTest::TestSuite
{
	public:

		void setUp()
		{;}

		void tearDown()
		{;}


		void calcStatisitics(Stats *stats, double* data,unsigned int histSize, unsigned int dataSize) {
			//std::cout << "TEST VALUES ARE: " << std::endl;
			/*for (int i = 0; i < dataSize; ++i) {
				std::cout << data[i] << " ";
			}
			std::cout << std::endl;*/
			
			double total = 0;
			short min = data[0];
			short max = data[0];

			/*Find total sum of data*/
			for (size_t i = 0; i < dataSize; ++i) {
				total += data[i];

				if (data[i]  > max){
					max = data[i];
				}

				if (data[i] < min) {
					min = data[i];
				}
			}
			

			short num_bins = 128;
			//std::cout << "num bins = " << num_bins << std::endl;
			//std::cout << "min = " << min << " max = " << max << std::endl;
			short bin_width = (max - min) / num_bins;
			if (bin_width <= 0) {
				bin_width = 1;
			}

			//std::cout << "bin width = " << bin_width << std::endl;
			short hist[num_bins];
			memset(hist,0,sizeof(short) * num_bins);

			float pdf[num_bins];
			
			short bin_idx = 0;
			/* Histogram Calculation */
			for (unsigned int j = 0; j < dataSize; ++j) {
				bin_idx = (short) ((data[j] - min) / bin_width);
				if (bin_idx >= 0 && bin_idx < num_bins) {
					hist[bin_idx]++;
				}
				else
					hist[127]++;
			}
			/*
			 * PDF calculation */
			for (short i = 0; i < num_bins;++i) {
				pdf[i] = (float) hist[i] / dataSize;
			}


			
			/*calculate mean*/
			double mean  = (double) total/dataSize;

			/*Varaince calculation*/
			double var = 0;
			for (unsigned int i = 0; i < dataSize; ++i) {
				const double res = data[i] - mean;
				var = var + (res * res); 
			}
			var = (double) var/(dataSize);
			/*calculate std */
			
			double std = sqrtf(var);
			double skewness = 0;

			double kurtosis = 0;

			/* Calculate Entropy */
			double entropy = 0;
			for (short i = 0; i < num_bins; ++i) {
				if (pdf[i] != 0) {
					entropy += (pdf[i] * log2(pdf[i]));
				}
			}
			if (std == 0 || var == 0) {
				stats->mean = (float) mean;
				stats->entropy = (float) (entropy * -1);
				stats->variance = (float) var;
				stats->skewness = (float) skewness;
				stats->kurtosis = (float) kurtosis;
				return;
			}
			//std::cout << "std = " << std << std ::endl;

			/*Calculate Skewness*/
			for (unsigned int i = 0; i < dataSize; ++i) {
				const double tmp = (data[i] - mean);
				skewness = skewness + (tmp * tmp * tmp);
			}

			skewness = (double) skewness/(dataSize * var * std);


			/*Calculate kurtosis*/

			for (unsigned int i = 0; i < dataSize; ++i) {
				const double tmp = (data[i] - mean);
				kurtosis = kurtosis + ( tmp * tmp * tmp * tmp );
			}

			kurtosis = (double) kurtosis/(dataSize * var * var);



			//kurtosis -= 3;


			stats->mean = (float) mean;
			stats->entropy = (float) (entropy * -1);
			stats->variance = (float) var;
			stats->skewness = (float) skewness;
			stats->kurtosis = (float) kurtosis;
		
		}

		void testWindowHistogramSingleBandImage () {
			std::cout << std::endl << "GPU WHS VERIFICATION TEST" << std::endl;
			int cuda_device_id = 0;
			//unsigned int window_size = 1;



			cvt::Tiler read_tiler;

			cv::Size2i tSize(256,256);
			read_tiler.setCvTileSize(tSize);

			read_tiler.open("test1-1.tif");

			cvt::cvTile<short> inputTile;
			//cvt::cvTile<float> *outputTile;
			

			/* Loop through all the tiles in the image */
			for (int window = 0; window <= 11; window++) {	
					inputTile = read_tiler.getCvTile<short>(4, window);
					cvt::gpu::GpuWHS<short,1,float,5> whs(cuda_device_id,
					inputTile.getROI().width,inputTile.getROI().height,window);
					whs.initializeDevice(cvt::gpu::SQUARE);

					cvt::cvTile<float> *outputTile = NULL;
					whs(inputTile,(const cvt::cvTile<float> **)&outputTile);
					if (!outputTile) {
						std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
						std::cout << "HERE" <<std::endl;
						exit(1);
					}
					TS_ASSERT_EQUALS(outputTile->getBandCount(),5);	
					/*Calculate Window Histogram Statistics for each pixel*/
					cv::Size2i dims = inputTile.getSize();
					cv::Rect roiDims = inputTile.getROI();
					
					const int imageArea = dims.width * dims.height;
					const int outArea = roiDims.width * roiDims.height;
					std::vector<Stats> stats;
					stats.resize(outArea);
					int r = 0;

					for (r = 0; r < outArea; ++r){
						std::vector<double> data;

						for (int x = 0 - window; x <= window; ++x) {
							for (int y = 0 - window; y <= window; ++y) {

								const int X = (r/roiDims.width) + x + window;
								const int Y = (r%roiDims.height) + y + window;
								data.push_back(inputTile[0].at<short>(X,Y));
							}

						}
						
						calcStatisitics(&stats[r], data.data(),32, data.size());
					}
					//return;	
					for (size_t s = 0; s < stats.size(); ++s) {
						const size_t row = s / roiDims.width;
						const size_t col = s % roiDims.height;
						TS_ASSERT_DELTA(stats[s].entropy,(*outputTile)[0].at<float>(row,col),1e-5);
						TS_ASSERT_DELTA(stats[s].mean,(*outputTile)[1].at<float>(row,col),1e-5);
						TS_ASSERT_DELTA(stats[s].variance,(*outputTile)[2].at<float>(row,col),1e-5);
						TS_ASSERT_DELTA(stats[s].skewness,(*outputTile)[3].at<float>(row,col),1e-5);
						TS_ASSERT_DELTA(stats[s].kurtosis,(*outputTile)[4].at<float>(row,col),1e-5);

					}
					delete outputTile;
					

			}
			read_tiler.close();

		
		}
};

#endif
