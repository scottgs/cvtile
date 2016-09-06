#ifndef WINDOW_HISTOGRAM_STATISTICS_
#define WINDOW_HISTOGRAM_STATISTICS_

#include <vector>
#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include <limits>
#include <map>
#include <set>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <utility>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cvtile/cvtile.hpp> // required for tiles

//namespace cvt
//{

//namespace algorithms {

struct PixelStats{
	float mean;
	float entropy;
	float variance;
	float skewness;
	float kurtosis;
};


class WindowHistogramStatistics {

	public:
		WindowHistogramStatistics(ssize_t buffer_radius);
		~WindowHistogramStatistics();
		bool operator () (cvt::cvTile<short>& inputTile, cvt::cvTile<float>& outputTile);
		
	
	protected:
	
		bool calculateWindowStatsPerPixel(PixelStats *stats, short* data,unsigned int dataSize);

		ssize_t _buffer_radius;

};


WindowHistogramStatistics::WindowHistogramStatistics(ssize_t buffer_radius) : _buffer_radius(buffer_radius)
{
	;
}


WindowHistogramStatistics::~WindowHistogramStatistics() {
	;
}


bool WindowHistogramStatistics::operator()(cvt::cvTile<short>& inputTile, cvt::cvTile<float>& outputTile) {
				
					if (inputTile.getBandCount() != 1) {
						//throw std::runtime_error("BAND COUNTS FOR INPUTS ARE WRONG");
						return false;
					}

					cv::Rect roiDims = inputTile.getROI();
					
					const int outArea = roiDims.width * roiDims.height;
					std::vector<PixelStats> pixel_stats;
					pixel_stats.resize(outArea);
					// calculate a single pixel window results
					// store into a vector of PixelStats
					for (int r = 0; r < outArea; ++r) {
						std::vector<short> window_data;

						for (ssize_t x = 0 - _buffer_radius; x <= _buffer_radius; ++x) {
							for (ssize_t y = 0 - _buffer_radius; y <= _buffer_radius; ++y) {

								const int X = (r/roiDims.width) + x + _buffer_radius;
								const int Y = (r%roiDims.height) + y + _buffer_radius;
								window_data.push_back(inputTile[0].at<short>(Y,X));
							}

						}
						
						calculateWindowStatsPerPixel(&pixel_stats[r], window_data.data(),window_data.size());
					}
					// Copy results into correct bands	
					for (size_t s = 0; s < pixel_stats.size(); ++s) {
						const int row = s / roiDims.width;
						const int col = s % roiDims.height;
						outputTile[0].at<float>(row,col) =  pixel_stats[s].entropy;
						outputTile[1].at<float>(row,col) = pixel_stats[s].mean;
						outputTile[2].at<float>(row,col) = pixel_stats[s].variance;
						outputTile[3].at<float>(row,col) = pixel_stats[s].skewness;
						outputTile[4].at<float>(row,col) = pixel_stats[s].kurtosis;
					}
					return true;
		
}

bool WindowHistogramStatistics::calculateWindowStatsPerPixel(PixelStats *stats, short* data, unsigned int dataSize) {
			
			size_t total = 0;
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
				return true;
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
			return true;
}

//} // END of ALGORITHMS namespace
//} // END OF CVT namespace

#endif
