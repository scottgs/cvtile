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

#ifndef CVTILE_ALGORITHM_DMP_H_
#define CVTILE_ALGORITHM_DMP_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <limits>
#include <map>
#include <set>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <utility>


namespace cvt {

namespace algorithms {

///
/// Generate a opening DMP profile
/// \param inputImage The image that will be filtered for light objects
/// \param ses The set of strictly increasing geodesic structuring elements
///
template <typename OperatingPixelType>
std::vector<cv::Mat> opencvOpenDmp(const cv::Mat& inputImage, const std::vector<int>& ses);

///
/// Dilation by Reconstruction using the Downhill Filter algorithm (Robinson,Whelan)
///
/// \param mask The image/surface that bounds the reconstruction of the marker
/// \param mark The image/surface that initializes the reconstruction
/// \returns The dilation reconstruction
template <typename OperatingPixelType>
cv::Mat dilationByReconstructionDownhill(const cv::Mat& mask, const cv::Mat& mark);


///
/// Generate a closing DMP profile
/// \param inputImage The image that will be filtered for dark objects
/// \param ses The set of strictly increasing geodesic structuring elements
///
template <typename OperatingPixelType>
std::vector<cv::Mat> opencvCloseDmp(const cv::Mat& inputImage, const std::vector<int>& ses);

///
/// Erosion by Reconstruction using an Uphill Filter algorithm (insp. by Robinson,Whelan)
///
/// \param mask The image/surface that bounds the reconstruction of the marker
/// \param mark The image/surface that initializes the reconstruction
/// \returns The erosion reconstruction
template <typename OperatingPixelType>
cv::Mat erosionByReconstructionUphill(const cv::Mat& mask, const cv::Mat& mark);

}; // END cvt::algorithms namespace
}; // END cvt namespace


// ============================
// ============================
// == BEGIN Template Functions
// ============================
// ============================

template <typename OperatingPixelType>
std::vector<cv::Mat> cvt::algorithms::opencvOpenDmp(const cv::Mat& inputImage, const std::vector<int>& ses)
{
	// ++ Open, then dilate by reconstruction

	// Build Structuring Elements
	std::vector<cv::Mat> structuringElements;
	for (std::vector<int>::const_iterator level = ses.begin();level!=ses.end();++level)
	{
		const int lev = *level;
		// Circular structuring element
		const int edgeSize = lev+1+lev; // turn radius into edge size
		structuringElements.push_back(
					cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(edgeSize,edgeSize))
					);
	}

	// Build Morphological Operation Stack (open)
	std::vector<cv::Mat> mos;
	for (size_t level = 0;level<structuringElements.size();++level)
	{
		cv::Mat filteredImage(inputImage.size().height, inputImage.size().width, cv::DataType<OperatingPixelType>::type);
		cv::morphologyEx(inputImage, filteredImage, cv::MORPH_OPEN, structuringElements.at(level));
		mos.push_back(filteredImage);
	}

	const int n_lvls = mos.size();

	// Set the Tile band count to the SEs array size
	std::vector<cv::Mat> profile;
	profile.reserve(n_lvls);

	// ////////////////////////////////////////////////////////////////////////
	// USE: dilationByReconstructionDownhill(const cv::Mat& mask,const cv::Mat& mark)


	cv::Mat tempRec;			// Temporary Open Record
	cv::Mat lastRec;			// previous level Open Record

	tempRec = cvt::algorithms::dilationByReconstructionDownhill<OperatingPixelType>( inputImage, mos.at(0) ).clone();

	// get the absolute difference of the two images
	cv::Mat diffImage(inputImage.size().height, inputImage.size().width, cv::DataType<OperatingPixelType>::type);
	absdiff(inputImage, tempRec, diffImage);

	profile.push_back(diffImage.clone());
	tempRec.copyTo(lastRec);

	// Subsequent Levels
	for (int i = 1;i < n_lvls;++i)
	{
		//  reconstruction(mask     ,  mark)
		tempRec = cvt::algorithms::dilationByReconstructionDownhill<OperatingPixelType>( inputImage, mos.at(i) ).clone();

		// get the absolute difference of the two images
		absdiff(lastRec, tempRec, diffImage);
		profile.push_back(diffImage.clone());

		tempRec.copyTo(lastRec);

	}


	return profile;
}

template <typename OperatingPixelType>
cv::Mat cvt::algorithms::dilationByReconstructionDownhill(const cv::Mat& mask, const cv::Mat& mark)
{
	//typedef unsigned short OperatingPixelType;

	if (false == std::numeric_limits<OperatingPixelType>::is_integer)
		throw std::logic_error("Invalid template / operating-pixel type");

	const cv::Size size = mask.size();

	if (size != mark.size())
	{
		std::stringstream errmsg;
		errmsg << "ERROR::dilationByReconstructionDownhill "
		<< "Size mismatch: ("
		<< size.width << "," << size.height << ") != ("
		<< mark.size().width << "," << mark.size().height << ")";
		throw std::runtime_error(errmsg.str());
	}

	// --------------------------------
	// Place each p : I(p) > 0 into List[I(p)]
	// --------------------------------
	// Copy the marker into the reconstruction
	//	we can then grow it in place
	// value i has a list of pixels at position (y,x)
	cv::Mat reconstruction(mark.size().height, mark.size().width, cv::DataType<OperatingPixelType>::type);
	mark.copyTo(reconstruction);

	// for template, use typename
	//	typedef typename std::list<std::pair<int,int> > LocationListType;
	//	typedef typename std::map< OperatingPixelType, LocationListType > MapOfLocationListType;
	typedef std::list<std::pair<int,int> > LocationListType;
	typedef std::map< OperatingPixelType, LocationListType > MapOfLocationListType;

	// This is essentially an indexed image version of the input `mark' raster
	MapOfLocationListType valueLocationListMap;

	// Build an in-memory indexed image
	// but while we are at it, verify that it is properly bounded by `mask' raster
	for (int i = 0; i < size.height; ++i)
		for (int j = 0; j < size.width; ++j)
		{
			const OperatingPixelType pixelValue =  reconstruction.at<OperatingPixelType>(i, j);
			if (pixelValue > mask.at<OperatingPixelType>(i, j))
			{
				//std::cout << "Storing ("<< i << "," << j <<") into valueLocationListMap["<< static_cast<int>(pixelValue) << "]" << std::endl;
				std::stringstream msg;
				msg << "(pixelValue > mask[0](i, j)), for raster location = ("<<i<<","<<j<<")";
				throw std::runtime_error(msg.str()); // the marker must always be LTE the mask, by definition
			}

			valueLocationListMap[pixelValue].push_back(std::pair<int,int>(i,j) );
		}

	// No valid pixels or values above floor?  Return the input as output
	if (valueLocationListMap.size() == 0)
		return reconstruction;

	// --------------------------------
	// The farthest downhill we will go
	//	is to the current min(mark[0])
	//	therefore, we do not need to
	//	process the current min, we
	//	already copied them to output
	const OperatingPixelType minValue =  valueLocationListMap.begin()->first;
	valueLocationListMap[minValue].clear();

	std::set<std::pair<int,int> > finalizedPixels;

	// --------------------------------
	// Do a backward iteration (downhill)
	// through the
	//	valueLocationListMap[values]
	// --------------------------------
	// use typename for template version
	//for (typename MapOfLocationListType::reverse_iterator valueList = valueLocationListMap.rbegin();
	//			valueList != valueLocationListMap.rend(); ++valueList)
	for (typename MapOfLocationListType::reverse_iterator valueList = valueLocationListMap.rbegin();
					valueList != valueLocationListMap.rend(); ++valueList)
	{
		// The gray value
		const OperatingPixelType currentValue = valueList->first;
		// The list of (y,x) tuples
		LocationListType locations = valueList->second;

		// for each location indexed in the value list
		while (!locations.empty())
		{

			// pull/pop the first position from the list, mark it as finalized
			std::pair<int,int> frontPosition = locations.front();
			finalizedPixels.insert(frontPosition);
			locations.pop_front();

			const int y = frontPosition.first;
			const int x = frontPosition.second;

			const int pre_x = x - 1;
			const int post_x = x + 1;
			const int pre_y = y - 1;
			const int post_y = y + 1;

			// For each neighbor
			// - a -
			// b p d
			// - c -
			// a = (pre_y,x), b = (y,pre_x), c = (post_y,x), d = (y,post_x)

			// Neighbor Pixel 'a'
			const std::pair<int,int> a = std::pair<int,int>(pre_y,x);
			const OperatingPixelType mask_a = mask.at<OperatingPixelType>(pre_y,x);
			// if neighbor index is within bounds and not finalized
			if ((finalizedPixels.find(a) == finalizedPixels.end()) && (pre_y >= 0) && (mask_a > 0))
			{
				OperatingPixelType neighborValue = reconstruction.at<OperatingPixelType>(pre_y,x);
				OperatingPixelType constraintValue = std::min<OperatingPixelType>(currentValue,mask_a);

				if (neighborValue < constraintValue)
				{
					reconstruction.at<OperatingPixelType>(pre_y,x) = constraintValue;
					if ( constraintValue == currentValue )
					{
						locations.push_back( a );
					}
					else
					{
						valueLocationListMap[neighborValue].remove( a );
						valueLocationListMap[constraintValue].push_back( a );
					}
				}
			}

			// Neighbor Pixel 'b'
			const std::pair<int,int> b = std::pair<int,int>(y,pre_x);
			const OperatingPixelType mask_b = mask.at<OperatingPixelType>(y,pre_x);
			if ((finalizedPixels.find(b) == finalizedPixels.end()) && (pre_x >= 0) && (mask_b > 0))
			{
				OperatingPixelType neighborValue = reconstruction.at<OperatingPixelType>(y,pre_x);
				OperatingPixelType constraintValue = std::min<OperatingPixelType>(currentValue,mask_b);

				if (neighborValue < constraintValue)
				{
					reconstruction.at<OperatingPixelType>(y,pre_x) = constraintValue;
					if ( constraintValue == currentValue )
					{
						locations.push_back( b );
					}
					else
					{
						valueLocationListMap[neighborValue].remove( b );
						valueLocationListMap[constraintValue].push_back( b );
					}
				}
			}

			// Neighbor Pixel 'c'
			const std::pair<int,int> c = std::pair<int,int>(post_y,x);
			const OperatingPixelType mask_c = mask.at<OperatingPixelType>(post_y,x);
			if ((finalizedPixels.find(c) == finalizedPixels.end()) && (post_y < size.height) && (mask_c > 0))
			{
				OperatingPixelType neighborValue = reconstruction.at<OperatingPixelType>(post_y,x);
				OperatingPixelType constraintValue = std::min<OperatingPixelType>(currentValue,mask_c);

				if (neighborValue < constraintValue)
				{
					reconstruction.at<OperatingPixelType>(post_y,x) = constraintValue;
					if ( constraintValue == currentValue )
					{
						locations.push_back( c );
					}
					else
					{
						valueLocationListMap[neighborValue].remove( c );
						valueLocationListMap[constraintValue].push_back( c );
					}
				}
			}

			// Neighbor Pixel 'd'
			const std::pair<int,int> d = std::pair<int,int>(y,post_x);
			const OperatingPixelType mask_d = mask.at<OperatingPixelType>(y,post_x);
			if ((finalizedPixels.find(d) == finalizedPixels.end()) && (post_x < size.width) && (mask_d > 0))
			{
				OperatingPixelType neighborValue = reconstruction.at<OperatingPixelType>(y,post_x);
				OperatingPixelType constraintValue = std::min<OperatingPixelType>(currentValue,mask_d);

				if (neighborValue < constraintValue)
				{
					reconstruction.at<OperatingPixelType>(y,post_x) = constraintValue;

					if ( constraintValue == currentValue )
					{
						locations.push_back( d );
					}
					else
					{
						valueLocationListMap[neighborValue].remove( d );
						valueLocationListMap[constraintValue].push_back( d );
					}
				}
			}



		} // end, a value's position list is emptied.

	} // end for each value in the marker



	return reconstruction;
};


template <typename OperatingPixelType>
std::vector<cv::Mat> cvt::algorithms::opencvCloseDmp(const cv::Mat& inputImage, const std::vector<int>& ses)
{
	// ++ Close, then erode by reconstruction


	// Build Structuring Elements
	std::vector<cv::Mat> structuringElements;
	for (std::vector<int>::const_iterator level = ses.begin();level!=ses.end();++level)
	{
		const int lev = *level;
		// Circular structuring element
		const int edgeSize = lev+1+lev; // turn radius into edge size
		structuringElements.push_back(
					cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(edgeSize,edgeSize))
					);
	}

	// Build Morphological Operation Stack (close)
	std::vector<cv::Mat> mos;
	for (size_t level = 0;level<structuringElements.size();++level)
	{
		cv::Mat filteredImage(inputImage.size().height, inputImage.size().width, cv::DataType<OperatingPixelType>::type);
		cv::morphologyEx(inputImage, filteredImage, cv::MORPH_CLOSE, structuringElements.at(level));
		mos.push_back(filteredImage);
	}

	const int n_lvls = mos.size();

	// Set the Tile band count to the SEs array size
	std::vector<cv::Mat> profile;
	profile.reserve(n_lvls);

	// ////////////////////////////////////////////////////////////////////////
	// USE: erosionByReconstructionUphill(const cv::Mat& mask,const cv::Mat& mark)


	cv::Mat tempRec;			// Temporary Close Record
	cv::Mat lastRec;			// previous level Close Record

	tempRec = cvt::algorithms::erosionByReconstructionUphill<OperatingPixelType>( inputImage, mos.at(0) ).clone();

	// get the absolute difference of the two images
	cv::Mat diffImage(inputImage.size().height, inputImage.size().width, cv::DataType<OperatingPixelType>::type);
	absdiff(inputImage, tempRec, diffImage);

	profile.push_back(diffImage.clone());
	tempRec.copyTo(lastRec);

	// Subsequent Levels
	for (int i = 1;i < n_lvls;++i)
	{
		//  reconstruction(mask     ,  mark)
		tempRec = cvt::algorithms::erosionByReconstructionUphill<OperatingPixelType>( inputImage, mos.at(i) ).clone();

		// get the absolute difference of the two images
		absdiff(lastRec, tempRec, diffImage);
		profile.push_back(diffImage.clone());

		tempRec.copyTo(lastRec);

	}


	return profile;

};

template <typename OperatingPixelType>
cv::Mat cvt::algorithms::erosionByReconstructionUphill(const cv::Mat& mask, const cv::Mat& mark)
{

	if (false == std::numeric_limits<OperatingPixelType>::is_integer)
		throw std::logic_error("Invalid template / operating-pixel type");

	const cv::Size size = mask.size();

	//if (size.width != mark.size().width || size.height != mark.size().height)
	if (size != mark.size())
	{
		std::stringstream errmsg;
		errmsg << "ERROR::erosionByReconstructionUphill "
		<< "Size mismatch: ("
		<< size.width << "," << size.height << ") != ("
		<< mark.size().width << "," << mark.size().height << ")";
		throw std::runtime_error(errmsg.str());
	}

	// --------------------------------
	// Place each p : I(p) > 0 into List[I(p)]
	// --------------------------------
	// Copy the marker into the reconstruction
	//	we can then grow it in place
	// value i has a list of pixels at position (y,x)
	cv::Mat reconstruction(mark.size().height, mark.size().width, cv::DataType<OperatingPixelType>::type);
	mark.copyTo(reconstruction);

	// for template, use typename
	typedef std::list<std::pair<int,int> > LocationListType;
	typedef std::map< OperatingPixelType, LocationListType > MapOfLocationListType;

	// This is essentially an indexed image version of the input `mark' raster
	MapOfLocationListType valueLocationListMap;

	// Build an in-memory indexed image
	// but while we are at it, verify that it is properly bounded by `mask' raster
	for (int i = 0; i < size.height; ++i)
		for (int j = 0; j < size.width; ++j)
		{
			const OperatingPixelType pixelValue =  reconstruction.at<OperatingPixelType>(i, j);
			if (pixelValue < mask.at<OperatingPixelType>(i, j))
			{
				//std::cout << "Storing ("<< i << "," << j <<") into valueLocationListMap["<< static_cast<int>(pixelValue) << "]" << std::endl;
				std::stringstream msg;
				msg << "(pixelValue < mask[0](i, j)), for raster location = ("<<i<<","<<j<<")";
				throw std::logic_error(msg.str()); // the marker must always be GTE the mask, by definition
			}

			valueLocationListMap[pixelValue].push_back( std::pair<int,int>(i,j) );
		}

	// No valid pixels or values above floor?  Return the input as output
	if (valueLocationListMap.size() == 0)
		return reconstruction;

	// --------------------------------
	// The farthest uphill we will go
	//	is to the current max(mark[0])
	//	therefore, we do not need to
	//	process the current max, we
	//	already copied them to output
	const OperatingPixelType maxValue =  valueLocationListMap.rbegin()->first;
	valueLocationListMap[maxValue].clear();

	std::set<std::pair<int,int> > finalizedPixels;

	// --------------------------------
	// Do a forward iteration (uphill)
	// through the
	//	valueLocationListMap[values]
	// --------------------------------
	// use typename for template version
	//for (typename MapOfLocationListType::iterator valueList = valueLocationListMap.begin();
	//			valueList != valueLocationListMap.end(); ++valueList)
	for (typename MapOfLocationListType::iterator valueList = valueLocationListMap.begin();
					valueList != valueLocationListMap.end(); ++valueList)
	{
		// The gray value
		const OperatingPixelType currentValue = valueList->first;
		// The list of (y,x) tuples
		LocationListType locations = valueList->second;

		// for each location indexed in the value list
		while (!locations.empty())
		{
			// pull/pop the first position from the list, mark it as finalized
			std::pair<int,int> frontPosition = locations.front();
			finalizedPixels.insert(frontPosition);
			locations.pop_front();

			const int y = frontPosition.first;
			const int x = frontPosition.second;

			const int pre_x = x - 1;
			const int post_x = x + 1;
			const int pre_y = y - 1;
			const int post_y = y + 1;

			// For each neighbor
			// - a -
			// b p d
			// - c -
			// a = (pre_y,x), b = (y,pre_x), c = (post_y,x), d = (y,post_x)

			// Neighbor Pixel 'a'
			const std::pair<int,int> a = std::pair<int,int>(pre_y,x);
			const OperatingPixelType mask_a = mask.at<OperatingPixelType>(pre_y,x);
			// if neighbor index is within bounds and not finalized
			if ((finalizedPixels.find(a) == finalizedPixels.end()) && (pre_y >= 0) && (mask_a > 0))
			{
				OperatingPixelType neighborValue = reconstruction.at<OperatingPixelType>(pre_y,x);
				OperatingPixelType constraintValue = std::max<OperatingPixelType>(currentValue,mask_a);

				if (neighborValue > constraintValue)
				{
					reconstruction.at<OperatingPixelType>(pre_y,x) = constraintValue;
					if ( constraintValue == currentValue )
					{
						locations.push_back( a );
					}
					else
					{
						valueLocationListMap[neighborValue].remove( a );
						valueLocationListMap[constraintValue].push_back( a );
					}
				}
			}

			// Neighbor Pixel 'b'
			const std::pair<int,int> b = std::pair<int,int>(y,pre_x);
			const OperatingPixelType mask_b = mask.at<OperatingPixelType>(y,pre_x);
			if ((finalizedPixels.find(b) == finalizedPixels.end()) && (pre_x >= 0) && (mask_b > 0))
			{
				OperatingPixelType neighborValue = reconstruction.at<OperatingPixelType>(y,pre_x);
				OperatingPixelType constraintValue = std::max<OperatingPixelType>(currentValue,mask_b);

				if (neighborValue > constraintValue)
				{
					reconstruction.at<OperatingPixelType>(y,pre_x) = constraintValue;
					if ( constraintValue == currentValue )
					{
						locations.push_back( b );
					}
					else
					{
						valueLocationListMap[neighborValue].remove( b );
						valueLocationListMap[constraintValue].push_back( b );
					}
				}
			}

			// Neighbor Pixel 'c'
			const std::pair<int,int> c = std::pair<int,int>(post_y,x);
			const OperatingPixelType mask_c = mask.at<OperatingPixelType>(post_y,x);
			if ((finalizedPixels.find(c) == finalizedPixels.end()) && (post_y < size.height) && (mask_c > 0))
			{
				OperatingPixelType neighborValue = reconstruction.at<OperatingPixelType>(post_y,x);
				OperatingPixelType constraintValue = std::max<OperatingPixelType>(currentValue,mask_c);

				if (neighborValue > constraintValue)
				{
					reconstruction.at<OperatingPixelType>(post_y,x) = constraintValue;
					if ( constraintValue == currentValue )
					{
						locations.push_back( c );
					}
					else
					{
						valueLocationListMap[neighborValue].remove( c );
						valueLocationListMap[constraintValue].push_back( c );
					}
				}
			}

			// Neighbor Pixel 'd'
			const std::pair<int,int> d = std::pair<int,int>(y,post_x);
			const OperatingPixelType mask_d = mask.at<OperatingPixelType>(y,post_x);
			if ((finalizedPixels.find(d) == finalizedPixels.end()) && (post_x < size.width) && (mask_d > 0))
			{
				OperatingPixelType neighborValue = reconstruction.at<OperatingPixelType>(y,post_x);
				OperatingPixelType constraintValue = std::max<OperatingPixelType>(currentValue,mask_d);

				if (neighborValue > constraintValue)
				{
					reconstruction.at<OperatingPixelType>(y,post_x) = constraintValue;

					if ( constraintValue == currentValue )
					{
						locations.push_back( d );
					}
					else
					{
						valueLocationListMap[neighborValue].remove( d );
						valueLocationListMap[constraintValue].push_back( d );
					}
				}
			}



		} // end, a value's position list is emptied.

	} // end for each value in the marker



	return reconstruction;


};



#endif
// CVTILE_ALGORITHM_DMP_H_

