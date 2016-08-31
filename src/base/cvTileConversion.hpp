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


#include "cvTile.hpp"
// TODO: Remove dependence on cgi library, or include the code.
#include <cgi/tile/Tile.hpp>

namespace cvt {
class cvTileConversion{

public:
	///
	/// Convert a cvTile to a Tile.
	/// \return The Tile object which is a copy of the cvTile object passed to the function.
	///
	template <typename T>
	static cgi::Tile<T> getTileCopy(const cvTile<T> m);

	///
	/// Convert a Tile to a cvTile.
	/// \return The cvTile object which is a copy of the Tile object passed to the function.
	///
	template <typename T>
	static cvTile<T> getCvTileCopy(const cgi::Tile<T> m);
};

template <typename T>
cgi::Tile<T> cvTileConversion::getTileCopy(const cvTile<T> m)
{
	int bandCount = m.getBandCount();

	std::vector< boost::numeric::ublas::matrix<T> > t_data;
	cgi::Tile<T> t;

	//INSERT THE BANDS IF THERE ARE BANDS
	boost::bimap<int, std::string> t_band_map;

	for(int i=0; i < bandCount; i++){
		cv::Mat temp_data = m[i];
		boost::numeric::ublas::matrix<T> band(temp_data.rows, temp_data.cols);

		for(int ii=0; ii < temp_data.rows; ii++){
			for(int jj=0; jj < temp_data.cols; jj++){
				band(ii,jj) = temp_data.at<T>(ii,jj);
			}
		}

		t_data.push_back(band);
	}

	if(t_data.size()!=0){
		cgi::Tile<T> m_local(t_data);
		t = m_local;
	}

	for(int i=0; i < bandCount; i++)
		//set the band
		t.setBandName(i, m.getBandName(i));

	//set the ROI
	t.setROI(m.getROI());

	//COPY THE METADATA
	std::set<std::string> t_metadata_keys = m.getMetadataKeys();

	// a map of strings for the _metadata
	std::set<std::string>::iterator it;

	//iterate over the metadata keys and inser the metadata into the Tile
	for (it = t_metadata_keys.begin(); it != t_metadata_keys.end(); ++it)
	    t.setMetadata(*it, m.getMetadata(*it));


	//get the no data value
	T t_nodata_value = m.getNoDataValue();

	//if there is no data value, then set it
	if(m.hasNoDataValue())
		t.setNoDataValue(t_nodata_value);

	if(m.hasMask())
	{
		// CONVERT THE MASK TO BOOST MATRIX
		cv::Mat t_mask = m.getMask();
		boost::numeric::ublas::matrix<unsigned char> t_boost_mask(t_mask.rows, t_mask.cols);
		//copy the values from the mask into the boost matrix
		for(int i=0; i<t_mask.rows; i++){
			for(int j=0; j<t_mask.cols; j++){
				t_boost_mask(i,j) = t_mask.at<unsigned char>(i,j);
			}
		}
		//set the mask
		t.setMask(t_boost_mask);
	}

	return t;
}

template <typename T>
cvTile<T> cvTileConversion::getCvTileCopy(const cgi::Tile<T> m)
{
	int bandCount = m.getBandCount();

	std::vector< cv::Mat > t_data;
	cvTile<T> t;

	//INSERT THE BANDS IF THERE ARE BANDS
	boost::bimap<int, std::string> t_band_map;

	for(int i=0; i < bandCount; i++){
		boost::numeric::ublas::matrix<T> temp_data = m[i];
		cv::Mat band(temp_data.size1(), temp_data.size2(), cv::DataType<T>::type);

		for(unsigned int ii=0; ii < temp_data.size1(); ii++){
			for(unsigned int jj=0; jj < temp_data.size2(); jj++){
				band.at<T>(ii,jj) = temp_data(ii,jj);
			}
		}

		t_data.push_back(band);
	}

	if(t_data.size()!=0){
		cvTile<T> m(t_data);
		t = m;
	}

	for(int i=0; i < bandCount; i++){
		//set the band
		t.setBandName(i, m.getBandName(i));
	}

	//set the ROI
	t.setROI(m.getROI());

	//COPY THE METADATA
	std::set<std::string> t_metadata_keys = m.getMetadataKeys();

	// a map of strings for the _metadata
	std::set<std::string>::iterator it;

	//iterate over the metadata keys and inser the metadata into the Tile
	for (it = t_metadata_keys.begin(); it != t_metadata_keys.end(); ++it)
	    t.setMetadata(*it, m.getMetadata(*it));

	//get the no data value
	T t_nodata_value = m.getNoDataValue();

	//if there is no data value, then set it
	if(m.hasNoDataValue())
		t.setNoDataValue(t_nodata_value);

	if(m.hasMask())
	{
		// CONVERT THE MASK TO BOOST MATRIX
		boost::numeric::ublas::matrix<unsigned char> t_mask = m.getMask();
		cv::Mat mask(t_mask.size1(), t_mask.size2(), cv::DataType<unsigned char>::type);
		//copy the values from the mask into the boost matrix
		for(unsigned int i=0; i<t_mask.size1(); i++){
			for(unsigned int j=0; j<t_mask.size2(); j++){
				mask.at<unsigned char>(i,j) = t_mask(i,j);
			}
		}
		//set the mask
		t.setMask(mask);
	}

	return t;
}

}//end namespace cvt
