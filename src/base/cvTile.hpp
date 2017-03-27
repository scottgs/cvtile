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


#ifndef CVTILE_HPP_
#define CVTILE_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bimap.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <vrtdataset.h>
#include <ogr_spatialref.h>
#include <gdal_priv.h>
#include <map>
#include <string>
#include <set>
#include <functional>
#include <algorithm>
#include <cmath>
#include "cvTileVectorProxy.hpp"

#define BOOST_UBLAS_NDEBUG

// TODO: Remove using std for readability.
using namespace std;

namespace cvt {

// forward declarations
template <typename ContainerType, typename ValueType, typename ReferenceType>
class cvTileIterator;

template<typename T>
class cvTile {
private:

	///
	/// A rectangle object representing the size of the cvTile object
	/// \sa cvt::core::Size2d
	///
	cv::Size2i _size;

	///
	/// A vector of matrix objects that hold the actual data.
	///
	std::vector<cv::Mat> _data;

	///
	/// A boolean object that indicates whether the NODATA value is valid.
	///
	bool _has_nodata_value;

	///
	/// The NODATA value for the cvTile.  This is the NODATA value for all bands.
	///
	T _nodata_value;

	///
	/// A boolean indicating if the cvTile has a mask
	///
	bool _has_mask;

	///
	/// A matrix object that contains the cvTile mask, if it exists
	///
	cv::Mat _mask;

	///
	/// A holder for metadata.  The metadata values are indexed by a key and
	/// can be pulled out using the index string.
	///
	std::map<std::string, std::string> _metadata;

	///
	/// A bimap for the band names.  Names can be converted to indexes and
	/// indexes can be converted to names.
	///
	boost::bimap<int, std::string> _band_map;

	///
	/// A region of interest (ROI) can be specified for the cvTile object.  This should
	/// used when a cvTile requires a buffer (due to some image processing specific
	/// reason) but we're only interested in the results from some core section
	/// of the image.  By default the ROI will be set to be the entire image, however
	/// this can be adjusted using the setROI method.
	/// \sa setROI
	///
	cv::Rect _roi;

public:

	///
	/// Typedef for the value type of the tile
	///
	typedef T value_type;

	///
	/// Typedef for the matrix type used for storage
	///
	typedef cv::Mat matrix_type;

	///
	/// Typedef for the tile iterator class
	///
	typedef cvTileIterator< cvTile<T>, T, cvTileVectorProxy<T> > iterator;

	///
	/// Typedef for the tile const_iterator class
	///
	typedef cvTileIterator< const cvTile<T>, const T, const cvTileVectorProxy<T> > const_iterator;

	///
	/// Default constructor to zombie cvTile object with no size and no bands
	///
	explicit cvTile();

	///
	/// Constructor to create a cvTile object based on dimensions and the number of bands
	/// \param size Dimensions
	/// \param bands Number of bands
	/// \sa Size2d
	///
	explicit cvTile(const cv::Size2i& size, int bands);

	///
	/// Constructor to create a cvTile object based on dimensions and the number of bands
	/// \param size Dimensions
	/// \param bands Number of bands
	/// \param initValue Value that should be used to initialize all data values
	/// \sa Size2d
	///
	explicit cvTile(const cv::Size2i& size, int bands, T initValue);

	///
	/// Constructor to create a 1-band cvTile object from a matrix of data
	/// \param m Matrix of data
	///
	explicit cvTile(const cv::Mat& m);

	///
	/// Constructor to create an n-band cvTile object from a vector of matrices
	/// \param v Vector of matrix objects
	///
	explicit cvTile(const std::vector<cv::Mat>& v);

	///
	/// Constructor to create an n-band cvTile object from a buffer of data given
	/// the dimensions and the number of bands.  The data in the buffer should be
	/// given in BSQ order.
	/// \param buffer Raw buffer of image data
	/// \param size Dimensions
	/// \param bands Number of bands
	///
	explicit cvTile(T* buffer, const cv::Size2i& size,
			int bands);

	int getType();

	///
	/// Set all data element in the Tile object to the given value
	/// \param value Value that should be used to initialize all data values
	///
	void set(T value);

	/* ********************************************************************* */

	///
	/// Set masked data elements in the Tile object to the given value
	/// \param value Value that should be used to initialize all data values
	///
	void set(T value, const cv::Mat& mask);

	/* ********************************************************************* */

	///
	/// Get the dimensions of the image
	/// \return A rectangle with the dimensions of the image
	/// \sa cv::Size2i
	///
	const cv::Size2i& getSize() const;

	///
	/// Get the number of bands in the image
	/// \return The number of bands in the image
	///
	int getBandCount() const;

	///
	/// Get the human readable band name by index
	/// \param band The band index
	/// \return The band name
	///
	const std::string getBandName(int band) const;

	///
	/// Get the band index by passing the human readable band name
	/// \param name The band name
	/// \return The band index
	///
	int getBandIndex(const std::string& name) const;

	///
	/// Set name for a given band based on an index
	/// \param band The band index
	/// \param name The band name
	/// \return A boolean value indicating whether the band name was successfully set.
	///
	bool setBandName(int band, const std::string& name);

	///
	/// Get the ROI for the image
	/// \return A const reference to the ROI for the image
	///
	const cv::Rect& getROI() const;

	///
	/// Set a metdata value by passing a key and value
	/// \param key The metadata key
	/// \param value The metadata value
	/// \return A boolean value indicating whether the metadata value was successfully set.
	///
	bool setMetadata(const std::string& key, const std::string& value);

	///
	/// Retrieve a metadata value by key
	/// \param key The metadata key
	/// \return The metadata value
	///
	const std::string getMetadata(const std::string& key) const;

	///
	/// Retrieve the set of metadata keys
	/// \return The set of metadata keys
	///
	const std::set<std::string> getMetadataKeys() const;

	///
	/// Retrieve a metadata value by key
	/// and lexically cast the result
	/// \tparam U desired type
	/// \param key The metadata key
	/// \return The metadata value
	///    lexically casted to U
	///
	template<class U>
	U getMetadataAs(const std::string& key) const;

	// NOTE: t[band](row, column) is more efficient than t(row, column)[band]
	// with cvTileVectorProxy this won't actually be so! stay tuned

	///
	/// Return a const reference to a matrix of data for a given band by index
	/// \param band The band index
	/// \return The reference to the data
	///
	const cv::Mat& operator[](int band) const;

	///
	/// Return a reference to a matrix of data for a given band by index
	/// \param band The band index
	/// \return The reference to the data
	///
	cv::Mat& operator[](int band);

	/* *************************************** */

	///
	/// Return a const reference to a matrix of data for a given band by name
	/// \param band The band name
	/// \return The reference to the data
	///
	const cv::Mat& operator[](const std::string& name) const;

	///
	/// Return a reference to a matrix of data for a given band by name
	/// \param band The band name
	/// \return The reference to the data
	///
	cv::Mat& operator[](const std::string& name);

	/* *************************************** */

	///
	/// Set the ROI for the image
	/// \param roi The new ROI
	/// \return A boolean value indicating whether the ROI was successfully set.
	///
	bool setROI(const cv::Rect& roi);

	///
	/// Expand the ROI for the image by adjusting the ROI corners outward
	/// \param expansion The amount to expand the ROI rectangle
	/// \return A boolean value indicating whether the ROI was successfully expanded.
	///
	bool expandROI(int expansion);

	///
	/// Expand the ROI for the image by adjusting the ROI corners outward
	/// \param expansionX The amount to expand the x-axis of the ROI rectangle
	/// \param expansionY The amount to expand the y-axis of the ROI rectangle
	/// \return A boolean value indicating whether the ROI was successfully expanded.
	///
	bool expandROI(int expansionX, int expansionY);

	///
	/// Constrict the ROI for the image by adjusting the ROI corners inward
	/// \param constriction The amount to constrict the ROI rectangle
	/// \return A boolean value indicating whether the ROI was successfully constricted.
	///
	bool constrictROI(int constriction);

	///
	/// Constrict the ROI for the image by adjusting the ROI corners inward
	/// \param constrictionX The amount to constrict the x-axis of the ROI rectangle
	/// \param constrictionY The amount to constrict the y-axis of the ROI rectangle
	/// \return A boolean value indicating whether the ROI was successfully constricted.
	///
	bool constrictROI(int constrictionX, int constrictionY);

	///
	/// Reduce the tile's internal raster to just the ROI, has no effect on metadata
	///
	void cropToROI();

	///
	/// Get a copy of the tile with an internal raster to just the ROI, has no effect on metadata
	/// \return a copy of the tile's ROI as a new tile with no buffer
	///
	cvTile<T> copyCropToROI() const;

	///
	/// Set the ROI for the image to the size of the image
	/// \return The ROI before the reset
	///
	cv::Rect resetROI();

	///
	/// Set the NODATA value for the tile object.  The same NODATA value is
	/// used for all bands.
	/// \param nodata_value The NODATA value.
	/// \return A boolean value indicating whether the NODATA value was successfully set.
	///
	bool setNoDataValue(T nodata_value);

	///
	/// Get the NODATA value for this tile.
	/// \return The NODATA value.
	///
	const T getNoDataValue() const;

	///
	/// Check to see if this tile has a NODATA value set or not.  By default
	/// Tile objects do not have a NODATA value set.
	/// \return A boolean value indicating whether the NODATA value has been set or not.
	///
	bool hasNoDataValue() const;

	/* ********************************************************************* */

	///
	/// Set the tile mask.
	/// \return A boolean value indiciating if the mask was set correctly.
	///
	bool setMask(const cv::Mat& mask);

	///
	/// Set the tile mask.
	/// \return A boolean value indiciating if the mask was set correctly.
	///
	bool setMask(const unsigned char* mask, const cv::Size2i& size);

	///
	/// Check to see if this tile has a mask.
	/// \return A boolean value indicating if this tile has a mask.
	///
	bool hasMask() const;

	///
	/// Get the 8-bit tile mask, if available. If not available, this function
	/// will return a matrix filled with 255 (valid).
	/// \return The 8-bit tile mask.
	///
	const cv::Mat getMask() const;

	/* ********************************************************************* */

	///
	/// Clone all the non-pixel information from one tile to
	/// produce a second tile with numBands.
	///
	/// \param numBands the number of tile bands in the clone.
	cvTile<T> cloneWithoutData(int numBands) const;

	///
	/// Clone all the non-pixel information from one tile to
	///  produce a second tile of type U with numBands
	///
	/// \tparam the type of the new tile
	/// \param numBands the number of tile bands in the clone.
	template<typename U>
	cvTile<U> cloneWithoutDataTo(int numBands) const;

	///
	/// Clone a subset of the tile's data.  Select the band by index
	///
	/// \param bandIndex Select the given band and clone it's data in the returned object.
	/// \return A partial clone of the object
	cvTile<T> cloneSubset(int bandIndex) const;

	///
	/// Clone a subset of the tile's data.  Select the band by name
	///
	/// \param bandIndex Select the given band and clone it's data in the returned object.
	/// \return A partial clone of the object
	cvTile<T> cloneSubset(std::string bandName) const;

	///
	/// Clone a subset of the tile's data.  Select the bands by index
	///
	/// \param bandIndex Select the given band and clone it's data in the returned object.
	/// \return A partial clone of the object
	cvTile<T> cloneSubset(std::set<int> bandIndex) const;

	///
	/// Clone a subset of the tile's data.  Select the bands by name
	///
	/// \param bandIndex Select the given band and clone it's data in the returned object.
	/// \return A partial clone of the object
	cvTile<T> cloneSubset(std::set<std::string> bandName) const;

	/* *************************************** */

	///
	/// Return a vector proxy for a given point in the tile.  This will return
	/// a vector containing info from all the bands at that point in the image.
	/// \param row The row (y) coordinate
	/// \param column The column (x) coordinate
	/// \return The vector proxy
	///
	const cvTileVectorProxy<T> operator()(int row, int column) const;

	///
	/// Return a vector proxy for a given point in the tile.  This will return
	/// a vector containing info from all the bands at that point in the image.
	/// \param row The row (y) coordinate
	/// \param column The column (x) coordinate
	/// \return The vector proxy
	///
	cvTileVectorProxy<T> operator()(int row, int column);

	/* ********************************************************************* */

	///
	/// A helper method for determining if the location pointed to by an iterator
	/// is valid according to the supplied validity type.
	/// \return A boolean value indicating whether the location pointed to by the iterator
	/// (represented by a cvTileVectorProxy) is valid or not.
	bool isValidVector(const cvTileVectorProxy<T>& tvp, valid_mask::Type validity_type) const;

	///
	/// This ignores any mask that may or may not be set and determines the validity of a pixel
	///   by no data valid alone.  Its return value is undefined if it is called on a tile
	///   which doesn't have a no data value set.
	///
	bool isValidVectorByValue(const cvTileVectorProxy<T> &tvp, valid_mask::Type validity_type) const;

	///
	/// Get the validity matrix for the tile according to the supplied validity type. The
	/// rules are as follows, in order of precedence:
	///   * If the tile has a mask, and the validity type supplied is ALL, the validity
	///     matrix will be derived from the tile mask.
	///   * If the tile has a nodata value, the validity matrix will be constructed based
	///     on the requested validity type.
	///   * If the tile has a mask, the validity matrix will be derived from the tile mask.
	///   * Otherwise, an all-true matrix will be returned.
	/// \return A matrix of boolean values indicating the validity of each location in the tile.
	/// The values will all be true if no nodata value is set.
	///
	const cv::Mat getValidMask(cvt::valid_mask::Type maskType) const;

	///
	/// Get the validity matrix for a given band. The rules are as follows, in order of
	/// precedence:
	///   * If the tile has a nodata value, the validity matrix will be constructed based on
	///     nodata values encountered in the requested band.
	///   * If the tile has a mask, the validity matrix will be derived from the tile mask.
	///   * Otherwise, an all-true matrix will be returned.
	/// \return A matrix of boolean values indicating the validity of each location in the band.
	/// The values will all be true if no nodata value is set.
	///
	const cv::Mat getValidMask(int band) const;

	///
	/// Get the validity matrix for a given named band. The rules are as follows, in order of
	/// precedence:
	///   * If the tile has a nodata value, the validity matrix will be constructed based on
	///     nodata values encountered in the requested band.
	///   * If the tile has a mask, the validity matrix will be derived from the tile mask.
	///   * Otherwise, an all-true matrix will be returned.
	/// \return A matrix of boolean values indicating the validity of each location in the band.
	/// The values will all be true if no nodata value is set.
	///
	const cv::Mat getValidMask(const std::string& name) const;


	///
	/// This gets a validity matrix based only upon the set nodata value if any, and the
	///   actual data in the tile. Any set mask is completely ignored.  Invalid pixels are
	///   represented in the output matrix as 0, valid pixels are represented by the
	///   valid_value parameter.  If there isn't a nodata value set, this returns a matrix
	///   of the size of the tile filled completely with the valid_value parameter.
	///
	template <typename U>
	const cv::Mat getValidMaskByValue(valid_mask::Type validity_type, U valid_value = 1) const;

	/* ********************************************************************* */

	///
	/// Get a const_iterator pointing at the beginning of the tile.
	/// \return The const_iterator pointing at the beginning of the tile.
	///
	const_iterator begin() const;

	///
	/// Get a const_iterator pointing at the end of the tile. Note that,
	/// as in the STL, the location pointed to by the end const_iterator is
	/// one increment beyond the last legal location in the data.
	/// \return The const_iterator pointing at the end of the tile.
	///
	const_iterator end() const;

	///
	/// Get an iterator pointing at the beginning of the tile.
	/// \return The iterator pointing at the beginning of the tile.
	///
	iterator begin();

	///
	/// Get an iterator pointing at the end of the tile. Note that,
	/// as in the STL, the location pointed to by the end iterator is one
	/// increment beyond the last legal location in the data.
	/// \return The iterator pointing at the end of the tile.
	///
	iterator end();

	///
	/// A function that gets the value of the specified row and column
	/// of the matrix passed to the function
	/// This is done because cvTileVectorProxy cannot access a Matrix by ".at<T>(row,col)"
	/// so I by-pass it by extracting it from this class
	const T& get(cv::Mat m, const int row,  const int col);
};
//end class

//define the default constructor that does not receive anything
template<typename T>
cvTile<T>::cvTile() :
		_size(cv::Size2i()), _has_nodata_value(false), _has_mask(
				false), _roi(cv::Rect(cv::Point2i(0,0),_size)) {
}

template<typename T>
cvTile<T>::cvTile(const cv::Size2i& size, int bands) :
		_size(size),
		_data(),
		_has_nodata_value(false),
		_has_mask(false),
		_roi(cv::Rect(cv::Point(0,0),size))
{
       //This is done because OpenCV's Mat object, if initialized inline with the constructor
       //just makes a copy of one reference, instead of creating new objects
       for(int i=0; i<bands; i++)
       {
           cv::Mat tmp(size.height, size.width, cv::DataType<T>::type);
	   _data.push_back(tmp);
       }
}

template<typename T>
cvTile<T>::cvTile(const cv::Size2i& size, int bands, T initValue) :
		_size(size),
		_data(),
		_has_nodata_value(false),
		_has_mask(false),
		_roi(cv::Rect(cv::Point2i(0,0), size))
{
	this->set(initValue);

       //This is done because OpenCV's Mat object, if initialized inline with the constructor
       //just makes a copy of one reference, instead of creating new objects
       for(int i=0; i<bands; i++)
       {
           cv::Mat tmp(size.height, size.width, cv::DataType<T>::type, cv::Scalar(initValue));
	   _data.push_back(tmp);
       }
}

template<typename T>
cvTile<T>::cvTile(const cv::Mat& m) :
		_size(cv::Size2i(m.cols, m.rows)),
		_data(std::vector<cv::Mat>(1, m)),
		_has_nodata_value(false),
		_has_mask(false),
		_roi(cv::Rect(cv::Point2i(0,0),cv::Size2i(m.cols, m.rows)))
{}

template<typename T>
cvTile<T>::cvTile(const std::vector<cv::Mat>& v) :
		_size(cv::Size2i(v.front().cols, v.front().rows)),
		_data(v),
		_has_nodata_value(false),
		_has_mask(false),
		_roi(cv::Rect(cv::Point2i(0,0),cv::Size2i(v.front().cols, v.front().rows)))
{
	// run through vector and assert sizes?
}

template<typename T>
cvTile<T>::cvTile(T* buffer, const cv::Size2i& size, int bands) :
		_size(size),
		_data(),
		_has_nodata_value(false),
		_has_mask(false),
		_roi(cv::Rect(cv::Point2i(0,0),size))
{
	T* buffer_start = buffer;
	const int band_offset = _size.height * _size.width;
	//const int row_offset = _size.width;

	for (int b = 0; b < bands; ++b)
	{
		//Cannot use the constructor's initialization list becuase the vector simply
		//copies one reference of the Mat class for each band, and does not create a new one
		//for each band. Weird!
		cv::Mat matrix_data(size.height, size.width, cv::DataType<T>::type);
		std::copy( buffer_start, buffer_start + band_offset, matrix_data.begin<T>() );
		buffer_start += band_offset;
		_data.push_back(matrix_data);
	}
}

template<typename T>
void cvTile<T>::set(T value) {
	for (unsigned int b = 0; b < _data.size(); ++b) {
		typename cv::Mat& matrix_data(_data[b]);
		std::fill(matrix_data.begin<T>(), matrix_data.end<T>(), value);
	}
}

template<typename T>
void cvTile<T>::set(T value, const cv::Mat& mask) {
	if (_data.size() == 0)
		return;
	if (mask.rows != _data[0].rows || mask.cols != _data[0].cols)
		return;

	//loop through the bands
	for (unsigned int b = 0; b < _data.size(); ++b) {
		//check if the value is supposed to be set, and if so
		//set it equal to the value variable passed to the function
		//TODO : optimize this using 1D array representation - cv::Mat data is "unsigned char* const"
		//so I have to figure out how to convert it correctly
		for (int i = 0; i < _data[b].rows; i++) {
			for (int j = 0; j < _data[b].cols; j++) {
				if (mask.at<bool>(i, j))
					_data[b].template at < T > (i, j) = value;
			}
		}
	}
}

template<typename T>
const cv::Size2i& cvTile<T>::getSize() const {
	return _size;
}

template<typename T>
int cvTile<T>::getBandCount() const {
	return _data.size();
}

template<typename T>
int cvTile<T>::getType() {
	return cv::DataType < T > ::type;
}

template<typename T>
const cv::Rect& cvTile<T>::getROI() const {
	return _roi;
}

template<typename T>
bool cvTile<T>::expandROI(int expansion) {
	return expandROI(expansion, expansion);
}

template<typename T>
bool cvTile<T>::expandROI(int expansionX, int expansionY) {
	cv::Rect tempRoi = _roi;

	//Replace this with actual code to expand the ROI
	tempRoi.x -= expansionX;
	tempRoi.y -= expansionY;
	tempRoi.width += (expansionX + expansionX);
	tempRoi.height += (expansionY + expansionY);

	// make sure the offset isn't too small
	if (tempRoi.x < 0)
		return false;
	if (tempRoi.y < 0)
		return false;

	// nor too big
	if (tempRoi.x >= this->getSize().width)
		return false;
	if (tempRoi.y >= this->getSize().height)
		return false;

	// make sure the size is positive
	if (tempRoi.width <= 0)
		return false;
	if (tempRoi.height <= 0)
		return false;

	// make sure that offset + size isn't too big
	if ((tempRoi.x + tempRoi.width)
			> this->getSize().width)
		return false;
	if ((tempRoi.y + tempRoi.height)
			> this->getSize().height)
		return false;

	_roi = tempRoi;

	return true;
}

template<typename T>
bool cvTile<T>::constrictROI(int constriction) {
	return expandROI(-constriction, -constriction);
}

template<typename T>
bool cvTile<T>::constrictROI(int constrictionX, int constrictionY) {
	return expandROI(-constrictionX, -constrictionY);
}

template<typename T>
cv::Rect cvTile<T>::resetROI() {
	cv::Rect roi = _roi;

	_roi = cv::Rect(cv::Point2i(0, 0),_size);

	return roi;
}

template<typename T>
void cvTile<T>::cropToROI() {
	// NOTE: Purposely leaving the metadata un-touched

	// get data dimensions
	const int width = _roi.width;
	const int height = _roi.height;
	const int bands = getBandCount();
	const cv::Size2i roiSize = _roi.size();

	// allocate new raster, copy and assign
	std::vector<cv::Mat> newData;

	for (int b = 0; b < bands; ++b) {
		//This has to be done here because cv::Mat does not get a new reference
		//if initialized directly when initializing the vector itself
		cv::Mat tmp(height, width, cv::DataType<T>::type);
		newData.push_back(tmp);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				newData[b].template at<T>(y, x) = _data[b].template at<T>(y + _roi.y, x + _roi.x);
			}
		}
	}

	_data = newData;

	// reset the _size, then the ROI to 0,0 -> size
	_size = roiSize;
	_roi = cv::Rect(_size, cv::Point2i(0, 0));
}

template<typename T>
cvTile<T> cvTile<T>::copyCropToROI() const {
	cvTile<T> copy(*this);
	copy.cropToROI();
	return copy;
}

template<typename T>
bool cvTile<T>::setROI(const cv::Rect& roi) {

	// make sure the offset isn't too small
	if (roi.x < 0)
		return false;

	if (roi.y < 0)
		return false;

	// nor too big
	if (roi.x >= this->getSize().width)
		return false;

	if (roi.y >= this->getSize().height)
		return false;

	// make sure the size is positive
	if (roi.width <= 0)
		return false;

	if (roi.height <= 0)
		return false;

	// make sure that offset + size isn't too big
	if ((roi.x + roi.width)
			> this->getSize().width)
		return false;

	if ((roi.y + roi.height)
			> this->getSize().height)

		return false;
	_roi = roi;
	return true;
}

template<typename T>
bool cvTile<T>::setMetadata(const std::string& key,
		const std::string& value) {
	_metadata[key] = value;

	return true;
}

template<typename T>
const std::string cvTile<T>::getMetadata(const std::string& key) const {
	std::map<std::string, std::string>::const_iterator iter = _metadata.find(
			key);

	if (iter == _metadata.end()) {
		return std::string();
	} else {
		return iter->second;
	}
}

template<typename T>
const std::set<std::string> cvTile<T>::getMetadataKeys() const {
	std::set<std::string> keys;

	for (std::map<std::string, std::string>::const_iterator iter =
			_metadata.begin(); iter != _metadata.end(); ++iter)
		keys.insert(iter->first);

	return keys;
}

template<typename T> template<class U>
U cvTile<T>::getMetadataAs(const std::string& key) const {
	return boost::lexical_cast<U>(this->getMetadata(key));
}

template<typename T>
const std::string cvTile<T>::getBandName(int band) const {
	boost::bimap<int, std::string>::left_const_iterator left_iter =
			_band_map.left.find(band);

	if (left_iter == _band_map.left.end()) {
		return std::string();
	} else {
		return left_iter->second;
	}
}

template<typename T>
int cvTile<T>::getBandIndex(const std::string& name) const {
	boost::bimap<int, std::string>::right_const_iterator right_iter =
			_band_map.right.find(name);

	if (right_iter == _band_map.right.end()) {
		return -1;
	} else {
		return right_iter->second;
	}
}

template<typename T>
bool cvTile<T>::setBandName(int band, const std::string& name) {
	_band_map.insert(boost::bimap<int, std::string>::value_type(band, name));

	return true;
}

template<typename T>
const cv::Mat& cvTile<T>::operator[](int band) const {
	return _data.at(band);
}

template<typename T>
cv::Mat& cvTile<T>::operator[](int band) {
	return const_cast<cv::Mat&>(static_cast<const cvTile<T>&>(*this)[band]);
}

template<typename T>
const cv::Mat& cvTile<T>::operator[](const std::string& name) const {
	const int band = getBandIndex(name);

	return _data.at(band);
}

template<typename T>
cv::Mat& cvTile<T>::operator[](const std::string& name) {
	return const_cast<cv::Mat&>(static_cast<const cvTile<T>&>(*this)[name]);
}

template <typename T>
const cvTileVectorProxy<T> cvTile<T>::operator()(int row, int column) const
{
	return cvTileVectorProxy<T>(this, row, column);
}

template <typename T>
cvTileVectorProxy<T> cvTile<T>::operator()(int row, int column)
{
	return static_cast<const cvTile&>(*this)(row, column);
}

template<typename T>
bool cvTile<T>::setNoDataValue(T nodata_value) {
	_nodata_value = nodata_value;
	_has_nodata_value = true;

	return true;
}

template<typename T>
const T cvTile<T>::getNoDataValue() const {
	return _nodata_value;
}

template<typename T>
bool cvTile<T>::hasNoDataValue() const {
	return _has_nodata_value;
}
//////
template<typename T>
bool cvTile<T>::setMask(const cv::Mat& mask) {
	if (mask.rows != _data[0].rows || mask.cols != _data[0].cols)
		return false;

	_mask = mask;
	_has_mask = true;

	return true;
}

template<typename T>
bool cvTile<T>::setMask(const unsigned char* mask,
		const cv::Size2i& size) {
	if (_mask.rows != size.height || _mask.cols != size.width)
		cv::resize(_mask, _mask, cv::Size(size.width, size.height));

	std::copy(mask, mask + size.area(), _mask.begin<T>());

	_has_mask = true;

	return true;
}

template<typename T>
bool cvTile<T>::hasMask() const {
	return _has_mask;
}

template<typename T>
const cv::Mat cvTile<T>::getMask() const {
	if (_has_mask)
		return _mask;

	//return a new matrix initialized with the value 255
	return cv::Mat(_size.height, _size.width,
			cv::DataType<unsigned char>::type, cv::Scalar(255));
}

template <typename T>
bool cvTile<T>::isValidVectorByValue(const cvTileVectorProxy<T> &tvp, valid_mask::Type validity_type) const
{
	int count = 0;
	for(int b = 0; b < getBandCount(); ++b)
		count += (tvp[b] == getNoDataValue()) ? 1 : 0;

	return (validity_type == valid_mask::ANY && count < getBandCount()) ||
		   (validity_type == valid_mask::ALL && count == 0) ||
	   (validity_type == valid_mask::MAJORITY && count <= getBandCount() / 2);
}

template <typename T>
bool cvTile<T>::isValidVector(const cvTileVectorProxy<T>& tvp, valid_mask::Type validity_type) const
{
	if (_has_mask && (validity_type == valid_mask::ALL || !_has_nodata_value))
	{
		return (_mask.at<unsigned char>(tvp._row, tvp._column) != 0);
	}
	else if (_has_nodata_value)
	{
		return isValidVectorByValue(tvp,validity_type);
	}
	// else
		return true;
}


template <typename T>
template <typename U>
const cv::Mat cvTile<T>::getValidMaskByValue(valid_mask::Type validity_type, U valid_value) const
{
	cv::Mat r_mask(getSize().height, getSize().width, cv::DataType<U>::type);

	if(hasNoDataValue())
	{
		for(typename cvt::cvTile<T>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
			r_mask.at<U>(iter.position().y, iter.position().x) = (isValidVectorByValue(*iter,validity_type)) ? valid_value : 0;
	}
	else
	{
		std::fill(r_mask.begin<U>(), r_mask.end<U>(), valid_value);
	}
	return r_mask;
}

template <typename T>
const cv::Mat cvTile<T>::getValidMask(valid_mask::Type validity_type) const
{
	cv::Mat mask(_size.height, _size.width, cv::DataType<bool>::type);

	if (_has_nodata_value || _has_mask)
	{
		std::transform(this->begin(), this->end(),
					   mask.begin<bool>(),
		               std::bind(&cvTile<T>::isValidVector, this, std::placeholders::_1, validity_type));
	}
	else
	{
		std::fill(mask.begin<bool>(), mask.end<bool>(), true);
	}

	return mask;
}

template <typename T>
const cv::Mat cvTile<T>::getValidMask(int band) const
{
	cv::Mat mask(_size.height, _size.width, cv::DataType<bool>::type);

	if (_has_nodata_value)
	{
		const typename cv::Mat band_data = _data.at(band);
		std::transform(band_data.begin<T>(), band_data.end<T>(),
		               mask.begin<bool>(),
		               std::bind2nd(std::not_equal_to<T>(), _nodata_value));
	}
	else if (_has_mask)
	{
		std::transform(_mask.begin<bool>(), _mask.end<bool>(),
					   mask.begin<bool>(),
					   std::bind2nd(std::not_equal_to<unsigned char>(), 0));
	}
	else
	{
		std::fill(mask.begin<bool>(), mask.end<bool>(), true);
	}

	return mask;
}

template <typename T>
const cv::Mat cvTile<T>::getValidMask(const std::string& name) const
{
	return getValidMask(getBandIndex(name));
}

template<typename T>
cvTile<T> cvTile<T>::cloneWithoutData(int numBands) const {
	const cv::Size2i& size = getSize();
	cvTile<T> returnTile(size, numBands);

	returnTile._has_nodata_value = _has_nodata_value;
	returnTile._nodata_value = _nodata_value;
	returnTile._has_mask = _has_mask;
	returnTile._mask = _mask;
	returnTile._metadata = _metadata;
	returnTile._roi = _roi;

	return returnTile;
}

template<typename T>
template<typename U>
cvTile<U> cvTile<T>::cloneWithoutDataTo(int numBands) const {
	cvTile<U> returnTile(getSize(), numBands);

	if (_has_nodata_value) {
		returnTile.setNoDataValue(static_cast<U>(getNoDataValue()));
	}

	if (_has_mask) {
		returnTile.setMask(_mask);
	}

	returnTile.setROI(getROI());

	for (std::map<std::string, std::string>::const_iterator iter =
			_metadata.begin(); iter != _metadata.end(); ++iter) {
		returnTile.setMetadata(iter->first, iter->second);
	}

	return returnTile;
}

template<typename T>
cvTile<T> cvTile<T>::cloneSubset(int bandIndex) const {
	std::set<int> tmp;
	tmp.insert(bandIndex);
	return this->cloneSubset(tmp);
}

template<typename T>
cvTile<T> cvTile<T>::cloneSubset(std::string bandName) const {
	std::set<int> tmp;
	tmp.insert(this->getBandIndex(bandName));
	return this->cloneSubset(tmp);
}

template<typename T>
cvTile<T> cvTile<T>::cloneSubset(std::set<int> bandIndex) const {
	cvTile<T> tmp = this->cloneWithoutData(bandIndex.size());

	unsigned int i = 0;
	for (std::set<int>::const_iterator it = bandIndex.begin();
			it != bandIndex.end(); ++it) {
		tmp.setBandName(i, this->getBandName(*it));
		tmp._data[i++] = this->_data[*it];
	}

	return tmp;
}

template<typename T>
cvTile<T> cvTile<T>::cloneSubset(std::set<std::string> bandName) const {
	std::set<int> tmp;
	for (std::set<std::string>::const_iterator it = bandName.begin();
			it != bandName.end(); ++it)
		tmp.insert(this->getBandIndex(*it));
	return this->cloneSubset(tmp);
}

template <typename T>
typename cvTile<T>::const_iterator cvTile<T>::begin() const
{
	return const_iterator::begin(this);
}

template <typename T>
typename cvTile<T>::const_iterator cvTile<T>::end() const
{
	return const_iterator::end(this);
}

template <typename T>
typename cvTile<T>::iterator cvTile<T>::begin()
{
	return iterator::begin(this);
}

template <typename T>
typename cvTile<T>::iterator cvTile<T>::end()
{
	return iterator::end(this);
}

///
/// Implement this up in the headers
template<typename T>
const T& cvTile<T>::get(cv::Mat m, const int row,  const int col)
{
	//unsigned char *input = (unsigned char*)(m.data);
	//return (T&)(input[m.cols * row + col]);
	return const_cast<T&>(m.at<T>(row, col));
}

}    	//end namespace

// ew, i don't like including these files way down here
#include "cvTileIterator.hpp"

#endif /*CVTILE_HPP_*/
