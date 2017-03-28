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


#ifndef Tiler_H_
#define Tiler_H_

#include <boost/lexical_cast.hpp>
#include <boost/thread/mutex.hpp>
#include <gdal/vrtdataset.h>
#include <gdal/ogr_spatialref.h>
#include <gdal/gdal_priv.h>
#include <sstream>
#include "cvTile.hpp"
#include "cvTileGdalExtensions.hpp"

namespace cvt
{

// ///////////////////////////////////////////////

/** @brief Method return values. */
enum ErrorType {
    /** @brief No error. */
    NoError,
    /** @brief The GDAL driver returned an unknown/unhandled error. */
    DriverError,
    /** @brief The file already exists. */
    FileExists,
    /** @brief The file is missing. */
    FileMissing,
    /** @brief The GDAL driver returned a creation error. */
    CreateError,
    /** @brief The file couldn't be opened. */
    OpenError,
    /** @brief Couldn't write to the file. */
    WriteError,
    /** @brief  Couldn't read from the file. */
    ReadError,
    /** @brief A tile index or band index exceeded the available bounds. */
    BoundsError,
    /** @brief  A null pointer was encountered. */
    NullPointer,
    /** @brief  Invalid data was provided */
    InvalidData,

};

/** @brief Supported file creation data types. */
enum DepthType {
    /** @brief 8-bit unsigned integer */
    Depth8U,
    /** @brief  16-bit signed integer */
    Depth16S,
    /** @brief  16-bit unsigned integer */
    Depth16U,
    /** @brief  32-bit signed integer */
    Depth32S,
    /** @brief 32-bit unsigned integer */
    Depth32U,
    /** @brief  32-bit (single-precision) float */
    Depth32F,
    /** @brief 64-bit (double-precision) float */
    Depth64F
};

/** @brief Available I/O modes */
enum IOType {
    /** @brief Update mode. */
    Update,
    /** @brief  Read-only mode. */
    ReadOnly
};

/** @brief Supported reprojection modes. */
enum ProjectionType {
    /** @brief Lat/long reprojection mode. */
    LatLong
};

class Tiler
{
	public:

		/** @brief Default constructor. */
		explicit Tiler();

		/** @brief Default destructor. */
		~Tiler();

		////////////////////
		// tiling options //
		////////////////////

		/** @brief Sets mosaic tile size.
		 *
		 *  @param s cvTile size.
		 */
		void setCvTileSize(const cv::Size2i& s);

		/** @brief Returns mosaic cvTile size. */
		const cv::Size2i getCvTileSize() const;

		////////End Tiling options/////////

		//////////////////////////
		// file access routines //
		//////////////////////////

		/** @brief  Attempts to open the given filename.
		 *    Returns an error if the open fails.
		 *
		 *  @param filename Name of file to open.
		 *  @param mode  I/O mode (ReadOnly, Update)
		 */
		ErrorType open(const std::string& filename, IOType mode = ReadOnly);

		/** @brief Attempts to create the given file.
		 *    Returns an error if the creation fails.
		 *
		 *  @param filename Name of file to create.
		 *  @param driverName GDAL driver name to use for file creation.
		 *  @param rasterSize Size of the image to create.
		 *  @param nBands  Number of bands to create.
		 *  @param depth  Data type to use for file creation.
		 */
		ErrorType create(const std::string& filename, const char* driverName,
		                       const cv::Size2i& rasterSize, int nBands, DepthType depth);

		/** @brief Attempts to create the given file.
		 *    Returns an error if the creation fails.
		 *
		 *  @param filename Name of file to create.
		 *  @param driverName GDAL driver name to use for file creation.
		 *  @param rasterSize Size of the image to create.
		 *  @param nBands  Number of bands to create.
		 *  @param depth  Data type to use for file creation.
		 *  @param creationOptions Options to use during file creation
		 */
		ErrorType create(const std::string& filename, const char* driverName,
		                       const cv::Size2i& rasterSize, int nBands, DepthType depth,
		                       std::map<std::string, std::string> creationOptions);

		/** @brief Reopens the file with the specified I/O mode.
		 *
		 *  @param mode  I/O mode (ReadOnly, Update)
		 */
		ErrorType reopen(IOType mode = ReadOnly);

		/** @brief Flushes the write cache and closes the file. */
		void close();

		///////////////////////
		// metadata routines //
		///////////////////////

		/** @brief Returns the size of the image. */
		const cv::Size2i getRasterSize() const;

		/** @brief Returns the number of cvTiles in the image. */
		int getCvTileCount() const;

		/** @brief Returns the number of cvTile rows in the image. */
		int getRowCount() const;

		/** @brief  Returns the number of cvTile columns in the image. */
		int getColumnCount() const;

		/** @brief Returns the number of bands in the image. */
		int getBandCount() const;

		/** @brief Attempts to look up a band index for the specified name.
		 *
		 *  @param bandName  Band name to look up.
		 */
		int getBandIndexByName(const std::string& bandName) const;

		/** @brief Returns the pixel-space upper-left coordinate of the cvTile.
		 *
		 *  @param cvTileIndex  Index of the cvTile.
		 */
		const cv::Point2i getCvTileUL(int cvTileIndex) const;

		/** @brief Attempts to retrieve the band no-data value.
		 *    Not all GDAL drivers support this.
		 *
		 *  @param bandIndex  Index of the band.
		 *  @param noDataValue  Pointer to double in which to store the
		 *        retrieved no-data value.
		 *  @param hasNoDataValue If not NULL, upon return will contain
		 *        true if the band had a no-data value,
		 *        false otherwise.
		 */
		ErrorType getBandNoDataValue(int bandIndex, double* noDataValue,
		                                   bool* hasNoDataValue = NULL) const;

		/** @brief Attempts to set the band name for the specified band.
		 *    Not all GDAL drivers support this.
		 *
		 *  @param bandIndex  Index of the band.
		 *  @param bandName  Band name to set.
		 */
		ErrorType setBandName(int bandIndex, const std::string& bandName);

		/** @brief Attempts to retrieve the band name for the specified band.
		 *    Not all GDAL drivers support this.
		 *
		 *  @param bandIndex  Index of the band.
		 *  @param bandName  Pointer to string in which to store the
		 *        band name.
		 */
		ErrorType getBandName(int bandIndex, std::string* bandName) const;

		/** @brief Converts a 1D cvTile index to a row-major 2D cvTile index.
		 *
		 *  @param cvTileIndex  Index of the cvTile.
		 */
		const cv::Point2i getCvTileIndex2D(int cvTileIndex) const;


		/** @brief Returns the geotransform coefficients.
		 *    The default geotransform is [0 1 0 0 0 1], if the image
		 *    has no associated map data.
		 *
		 *  @param geoTransform Array of six (6) doubles in which to
		 *        store the geotransform coefficients.
		 */
		ErrorType getGeoTransform(double* geoTransform) const;

		/** @brief Returns the projection reference, if it exists.
		 *    The projection reference is in the OpenGIS WKT format.
		 *
		 *  @param projRef  String in which to place the projection
		 *       reference.
		 */
		ErrorType getProjectionReference(std::string& projRef) const;


		/** @brief Copies (limited) metadata from the reference image.
                 *
                 *  @param referenceMosaic  Reference image from which to copy
                 *         metadata.
                 */
                ErrorType copyMetadata(Tiler& referenceTiler);

                /** @brief Copies the dataset mask from the reference image.
                 *
                 *  @param referenceMosaic  Reference image from which to copy
                 *         the dataset mask.
                 */
                ErrorType copyMask(const Tiler& referenceTiler);


		/** @brief Attempts to set the band no-data value.
		 *    Not all GDAL drivers support this.
		 *
		 *  @param bandIndex  Index of the band.
		 *  @param value   No-data value to set.
		 */
		ErrorType setBandNoDataValue(int bandIndex, double value);

		///////////////////////
		// cvTile I/O routines //
		///////////////////////

		/////////////
		// putTile //
		/////////////


		/** @brief Writes the multi-band cvTile data to the specified cvTile.
		 *    This routine will automatically convert from the cvTile
		 *    data type to the file data type, if necessary.
		 *
		 *  @param cvTile   cvTile object containing the multi-band data.
		 *  @param cvTileIndex  Destination cvTile index.
		 */
		template <typename T>
		ErrorType putCvTile(const cvTile<T>& cvTile, int cvTileIndex);

		/** @brief Writes the multi-band buffer data to the specified cvTile.
		 *    This routine will automatically convert from the buffer
		 *    data type to the file data type, if necessary.
		 *
		 *  @param cvTileBuffer  Buffer containing the multi-band data.
		 *  @param cvTileIndex  Destination cvTile index.
		 */
		template <typename T>
		ErrorType putCvTile(const T* cvTileBuffer, int cvTileIndex);

		/** @brief Writes the buffer data to the specified cvTile and band.
		 *    This routine will automatically convert from the buffer
		 *    data type to the image data type, if necessary.
		 *
		 *  @param bandBuffer  Buffer containing the data.
		 *  @param cvTileIndex  Destination cvTile index.
		 *  @param bandIndex  Destination band index.
		 */
		template <typename T>
		ErrorType putCvTile(const T* bandBuffer, int cvTileIndex,
		                        int bandIndex);

		/////////////
		// getCvTile //
		/////////////

		/** @brief Fills the buffer with data from the specified cvTile and band.
		 *    This routine will automatically convert from the image
		 *    data type to the buffer data type, if necessary. cvTile
		 *    buffering is also supported.
		 *
		 *  @param cvTileIndex  Source cvTile index.
		 *  @param bandIndex  Source band index.
		 *  @param bandBuffer  Destination buffer.
		 *  @param cvTileBufferWidth Optional cvTile buffer width.
		 */
		template <typename T>
		ErrorType getCvTile(int cvTileIndex, int bandIndex, T* bandBuffer,
								int cvTileBufferWidth = 0) const;

		/** @brief Fills the buffer with data from the specified cvTile and band.
		 *    This routine will automatically convert from the image
		 *    data type to the buffer data type, if necessary. cvTile
		 *    buffering is also supported.
		 *
		 *  @param cvTileIndex  Source cvTile index.
		 *  @param bandName  Source band name.
		 *  @param bandBuffer  Destination buffer.
		 *  @param cvTileBufferWidth Optional cvTile buffer width.
		 */
		template <typename T>
		ErrorType getCvTile(int cvTileIndex, const std::string& bandName,
								T* bandBuffer, int cvTileBufferWidth = 0) const;

		/** @brief Fills the buffer with data from the specified cvTile.
		 *    This routine will automatically convert from the image
		 *    data type to the buffer data type, if necessary. cvTile
		 *    buffering is also supported.
		 *
		 *  @param cvTileIndex  Source cvTile index.
		 *  @param tileBuffer  Destination buffer.
		 *  @param cvTileBufferWidth Optional tile buffer width.
		 */
		template <typename T>
		ErrorType getCvTile(int cvTileIndex, T* tileBuffer,
								int cvTileBufferWidth = 0) const;

		/** @brief Fills the tile object with data from the specified tile.
		 *    This routine will automatically convert from the image
		 *    data type to the tile data type, if necessary. Tile
		 *    buffering is also supported.
		 *
		 *    Note that if for all bands which contain a noDataValue
		 *    the noDataValue is the same, this value will be set
		 *    as the noDataValue in the returned cvTile.  Otherwise
		 *    (either the noDataValues are not homogenous or do
		 *    not exist for any bands) a NoDataValue will not be
		 *    set in the cvTile.
		 *
		 *  @param cvTileIndex  Source tile index.
		 *  @param cvTileBufferWidth Optional tile buffer width.
		 */
		template <typename T>
		const cvTile<T> getCvTile(int cvTileIndex,
							  int cvTileBufferWidth = 0) const;

		/** @brief Fills the tile object with data from the specified tile and bands.
		 *    This routine will automatically convert from the image
		 *    data type to the buffer data type, if necessary. cvTile
		 *    buffering is also supported.
		 *
		 *  @param cvTileIndex	Source tile index.
		 *  @param bandIndexList Vector of source band indices.
		 *  @param cvTileBufferWidth Optional tile buffer width.
		 */
		template <typename T>
		const cvTile<T> getCvTile(int cvTileIndex, const std::vector<int>& bandIndexList,
												  int cvTileBufferWidth = 0) const;

		/** @brief Fills the buffer with data from the specified tile and bands.
		 *    This routine will automatically convert from the image
		 *    data type to the buffer data type, if necessary. Tile
		 *    buffering is also supported.
		 *
		 *  @param cvTileIndex  Source tile index.
		 *  @param bandNameList Vector of source band names.
		 *  @param bandBuffer  Destination buffer.
		 *  @param cvTileBufferWidth Optional tile buffer width.
		 */
		template <typename T>
		ErrorType getCvTile(int cvTileIndex,
		                        const std::vector<std::string>& bandNameList, T* bandBuffer,
		                        int cvTileBufferWidth = 0) const;

		/** @brief Fills the buffer with data from the specified tile and bands.
		 *    This routine will automatically convert from the image
		 *    data type to the buffer data type, if necessary. Tile
		 *    buffering is also supported.
		 *
		 *  @param cvTileIndex  Source tile index.
		 *  @param bandIndexList Vector of source band indices.
		 *  @param bandBuffer  Destination buffer.
		 *  @param cvTileBufferWidth Optional tile buffer width.
		 */
		template <typename T>
		ErrorType getCvTile(int cvTileIndex,
		                        const std::vector<int>& bandIndexList, T* bandBuffer,
		                        int cvTileBufferWidth = 0) const;

		/** @brief Creates a tile with data from the internal mask file
		 *    Tile buffering is supported, and if the mosaic does not have a
		 *    mask associated with it fails with an exception
		 *
		 *  @param cvTileIndex  Source tile index.
		 *  @param tileBuffer  Destination buffer.
		 *  @param cvTileBufferWidth Optional tile buffer width.
		 */
		ErrorType getMaskTile(int cvTileIndex,
					  unsigned char* bandBuffer,
									  int cvTileBufferWidth = 0) const;

		/** @brief Fills the tile object with data from the specified tile
		 * 	  by using 2D cvTile indexing.
		 *    This routine will automatically convert from the image
		 *    data type to the tile data type, if necessary. Tile
		 *    buffering is also supported.
		 *
		 *    Note that if for all bands which contain a noDataValue
		 *    the noDataValue is the same, this value will be set
		 *    as the noDataValue in the returned cvTile.  Otherwise
		 *    (either the noDataValues are not homogenous or do
		 *    not exist for any bands) a NoDataValue will not be
		 *    set in the cvTile.
		 *
		 *  @param x_index  Source global tile index x.
		 *  @param y_index  Source global tile index y.
		 *  @param cvTileBufferWidth Optional cvTile buffer width.
		 */
		template <typename T>
		const cvTile<T> getCvTile2D(int x_index, int y_index,
												  int cvTileBufferWidth = 0) const;

		/** @brief Fills the tile object with data from the specified
		 *    cvTile and bands by using a 2D indexing.
		 *    This routine will automatically convert from the image
		 *    data type to the buffer data type, if necessary. cvTile
		 *    buffering is also supported.
		 *
		 *  @param x_index  Source cvTile index x.
		 *  @param y_index  Source cvTile index y.
		 *  @param bandIndexList Vector of source band indices.
		 *  @param cvTileBufferWidth Optional cvTile buffer width.
		 */
		template <typename T>
		const cvTile<T> getCvTile2D(int x_index, int y_index,
												  const std::vector<int>& bandIndexList,
												  int cvTileBufferWidth = 0) const;

		// ////////////////////////////////////////////////

		///
		/// Conversion operator that allows a Tiler object to be treated like a GDALDataset
		///
		operator GDALDataset*() const
		{
			return dataset;
		}

		///
		/// Conversion operator that allows a Tiler object to be treated like a const GDALDataset
		///
		operator const GDALDataset*() const
		{
			return dataset;
		}

		// ////////////////////////////////////////////////
	private:

		GDALDataset* dataset;   // GDAL dataset for the image

		// associative array for band lookups
		std::map<std::string, int> bandNameToIndex;

		cv::Size2i cvTileSize;    // tile (and block, for HFA) size

		std::string previousFilename;

		static boost::mutex io_mutex;

		bool checkTileIndex(int cvTileIndex) const;
		bool checkBandIndex(int bandIndex) const;

		template <typename T>
		void setCvTileNoDataValue(cvTile<T>& tile) const;

		template <typename T>
		void setCvTileMetadata(int cvTileIndex, int cvTileBufferWidth, cvTile<T>& tile) const;

		static const std::pair<double, double> getBounds(GDALDataType dataType);

		// ////////////////////////////////////////////////

	private:

		const cv::Size2i getBlockSize() const;

		ErrorType generateBandNameMap();

		static GDALDataType getGDALDataType(DepthType _dataType);

		char** getDriverOptions(const std::string& driverName);

		static void updateCreationOptions(char**& options, const std::map<std::string, std::string>& optionsToAdd);

		GDALDataType getGDALDataType() const;

		static bool isWithinBoundsInclusive(double value, std::pair<double, double> bounds);

		template <typename T>
		void setGeneralCvTileMetadata(int ul_x_pixel, int ul_y_pixel, int cvTileBufferWidth, cvTile<T>& tile) const;

		int getLocalCvTileIndex(int globalTileIndex_x, int globalTileIndex_y) const;

		// Metadata Management Helpers
		GDALRasterBand* getMaskGDALRasterBand(int index = 1);
		ErrorType deepCopyMask(GDALRasterBand* source, GDALRasterBand* target);
		ErrorType bandCopySanityCheck(const Tiler& referenceTiler, const int band) const;
		ErrorType setUpDatasetMask();
};

template <typename T>
ErrorType Tiler::getCvTile(int cvTileIndex, int bandIndex, T *bandBuffer,
                                int cvTileBufferWidth) const
{
	if (dataset == NULL) return NullPointer;
	if (!checkTileIndex(cvTileIndex) || !checkBandIndex(bandIndex)) return BoundsError;

	GDALRasterBand* band = dataset->GetRasterBand(bandIndex + 1);
	if (band == NULL) return NullPointer;

	cv::Point2i srcUL(getCvTileUL(cvTileIndex));
	cv::Point2i dstUL(0, 0);

	//
	// constrain the size of the source region to the actual size available
	//

	if (srcUL.x < cvTileBufferWidth)
	{
		dstUL.x = cvTileBufferWidth - srcUL.x;
		srcUL.x = 0;
	}
	else
	{
		srcUL.x -= cvTileBufferWidth;
	}

	if (srcUL.y < cvTileBufferWidth)
	{
		dstUL.y = cvTileBufferWidth - srcUL.y;
		srcUL.y = 0;
	}
	else
	{
		srcUL.y -= cvTileBufferWidth;
	}

	cv::Size2i dstSize(cvTileSize);
	dstSize.width += 2 * cvTileBufferWidth;
	dstSize.height += 2 * cvTileBufferWidth;

	cv::Size2i srcSize(dstSize);
	srcSize.width -= dstUL.x;
	srcSize.height -= dstUL.y;

	cv::Size2i rasterSize(getRasterSize());

	if (srcUL.x + srcSize.width > rasterSize.width)
	{
		srcSize.width = rasterSize.width - srcUL.x;
	}

	if (srcUL.y + srcSize.height > rasterSize.height)
	{
		srcSize.height = rasterSize.height - srcUL.y;
	}

	T* srcBuffer = new T[srcSize.width * srcSize.height];

	GDALDataType dataType = gdalext::traits::gdal_traits<T>::type_id();

	{
		boost::mutex::scoped_lock lock(io_mutex);

		if (band->RasterIO(GF_Read, srcUL.x, srcUL.y,
						   srcSize.width, srcSize.height, srcBuffer,
						   srcSize.width, srcSize.height, dataType, 0, 0)
				== CE_Failure)
		{
			return ReadError;
		}
	}

	// we'll check the file and see if it has a no-data value.
	// if it does, we blast the band buffer with it before we copy
	// our data in.

	double noDataValue = 0.0;
	bool hasNoDataValue = false;
	getBandNoDataValue(bandIndex, &noDataValue, &hasNoDataValue);

	if (hasNoDataValue)
	{
		unsigned int nDstPixels = dstSize.width * dstSize.height;
		T castNoDataValue = (T) noDataValue;

		for (unsigned int i = 0; i < nDstPixels; i++) bandBuffer[i] = castNoDataValue;
	}
	else
	{
		// if there isn't a no-data value available, fill with zeros
		memset(bandBuffer, '\0', dstSize.width*dstSize.height*sizeof(T));
	}

	T* dstPtr = bandBuffer + dstUL.y * dstSize.width + dstUL.x;
	T* srcPtr = srcBuffer;

	for (int i = 0; i < srcSize.height; i++)
	{
		memcpy(dstPtr, srcPtr, srcSize.width*sizeof(T));
		dstPtr += dstSize.width;
		srcPtr += srcSize.width;
	}

	delete[] srcBuffer;

	return NoError;
}

template <typename T>
ErrorType Tiler::getCvTile(int cvTileIndex, const std::string &bandName, T *bandBuffer,
                                int cvTileBufferWidth) const
{
	int bandIndex = getBandIndexByName(bandName);
	if (bandIndex < 0) return BoundsError;

	return getCvTile(cvTileIndex, bandIndex, bandBuffer, cvTileBufferWidth);
}

template <typename T>
const cvTile<T> Tiler::getCvTile(int cvTileIndex, int cvTileBufferWidth) const
{
	cv::Size2i cvTileObjectSize(cvTileSize.width + 2*cvTileBufferWidth,
	                                 cvTileSize.height + 2*cvTileBufferWidth);
	std::vector<T> tileBuffer(cvTileObjectSize.area() * getBandCount());

	if (getCvTile(cvTileIndex, &tileBuffer[0], cvTileBufferWidth) != NoError)
	{
		// I'm not sure how this error should be handled...
		return cvTile<T>(cvTileObjectSize, getBandCount());
	}

	cvTile<T> tile(&tileBuffer[0], cvTileObjectSize, getBandCount());

	setCvTileNoDataValue(tile);
	setCvTileMetadata(cvTileIndex, cvTileBufferWidth, tile);

	// populate band names int the tile
	for(std::map< std::string, int >::const_iterator it = this->bandNameToIndex.begin();
	    it != this->bandNameToIndex.end();
	    ++it)
	{
		tile.setBandName(it->second, it->first);
	}

	// If the dataset has a per-dataset mask, fetch the mask and set
	// it on the tile object.

	if (dataset->GetRasterBand(1)->GetMaskFlags() == GMF_PER_DATASET)
	{
		std::vector<unsigned char> mask(cvTileObjectSize.area());
		getMaskTile(cvTileIndex, &mask[0], cvTileBufferWidth);
		tile.setMask(&mask[0], cvTileObjectSize);
	}

	return tile;
}

template <typename T>
const cvTile<T> Tiler::getCvTile(int cvTileIndex, const std::vector<int>& bandIndexList,
                              int cvTileBufferWidth) const
{
	int bandCount = bandIndexList.size();

	cv::Size2i cvTileObjectSize(cvTileSize.width + 2*cvTileBufferWidth,
		                             cvTileSize.height + 2*cvTileBufferWidth);
	std::vector<T> tileBuffer(cvTileObjectSize.area() * bandCount);

	if (getCvTile(cvTileIndex, bandIndexList, &tileBuffer[0], cvTileBufferWidth) != NoError)
	{
		// I'm not sure how this error should be handled...
		return cvTile<T>(cvTileObjectSize, bandCount);
	}

	cvTile<T> tile(&tileBuffer[0], cvTileObjectSize, bandCount);
	setCvTileNoDataValue(bandIndexList, tile);
	setCvTileMetadata(cvTileIndex, cvTileBufferWidth, tile);

	// populate band names int the tile
	for(std::map< std::string, int >::const_iterator it = this->bandNameToIndex.begin();
	    it != this->bandNameToIndex.end();
	    ++it)
	{
		tile.setBandName(it->second, it->first);
	}

	// If the dataset has a per-dataset mask, fetch the mask and set
	// it on the tile object.

	if (dataset->GetRasterBand(1)->GetMaskFlags() == GMF_PER_DATASET)
	{
		std::vector<unsigned char> mask(cvTileObjectSize.area());
		getMaskTile(cvTileIndex, &mask[0], cvTileBufferWidth);
		tile.setMask(&mask[0], cvTileObjectSize);
	}

	return tile;
}

template <typename T>
ErrorType Tiler::getCvTile(int cvTileIndex, const std::vector<std::string> &bandNameList,
                          T *tileBuffer, int cvTileBufferWidth) const
{
	std::vector<int> bandIndexList;

	for (unsigned int i = 0; i < bandNameList.size(); i++)
	{
		int bandIndex = getBandIndexByName(bandNameList.at(i));
		if (bandIndex < 0) return BoundsError;

		bandIndexList.push_back(bandIndex);
	}

	return getCvTile(cvTileIndex, bandIndexList, tileBuffer, cvTileBufferWidth);
}

template <typename T>
ErrorType Tiler::getCvTile(int cvTileIndex, const std::vector<int> &bandIndexList,
                          T* tileBuffer, int cvTileBufferWidth) const
{
	int bandBufferLength =
	    (cvTileSize.width + 2 * cvTileBufferWidth) * (cvTileSize.height + 2 * cvTileBufferWidth);
	T* bandBuffer = tileBuffer;

	ErrorType err = NoError;

	for (unsigned int b = 0; b < bandIndexList.size(); b++)
	{
		err = getTile(cvTileIndex, bandIndexList.at(b), bandBuffer,
		              cvTileBufferWidth);
		if (err != NoError) break;

		bandBuffer += bandBufferLength;
	}

	return err;
}

template <typename T>
ErrorType Tiler::getCvTile(int cvTileIndex, T* tileBuffer,
                          int cvTileBufferWidth) const
{
	int bandBufferLength =
	    (cvTileSize.width + 2 * cvTileBufferWidth) * (cvTileSize.height + 2 * cvTileBufferWidth);
	T* bandBuffer = tileBuffer;

	ErrorType err = NoError;

	for (int b = 0; b < getBandCount(); b++)
	{
		err = getCvTile(cvTileIndex, b, bandBuffer, cvTileBufferWidth);
		if (err != NoError) break;

		bandBuffer += bandBufferLength;
	}

	return err;
}

template <typename T>
void Tiler::setCvTileNoDataValue(cvTile<T> &tile) const
{
	std::vector<double> values;
	// for all bands get the nodata value
	for (int bandIndex = 0; bandIndex < getBandCount(); bandIndex++)
	{
		double NDV;
		bool hasNDV;
		ErrorType err = getBandNoDataValue(bandIndex, &NDV, &hasNDV);
		if (err == NoError && hasNDV)
			values.push_back(NDV);
	}
	if (values.empty() == false) // if exist any nodata values
	{
		bool all_identical = true;
		typename std::vector<double>::iterator i = values.begin();
		double first = *i;

		while (++i < values.end())
		{
			if (first != *i)
			{
				all_identical = false;
				break;
			}
		}
		if (all_identical)
			tile.setNoDataValue(static_cast<T>(first));
	}
}

template <typename T>
void Tiler::setCvTileMetadata(int cvTileIndex, int cvTileBufferWidth, cvTile<T>& tile) const
{

	const cv::Point2i two_d_idxs = this->getCvTileIndex2D(cvTileIndex);
	const cv::Size2i tile_size = this->getCvTileSize();


	const int ul_x_pixel = two_d_idxs.x * tile_size.width;
	const int ul_y_pixel = two_d_idxs.y * tile_size.height;
	setGeneralCvTileMetadata(ul_x_pixel, ul_y_pixel, cvTileBufferWidth, tile);

	// ////////////////////////////////////////////////////////////

	// declare some temporary variables to use when we're storing
	// the global tile index
	double utm_easting = 0, utm_northing = 0;

	// ////////////////////////////////////////////////////////////

	std::vector<double> geoTransformArray(6);
	const bool hasGeoTransform = (getGeoTransform(&geoTransformArray[0]) == NoError);

	if(hasGeoTransform)
	{
		///// Translate the corners to fits this subset of the original dataset, this is kept for backwards compatability as
		////    it is based upon a restriction to the UTM coordinate system
		// top left x, top left easting
		utm_easting = (geoTransformArray[0] + two_d_idxs.x * tile_size.width * geoTransformArray[1]);
		tile.setMetadata("UL_EASTING", boost::lexical_cast<std::string>(utm_easting));

		// top left y, top left northing
		// NOTE: The element geoTransformArray[5] should be negative, so we use + instead of - in the calculation of utm_northing
		utm_northing = (geoTransformArray[3] + two_d_idxs.y * tile_size.height * geoTransformArray[5]);
		tile.setMetadata("UL_NORTHING", boost::lexical_cast<std::string>(utm_northing));

		//////////////////////////////////////////////////////////////////////////////////////////////////
		// Leaving this for backwards compatability - hudsonnj 2/22/13
			// see GDAL Specs at: http://www.gdal.org/gdal_tutorial.html, Getting Dataset Information
			// w-e pixel resolution
			tile.setMetadata("GSD", boost::lexical_cast<std::string>(geoTransformArray[1]));
		//////////////////////////////////////////////////////////////////////////////////////////////////
	}

	int isNorth = 0;
	int utm_zone = -1;
	std::string geoReferenceString;
	const bool hasProjReference = (NoError == getProjectionReference(geoReferenceString));

	if(hasProjReference)
	{
		OGRSpatialReference ogrSR( geoReferenceString.c_str() );
		utm_zone = ogrSR.GetUTMZone(&isNorth);

		if (isNorth)
			tile.setMetadata("UTM_HEMISPHERE", "N");
		else
			tile.setMetadata("UTM_HEMISPHERE", "S");

		tile.setMetadata("UTM_ZONE", boost::lexical_cast<std::string>(utm_zone));
	}

	// ////////////////////////////////////////////////////////////
}

template <typename T>
void Tiler::setGeneralCvTileMetadata(int ul_x_pixel, int ul_y_pixel, int cvTileBufferWidth, cvTile<T>& tile) const
{
	// set ROI on the tile if the buffer width is non-zero
	if (cvTileBufferWidth > 0)
	{
		//cv::Size2i bufferedTileSize = tile.getSize();
		cv::Point2i offset(cvTileBufferWidth, cvTileBufferWidth);
		cv::Rect roi(offset,this->cvTileSize);
		tile.setROI(roi);
	}

	tile.setMetadata("UL_PIXEL_Y", boost::lexical_cast<std::string>(ul_y_pixel));
	tile.setMetadata("UL_PIXEL_X", boost::lexical_cast<std::string>(ul_x_pixel));

	if(cvTileBufferWidth > 0)
	{
		tile.setMetadata("UL_ROI_PIXEL_X", boost::lexical_cast<std::string>(ul_x_pixel - cvTileBufferWidth));
		tile.setMetadata("UL_ROI_PIXEL_Y", boost::lexical_cast<std::string>(ul_y_pixel - cvTileBufferWidth));
	}

	std::vector<double> geoTransformArray(6);
	const bool hasGeoTransform = (getGeoTransform(&geoTransformArray[0]) == NoError);
	if(hasGeoTransform)
	{
		// Storing the oriented GSD, no need for adjustment
		tile.setMetadata("GSD_X", boost::lexical_cast<std::string>(geoTransformArray[1]));
		tile.setMetadata("GSD_Y", boost::lexical_cast<std::string>(geoTransformArray[5]));

		// Rotation information in the geotransform similarly requires no adjustment
		tile.setMetadata("ROTATION_X", boost::lexical_cast<std::string>(geoTransformArray[2]));
		tile.setMetadata("ROTATION_Y", boost::lexical_cast<std::string>(geoTransformArray[4]));

		//////////////////////////////////////////////////////////////////////////////////////////////////
		// This will work for finding the upper left x and y coordinates even if we're not in UTM
		{
			const double ul_x = geoTransformArray[0] + ul_x_pixel * geoTransformArray[1] + ul_y_pixel * geoTransformArray[2];
			tile.setMetadata("UL_TRANSFORM_X", boost::lexical_cast<std::string>(ul_x));
			const double ul_y = geoTransformArray[3] + ul_x_pixel * geoTransformArray[4] + ul_y_pixel * geoTransformArray[5];
			tile.setMetadata("UL_TRANSFORM_Y", boost::lexical_cast<std::string>(ul_y));
		}

		// Include the ROI version of the transform if applicable
		if(cvTileBufferWidth > 0)
		{
			const double ul_x = geoTransformArray[0] + (ul_x_pixel - cvTileBufferWidth) * geoTransformArray[1] + (ul_y_pixel - cvTileBufferWidth) * geoTransformArray[2];
			tile.setMetadata("UL_ROI_TRANSFORM_X", boost::lexical_cast<std::string>(ul_x));
			const double ul_y = geoTransformArray[3] + (ul_x_pixel - cvTileBufferWidth) * geoTransformArray[4] + (ul_y_pixel - cvTileBufferWidth) * geoTransformArray[5];
			tile.setMetadata("UL_ROI_TRANSFORM_Y", boost::lexical_cast<std::string>(ul_y));
		}

	}

	std::string geoReferenceString;
	if(NoError == getProjectionReference(geoReferenceString))
		tile.setMetadata("GEO_REFERENCE",geoReferenceString);
}


template <typename T>
ErrorType Tiler::putCvTile(const cvTile<T>& tile, int cvTileIndex)
{
	int tileBandCount = tile.getBandCount();
	cv::Rect roi = tile.getROI();
	cv::Size2i roiSize(roi.size());

	if (tileBandCount != getBandCount() ||
	        roiSize.width != cvTileSize.width ||
	        roiSize.height != cvTileSize.height ||
	        cvTileIndex < 0 ||
	        cvTileIndex >= getCvTileCount())
	{
		//std::cout << "BOUNDS ERROR!!" << std::endl;
		return BoundsError;
	}

	cv::Point2i roiOffset(roi.x, roi.y);
	ErrorType err = NoError;

	if (roiOffset.x == 0 && roiOffset.y == 0)
	{
		// easy case: the ROI is the entire tile

		for (int bandIndex = 0; bandIndex < tileBandCount; ++bandIndex)
		{
			// get a handle to the band data
			unsigned char* bandData = tile[bandIndex].data;

			T* bandDataCast = (T*)(bandData);

			// write band data to diMosaicsk using putCvTile
			err = putCvTile(&bandDataCast[0], cvTileIndex, bandIndex);
			if (err != NoError) break;
		}
	}
	else
	{
		// hard case: the ROI is *not* the entire tile
		cv::Size2i cvTileObjectSize(tile.getSize());
		int srcOffset = roiOffset.y * cvTileObjectSize.width + roiOffset.x;

		std::vector<T> roiData(roiSize.area());

		for (int bandIndex = 0; bandIndex < tileBandCount; ++bandIndex)
		{
			// get a handle to the band data
			cv::Mat bandData = tile[bandIndex];

			// this iterator will point to the beginning of each row in the ROI
			cv::MatConstIterator_<T> srcStart = bandData.begin<T>() + srcOffset;
			typename std::vector<T>::iterator dstStart = roiData.begin();

			for (int row = 0; row < roiSize.height; ++row)
			{
				std::copy(srcStart, srcStart + roiSize.width, dstStart);
				srcStart += cvTileObjectSize.width;
				dstStart += roiSize.width;
			}

			// write band data to disk using putCvTile
			err = putCvTile(&roiData[0], cvTileIndex, bandIndex);
			if (err != NoError) break;
		}
	}

	return err;
}

template <typename T>
ErrorType Tiler::putCvTile(const T* tileBuffer, int cvTileIndex)
{
	int bandBufferLength = cvTileSize.width * cvTileSize.height;
	const T* bandBuffer = tileBuffer;

	ErrorType err = NoError;

	for (int b = 0; b < getBandCount(); b++)
	{
		err = putCvTile(bandBuffer, cvTileIndex, b);
		if (err != NoError)
			break;

		bandBuffer += bandBufferLength;
	}

	return err;
}

template <typename T>
ErrorType Tiler::putCvTile(const T *bandBuffer, int cvTileIndex, int bandIndex)
{
	if (dataset == NULL) return NullPointer;
	if (dataset->GetAccess() != GA_Update)return WriteError;
	if (!checkTileIndex(cvTileIndex) || !checkBandIndex(bandIndex)) return BoundsError;

	GDALRasterBand* band = dataset->GetRasterBand(bandIndex + 1);
	if (band == NULL) return NullPointer;

	cv::Point2i tileUL(getCvTileUL(cvTileIndex));

	// constrain the size of the region to write to the actual size available
	cv::Size2i dstSize(0, 0);
	cv::Size2i rasterSize(getRasterSize());

	dstSize.width = tileUL.x + cvTileSize.width > rasterSize.width ?
	                rasterSize.width - tileUL.x : cvTileSize.width;
	dstSize.height = tileUL.y + cvTileSize.height > rasterSize.height ?
	                 rasterSize.height - tileUL.y : cvTileSize.height;

	T* dstBuffer = new T[dstSize.width * dstSize.height];
	T* dstPtr = dstBuffer;
	T* srcPtr = const_cast<T*>(bandBuffer);

	for (int i = 0; i < dstSize.height; i++)
	{
		memcpy(dstPtr, srcPtr, dstSize.width*sizeof(T));
		dstPtr += dstSize.width;
		srcPtr += cvTileSize.width;
	}

	GDALDataType dataType = gdalext::traits::gdal_traits<T>::type_id();

	ErrorType err = NoError;

	{
		boost::mutex::scoped_lock lock(io_mutex);

		if (band->RasterIO(GF_Write, tileUL.x, tileUL.y, dstSize.width,
						   dstSize.height, dstBuffer, dstSize.width,
						   dstSize.height, dataType, 0, 0) == CE_Failure)
		{
			err = WriteError;
		}
	}

	delete[] dstBuffer;
	return err;
}

template <typename T>
const cvTile<T> Tiler::getCvTile2D(int x_index, int y_index,
								int cvTileBufferWidth) const
{
	int localTileIndex = getLocalCvTileIndex(x_index, y_index);
	cvTile<T> t = this->getCvTile<T>(localTileIndex, cvTileBufferWidth);

	return t;
}

template <typename T>
const cvTile<T> Tiler::getCvTile2D(int x_index, int y_index,
                                                  const std::vector<int>& bandIndexList,
                                                  int cvTileBufferWidth) const
{
	int localTileIndex = getLocalCvTileIndex(x_index, y_index);
	cvTile<T> t = this->getCvTile<T>(localTileIndex, bandIndexList, cvTileBufferWidth);

	return t;
}

} // namespace cvt

#endif /*Tiler_H_*/
