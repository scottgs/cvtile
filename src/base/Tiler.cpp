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


#include "Tiler.hpp"

#include <string>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/regex.hpp>
#include <boost/thread/mutex.hpp>
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <vrtdataset.h>

namespace cvt
{

boost::mutex Tiler::io_mutex;

Tiler::Tiler() : dataset(NULL),cvTileSize(0, 0)
{
	if (GDALGetDriverCount() == 0) GDALAllRegister();
}

Tiler::~Tiler()
{
	// Relying on deconstructor to automatically close Tilers results in seg fault. For now, Tiler objects must be 
	// closed manually.
	//close();
}

void Tiler::close()
{
	// close dataset if it's already open
	if (dataset != NULL)
	{
		dataset->FlushCache();
		GDALClose(dataset);
		dataset = NULL;
	}
}

// ////////////////////////////////////////////////

ErrorType Tiler::open(const std::string& filename, IOType mode)
{
	GDALAccess gdalAccessMode;

	switch (mode)
	{
		case Update:
			gdalAccessMode = GA_Update;
			break;

		case ReadOnly:
			gdalAccessMode = GA_ReadOnly;
			break;

		default:
			return OpenError;
			break;
	}

	// In #1160, we introduced a requirement that Tiler::open be able to
	// accept NITF subdatasets as arguments.  Let's produce a regex that can
	// identify them.
	boost::regex re("^NITF_IM:\\d+:(.+)");
	boost::match_results<std::string::const_iterator> what;

	// If the file we're trying to open exists in GDAL's vsimem virtual
	// file system, bypass the existence check and allow GDALOpen to fail
	// instead if it doesn't exist
	if ((filename.substr(0, 8) != "/vsimem/")
		&& (filename.substr(0,6) != "MEM:::")
		&& (!boost::regex_match(filename, what, re))
		&& (!boost::filesystem::exists(filename)))
		return FileMissing;

	dataset = (GDALDataset *) GDALOpen(filename.c_str(), gdalAccessMode);
	if (dataset == NULL){
		//std::cout << "open(const std::string& filename, IOType mode) - dataset == NULL" << std::endl;
		return OpenError;
	}


	// if the tile size wasn't set before opening the file, read the
	// intrinsic block size from the file and use that
	if (cvTileSize.area() == 0){
		//std::cout << "in open(...)cvTileSize.area() is 0" << std::endl;
		setCvTileSize(getBlockSize());
	}

	// read the band names into our hash table
	generateBandNameMap();

	previousFilename = filename;

	return NoError;
}

void Tiler::setCvTileSize(const cv::Size2i& s)
{
	cvTileSize = s;
}

const cv::Size2i Tiler::getRasterSize() const
{
	if (dataset == NULL) return cv::Size2i(0, 0);

	return cv::Size2i(dataset->GetRasterXSize(), dataset->GetRasterYSize());
}

const cv::Size2i Tiler::getBlockSize() const
{
	if (dataset == NULL) return cv::Size2i(0, 0);

	GDALRasterBand* band = dataset->GetRasterBand(1);
	if (band == NULL) return cv::Size2i(0, 0);

	cv::Size2i blockSize(0, 0);
	band->GetBlockSize(&blockSize.width, &blockSize.height);

	return blockSize;
}

ErrorType Tiler::generateBandNameMap()
{
	if (dataset == NULL) return NullPointer;

	int nBands = getBandCount();
	for (int b = 0; b < nBands; b++)
	{
		GDALRasterBand* band = dataset->GetRasterBand(b + 1);
		if (band == NULL) return NullPointer;

		std::string bandName = std::string(band->GetDescription());
		bandNameToIndex[bandName] = b;
	}

	return NoError;
}

bool Tiler::checkTileIndex(int tileIndex) const
{
	return (tileIndex >= 0 && tileIndex < getCvTileCount());
}

bool Tiler::checkBandIndex(int bandIndex) const
{
	return (bandIndex >= 0 && bandIndex < getBandCount());
}

const cv::Point2i Tiler::getCvTileUL(int tileIndex) const
{
	if (!checkTileIndex(tileIndex)) return cv::Point2i( -1, -1);

	int ulx, uly;

	ulx = (tileIndex % getColumnCount()) *cvTileSize.width;
	uly = (tileIndex / getColumnCount()) *cvTileSize.height;

	return cv::Point2i(ulx, uly);
}

ErrorType Tiler::getBandNoDataValue(int bandIndex, double* noDataValue,
        bool* hasNoDataValue) const
{
	if (dataset == NULL) return NullPointer;
	if (!checkBandIndex(bandIndex)) return BoundsError;

	GDALRasterBand* band = dataset->GetRasterBand(bandIndex + 1);
	if (band == NULL) return NullPointer;

	int success;
	*noDataValue = band->GetNoDataValue(&success);

	if (hasNoDataValue != NULL)
	{
		switch (success)
		{
			case TRUE:
				*hasNoDataValue = true;
				break;
			case FALSE:
				*hasNoDataValue = false;
				break;
			default:
				break;
		}
	}

	return NoError;
}

int Tiler::getCvTileCount() const
{
	if (dataset == NULL ||cvTileSize.area() == 0) return 0;

	return getRowCount()*getColumnCount();
}

int Tiler::getRowCount() const
{
	if (dataset == NULL ||cvTileSize.area() == 0) return 0;

	cv::Size2i rasterSize(getRasterSize());

	return ((int) ceil((double) rasterSize.height / (double)cvTileSize.height));
}


int Tiler::getColumnCount() const
{
	if (dataset == NULL ||cvTileSize.area() == 0) return 0;

	cv::Size2i rasterSize(getRasterSize());

	return ((int) ceil((double) rasterSize.width / (double)cvTileSize.width));
}

int Tiler::getBandCount() const
{
	if (dataset == NULL){
		//std::cout << "in getBandCount() and \\dataset == NULL\\" << std::endl;
		return 0;
	}

	return dataset->GetRasterCount();
}

int Tiler::getBandIndexByName(const std::string& bandName) const
{
	std::map<std::string, int>::const_iterator iter = bandNameToIndex.find(bandName);
	if (iter == bandNameToIndex.end()) return -1;

	return iter->second;
}

ErrorType Tiler::setBandName(int bandIndex, const std::string& bandName)
{
	if (dataset == NULL) return NullPointer;
	if (!checkBandIndex(bandIndex)) return BoundsError;

	GDALRasterBand* band = dataset->GetRasterBand(bandIndex + 1);
	if (band == NULL) return NullPointer;

	// remove the old band name from the lookup table
	std::string oldBandName;
	if (getBandName(bandIndex, &oldBandName) == NoError)
		bandNameToIndex.erase(oldBandName);

	// set the new band name
	bandNameToIndex[bandName] = bandIndex;

	// If we have don't have update access to the dataset, we can't
	// set the new band name on the dataset itself, but it will persist
	// for the lifetime of the Tiler object.

	if (dataset->GetAccess() != GA_Update)
		return WriteError;

	// Otherwise, we can set the new band name on the dataset too.
	band->SetDescription(bandName.c_str());

	return NoError;
}

ErrorType Tiler::getBandName(int bandIndex, std::string* bandName) const
{
	if (dataset == NULL) return NullPointer;
	if (!checkBandIndex(bandIndex)) return BoundsError;

	GDALRasterBand* band = dataset->GetRasterBand(bandIndex + 1);
	if (band == NULL) return NullPointer;

	*bandName = std::string(band->GetDescription());

	return NoError;
}

ErrorType Tiler::getMaskTile(int tileIndex, unsigned char* bandBuffer,
                                	int tileBufferWidth) const
{
	if (dataset == NULL) return NullPointer;
	if (!checkTileIndex(tileIndex) || !checkBandIndex(0)) return BoundsError;

	GDALRasterBand* band = dataset->GetRasterBand(1)->GetMaskBand();
	if (band == NULL) return NullPointer;

	cv::Point2i srcUL(getCvTileUL(tileIndex));
	cv::Point2i dstUL(0, 0);

	//
	// constrain the size of the source region to the actual size available
	//

	if (srcUL.x < tileBufferWidth)
	{
		dstUL.x = tileBufferWidth - srcUL.x;
		srcUL.x = 0;
	}
	else
	{
		srcUL.x -= tileBufferWidth;
	}

	if (srcUL.y < tileBufferWidth)
	{
		dstUL.y = tileBufferWidth - srcUL.y;
		srcUL.y = 0;
	}
	else
	{
		srcUL.y -= tileBufferWidth;
	}

	cv::Size2i dstSize(cvTileSize);
	dstSize.width += 2 * tileBufferWidth;
	dstSize.height += 2 * tileBufferWidth;

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

	unsigned char* srcBuffer = new unsigned char[srcSize.width * srcSize.height];

	{
		boost::mutex::scoped_lock lock(io_mutex);

		if (band->RasterIO(GF_Read, srcUL.x, srcUL.y,
						   srcSize.width, srcSize.height, srcBuffer,
						   srcSize.width, srcSize.height, GDT_Byte, 0, 0)
				== CE_Failure)
		{
			return ReadError;
		}
	}

	unsigned char* dstPtr = bandBuffer + dstUL.y * dstSize.width + dstUL.x;
	unsigned char* srcPtr = srcBuffer;

	for (int i = 0; i < srcSize.height; i++)
	{
		memcpy(dstPtr, srcPtr, srcSize.width*sizeof(unsigned char));
		dstPtr += dstSize.width;
		srcPtr += srcSize.width;
	}

	delete[] srcBuffer;

	return NoError;
}

const cv::Point2i Tiler::getCvTileIndex2D(int tileIndex) const
{
	return cv::Point2i(tileIndex % getColumnCount(),
	                      tileIndex / getColumnCount());
}

ErrorType Tiler::getGeoTransform(double* geoTransform) const
{
	if (dataset == NULL) return NullPointer;

	dataset->GetGeoTransform(geoTransform);

	return NoError;
}

ErrorType Tiler::getProjectionReference(std::string& projRef) const
{
	if (dataset == NULL) return NullPointer;

	projRef = std::string(dataset->GetProjectionRef());

	return NoError;
}

ErrorType Tiler::copyMetadata(Tiler& referenceTiler)
{
	if (dataset == NULL) return NullPointer;

	double geoTransform[6];
	std::string projRef;
	ErrorType err = NoError;

	if (referenceTiler.getProjectionReference(projRef) != NoError) return ReadError;

	// non-fatal error
	if (dataset->SetProjection(projRef.c_str()) == CE_Failure) err = WriteError;

	if (referenceTiler.getGeoTransform(geoTransform) != NoError) return ReadError;

	// non-fatal error
	if (dataset->SetGeoTransform(geoTransform) == CE_Failure) err = WriteError;

	return err;
}

ErrorType Tiler::deepCopyMask(GDALRasterBand* source, GDALRasterBand* target)
{
	if (target->GetAccess() == GA_ReadOnly)
		return WriteError;

	if (GDALRasterBandCopyWholeRaster(source, target, NULL, NULL, NULL) != CE_None)
		return WriteError;

	return NoError;
}

ErrorType Tiler::bandCopySanityCheck(const Tiler& referenceTiler, const int band) const
{
	if (dataset == NULL)
		return NullPointer;

	if (dataset->GetRasterCount() == 0)
		return BoundsError;

	if (dataset->GetAccess() != GA_Update)
		return WriteError;

	GDALDataset* reference_dataset = referenceTiler.dataset;

	if (reference_dataset == NULL)
		return NullPointer;

	if (reference_dataset->GetRasterCount() <= band)
		return BoundsError;

	return NoError;
}

GDALRasterBand* Tiler::getMaskGDALRasterBand(int index)
{
	return dataset->GetRasterBand(index)->GDALRasterBand::GetMaskBand(); 
}

ErrorType Tiler::setUpDatasetMask()
{
	if(boost::lexical_cast<int>(GDALVersionInfo("VERSION_NUM")) >= 1800 && dataset->GetDriver() == GDALGetDriverByName("VRT"))
	{
		//
		// If this is a VRT dataset, we can do something special -- namely, use
		// the default implementations of CreateMaskBand and GetMaskBand to create
		// and retrieve the mask band, and then use the virtual CreateMaskBand
		// function in VRTDataset to add the MaskBand node to the VRT.
		//

		if(dataset->GDALDataset::CreateMaskBand(GMF_PER_DATASET) != CE_None)
			return CreateError;

		// Since the mask band is GMF_PER_DATASET, all the bands share the same mask
		GDALRasterBand* dataset_mask = getMaskGDALRasterBand(1);
		if(dataset_mask == NULL)
			return NullPointer;

		if(dataset->CreateMaskBand(GMF_PER_DATASET) != CE_None)
			return CreateError;

		VRTSourcedRasterBand* vrt_mask = static_cast<VRTSourcedRasterBand*>(dataset->GetRasterBand(1)->GetMaskBand());
		if(vrt_mask->AddSimpleSource(dataset_mask) != CE_None)
			return WriteError;
	}
	else if(dataset->CreateMaskBand(GMF_PER_DATASET) != CE_None)
	{
		return CreateError;
	}

	return NoError;
}

ErrorType Tiler::copyMask(const Tiler& referenceTiler)
{
	{
		ErrorType err = bandCopySanityCheck(referenceTiler, 0);
		if(NoError != err)
			return err;
		
		err = setUpDatasetMask();
		if(NoError != err)
			return err;
	}
	// If the reference dataset has per-band masks rather than a per-dataset mask, give up.
	GDALRasterBand* reference_band = referenceTiler.dataset->GetRasterBand(1);
	if (!(reference_band->GetMaskFlags() & GMF_PER_DATASET))
		return FileMissing;

	return deepCopyMask(reference_band->GetMaskBand(), getMaskGDALRasterBand());
}

const cv::Size2i Tiler::getCvTileSize() const
{
	return cvTileSize;
}

ErrorType Tiler::create(const std::string& filename, const char* driverName,
                               const cv::Size2i& rasterSize, int nBands, DepthType depth) {
	std::map<std::string, std::string> creationOptions;
	return create(filename, driverName, rasterSize, nBands, depth, creationOptions);
}

ErrorType Tiler::create(const std::string& filename, const char* driverName,
                               const cv::Size2i& rasterSize, int nBands, DepthType depth,
                               std::map<std::string, std::string> creationOptions)
{
	if (boost::filesystem::exists(filename)) return FileExists;

	// make sure there's no dataset open -- should we throw an error if
	// there is one?
	close();

	GDALDataType dataType = getGDALDataType(depth);

	// load the requested driver
	GDALDriver* driver = GetGDALDriverManager()->GetDriverByName(driverName);
	if (driver == NULL) return DriverError;

	char** driverMetadata = driver->GetMetadata();

	// ensure that the driver supports the Create() method
	if (!CSLFetchBoolean(driverMetadata, GDAL_DCAP_CREATE, FALSE)) return DriverError;

	// get the default options based on the driver type
	char** options = getDriverOptions(GDALGetDriverShortName(driver));

	// now add any more (or mod existing ones) based on what the user specified
	updateCreationOptions(options, creationOptions);

	dataset = (GDALDataset*) driver->Create(filename.c_str(), rasterSize.width,
	                                        rasterSize.height, nBands, dataType, options);

	// clean up this memory before we check for failure
	CSLDestroy(options);

	if (dataset == NULL) return CreateError;

	// if the tile size wasn't set before opening the file, read the
	// intrinsic block size from the file and use that
	if (cvTileSize.area() == 0){
		//std::cout << "in create(...)cvTileSize.area() is 0. Setting new size" << std::endl;
		setCvTileSize(getBlockSize());
	}

	// read the (default) band names into our hash table
	generateBandNameMap();

	previousFilename = filename;

	// if the caller segfaults or throws an exception, 
	// we call GDALDataset::FlushCache() just to make certain
	// that at least the complete, empty file is written to disk before we return
	// control to the caller.
	dataset->FlushCache();

	return NoError;
}

GDALDataType Tiler::getGDALDataType(DepthType _dataType)
{
	GDALDataType dataType;

	switch (_dataType)
	{
		case Depth8U:
			dataType = GDT_Byte;
			break;

		case Depth16S:
			dataType = GDT_Int16;
			break;

		case Depth16U:
			dataType = GDT_UInt16;
			break;

		case Depth32S:
			dataType = GDT_Int32;
			break;

		case Depth32U:
			dataType = GDT_UInt32;
			break;

		case Depth32F:
			dataType = GDT_Float32;
			break;

		case Depth64F:
			dataType = GDT_Float64;
			break;

		default:
			dataType = (GDALDataType) -1;
			break;
	}

	return dataType;
}

char** Tiler::getDriverOptions(const std::string& driverName)
{
	char** driverOptions = NULL;

	if ("HFA" == driverName)
	{
		// set block size to the tile size, if it's available
		if (cvTileSize.area() > 0)
		{
			// block size is always square in HFA files, so we can only
			// use one dimension of the tile size
			driverOptions = CSLSetNameValue(driverOptions, "BLOCKSIZE",
											boost::lexical_cast<std::string>(cvTileSize.width).c_str());
		}
	}
	else if ("GTiff" == driverName)
	{
		driverOptions = CSLSetNameValue(driverOptions, "TILED", "YES");

		// set block size to the tile size, if it's available
		if (cvTileSize.area() > 0)
		{
			driverOptions = CSLSetNameValue(driverOptions, "BLOCKXSIZE",
											boost::lexical_cast<std::string>(cvTileSize.width).c_str());
			driverOptions = CSLSetNameValue(driverOptions, "BLOCKYSIZE",
											boost::lexical_cast<std::string>(cvTileSize.height).c_str());
		}
	}
	else if ("NITF" == driverName)
	{
		// set block size to the tile size, if it's available
		if (cvTileSize.area() > 0)
		{
			driverOptions = CSLSetNameValue(driverOptions, "BLOCKXSIZE",
											boost::lexical_cast<std::string>(cvTileSize.width).c_str());
			driverOptions = CSLSetNameValue(driverOptions, "BLOCKYSIZE",
											boost::lexical_cast<std::string>(cvTileSize.height).c_str());
		}
	}

	return driverOptions;
}

void Tiler::updateCreationOptions(char**& options, const std::map<std::string, std::string>& optionsToAdd) {

	for(std::map<std::string, std::string>::const_iterator iter=optionsToAdd.begin(); iter!=optionsToAdd.end(); ++iter) {
		options = CSLSetNameValue(options, iter->first.c_str(), iter->second.c_str());
	}

}

ErrorType Tiler::setBandNoDataValue(int bandIndex,
        double value)
{
	if (dataset == NULL) return NullPointer;
	if (dataset->GetAccess() != GA_Update) return WriteError;
	if (!checkBandIndex(bandIndex)) return BoundsError;
	if (false == isWithinBoundsInclusive(value, getBounds(this->getGDALDataType()))) return InvalidData;

	GDALRasterBand *band = dataset->GetRasterBand(bandIndex + 1);
	if (band == NULL) return NullPointer;

	if (band->SetNoDataValue(value) == CE_Failure) return DriverError;

	return NoError;
}

GDALDataType Tiler::getGDALDataType() const {

	if (dataset == NULL) return GDT_Unknown ;

	GDALRasterBand* band = dataset->GetRasterBand(1);
	if (band == NULL) return GDT_Unknown;

	return band->GetRasterDataType();
}

const std::pair<double, double> Tiler::getBounds(GDALDataType dataType) {

	// create the bounds object and set the default
	// to be the bounds of the type double
	std::pair<double, double> bounds;
	bounds.first = boost::numeric::bounds<double>::lowest();
	bounds.second = boost::numeric::bounds<double>::highest();

	// now lookup the type and set the bounds
	switch(dataType) {

	case GDT_Byte:
		bounds.first = boost::numeric::bounds<unsigned char>::lowest();
		bounds.second = boost::numeric::bounds<unsigned char>::highest();
		break;

	case GDT_UInt16:
		bounds.first = boost::numeric::bounds<unsigned short>::lowest();
		bounds.second = boost::numeric::bounds<unsigned short>::highest();
		break;

	case GDT_Int16:
	case GDT_CInt16:
		bounds.first = boost::numeric::bounds<short>::lowest();
		bounds.second = boost::numeric::bounds<short>::highest();
		break;


	case GDT_UInt32:
		bounds.first = boost::numeric::bounds<unsigned int>::lowest();
		bounds.second = boost::numeric::bounds<unsigned int>::highest();
		break;

	case GDT_Int32:
	case GDT_CInt32:
		bounds.first = boost::numeric::bounds<int>::lowest();
		bounds.second = boost::numeric::bounds<int>::highest();
		break;

	case GDT_Float32:
	case GDT_CFloat32:
		bounds.first = boost::numeric::bounds<float>::lowest();
		bounds.second = boost::numeric::bounds<float>::highest();
		break;

	case GDT_Float64:
	case GDT_CFloat64:
		bounds.first = boost::numeric::bounds<double>::lowest();
		bounds.second = boost::numeric::bounds<double>::highest();
		break;

	default:
		break;
	}

	return bounds;
}

bool Tiler::isWithinBoundsInclusive(double value, std::pair<double, double> bounds) {

	if((value >= bounds.first) && (value <= bounds.second))
		return true;

	return false;
}

ErrorType Tiler::reopen(IOType mode)
{
	if (dataset != NULL) close();

	return open(previousFilename.c_str(), mode);
}

int Tiler::getLocalCvTileIndex(int x_index, int y_index) const
{
	// convert these into a single tile index
	int local_1d_idx = (x_index * this->getColumnCount()) + y_index;

	return local_1d_idx;
}

} // namespace cvt
