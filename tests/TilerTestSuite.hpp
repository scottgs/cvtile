#ifndef Tiler_TEST_SUITE_H_
#define Tiler_TEST_SUITE_H_

#include <boost/numeric/ublas/io.hpp>
#include <boost/filesystem.hpp>
#include <cxxtest/TestSuite.h>
#include <algorithm>
#include <numeric>
#include "../src/base/Tiler.hpp"
#include "TilerTestSuiteTestData.hpp"

// THIS HAS TO BE IFDEF CGI then use this, else just include cvTile.hpp
#ifdef HAVE_CGI
	#include "../src/base/cvTileConversion.hpp"
#else
	#include "../src/base/cvTile.hpp"
#endif


using namespace std;
using namespace cvt;

cv::Size2i rasterSize(5, 5);
cv::Size2i tileSize(3, 3);
cv::Size2i testImageSize(768, 768);
int bandCount = 1;
int rowCount = 2;
int columnCount = 2;
int tileCount = 4;
int nodataValue = -1;
int bufferWidth = 1;

Tiler mt;
Tiler mtw;

string inputFilename1024("mosaic-1024-test.tif");
string inputFilename("mosaic-test-5x5.ntf");
string scratchFilename("tdata.mosaic-test-5x5.ntf");
string driverName("NITF");
/// An exception for runtime file errors.

/// An exception for runtime file errors.
class file_error : public std::runtime_error
{
	public:
		explicit file_error(const std::string& message);
};

class TilerTestSuite : public CxxTest::TestSuite
{
	public:

		void setUp()
		{
			mt.setCvTileSize(tileSize);
			mt.open(inputFilename);

			// if the scratch file has been created, recopy the test data to it
			// before we run the next test
			if (NoError == mtw.open(scratchFilename, Update))
			{
				vector<int> buffer(tileSize.area()*bandCount);
				for (int i = 0; i < mt.getCvTileCount(); ++i)
				{
					mt.getCvTile(i, &buffer[0]);
					mtw.putCvTile(&buffer[0], i);
				}

				// make sure that it's available
				double readNodataValue = 0;
				bool hasNodataValue = false;
				//std::cout << "result of getBandNoDataValue() = " << mt.getBandNoDataValue(0, &readNodataValue, &hasNodataValue) << std::endl;
				if (hasNodataValue)
				{
					mtw.setBandNoDataValue(0, readNodataValue);
					//std::cout << "mt has a nodata value (" << readNodataValue << ")" << std::endl;
				}
				else
				{
					//std::cout << "Bzzzz" << std::endl;
				}

				mtw.reopen(ReadOnly);
			}
		}

		void tearDown()
		{
			mt.close();
			mtw.close();
		}

		void testLoadConverTileToCvTile()
		{
			cvt::Tiler read_tiler;
			read_tiler.open("test2.tif");
			cv::Size2i rasterSize = read_tiler.getRasterSize();
			read_tiler.setCvTileSize(rasterSize);

			cvt::cvTile<unsigned short> iTile = read_tiler.getCvTile<unsigned short>(0);

			vector<int> data((tileSize.width + 2*bufferWidth) * (tileSize.height + 2*bufferWidth));

			TS_ASSERT(NoError == mtw.getCvTile(0, &data[0], bufferWidth));
		}

		void testOpen()
		{
			Tiler m;
			m.setCvTileSize(tileSize);

			TS_ASSERT(NoError == m.open(inputFilename));

			TS_ASSERT(rasterSize == m.getRasterSize());
			TS_ASSERT(tileSize == m.getCvTileSize());
			TS_ASSERT(bandCount == m.getBandCount());
			TS_ASSERT(rowCount == m.getRowCount());
			TS_ASSERT(columnCount == m.getColumnCount());
			TS_ASSERT(tileCount == m.getCvTileCount());
		}

		void testOpenNitfSubdatasetFailure()
		{
			Tiler m;
			m.setCvTileSize(tileSize);

			// due to the way that we handle subdatasets, we expect an
			// OpenError and not a FileMissing error.
			TS_ASSERT_DIFFERS(FileMissing, m.open("NITF_IM:1:foobar.ntf"));
			TS_ASSERT_EQUALS(OpenError, m.open("NITF_IM:1:foobar.ntf"));
		}

		void testThrowOpen()
		{
			Tiler m;
			m.setCvTileSize(tileSize);

			TS_ASSERT(NoError == m.open(inputFilename));

			TS_ASSERT(rasterSize == m.getRasterSize());
			TS_ASSERT(tileSize == m.getCvTileSize());
			TS_ASSERT(bandCount == m.getBandCount());
			TS_ASSERT(rowCount == m.getRowCount());
			TS_ASSERT(columnCount == m.getColumnCount());
			TS_ASSERT(tileCount == m.getCvTileCount());
		}

		void testCreate()
		{
			mtw.setCvTileSize(tileSize);
			TS_ASSERT(NoError == mtw.create(scratchFilename, driverName.c_str(), rasterSize, bandCount, Depth32S));

			TS_ASSERT(rasterSize == mtw.getRasterSize());
			TS_ASSERT(tileSize == mtw.getCvTileSize());
			TS_ASSERT(bandCount == mtw.getBandCount());
			TS_ASSERT(rowCount == mtw.getRowCount());
			TS_ASSERT(columnCount == mtw.getColumnCount());
			TS_ASSERT(tileCount == mtw.getCvTileCount());
		}

		void testCreateFromOriginalFile()
		{
			Tiler read_tiler;
			std::string sourceFile("test1.tif");
			std::string outFile("test1-1.tif");

			TS_ASSERT(NoError == read_tiler.open(sourceFile));

			//make the size of the tile the size of the image so one tile is the whole image
			read_tiler.setCvTileSize(read_tiler.getRasterSize());

			cvTile<unsigned char> tile = read_tiler.getCvTile<unsigned char>(0);

			Tiler write_tiler;
			//set the cvTileSize for the writer image so it knows how the tiles are to be put back
			write_tiler.setCvTileSize(read_tiler.getCvTileSize());

			if(boost::filesystem::exists(outFile)) {
				boost::filesystem::remove(outFile);
			}

			TS_ASSERT(NoError == write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), read_tiler.getBandCount(), Depth8U));

			TS_ASSERT(NoError == write_tiler.putCvTile(tile, 0));

			write_tiler.close();
			read_tiler.close();
		}

		void testCreateFromOriginalFileUsing2Dindex()
		{
			Tiler read_tiler;
			std::string sourceFile("test2.tif");
			std::string outFile("test2-2.tif");

			TS_ASSERT(NoError == read_tiler.open("test2.tif"));

			//make the size of the tile the size of the whole image so one tile is the whole image
			read_tiler.setCvTileSize(read_tiler.getRasterSize());

			cvTile<unsigned char> tile = read_tiler.getCvTile2D<unsigned char>(0, 0);

			Tiler write_tiler;
			//set the cvTileSize for the writer image so it knows how the tiles are to be put back
			write_tiler.setCvTileSize(read_tiler.getCvTileSize());

			if(boost::filesystem::exists(outFile)) {
				boost::filesystem::remove(outFile);
			}

			TS_ASSERT(NoError == write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), read_tiler.getBandCount(), Depth8U));

			TS_ASSERT(NoError == write_tiler.putCvTile(tile, 0));

			write_tiler.close();
			read_tiler.close();
		}

		void testCreateFromOriginalFileUsing2DindexAnd256x256Tile()
		{
			Tiler read_tiler;
			std::string sourceFile("test2.tif");
			std::string outFile("test2-3.tif");

			TS_ASSERT(NoError == read_tiler.open(sourceFile));

			//tile Size
			const cv::Size2i sz(256,256);

			//set the cvTileSize so the image will be partitioned based on this
			read_tiler.setCvTileSize(sz);

			Tiler write_tiler;
			//set the cvTileSize for the writer image so it knows how the tiles are to be put back
			write_tiler.setCvTileSize(read_tiler.getCvTileSize());

			if(boost::filesystem::exists(outFile)) {
				boost::filesystem::remove(outFile);
			}

			TS_ASSERT(NoError == write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), read_tiler.getBandCount(), Depth8U));

			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(0, 0), 0));
			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(0, 1), 1));
			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(0, 2), 2));
			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(1, 0), 3));
			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(1, 1), 4));
			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(1, 2), 5));
			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(2, 0), 6));
			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(2, 1), 7));
			TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(2, 2), 8));

			write_tiler.close();
			read_tiler.close();
		}

		void testCreateFromOriginalFileUsing1DindexingAnd16x16Tile()
		{
			Tiler read_tiler;
			std::string sourceFile("test2.tif");
			std::string outFile("test2-4.tif");

			TS_ASSERT(NoError == read_tiler.open(sourceFile));

			//tile Size
			const cv::Size2i sz(16,16);

			//set the cvTileSize so the image will be partitioned based on this
			read_tiler.setCvTileSize(sz);

			Tiler write_tiler;
			//set the cvTileSize for the writer image so it knows how the tiles are to be put back
			write_tiler.setCvTileSize(read_tiler.getCvTileSize());

			if(boost::filesystem::exists(outFile)) {
				boost::filesystem::remove(outFile);
			}

			TS_ASSERT(NoError == write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), read_tiler.getBandCount(), Depth8U));

			//loop through all tiles and put them into the writer Tiler
			for (int i = 0; i < read_tiler.getCvTileCount(); ++i)
			{
				TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile<unsigned char>(i), i));
			}

			write_tiler.close();
			read_tiler.close();
		}

		void testRead16x16TileWithBuffer()
		{
			Tiler read_tiler;
			std::string sourceFile("test2.tif");

			TS_ASSERT(NoError == read_tiler.open(sourceFile));

			//tile Size
			const cv::Size2i sz(256,256);

			//set the cvTileSize so the image will be partitioned based on this
			read_tiler.setCvTileSize(sz);

				// READ WITH BUFFER
				// Tile 4, buffer 10
			cvt::cvTile<unsigned char> iTile = read_tiler.getCvTile<unsigned char>(4, 10);

			TS_ASSERT_EQUALS(iTile.getSize().width, 276);
			TS_ASSERT_EQUALS(iTile.getSize().width, 276);

			TS_ASSERT_EQUALS(iTile.getROI().x,10);
			TS_ASSERT_EQUALS(iTile.getROI().y,10);

			TS_ASSERT_EQUALS(iTile.getROI().size().width,256);
			TS_ASSERT_EQUALS(iTile.getROI().size().height,256);

			read_tiler.close();
		}


		// this currently fails for GTiff
		void testOpenWithImplicitTileSize()
		{
			Tiler m;
			TS_ASSERT(NoError == m.open(scratchFilename));

			TS_ASSERT(rasterSize == m.getRasterSize());
			TS_ASSERT(tileSize == m.getCvTileSize());
			TS_ASSERT(bandCount == m.getBandCount());
			TS_ASSERT(rowCount == m.getRowCount());
			TS_ASSERT(columnCount == m.getColumnCount());
			TS_ASSERT(tileCount == m.getCvTileCount());
		}

		void testReadOnly()
		{
			TS_ASSERT(WriteError == mtw.putCvTile(&tile1_nd_nb[0], 0));

			vector<int> buffer(tileSize.area()*bandCount);
			TS_ASSERT(NoError == mtw.getCvTile(0, &buffer[0]));
			TS_ASSERT(std::equal(buffer.begin(), buffer.end(), &tile1_nnd_nb[0]));
		}

		void testUpdate()
		{
			TS_ASSERT(WriteError == mtw.putCvTile(&tile1_nd_nb[0], 0));

			mtw.reopen(Update);
			TS_ASSERT(NoError == mtw.putCvTile(&tile1_nd_nb[0], 0));

			vector<int> buffer(tileSize.area()*bandCount);
			TS_ASSERT(NoError == mtw.getCvTile(0, &buffer[0]));
			TS_ASSERT(std::equal(buffer.begin(), buffer.end(), &tile1_nd_nb[0]));
		}

		void testNoNodataValue()
		{
			double readNodataValue = 0;
			bool hasNodataValue = false;

			TS_ASSERT(NoError == mt.getBandNoDataValue(0, &readNodataValue, &hasNodataValue));
			TS_ASSERT(!hasNodataValue);
		}

		void testTileGeotransformData()
		{
			Tiler m;
			std::vector<double> georef(6);
			if(NoError != m.open("mosaic-1024-test.tif"))
			{
				TS_FAIL("Unable to open \"mosaic-1024-test.tif\"");
			}
			else if(NoError != m.getGeoTransform(&georef[0]))
			{
				TS_FAIL("Unable to get projection reference from \"mosaic-1024-test.tif\"");
			}
			else
			{
				TS_ASSERT_DIFFERS(m.getCvTileCount(), 0);
				TS_ASSERT_DIFFERS(georef[1], 0.0);
				TS_ASSERT_DIFFERS(georef[5], 0.0);
				for(int t = 0; t < m.getCvTileCount(); ++t)
				{
					const cvTile<double> tile = m.getCvTile<double>(t);
					TS_ASSERT_EQUALS((georef[1]), (tile.getMetadataAs<double>("GSD_X")));
					TS_ASSERT_EQUALS((georef[2]), (tile.getMetadataAs<double>("ROTATION_X")));
					TS_ASSERT_EQUALS((georef[5]), (tile.getMetadataAs<double>("GSD_Y")));
					TS_ASSERT_EQUALS((georef[4]), (tile.getMetadataAs<double>("ROTATION_Y")));

					TS_ASSERT_EQUALS((tile.getMetadata("UL_EASTING")), (tile.getMetadata("UL_TRANSFORM_X")));
					TS_ASSERT_EQUALS((tile.getMetadata("UL_NORTHING")), (tile.getMetadata("UL_TRANSFORM_Y")));

					const cv::Point2i two_d_index = m.getCvTileIndex2D(t);
					const cv::Size2i tile_size = m.getCvTileSize();
					const int x_coord = tile_size.width * two_d_index.x;
					const int y_coord = tile_size.height * two_d_index.y;
					TS_ASSERT_EQUALS((x_coord),(tile.getMetadataAs<int>("UL_PIXEL_X")));
					TS_ASSERT_EQUALS((y_coord),(tile.getMetadataAs<int>("UL_PIXEL_Y")));

					TS_ASSERT((tile.getMetadata("UL_ROI_TRANSFORM_X").empty()));
					TS_ASSERT((tile.getMetadata("UL_ROI_TRANSFORM_Y").empty()));
					TS_ASSERT((tile.getMetadata("UL_ROI_PIXEL_X").empty()));
					TS_ASSERT((tile.getMetadata("UL_ROI_PIXEL_Y").empty()));
				}

				for(int t = 0; t < m.getCvTileCount(); ++t)
				{
					const cvt::cvTile<double> tile = m.getCvTile<double>(t,1);
					const double gsd_x = tile.getMetadataAs<double>("GSD_X");
					const double gsd_y = tile.getMetadataAs<double>("GSD_Y");

					TS_ASSERT_EQUALS((georef[1]), (gsd_x));
					TS_ASSERT_EQUALS((georef[5]), (gsd_y));

					TS_ASSERT_EQUALS((georef[2]), (tile.getMetadataAs<double>("ROTATION_X")));
					TS_ASSERT_EQUALS((georef[4]), (tile.getMetadataAs<double>("ROTATION_Y")));

					TS_ASSERT_EQUALS((tile.getMetadata("UL_EASTING")), (tile.getMetadata("UL_TRANSFORM_X")));
					TS_ASSERT_EQUALS((tile.getMetadata("UL_NORTHING")), (tile.getMetadata("UL_TRANSFORM_Y")));

					const cv::Point2i two_d_index = m.getCvTileIndex2D(t);
					const cv::Size2i tile_size = m.getCvTileSize();
					const int x_coord = tile_size.width * two_d_index.x;
					const int y_coord = tile_size.height * two_d_index.y;

					TS_ASSERT_EQUALS((x_coord),(tile.getMetadataAs<int>("UL_PIXEL_X")));
					TS_ASSERT_EQUALS((y_coord),(tile.getMetadataAs<int>("UL_PIXEL_Y")));

					TS_ASSERT_EQUALS((x_coord-1),(tile.getMetadataAs<int>("UL_ROI_PIXEL_X")));
					TS_ASSERT_EQUALS((y_coord-1),(tile.getMetadataAs<int>("UL_ROI_PIXEL_Y")));


					// TODO: Adjust this to take the rotation field of the georef into account
					if(gsd_x > 0)
					{
						TS_ASSERT_LESS_THAN((tile.getMetadataAs<double>("UL_ROI_TRANSFORM_X")),(tile.getMetadataAs<double>("UL_TRANSFORM_X")));
					}
					else if (gsd_x < 0)
					{
						TS_ASSERT_LESS_THAN((tile.getMetadataAs<double>("UL_TRANSFORM_X")),(tile.getMetadataAs<double>("UL_ROI_TRANSFORM_X")));
					}
					else
					{
						TS_FAIL("0 GSD_X value encountered"); // THIS SHOULD NEVER HAPPEN
					}

					if(gsd_y > 0)
					{
						TS_ASSERT_LESS_THAN((tile.getMetadataAs<double>("UL_ROI_TRANSFORM_Y")),(tile.getMetadataAs<double>("UL_TRANSFORM_Y")));
					}
					else if (gsd_y < 0)
					{
						TS_ASSERT_LESS_THAN((tile.getMetadataAs<double>("UL_TRANSFORM_Y")),(tile.getMetadataAs<double>("UL_ROI_TRANSFORM_Y")));
					}
					else
					{
						TS_FAIL("0 GSD_Y value encountered"); // THIS SHOULD NEVER HAPPEN
					}
				}
			}
		}

		void xtestInMemDataset()
		{
#if 0
			const cv::Size2i sz(32,24);

			std::vector<uint8_t> data(sz.width * sz.height,0);
			const std::vector<uint8_t>& cdata = data;

			Tiler m;
			TS_ASSERT_EQUALS(NoError, m.open<uint8_t>(&data[0],sz,cvt::Update));
			m.setCvTileSize(sz);

			cvTile<uint8_t> char_tile = m.getCvTile<uint8_t>(0);
			TS_ASSERT(std::equal(data.begin(), data.end(), char_tile[0].begin<uint8_t>()))
			cvTile<uint16_t> shocvTilert_tile = m.getCvTile<uint16_t>(0);
			TS_ASSERT(std::equal(data.begin(), data.end(), short_tile[0].begin<uint16_t>()))

			short_tile.set(20);

			m.putCvTile(short_tile,0);
			m.close();

			TS_ASSERT(!(std::equal(data.begin(), data.end(), char_tile[0].begin<uint8_t>())));
			TS_ASSERT_EQUALS((std::count(data.begin(), data.end(), 20)), (data.size()))

			TS_ASSERT_EQUALS(NoError, m.open(&cdata[0],sz));

			char_tile = m.getCvTile<uint8_t>(0);
			TS_ASSERT(std::equal(data.begin(), data.end(), char_tile[0].begin<uint8_t>()));
#endif
		}
};

#endif /* Tiler_TEST_SUITE_H_ */
