#include <iostream>
#include <boost/program_options.hpp>
#include <memory>
#include <exception>

#include "AppModuleFactory.hpp"
#include "CvTileAlgorithmFactory.hpp"
#include <cvtile/cvtile.hpp>
#include <cvtile/gpu/gpu.hpp>

namespace po = boost::program_options;
using namespace std;

int main (int argc, char** argv) {
	AppModuleFactory appModuleFactory;
	po::variables_map algorithm_config;	
	po::options_description program_options;

	if (! appModuleFactory.getAppModuleConfig(argc,argv,program_options,algorithm_config)) {
		std::cout << program_options << std::endl;
		return 0;
	}
	

	CvTileAlgorithmFactory<short,1,short,1> factory; 
	std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,short,1> > ga = factory.makeCvTileAlgorithm(algorithm_config);

	//vm["algorithm"].as<std::string>();
	const size_t tile_width = algorithm_config["tile-wdith"].as<size_t>();
	const size_t tile_height = algorithm_config["tile-height"].as<size_t>();

	cvt::Tiler read_tiler;
	cvt::Tiler write_tiler;
	// raster size	
	cv::Size2i rSize(tile_width,tile_height);
	// tile size
	cv::Size2i tSize(tile_width,tile_height);
	// cvTile chip sizes
	read_tiler.setCvTileSize(tSize);
	write_tiler.setCvTileSize(tSize);

	if (cvt::NoError != read_tiler.open(algorithm_config["input-image-1"].as<std::string>())) {
		throw std::runtime_error("FAILED TO OPEN INPUT FILE");
	}

	std::string driver_name("GTIFF");

	if (cvt::NoError != write_tiler.create(algorithm_config["output-image"].as<std::string>(), driver_name.c_str(), rSize, 1, cvt::Depth16S)) {
		throw std::runtime_error("FAILED TO CREATE OUTPUT FILE");
	}


	cvt::cvTile<short> inputTile;
	for (auto i = 0; i < read_tiler.getCvTileCount(); ++i) {
		inputTile = read_tiler.getCvTile<short>(i, 4);
		cvt::cvTile<short> *outputTile = NULL;
		(*ga)(inputTile,(const cvt::cvTile<short> **)&outputTile);
		if (!outputTile) {
			std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
			std::cout << "HERE" <<std::endl;
			exit(1);
		}
		write_tiler.putCvTile(*outputTile,i);
	}

	write_tiler.close();
	read_tiler.close();


	return 0;
}
