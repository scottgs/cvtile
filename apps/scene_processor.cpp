#include <iostream>
#include <boost/program_options.hpp>
#include <memory>

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
	}

	CvTileAlgorithmFactory<short,1,short,1> factory(); 
	//std::unique_ptr<cvt::gpu::GpuAlgorithm<short,1,short,1>> ga = factory.makeCvTileAlgorithm(algorithm_config);

	//vm["algorithm"].as<std::string>();
	const size_t tile_width = algorithm_config["tile-wdith"].as<size_t>();
	const size_t tile_height = algorithm_config["tile-height"].as<size_t>();

	cvt::Tiler read_tiler;
	cv::Size2i tSize(tile_width,tile_height);
	read_tiler.setCvTileSize(tSize);
	read_tiler.open(algorithm_config["input-image-1"].as<std::string>());

	cvt::cvTile<short> inputTile;
	for (auto i = 0; i < read_tiler.getCvTileCount(); ++i) {
		inputTile = read_tiler.getCvTile<short>(i, 4);
	}



	return 0;
}
