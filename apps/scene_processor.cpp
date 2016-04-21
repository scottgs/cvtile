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

	//vm["algorithm"].as<std::string>();

	if (algorithm_config.count("tile-width") == 0 || algorithm_config.count("tile-height") == 0) { 
		std::cout << "NO TILE WIDTH SET OR WIDTH SET\n";
		return 0;
	}
	const size_t tile_width = algorithm_config["tile-width"].as<size_t>();
	const size_t tile_height = algorithm_config["tile-height"].as<size_t>();


	ssize_t buffer_radius = static_cast<ssize_t>(algorithm_config["buffer-radius"].as<size_t>());

	cvt::Tiler read_tiler;
	cvt::Tiler write_tiler;
	// tile size
	cv::Size2i tSize(tile_width,tile_height);
	// cvTile chip sizes
	read_tiler.setCvTileSize(tSize);
	write_tiler.setCvTileSize(tSize);

	std::cout << algorithm_config["input-image-1"].as<std::string>() << std::endl; 

	if (cvt::NoError != read_tiler.open(algorithm_config["input-image-1"].as<std::string>())) {
		throw std::runtime_error("FAILED TO OPEN INPUT FILE");
	}

	std::string driver_name("GTIFF");
	
	std::string out_file = algorithm_config["output-image"].as<std::string>();

  if(boost::filesystem::exists(out_file)) {
			boost::filesystem::remove(out_file);
	}

	if (cvt::NoError != write_tiler.create(out_file, driver_name.c_str(), tSize, 5, cvt::Depth32F)) {
		throw std::runtime_error("FAILED TO CREATE OUTPUT FILE");
	}	

  CvTileAlgorithmFactory<short,1,float,5> factory;
	std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> > ga = factory.makeCvTileAlgorithm(algorithm_config);
	cvt::cvTile<short> inputTile;

	if (algorithm_config.count("input-image-1") && algorithm_config.count("input-image-2") == 0) {

		for (auto i = 0; i < read_tiler.getCvTileCount(); ++i) {
			inputTile = read_tiler.getCvTile<short>(i, buffer_radius);
			cvt::cvTile<float> *outputTile = NULL;
			(*ga)(inputTile,(const cvt::cvTile<float> **)&outputTile);
			if (!outputTile) {
				std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
				std::cout << "HERE" <<std::endl;
				exit(1);
			}
			write_tiler.putCvTile(*outputTile,i);
		}
	}
	else {
		std::cout << algorithm_config["input-image-2"].as<std::string>() << std::endl;
		cvt::Tiler read_tiler_two;
		read_tiler_two.setCvTileSize(tSize);
		cvt::cvTile<short> inputTileTwo;

		for (auto i = 0; i < read_tiler.getCvTileCount(); ++i) {

			inputTile = read_tiler.getCvTile<short>(i, buffer_radius);
			inputTileTwo = read_tiler_two.getCvTile<short>(i, buffer_radius);

			cvt::cvTile<float> *outputTile = NULL;
			(*ga)(inputTile,inputTileTwo,(const cvt::cvTile<float> **)&outputTile);
			if (!outputTile) {
				std::cout << "FAILURE TO GET DATA FROM DEVICE" << std::endl;
				std::cout << "HERE" <<std::endl;
				exit(1);
			}
			write_tiler.putCvTile(*outputTile,i);
		}
		read_tiler_two.close();
	}
	write_tiler.close();
	read_tiler.close();


	return 0;
}
