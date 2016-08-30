#ifndef APP_MODULE_FACTORY_
#define APP_MODULE_FACTORY_

#include <string>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>

class AppModuleFactory {

	public :
		AppModuleFactory ();
		bool getAppModuleConfig (int ac, char** av, boost::program_options::options_description& od, boost::program_options::variables_map& vm);

};

AppModuleFactory::AppModuleFactory () {
	;
}

bool AppModuleFactory::getAppModuleConfig (int ac, char** av, boost::program_options::options_description& od, boost::program_options::variables_map& vm) {
	// handle a incorrect passed command line
	boost::program_options::options_description base("Command Line Options");
	size_t t0;
	base.add_options()
		("help,h", "usage ./process_task <algorithm> <config_file> <outputfile>")
		("algorithm,a", boost::program_options::value<std::string>(), "algorithm to use")
	  ("config-file,c", boost::program_options::value<std::string>(), "configuration for algorithm")
		("output-image,o", boost::program_options::value<std::string>(), "the name and path of the outputfile")
		("threads,t", boost::program_options::value<size_t>(&t0)->default_value(1), "the number of threads to spawn upto max possible concurrent hardware threads ")
	  ;
	od.add(base);

	if (ac <= 5 || !av || !(*av)) {
		return false;
	}
		//Declare a group of options that will be
		// allowed both on command line and in
		// config file
		size_t t1, t2, t3, t4, t5;
		boost::program_options::options_description config("Configuration File Options");
		config.add_options()
 			("input-image-1", boost::program_options::value<std::string>(), "the image used for uniary and binary image processing algorithms")
   			("tile-width", boost::program_options::value<size_t>(&t1)->default_value(256), "the tile width per image")
 			("tile-height", boost::program_options::value<size_t>(&t2)->default_value(256), "the tile height per image")
			("band-depth", boost::program_options::value<size_t>(&t3)->default_value(3), "the number of bands to process")
			("buffer-radius", boost::program_options::value<size_t>(&t4)->default_value(5),"the image used with input_image_1 for binary image processing algorithms");
			;


	store(boost::program_options::parse_command_line(ac, av,base), vm);
 	boost::program_options::notify(vm);

	// GpuAbsoluteDiff is binary so it requires the second image
	// GpuWHS, GpuErode, and GpuDilate require filter-radius and filter-type
	if (vm.count("algorithm") == 0) {
		std::cout << "No algorithm selected" << std::endl;
		return false;
	}


	std::string algorithm = vm["algorithm"].as<std::string>();

	if (boost::iequals(algorithm,"GpuAbsoluteDiff")) {
		config.add_options()("input-image-2", boost::program_options::value<std::string>(), "the image used with input_image_1 for binary image processing algorithms");
	}
	else if (boost::iequals(algorithm,"GpuWHS") || boost::iequals(algorithm,"GpuErode") || boost::iequals(algorithm,"GpuDilate") ) {
		config.add_options()("filter-type", boost::program_options::value<size_t>(&t5)->default_value(0), "The structuring element type {Sqaure = 0, Circle = 1}");
	}
	else {
		throw std::runtime_error("NOT POSSIBLE ALG\n");
	}


	od.add(config);
	if (vm.count("config-file") > 0) {
		std::string config_file = vm["config-file"].as<std::string>();
		std::ifstream ifs(config_file.c_str());
    if (!ifs)
		{
			std::cout << "can not open config file: " << config_file << "\n";
			return false;
		}
		else
		{
			store(parse_config_file(ifs, config), vm);
			boost::program_options::notify(vm);
		}

		ifs.close();
	}
	else {
		return false;
	}
	return true;
}

#endif
