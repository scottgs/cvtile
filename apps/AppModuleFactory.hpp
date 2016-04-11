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
	base.add_options()
		("help,h", "usage ./process_task <algorithm> <config_file> <outputfile>")
		("algorithm,a", boost::program_options::value<std::string>(), "algorithm to use")	
	    ("config-file,c", boost::program_options::value<std::string>(), "configuration for algorithm")
		("output-image,o", boost::program_options::value<std::string>(), "the name and path of the outputfile")
	    ;
	od.add(base);
	
	if (ac <= 5 || !av || !(*av)) {
		return false;
	}
		//Declare a group of options that will be 
		// allowed both on command line and in
		// config file
		int t1, t2,t3,t4;
		boost::program_options::options_description config("Configuration File Options");
		config.add_options()
 			("input-image-1", boost::program_options::value<std::string>(), "the image used for uniary and binary image processing algorithms")
   			("tile-width", boost::program_options::value<int>(&t1)->default_value(256), "the tile width per image")
 			("tile-height", boost::program_options::value<int>(&t2)->default_value(256), "the tile height per image")
			("band-depth", boost::program_options::value<int>(&t3)->default_value(3), "the number of bands to process")
			("gpu-number", boost::program_options::value<int>(&t4)->default_value(0), "the GPU card to use")
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
		config.add_options()("filter-radius", boost::program_options::value<std::string>(), "the image used with input_image_1 for binary image processing algorithms");
		config.add_options()("filter-type", boost::program_options::value<std::string>(), "the image used with input_image_1 for binary image processing algorithms");
	}
	
	od.add(config);
	if (vm.count("config-file") > 0) { 
		std::string config_file = vm["config-file"].as<std::string>();
		std::ifstream ifs(config_file.c_str());
    if (!ifs)
		{
			std::cout << "can not open config file: " << config_file << "\n";
			return 0;
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
