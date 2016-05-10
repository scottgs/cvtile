#include <iostream>
#include <boost/program_options.hpp>
#include <memory>
#include <exception>
#include <thread>
#include <mutex>
#include "AppModuleFactory.hpp"
#include "CvTileAlgorithmFactory.hpp"
#include <cvtile/cvtile.hpp>
#include <cvtile/gpu/gpu.hpp>

namespace po = boost::program_options;
using namespace std;

std::mutex worker_mutex;

// interleaved task processing function
void run_unary_task( CvTileAlgorithmFactory<short,1,float,5>& factory,						
							po::variables_map& ac,
							cvt::Tiler& read_tiler,
							cvt::Tiler& output_tiler,
							ssize_t buffer_radius,
							size_t start_tile, 
							size_t num_workers,
							size_t cuda_device_id);

void run_binary_task(CvTileAlgorithmFactory<short,1,float,5>& factory,
							po::variables_map& ac,
							cvt::Tiler& read_tiler,
							cvt::Tiler& read_tiler_two,
							cvt::Tiler& write_tiler,
							ssize_t buffer_radius,
							size_t start_tile, 
							size_t num_workers,
							size_t cuda_device_id );


	// TODO: re-adjust the number of threads to use GPU, might want to do a CPU and GPU hybrid solution in the future



int main (int argc, char** argv) {
	AppModuleFactory appModuleFactory;
	po::variables_map algorithm_config;	
	po::options_description program_options;

	if (! appModuleFactory.getAppModuleConfig(argc,argv,program_options,algorithm_config)) {
		std::cout << program_options << std::endl;
		return 0;
	}
	
	


	if (algorithm_config.count("tile-width") == 0 || algorithm_config.count("tile-height") == 0) { 
		std::cout << "NO TILE WIDTH SET OR WIDTH SET\n";
		return 0;
	}
	const size_t tile_width = algorithm_config["tile-width"].as<size_t>();
	const size_t tile_height = algorithm_config["tile-height"].as<size_t>();
	
	size_t num_threads_requested = algorithm_config["threads"].as<size_t>();

	size_t max_num_concurrent_threads = std::thread::hardware_concurrency();

	// if requested threads is larger than max number of concurrent threads
	// requested threads is max number of concurrent threads
	if ( num_threads_requested > max_num_concurrent_threads) {
		num_threads_requested = max_num_concurrent_threads;
	}

	std::cout << num_threads_requested << std::endl;


	
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

	std::cout << read_tiler.getCvTileCount() << std::endl;

	std::string driver_name("GTiff");
	
	std::string out_file = algorithm_config["output-image"].as<std::string>();

  if(boost::filesystem::exists(out_file)) {
			boost::filesystem::remove(out_file);
	}

	if (cvt::NoError != write_tiler.create(out_file, driver_name.c_str(), read_tiler.getRasterSize(), 5, cvt::Depth32F)) {
		throw std::runtime_error("FAILED TO CREATE OUTPUT FILE");
	}	

  CvTileAlgorithmFactory<short,1,float,5> factory;

	std::vector<std::thread> thread_group;
	if (algorithm_config.count("input-image-1") && algorithm_config.count("input-image-2") == 0) { // UNARY CVTILE OPS
		
			// spawn threads
			for (size_t i = 0; i < num_threads_requested; ++i) {
				thread_group.emplace_back(run_unary_task,std::ref(factory),std::ref(algorithm_config), std::ref(read_tiler),std::ref(write_tiler),buffer_radius,i,num_threads_requested, i+ 2);
				std::cout << "spawned thread" << std::endl;
			}

			// join spawned threads when task completed
			for (size_t i = 0; i < num_threads_requested; ++i) {
				thread_group[i].join();
			}

	}
	else {  // BINARY CVTILE OPS
		cvt::Tiler read_tiler_two;
		read_tiler_two.setCvTileSize(tSize);

		if (cvt::NoError != read_tiler_two.open(algorithm_config["input-image-2"].as<std::string>())) {
			throw std::runtime_error("FAILED TO OPEN INPUT FILE");
		}
		// spawn threads
		for (size_t i = 0; i < num_threads_requested; ++i) {
			thread_group.emplace_back(run_binary_task,std::ref(factory),std::ref(algorithm_config), std::ref(read_tiler),std::ref(read_tiler_two),std::ref(write_tiler),buffer_radius,i,num_threads_requested, i+ 2);
		}

			// join spawned threads when task completed
		for (size_t i = 0; i < num_threads_requested; ++i) {
			thread_group[i].join();
		}
		read_tiler_two.close();
	}
	write_tiler.close();
	read_tiler.close();


	return 0;
}


void run_unary_task(CvTileAlgorithmFactory<short,1,float,5>& factory,
							po::variables_map& ac,
							cvt::Tiler& read_tiler,
							cvt::Tiler& write_tiler,
							ssize_t buffer_radius,
							size_t start_tile, 
							size_t num_workers,
							size_t cuda_device_id )
{
	/*
	 * Create instance of GPU algorithm based on requested algorithm
	 * */
	worker_mutex.lock();
	std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> > gpu_algorithm = factory.makeCvTileAlgorithm(ac,cuda_device_id);
	worker_mutex.unlock();

	cvt::cvTile<short> inputTile;

	/*
	 * How many tiles are present that can be worked up to
	 * */
	worker_mutex.lock();
	size_t total_tiles = read_tiler.getCvTileCount();
	worker_mutex.unlock();
	
	/*
	 * Process assigned tiles 
	 *  */
	for (size_t i = start_tile; i < total_tiles; i += num_workers) {
			/*
			 * Pull assigned tiles from raw image
			 * */
			worker_mutex.lock();
			inputTile = read_tiler.getCvTile<short>(i, buffer_radius);
			if (inputTile.getBandCount() == 0) {
				std::cout << "BAD TILE " << i << std::endl;
				abort();
			}
			worker_mutex.unlock();
			
			/*
			 * Run GPU Algorithm with assigned work tiles
			 * */
			cvt::cvTile<float> *outputTile = NULL;
			if ( (*gpu_algorithm)(inputTile,(const cvt::cvTile<float> **)&outputTile) != cvt::ErrorCode::Ok) {
				throw std::runtime_error("FAILED TO RUN ALG");
			}	

			if (!outputTile) {
				throw std::runtime_error("FAILED TO GET DATA FROM DEVICE");
			}

			/*
			 * Write assigned output tile
			 * */

			worker_mutex.lock();
			if (write_tiler.putCvTile(*outputTile,i) != cvt::NoError) {
				std::cout << "FAILED TO WRITE TILE " << i << std::endl;
				abort();
			}
			worker_mutex.unlock();
			delete outputTile;
		}
}


void run_binary_task(CvTileAlgorithmFactory<short,1,float,5>& factory,
							po::variables_map& ac,
							cvt::Tiler& read_tiler,
							cvt::Tiler& read_tiler_two,
							cvt::Tiler& write_tiler,
							ssize_t buffer_radius,
							size_t start_tile, 
							size_t num_workers,
							size_t cuda_device_id )

{
	/*
	 * Create instance of GPU algorithm based on requested algorithm
	 * */
	worker_mutex.lock();
	std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> > gpu_algorithm = factory.makeCvTileAlgorithm(ac,cuda_device_id);
	worker_mutex.unlock();

	cvt::cvTile<short> inputTile;
	cvt::cvTile<short> inputTileTwo;

	/*
	 * How many tiles are present that can be worked up to
	 * */
	worker_mutex.lock();
	const size_t total_tiles = read_tiler.getCvTileCount();
	worker_mutex.unlock();

	/*
	 * Process assigned tiles 
	 *  */
	for (size_t i = start_tile; i < total_tiles; i += num_workers) {
		
			/*
			 * Pull assigned tiles from raw image
			 * */
			worker_mutex.lock();
			inputTile = read_tiler.getCvTile<short>(i, buffer_radius);
			inputTileTwo = read_tiler_two.getCvTile<short>(i, buffer_radius);
			if (inputTile.getBandCount() == 0 || inputTileTwo.getBandCount() == 0 ) {
				std::cout << "BAD TILE " << i << std::endl;
				abort();
			}
			worker_mutex.unlock();

			/*
			 * Run GPU Algorithm with assigned work tiles
			 * */
			cvt::cvTile<float> *outputTile = NULL;
			if ( (*gpu_algorithm)(inputTile,inputTileTwo,(const cvt::cvTile<float> **)&outputTile) != cvt::ErrorCode::Ok) {
				throw std::runtime_error("FAILED TO RUN ALG");
			}	

			if (!outputTile) {
				throw std::runtime_error("FAILED TO GET DATA FROM DEVICE");
			}
			/*
			 * Write assigned output tile
			 * */
			worker_mutex.lock();
			if (write_tiler.putCvTile(*outputTile,i) != cvt::NoError) {
				std::cout << "FAILED TO WRITE TILE " << i << std::endl;
				abort();
			}
			worker_mutex.unlock();
			delete outputTile;
		}


}
