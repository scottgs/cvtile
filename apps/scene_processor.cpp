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

// interleaved task processing
void run_unary_task(const std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> >& gpu_algorithm, 
							cvt::Tiler& read_tiler,
							cvt::Tiler& output_tiler,
							ssize_t buffer_radius,
							size_t start_tile, 
							size_t num_workers);

void run_binary_task (const std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> >& gpu_algorithm, 
							cvt::Tiler& read_tiler,
							cvt::Tiler& read_tiler_two,
							cvt::Tiler& write_tiler,
							ssize_t buffer_radius,
							size_t start_tile, 
							size_t num_workers);

	// TODO: re-adjust the number of threads to use GPU, might want to do a CPU and GPU hybrid solution in the future



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
	
	size_t num_threads_requested = algorithm_config["threads"].as<size_t>();

	size_t max_num_concurrent_threads = std::thread::hardware_concurrency();

	// if requested threads is larger than max number of concurrent threads
	// // requested threads is max number of concurrent threads
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

	// re-adjust the number of threads to use GPU, might want to do a CPU and GPU hybrid solution in the future
	std::vector<int> gpu_ids = cvt::gpu::getGpuDeviceIds();

  CvTileAlgorithmFactory<short,1,float,5> factory;
	std::vector<std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> > > gpu_algorithms;

	std::vector<std::thread> thread_group;
	//num_threads_requested = 2;
	// create tiler processors per GPU
	for (size_t i = 0; i < gpu_ids.size() && i < num_threads_requested; ++i) {
		gpu_algorithms.push_back(factory.makeCvTileAlgorithm(algorithm_config,i + 2));
	}
	/*std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> > ga = factory.makeCvTileAlgorithm(algorithm_config);
	cvt::cvTile<short> inputTile;*/

	if (algorithm_config.count("input-image-1") && algorithm_config.count("input-image-2") == 0) {

		if (num_threads_requested > 1) {
			// spawn threads
			for (size_t i = 0; i < num_threads_requested; ++i) {
				thread_group.push_back(std::thread(run_unary_task,std::ref(gpu_algorithms[i]), std::ref(read_tiler),std::ref(write_tiler),buffer_radius,i,num_threads_requested));
				std::cout << "spawned thread" << std::endl;
			}

			// join spawned threads when task completed
			for (size_t i = 0; i < num_threads_requested; ++i) {
				thread_group[i].join();
			}

		}
		else {
				std::cout << "RUN SINGLE THREADED" << std::endl;
				run_unary_task (gpu_algorithms[0], read_tiler,write_tiler,buffer_radius,0, 1); 
		}
	}
	else {
		cvt::Tiler read_tiler_two;
		
		// greater than 1 thread or just run without threads
		if (num_threads_requested > 1) {
			// spawn threads
			for (size_t i = 0; i < num_threads_requested; ++i) {
				thread_group.push_back(std::thread(run_binary_task,std::ref(gpu_algorithms[i]), std::ref(read_tiler),std::ref(read_tiler_two),std::ref(write_tiler),std::ref(buffer_radius),i,num_threads_requested));
			}

			// join spawned threads when task completed
			for (size_t i = 0; i < num_threads_requested; ++i) {
				thread_group[i].join();
			}

		}
		else {
				// run in sequential order
				run_binary_task (gpu_algorithms[0], read_tiler,read_tiler_two,write_tiler,buffer_radius,0, 1); 
		}

		read_tiler_two.close();
	}
	write_tiler.close();
	read_tiler.close();


	return 0;
}


void run_unary_task(const std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> >& gpu_algorithm, 
							cvt::Tiler& read_tiler,
							cvt::Tiler& write_tiler,
							ssize_t buffer_radius,
							size_t start_tile, 
							size_t num_workers)
{

	cvt::cvTile<short> inputTile;
	worker_mutex.lock();
	size_t total_tiles = read_tiler.getCvTileCount();
	std::cout << "NUM TITLES " << total_tiles << std::endl;
	worker_mutex.unlock();

	for (size_t i = start_tile; i < total_tiles; i += num_workers) {
			
			worker_mutex.lock();
			inputTile = read_tiler.getCvTile<short>(i, buffer_radius);
			if (inputTile.getBandCount() == 0) {
				std::cout << "BAD TILE " << i << std::endl;
			}
			//std::cout << "I Worked: " << i << std::endl;
			worker_mutex.unlock();

			cvt::cvTile<float> *outputTile = NULL;
			(*gpu_algorithm)(inputTile,(const cvt::cvTile<float> **)&outputTile);
			if (!outputTile) {
				throw std::runtime_error("FAILED TO GET DATA FROM DEVICE");
			}
			worker_mutex.lock();
			if (cvt::NoError != write_tiler.putCvTile(*outputTile,i)) {
				std::cout << "FAILED TO WRITE TILE " << i << std::endl;
			}
			worker_mutex.unlock();
			delete outputTile;
		}
}


void run_binary_task (const std::shared_ptr<cvt::gpu::GpuAlgorithm<short,1,float,5> >& gpu_algorithm, 
							cvt::Tiler& read_tiler,
							cvt::Tiler& read_tiler_two,
							cvt::Tiler& write_tiler,
							ssize_t buffer_radius,
							size_t start_tile, 
							size_t num_workers) 
{
		cvt::cvTile<short> inputTile;
		size_t total_tiles = read_tiler.getCvTileCount();
		cvt::cvTile<short> inputTileTwo;

		for (size_t i = start_tile; i < total_tiles; i += num_workers) {
			
			worker_mutex.lock();
			inputTile = read_tiler.getCvTile<short>(i, buffer_radius);
			inputTileTwo = read_tiler_two.getCvTile<short>(i, buffer_radius);
			worker_mutex.unlock();

			cvt::cvTile<float> *outputTile = NULL;
			(*gpu_algorithm)(inputTile,inputTileTwo,(const cvt::cvTile<float> **)&outputTile);
			if (!outputTile) {
				throw std::runtime_error("FAILED TO GET DATA FROM DEVICE");
				exit(1);
			}
			worker_mutex.lock();
			write_tiler.putCvTile(*outputTile,i);
			worker_mutex.unlock();
			delete outputTile;
		}

}
