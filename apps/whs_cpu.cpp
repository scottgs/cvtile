#include <iostream>
#include <boost/program_options.hpp>
#include <memory>
#include <exception>
#include <thread>
#include <mutex>
#include <cvtile/cvtile.hpp>
#include "WindowHistogramStatistics.hpp"
#include <boost/filesystem.hpp>


std::mutex worker_mutex;

void worker(cvt::Tiler& read_tiler, cvt::Tiler& write_tiler, size_t start_tile, ssize_t buffer_radius,size_t num_workers);

int main (int argc, char **argv) {

	if (argc != 5) {
		std::cout << argv[0] << " <tile dim> <input file> <output file> <threads> " << std::endl;
		return 0;
	}

	std::string input_fname(argv[2]);
	std::string output_fname(argv[3]);
	int tile_dim = atoi(argv[1]);
	int thread_requested = atoi(argv[4]);
	ssize_t buffer_radius = 5;

	WindowHistogramStatistics whs(buffer_radius);

	cvt::Tiler read_tiler;
	cvt::Tiler write_tiler;
	// tile size
	cv::Size2i tSize(tile_dim,tile_dim);
	// cvTile chip sizes
	read_tiler.setCvTileSize(tSize);
	write_tiler.setCvTileSize(tSize);


	if (cvt::NoError != read_tiler.open(input_fname)) {
		throw std::runtime_error("FAILED TO OPEN INPUT FILE");
	}

	//std::cout << read_tiler.getCvTileCount() << std::endl;

	std::string driver_name("GTiff");
	
  if(boost::filesystem::exists(output_fname)) {
			boost::filesystem::remove(output_fname);
	}

	if (cvt::NoError != write_tiler.create(output_fname, driver_name.c_str(), read_tiler.getRasterSize(), 5, cvt::Depth32F)) {
		throw std::runtime_error("FAILED TO CREATE OUTPUT FILE");
	}	

	// threading now
	std::vector<std::thread> thread_group;
	for (size_t t = 0; t < thread_requested; ++t) {
		thread_group.push_back(std::thread(worker,std::ref(read_tiler), std::ref(write_tiler), t, buffer_radius,thread_requested));	
	}

	for (size_t t = 0; t < thread_requested; ++t) {
		thread_group[t].join();
	}
	/*cvt::cvTile<short> inputTile;
	size_t total_tiles = read_tiler.getCvTileCount();
	std::cout << total_tiles << std::endl;

	for (size_t i = 0; i < total_tiles; i++) {
			
			inputTile = read_tiler.getCvTile<short>(i, buffer_radius);
			cvt::cvTile<float> outputTile(tSize, 5);
			
			if (! whs(inputTile,outputTile)) {
				std::cout << "SOMETHING WENT WRONG" << std::endl;
				return 0;
			}

			write_tiler.putCvTile(outputTile,i);

	}*/
	
	read_tiler.close();
	write_tiler.close();

	return 0;

}

void worker(cvt::Tiler& read_tiler, cvt::Tiler& write_tiler, size_t start_tile, ssize_t buffer_radius,size_t num_workers) {
	
	cvt::cvTile<short> inputTile;
	
	worker_mutex.lock();
	size_t total_tiles = read_tiler.getCvTileCount();
	cv::Size tSize = read_tiler.getCvTileSize();
	worker_mutex.unlock();
	
	std::cout << total_tiles << std::endl;
	std::cout << tSize << std::endl;

	WindowHistogramStatistics whs(buffer_radius);


	for (size_t i = start_tile; i < total_tiles; i += num_workers) {

			worker_mutex.lock();	
			inputTile = read_tiler.getCvTile<short>(i, buffer_radius);
			worker_mutex.unlock();
			cvt::cvTile<float> outputTile(tSize, 5);
			
			if (! whs(inputTile,outputTile)) {
				std::cout << "SOMETHING WENT WRONG" << std::endl;
				return;
			}
			worker_mutex.lock();
			write_tiler.putCvTile(outputTile,i);
			worker_mutex.unlock();

	}
	return;
}
