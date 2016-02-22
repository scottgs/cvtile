#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <cvtile/gpu/gpu.hpp>

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class CvTileAlgorithmFactory {

	public:

		CvTileAlgorithmFactory();
		~CvTileAlgorithmFactory();
		cvt::GpuAlgorithm<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount> makeCvTileAlgorithm(boost::program_options::variables_map& gpu_alg_params);	

};

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount>

CvTileAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::CvTileAlgorithmFactory() {
	;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount>
CvTileAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>:~CvTileAlgorithmFactory() {
	;
}

// NOTE: Need a refactor on gpu id number
template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount>
cvt::GpuAlgorithm<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount> 
CvTileAlgorithm<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::makeCvTileAlgorithm (boost::program_options::variables_map& gpu_alg_params) {

	std::string algorithm = gpu_alg_params["algorithm"].as<std::string>();
	size_t tile_height = gpu_alg_params["tile-width"].as<size_t>();
	size_t tile_width = gpu_alg_params["tile-height"].as<size_t>();
	unsigned int cuda_device_id = gpu_alg_params["gpu-number"].as<std::string>();

	if (boost::iequals(algorithm,"GpuDilate")) {	

		ssize_t window_radius = gpu_alg_params["filter-radius"].as<std::string>();
		std::string filter_type = gpu_alg_params["filter-type"].as<std::string();
			
		cvt::gpu::GpuErode<InputPixelType,InputBandCount,OuputPixelType,OutputBandCount> erode(cuda_device_id,tile_width,tile_height,window_radius);
		
		return erode;
	}
	else if (boost::iequals(algorithm,"GpuAbsoluteDiff")) {
		
		return GpuAbsoluteDiff(cuda_device_id,tile_width,tile.height);
	}
	else if (aboost::iequals(algorithm,"GpuErode")) {
		ssize_t window_radius = gpu_alg_params["filter-radius"].as<std::string>();
		std::string filter_type = gpu_alg_params["filter-type"].as<std::string();
			
		cvt::gpu::GpuDilate<InputPixelType,InputBandCount,OuputPixelType,OutputBandCount> dilate(cuda_device_id,tile_width,tile_height,window_radius);
		
		return dilate
	}
	else if (aboost::iequals(algorithm,"GpuWHS")) {
		
		ssize_t window_radius = gpu_alg_params["filter-radius"].as<std::string>();
		std::string filter_type = gpu_alg_params["filter-type"].as<std::string();
			
		cvt::gpu::GpuWHS<InputPixelType,InputBandCount,OuputPixelType,OutputBandCount> whs(cuda_device_id,tile_width,tile_height,window_radius);
		
		return whs;
	}

}

