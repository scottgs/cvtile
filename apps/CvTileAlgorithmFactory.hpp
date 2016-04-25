#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <memory>
#include <cvtile/gpu/gpu.hpp>


template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount >
class CvTileAlgorithmFactory {

	using GpuAbs    = cvt::gpu::GpuAbsoluteDifference<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount>;
	using GpuWhs    = cvt::gpu::GpuWHS<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount>;
	using GpuErode  = cvt::gpu::GpuErode<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount>;
	using GpuDilate = cvt::gpu::GpuDilate<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount>;

	public:

		CvTileAlgorithmFactory();
		~CvTileAlgorithmFactory();
		//cvt::gpu::GpuAlgorithm<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount> makeCvTileAlgorithm(boost::program_options::variables_map& gpu_alg_params);	
		std::shared_ptr<cvt::gpu::GpuAlgorithm<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount> > makeCvTileAlgorithm(boost::program_options::variables_map& gpu_alg_params,const size_t cuda_device_id);
};

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount>
CvTileAlgorithmFactory<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::CvTileAlgorithmFactory() {
	;
}

template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount>
CvTileAlgorithmFactory<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::~CvTileAlgorithmFactory() {
	;
}

// NOTE: Need a refactor on gpu id number
template< typename InputPixelType, int InputBandCount, typename OutputPixelType, int OutputBandCount>
std::shared_ptr<cvt::gpu::GpuAlgorithm<InputPixelType,InputBandCount,OutputPixelType,OutputBandCount> > 
CvTileAlgorithmFactory<InputPixelType, InputBandCount, OutputPixelType, OutputBandCount>::makeCvTileAlgorithm (boost::program_options::variables_map& gpu_alg_params, const size_t cuda_device_id) {

	std::string algorithm = gpu_alg_params["algorithm"].as<std::string>();
	size_t tile_height = gpu_alg_params["tile-width"].as<size_t>();
	size_t tile_width = gpu_alg_params["tile-height"].as<size_t>();
	//size_t cuda_device_id = gpu_alg_params["gpu-number"].as<size_t>(); // passed in parameter
	ssize_t buffer_radius = static_cast<ssize_t>(gpu_alg_params["buffer-radius"].as<size_t>());

	size_t roi_height = tile_height;
	size_t roi_width = tile_width;

	std::cout << roi_height << std::endl;

	if (boost::iequals(algorithm,"GpuErode")) {	

		size_t filter_type = gpu_alg_params["filter-type"].as<size_t>();
			
		GpuErode *erode = new GpuErode(cuda_device_id,roi_width,roi_height,buffer_radius);
		std::shared_ptr<GpuErode> erode_ptr(erode);

		if (cvt::ErrorCode::Ok != erode_ptr->initializeDevice(static_cast<cvt::gpu::windowRadiusType>(filter_type))) {
			return nullptr;
		}

		return 	erode_ptr;
	}
		else if (boost::iequals(algorithm,"GpuAbsoluteDiff")) {
		GpuAbs *abs = new GpuAbs(cuda_device_id,tile_width,tile_height);	
		std::shared_ptr<GpuAbs> abs_ptr(abs);
		if (cvt::ErrorCode::Ok != abs_ptr->initializeDevice()) {
			return nullptr;
		}

		return abs_ptr;

	}
	else if (boost::iequals(algorithm,"GpuDilate")) {
		size_t filter_type = gpu_alg_params["filter-type"].as<size_t>();
			
		GpuDilate *dilate = new GpuDilate(cuda_device_id,roi_width,roi_height,buffer_radius);	
		std::shared_ptr<GpuDilate> dilate_ptr(dilate);
		if(cvt::ErrorCode::Ok != dilate_ptr->initializeDevice(static_cast<cvt::gpu::windowRadiusType>(filter_type))) {
			return nullptr;
		}
		return dilate_ptr;

	}
	else if (boost::iequals(algorithm,"GpuWHS")) {
		
		size_t filter_type = gpu_alg_params["filter-type"].as<size_t>();
			
		GpuWhs *whs = new GpuWhs(cuda_device_id,roi_width,roi_height,buffer_radius);		
		std::shared_ptr<GpuWhs> whs_ptr(whs);	
		if( cvt::ErrorCode::Ok != whs_ptr->initializeDevice(static_cast<cvt::gpu::windowRadiusType>(filter_type))) {
			return nullptr;
		}
		return whs_ptr;
	}
	return nullptr;
};

