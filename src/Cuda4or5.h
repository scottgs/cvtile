// One header to rule them all
// This header will include the CUDART and determine the
//		appropriate CUDA SDK headers to include
//CUDA includes
#include "/opt/cuda/include/cuda_runtime_api.h"

// The CUDART API sets MACRO CUDART_VERSION

#if CUDART_VERSION >= 7000

#pragma message("Using CUDA Toolkit 7")

//#include <help_cuda.h>
#include <cuda.h>

#elif CUDART_VERSION >= 5000

#pragma message("Using CUDA Toolkit 5")

#include <helper_functions.h>

#elif CUDART_VERSION >= 4000

#pragma message("Using CUDA Toolkit 4")

#include <sdkHelper.h>
#include <shrQATest.h>
#include <shrUtils.h>

#endif
