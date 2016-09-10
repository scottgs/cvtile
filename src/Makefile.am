SUBDIRS = base algorithms gpu
include $(top_srcdir)/cuda.mk

LINK = $(LIBTOOL) --mode=link $(CC) -o  $@ $(CUDA_LDFLAGS) -use-fast-math $(CUDA_LIBS) $(OPENCV_LIBS)

AM_CPPFLAGS = -I.. -std=c++0x

LIBRARY_VERSION=0:0:0

lib_LTLIBRARIES = libcvtile.la

libcvtile_la_LDFLAGS = -version-info $(LIBRARY_VERSION) @LTLIBOBJS@
libcvtile_la_SOURCES =
libcvtile_la_LIBADD = base/libcvtile_base.la \
						  algorithms/libcvtile_algorithms.la \
						  gpu/libcvtile_gpu.la


cvtiledir = $(includedir)/cvtile
cvtile_HEADERS = cvtile.hpp

basedir = ${includedir}/cvtile/base
base_HEADERS = base/cvTile.hpp \
				base/cvTileIterator.hpp \
				base/cvTileVectorProxy.hpp \
				base/Tiler.hpp \
				base/cvTileGdalExtensions.hpp

algorithmsdir = ${includedir}/cvtile/algorithms
algorithms_HEADERS = algorithms/algorithms.hpp \
				algorithms/dmp.hpp


gpudir = ${includedir}/cvtile/gpu
gpu_HEADERS = gpu/gpu.hpp


gpudriversdir = ${includedir}/cvtile/gpu/drivers
gpudrivers_HEADERS = gpu/drivers/GPUProperties.hpp \
			gpu/drivers/GpuAlgorithm.hpp \
			gpu/drivers/GpuWindowFilterAlgorithm.hpp \
			gpu/drivers/GpuWHS.hpp \
			gpu/drivers/GpuErode.hpp \
			gpu/drivers/GpuDilate.hpp \
			gpu/drivers/GpuBinaryImageAlgorithm.hpp \
			gpu/drivers/GpuAbsoluteDifference.hpp

gpukernelsdir = ${includedir}/cvtile/gpu/kernels
gpukernels_HEADERS = gpu/kernels/GpuAlgorithmKernels.hpp

