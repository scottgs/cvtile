include $(top_srcdir)/cuda.mk

LINK = $(LIBTOOL) --mode=link $(CXX) -o  $@ $(CUDA_LDFLAGS) -use-fast-math $(CUDA_LIBS) $(OPENCV_LIBS)

noinst_LTLIBRARIES = libcvtile_gpu_drivers.la


libcvtile_gpu_drivers_la_CPPFLAGS = $(CUDA_CFLAGS) -DNDEBUG -DBOOST_UBLAS_NDEBUG @AM_CPPFLAGS@
libcvtile_gpu_drivers_la_LDFLAGS = @LTLIBOBJS@
libcvtile_gpu_drivers_la_SOURCES = GPUProperties.cpp
libcvtile_gpu_drivers_la_LIBADD =

