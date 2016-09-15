SUBDIRS = drivers kernels
include $(top_srcdir)/cuda.mk

LINK = $(LIBTOOL) --mode=link $(CXX) -o  $@ $(CUDA_LDFLAGS) -use-fast-math $(CUDA_LIBS) $(OPENCV_LIBS)

noinst_LTLIBRARIES = libcvtile_gpu.la

libcvtile_gpu_la_LDFLAGS = @LTLIBOBJS@
libcvtile_gpu_la_SOURCES =
libcvtile_gpu_la_LIBADD = drivers/libcvtile_gpu_drivers.la kernels/libcvtile_gpu_kernels.la


