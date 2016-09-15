include $(top_srcdir)/cuda.mk

AM_CPPFLAGS = -I..

LINK = $(LIBTOOL) --mode=link $(CXX) -o  $@ $(CUDA_LDFLAGS) -use-fast-math $(CUDA_LIBS)

noinst_LTLIBRARIES = libcvtile_gpu_kernels.la

libcvtile_gpu_kernels_la_SOURCES = testing.cu GpuAlgorithmKernels.cu
libcvtile_gpu_kernels_la_CPPFLAGS = $(CUDA_CFLAGS) -DNDEBUG @AM_CPPFLAGS@


