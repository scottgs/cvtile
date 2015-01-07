include $(top_srcdir)/cuda.mk

LINK = $(LIBTOOL) --mode=link $(CC) -o  $@ $(CUDA_LDFLAGS) -use-fast-math $(CUDA_LIBS) $(OPENCV_LIBS)

noinst_LTLIBRARIES = libcvtile_gpu_drivers.la

libcvtile_gpu_drivers_la_LDFLAGS = @LTLIBOBJS@ 
libcvtile_gpu_drivers_la_SOURCES = 
libcvtile_gpu_drivers_la_LIBADD = 
				
