bin_PROGRAMS = scene_processor whs_cpu

AM_CXXFLAGS = -I.. -std=c++0x $(CUDA_CFLAGS) -g  -pthread -pg

scene_processor_SOURCES = scene_processor.cpp
#scene_processor_LDADD = $(top_builddir)/src/.libs/libcvtile.a $(CUDA_LIBS) $(CUDA_LDFLAGS) $(OPENCV_LIBS)

scene_processor_LDADD = $(top_builddir)/src/.libs/libcvtile.a  $(CUDA_LIBS) $(CUDA_LDFLAGS) $(OPENCV_LIBS) -lboost_system -lboost_program_options \
    -lgdal -lboost_regex

whs_cpu_SOURCES = whs_cpu.cpp
whs_cpu_LDADD = $(top_builddir)/src/.libs/libcvtile.a  $(CUDA_LIBS) $(CUDA_LDFLAGS) $(OPENCV_LIBS) -lboost_system -lboost_program_options -lgdal -lboost_regex
