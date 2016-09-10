bin_PROGRAMS = scene_processor

AM_CXXFLAGS = -I.. -std=c++11 $(CUDA_CFLAGS)


scene_processor_SOURCES = scene_processor.cpp
#scene_processor_LDADD = $(top_builddir)/src/.libs/libcvtile.a $(CUDA_LIBS) $(CUDA_LDFLAGS) $(OPENCV_LIBS)

scene_processor_LDADD = $(top_builddir)/src/.libs/libcvtile.a  $(CUDA_LIBS) $(CUDA_LDFLAGS) $(OPENCV_LIBS) -lboost_system -lboost_program_options -lgdal -lboost_regex -lboost_filesystem
