AM_CPPFLAGS = $(OPENCV_CFLAGS) -rdynamic -I..

LINK = $(LIBTOOL) --mode=link $(CXX) -std=c++11 -o $@ $(CUDA_LDFLAGS) -use-fast-math $(CUDA_LIBS) $(OPENCV_LIBS)

noinst_LTLIBRARIES = libcvtile_base.la


libcvtile_base_la_CPPFLAGS = -DNDEBUG -std=c++11  @AM_CPPFLAGS@

libcvtile_base_la_SOURCES = Tiler.cpp

