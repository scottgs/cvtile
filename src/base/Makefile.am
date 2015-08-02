AM_CPPFLAGS = $(OPENCV_CFLAGS) -rdynamic -I.. -std=c++0x

LINK = $(LIBTOOL) --mode=link $(CC) -std=c++0x -o  $@ $(CUDA_LDFLAGS) -use-fast-math $(CUDA_LIBS) $(OPENCV_LIBS)

noinst_LTLIBRARIES = libcvtile_base.la


libcvtile_base_la_CPPFLAGS = -DNDEBUG -std=c++0x  @AM_CPPFLAGS@

libcvtile_base_la_SOURCES = Tiler.cpp

