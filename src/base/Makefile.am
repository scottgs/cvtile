AM_CPPFLAGS = $(OPENCV_CFLAGS) -rdynamic -I..

LINK = $(LIBTOOL) --mode=link $(CC) -o  $@ $(CUDA_LDFLAGS) -use-fast-math $(CUDA_LIBS) $(OPENCV_LIBS)

noinst_LTLIBRARIES = libcvtile_base.la


libcvtile_base_la_CPPFLAGS = -DNDEBUG @AM_CPPFLAGS@

libcvtile_base_la_SOURCES = Tiler.cpp

