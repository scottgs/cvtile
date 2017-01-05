SUBDIRS = base algorithms

LINK = $(LIBTOOL) --mode=link $(CXX) -o $@ -use-fast-math  -lgdal $(OPENCV_LIBS)

AM_CPPFLAGS = -I.. -std=c++11

LIBRARY_VERSION=0:0:0

lib_LTLIBRARIES = libcvtile.la

libcvtile_la_LDFLAGS = -version-info $(LIBRARY_VERSION) @LTLIBOBJS@
libcvtile_la_SOURCES =
libcvtile_la_LIBADD = base/libcvtile_base.la \
						  algorithms/libcvtile_algorithms.la


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

