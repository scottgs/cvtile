AC_PREREQ(2.63)

AC_INIT([libcvtile], [1.0], [grantscott@missouri.edu])
AC_CONFIG_SRCDIR([src/base/cvTile.hpp])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE


# Checks for programs.
AC_PROG_CC([clang gcc])
AM_PROG_CC_C_O
AC_PROG_CPP([clang++ g++])
AC_PROG_CXX([clang++ g++])
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_PROG_LN_S
AC_PROG_MAKE_SET

AC_CONFIG_FILES([Makefile
		src/Makefile
		src/base/Makefile
		src/algorithms/Makefile
		src/gpu/Makefile
		src/gpu/drivers/Makefile
		src/gpu/kernels/Makefile
		tests/Makefile
		apps/Makefile
		])

#find out what version we are running
ARCH=`uname -m`
if [[ $ARCH == "x86_64" ]];
then
  SUFFIX="64"
else
  SUFFIX=""
fi

# -- These are capture as CUDA_CFLAGS,
# AM_CPPFLAGS="-I/opt/cuda/sdk/shared/inc/ -I/opt/cuda/sdk/C/common/inc/"
#   CUDART 5 = /opt/cuda/sdk/common/inc/
# Setup CUDA paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([cuda],
   [  --with-cuda=PATH    prefix where cuda is installed [default=/opt/cuda/]])

if test -n "$with_cuda"
then
   CUDA_CFLAGS="-I$with_cuda/include -I$with_cuda/sdk/C/common/inc/ -I$with_cuda/sdk/shared/inc/ -I$with_cuda/sdk/common/inc/"
   CUDA_LIBS="-lcudart"
   CUDA_LDFLAGS="-L$with_cuda/lib$SUFFIX"
   NVCC="$with_cuda/bin/nvcc"
else
   CUDA_CFLAGS="-I/opt/cuda/include -I/opt/cuda/sdk/C/common/inc/ -I/opt/cuda/sdk/shared/inc/ -I/opt/cuda/sdk/common/inc/"
   CUDA_LIBS="-lcudart"
   CUDA_LDFLAGS="-L/opt/cuda/lib$SUFFIX"
   NVCC="nvcc"
fi
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(NVCC)

# Global CPP Flags
AM_CPPFLAGS="-std=c++11 -Werror -Wall -Wextra -O3 -ffast-math -msse3 -march=native -mtune=native"
AC_SUBST([AM_CPPFLAGS])
CFLAGS=" "
AC_SUBST([CFLAGS])

# Setup OPENCV paths using PackageConfig, e.g., opencv.pc
# ------------------------------------------------------------------------------
PKG_CHECK_MODULES([OPENCV], opencv >= 2.3)
AC_SUBST(OPENCV_CFLAGS)
AC_SUBST(OPENCV_LIBS)

AC_PATH_PROG(CXXTEST, [cxxtestgen.pl])
if test -z "$CXXTEST"; then AC_MSG_ERROR([cxxtest not found]); fi

AC_LANG([C++])

AC_CHECK_HEADER([boost/bimap.hpp], [], AC_MSG_ERROR([boost/bimap.hpp not found]))
AC_CHECK_HEADER([boost/bind.hpp], [], AC_MSG_ERROR([boost/bind.hpp not found]))
AC_CHECK_HEADER([boost/iterator/iterator_facade.hpp], [], AC_MSG_ERROR([boost/iterator/iterator_facade.hpp not found]))
AC_CHECK_HEADER([boost/lexical_cast.hpp], [], AC_MSG_ERROR([boost/lexical_cast.hpp not found]))
AC_CHECK_HEADER([boost/mpl/and.hpp], [], AC_MSG_ERROR([boost/mpl/and.hpp not found]))
AC_CHECK_HEADER([boost/numeric/ublas/matrix.hpp], [], AC_MSG_ERROR([boost/numeric/ublas/matrix.hpp not found]))
AC_CHECK_HEADER([cxxtest/TestSuite.h], [], AC_MSG_ERROR([cxxtest/TestSuite.h not found]))
AC_CHECK_HEADER([gdal_priv.h], [], AC_MSG_ERROR([gdal_priv.h not found]))
AC_CHECK_HEADER([ogr_spatialref.h], [], AC_MSG_ERROR([ogr_spatialref.h not found]))

AC_CHECK_LIB([boost_filesystem-mt], [main], [], AC_MSG_ERROR([libboost_filesystem-mt not found]))
AC_CHECK_LIB([boost_thread-mt], [main], [], AC_MSG_ERROR([libboost_thread-mt not found]))

AC_CHECK_LIB([gdal], [main], [], AC_MSG_ERROR([libgdal not found]))

AC_SUBST([AM_CXXFLAGS])

AC_OUTPUT
