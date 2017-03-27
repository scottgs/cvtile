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
    tests/Makefile
		])

#find out what version we are running
ARCH=`uname -m`
if [[ $ARCH == "x86_64" ]];
then
  SUFFIX="64"
else
  SUFFIX=""
fi

# Global CPP Flags
AM_CPPFLAGS="-std=c++11 -Werror -Wall -Wextra"
AC_SUBST([AM_CPPFLAGS])
CFLAGS=" "
AC_SUBST([CFLAGS])

# Setup OPENCV paths using PackageConfig, e.g., opencv.pc
# ------------------------------------------------------------------------------
PKG_CHECK_MODULES([OPENCV], opencv >= 2.3)
AC_SUBST(OPENCV_CFLAGS)
AC_SUBST(OPENCV_LIBS)

AC_PATH_PROG(CXXTEST, [cxxtestgen])
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

AC_CHECK_LIB([boost_filesystem], [main], [], AC_MSG_ERROR([libboost_filesystem not found]))
AC_CHECK_LIB([boost_thread], [main], [], AC_MSG_ERROR([libboost_thread not found]))

AC_CHECK_LIB([gdal], [main], [], AC_MSG_ERROR([libgdal not found]))

AC_SUBST([AM_CXXFLAGS])

AC_OUTPUT
