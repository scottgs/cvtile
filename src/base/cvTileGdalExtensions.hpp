#ifndef GDALEXT_TRAITS_GDAL_TRAITS_HPP_
#define GDALEXT_TRAITS_GDAL_TRAITS_HPP_

#include <gdal_priv.h>
#include <complex>
#include <cstdint>

namespace gdalext {
namespace traits {

///
/// This struct wraps the type_id method such that it is templated.
/// It also provides access to a "type" value which is equal to T
/// Additionally it provides a string_id() to a string that reflects the type
///
template <typename T>
struct gdal_traits;

#define GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(data_type, gdal_type, string_value) \
	template<> \
	struct gdal_traits<data_type> { \
		typedef data_type type;\
		static inline GDALDataType type_id() { return gdal_type; }\
		static const char* string_id() { return string_value; }\
	};

GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(uint8_t,GDT_Byte,"Byte");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(uint16_t,GDT_UInt16,"UInt16");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(uint32_t,GDT_UInt32,"UInt32");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(int16_t,GDT_Int16,"Int16");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(int32_t,GDT_Int32,"Int32");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(float,GDT_Float32,"Float32");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(double,GDT_Float64,"Float64");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(std::complex<int16_t>,GDT_CInt16,"CInt16");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(std::complex<int32_t>,GDT_CInt32,"CInt32");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(std::complex<float>,GDT_CFloat32,"CFloat32");
GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS(std::complex<double>,GDT_CFloat64,"CFloat64");

#undef GDALEXT_TRAITS_GDAL_TRAITS_MAKE_TRAITS

}// namespace core
}// namespace cgi

#endif // GDALEXT_TRAITS_GDAL_TRAITS_HPP_
