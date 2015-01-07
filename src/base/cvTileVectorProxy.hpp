/*
*******************************************************************************

Copyright (c) 2015, The Curators of the University of Missouri
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************
*/


#ifndef CVcvTileVectorProxy_HPP_
#define CVcvTileVectorProxy_HPP_

#include <vector>

namespace cvt
{

#ifndef VALID_MASK_HPP_
#define VALID_MASK_HPP_

namespace valid_mask {
///
/// A typedef for the types of tile-wide valid masks that we can pull back
///
typedef enum { ALL, ANY, MAJORITY } Type;
}

#endif /* VALID_MASK_HPP_ */

template <typename T>
class cvTile;				// forward declaration

template <typename T>
class cvTileVectorProxy
{
	public:
		typedef T value_type;
	
	private:
		cvTile<T>* const _tile_ptr;
		const int _row;
		const int _column;

		friend const cvTileVectorProxy<T> cvTile<T>::operator()(int row, int column) const;
		friend cvTileVectorProxy<T> cvTile<T>::operator()(int row, int column);
		friend bool cvTile<T>::isValidVector(const cvTileVectorProxy<T>&, valid_mask::Type) const;

		explicit cvTileVectorProxy(cvTile<T>* tile_ptr, int row, int column);
		explicit cvTileVectorProxy(const cvTile<T>* tile_ptr, int row, int column);

	public:
		T& operator[](int band);
		const T& operator[](int band) const;

		cvTileVectorProxy& operator=(const cvTileVectorProxy<T>& rhs);

		bool operator==(const cvTileVectorProxy<T>& rhs) const;

		operator const std::vector<T>() const;
};

template <typename T>
cvTileVectorProxy<T>::cvTileVectorProxy(cvTile<T>* tile_ptr, int row, int column)
		: _tile_ptr(tile_ptr), _row(row), _column(column)
{ }

// is there a way to avoid this const_cast?
template <typename T>
cvTileVectorProxy<T>::cvTileVectorProxy(const cvTile<T>* tile_ptr, int row, int column)
		: _tile_ptr(const_cast<cvTile<T>*>(tile_ptr)), _row(row), _column(column)
{ }

template <typename T>
const T& cvTileVectorProxy<T>::operator[](int band) const
{
	return (*_tile_ptr).get((*_tile_ptr)[band],_row, _column);
}

template <typename T>
T& cvTileVectorProxy<T>::operator[](int band)
{
	return const_cast<T&>(
	           static_cast<const cvTileVectorProxy&>(*this)[band]
	       );
}

template <typename T>
cvTileVectorProxy<T>& cvTileVectorProxy<T>::operator=(const cvTileVectorProxy<T>& rhs)
{
	const int bands = _tile_ptr->getBandCount();

	for (int band = 0; band < bands; ++band)
	{
		operator[](band) = rhs[band];
	}

	return *this;
}

template <typename T>
bool cvTileVectorProxy<T>::operator==(const cvTileVectorProxy<T>& rhs) const
{
	std::vector<T> vlhs = *this;
	std::vector<T> vrhs = rhs;

	if (vlhs.size() != vrhs.size()) return false;
	return std::equal(vlhs.begin(), vlhs.end(), vrhs.begin());
}

template <typename T>
cvTileVectorProxy<T>::operator const std::vector<T>() const
{
	const int bands = _tile_ptr->getBandCount();
	std::vector<T> v(bands);

	for (int band = 0; band < bands; ++band)
	{
		v[band] = (*_tile_ptr).get((*_tile_ptr)[band],_row, _column);
	}

	return v;
}

} //namespace cgi

#endif /*cvTileVectorProxy_HPP_*/
