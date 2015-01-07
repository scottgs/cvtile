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


#ifndef CVTILE_ITERATOR_HPP_
#define CVTILE_ITERATOR_HPP_

#include <boost/iterator/iterator_facade.hpp>
#include <boost/mpl/and.hpp>
#include "cvTile.hpp"

namespace cvt
{
/*	typedefs inherited from iterator_facade

   protected:
      // For use by derived classes
      typedef iterator_facade<Derived,Value,CategoryOrTraversal,Reference,Difference> iterator_facade_;

   public:

      typedef typename associated_types::value_type value_type;
      typedef Reference reference;
      typedef Difference difference_type;
      typedef typename associated_types::pointer pointer;
      typedef typename associated_types::iterator_category iterator_category;

 */

template <typename ContainerType, typename ValueType, typename ReferenceType>
class cvTileIterator :
			public boost::iterator_facade <
			cvTileIterator<ContainerType, ValueType, ReferenceType>,
			ValueType,
			boost::bidirectional_traversal_tag,
			ReferenceType,
			const cv::Point2i
			>
{
	private:
		ContainerType* _tile_ptr;
		cv::Point2i _position;

		struct _enabler { };

	public:
		explicit cvTileIterator() :
				_tile_ptr(NULL), _position(0, 0)
		{ }

		template <typename OtherContainerType, typename OtherValueType, typename OtherReferenceType>
		cvTileIterator(
		    const cvTileIterator<OtherContainerType, OtherValueType, OtherReferenceType>& rhs,
		    typename boost::enable_if<
		    boost::mpl::and_<
		    boost::is_convertible<OtherContainerType, ContainerType>,
		    boost::is_convertible<OtherValueType, ValueType>,
		    boost::is_convertible<OtherReferenceType, ReferenceType>
		    >, _enabler
		    >::type = _enabler()
		) :
				_tile_ptr(rhs._tile_ptr), _position(rhs._position)
		{ }

	private:
		explicit cvTileIterator(ContainerType* tile_ptr, const cv::Point2i& position) :
				_tile_ptr(tile_ptr), _position(position)
		{ }

	public:
		ValueType& operator[](int band) const
		{
			//return (*_tile_ptr)[band].at<ValueType>(_position.y, _position.x);
			return (*_tile_ptr).get((*_tile_ptr)[band],_position.y, _position.x);
		}

		ValueType& operator[](const std::string& name) const
		{
			//return (*_tile_ptr)[name].at<ValueType>(_position.y, _position.x);
			return (*_tile_ptr).get((*_tile_ptr)[name],_position.y, _position.x);
		}

		const cv::Point2i position() const
		{
			return _position;
		}

		static cvTileIterator begin(ContainerType* tile_ptr)
		{
			const cv::Point2i position(0, 0);
			return cvTileIterator(tile_ptr, position);
		}

		static cvTileIterator end(ContainerType* tile_ptr)
		{
			// we create an iterator pointing at the last valid location in the data,
			// and then increment it to yield the end() iterator. this method is thus
			// independent of how increment is implemented (e.g., row-major or
			// column-major).

			const cv::Point2i position(tile_ptr->getSize().width - 1,
			                                  tile_ptr->getSize().height - 1);
			return ++cvTileIterator(tile_ptr, position);
		}

		// this method isn't defined for us by iterator_facade unless random access traversal is used.
		template <typename OtherContainerType, typename OtherValueType, typename OtherReferenceType>
		const cv::Point2i operator-(const cvTileIterator<OtherContainerType, OtherValueType, OtherReferenceType>& rhs) const
		{
			return distance_to(rhs);
		}

	private:
		friend class boost::iterator_core_access;

		ReferenceType dereference() const
		{
			//return _tile_ptr->operator().at<ValueType>(_position.y, _position.x);
			return _tile_ptr->operator()(_position.y, _position.x);
		}

		template <typename, typename, typename>
		friend class cvTileIterator;

		template <typename OtherContainerType, typename OtherValueType, typename OtherReferenceType>
		bool equal(const cvTileIterator<OtherContainerType, OtherValueType, OtherReferenceType>& rhs) const
		{
			return _tile_ptr == rhs._tile_ptr && _position == rhs._position;
		}

		void increment()
		{
			++_position.x;
			if (_position.x == _tile_ptr->getSize().width)
			{
				++_position.y;
				_position.x = 0;
			}
		}

		void decrement()
		{
			--_position.x;
			if (_position.x < 0)
			{
				--_position.y;
				_position.x = _tile_ptr->getSize().width - 1;
			}
		}

		template <typename OtherContainerType, typename OtherValueType, typename OtherReferenceType>
		const cv::Point2i distance_to(const cvTileIterator<OtherContainerType, OtherValueType, OtherReferenceType>& rhs) const
		{
			return rhs._position - _position;
		}
};

} // namespace cgi

#endif /*CVTILE_ITERATOR_HPP_*/
