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



#ifndef MOSAICTESTSUITETESTDATA_H_
#define MOSAICTESTSUITETESTDATA_H_

// the following arrays are what Mosaic *should* return
// for the given tiles under the given conditions

////////////////////////////////
// no nodata value, no buffer //
////////////////////////////////

// 		. . .
// 		. x x
//		x x x
int tile1_nnd_nb[] = {	0, 		0, 		0,
					  	0, 		255, 	255,
					  	255, 	255, 	255		};

//		x x 0
//		x . 0
//		x . 0
int tile2_nnd_nb[] = {	255,	255,	0,
					  	255,	0,		0,
					  	255,	0,		0		};

//		. x x
//		. . x
//		0 0 0
int tile3_nnd_nb[] = {	0,		255,	255,
		              	0,		0, 		255,
		              	0, 		0, 		0};

//		. . 0
//		. . 0
//		0 0 0
int tile4_nnd_nb[] = {	0,		0,		0,
						0,		0,		0,
						0,		0,		0		};

/////////////////////////////////////
// no nodata value, 1-pixel buffer //
/////////////////////////////////////

//		0 0 0 0 0
// 		0 . . . x
// 		0 . x x x
//	 	0 x x x x
// 		0 . x x .
int tile1_nnd_1b[] = {	0,		0,		0,		0,		0,
						0,		0,		0,		0,		255,
						0,		0,		255,	255,	255,
						0,		255,	255,	255,	255,
						0,		0,		255,	255,	0		};

//		0 0 0 0 0
// 		. x x 0 0
// 		x x . 0 0
//	 	x x . 0 0
// 		x . . 0 0
int tile2_nnd_1b[] = {	0,		0,		0,		0,		0,
						0,		255,	255,	0,		0,
						255,	255,	0,		0,		0,
						255,	255,	0,		0,		0,
						255,	0,		0,		0,		0		};

//		0 x x x x
//		0 . x x .
//		0 . . x .
//		0 0 0 0 0
//		0 0 0 0 0
int tile3_nnd_1b[] = {	0, 		255,	255,	255,	255,
						0,		0,		255,	255,	0,
						0,		0,		0,		255,	0,	
						0,		0,		0,		0,		0,
						0,		0,		0,		0,		0		};

//      x x . 0 0
//		x . . 0 0 
//		x . . 0 0
//		0 0 0 0 0
//      0 0 0 0 0
int tile4_nnd_1b[] = {	255,	255,	0,		0,		0,
						255,	0,		0,		0,		0,
						255,	0,		0,		0,		0,
						0,		0,		0,		0,		0,
						0,		0,		0,		0,		0		};

/////////////////////////////
// nodata value, no buffer //
/////////////////////////////

// 		. . .
// 		. x x
//		x x x
int tile1_nd_nb[] = {	0, 		0, 		0,
					  	0, 		255, 	255,
					  	255, 	255, 	255		};

//		x x N
//		x . N
//		x . N
int tile2_nd_nb[] = {	255,	255,	-1,
					  	255,	0,		-1,
					  	255,	0,		-1		};

//		. x x
//		. . x
//		N N N
int tile3_nd_nb[] = {	0,		255,	255,
		              	0,		0, 		255,
		              	-1, 	-1, 	-1};

//		. . N
//		. . N
//		N N N
int tile4_nd_nb[] = {	0,		0,		-1,
						0,		0,		-1,
						-1,		-1,		-1		};

/////////////////////////////////////
// no nodata value, 1-pixel buffer //
/////////////////////////////////////

//		N N N N N
// 		N . . . x
// 		N . x x x
//	 	N x x x x
// 		N . x x .
int tile1_nd_1b[] = {	-1,		-1,		-1,		-1,		-1,
						-1,		0,		0,		0,		255,
						-1,		0,		255,	255,	255,
						-1,		255,	255,	255,	255,
						-1,		0,		255,	255,	0		};

//		N N N N N
// 		. x x N N
// 		x x . N N
//	 	x x . N N
// 		x . . N N
int tile2_nd_1b[] = {	-1,		-1,		-1,		-1,		-1,
						0,		255,	255,	-1,		-1,
						255,	255,	0,		-1,		-1,
						255,	255,	0,		-1,		-1,
						255,	0,		0,		-1,		-1		};

//		N x x x x
//		N . x x .
//		N . . x .
//		N N N N N
//		N N N N N
int tile3_nd_1b[] = {	-1, 	255,	255,	255,	255,
						-1,		0,		255,	255,	0,
						-1,		0,		0,		255,	0,	
						-1,		-1,		-1,		-1,		-1,
						-1,		-1,		-1,		-1,		-1		};

//      x x . N N
//		x . . N N 
//		x . . N N
//		N N N N N
//      N N N N N
int tile4_nd_1b[] = {	255,	255,	0,		-1,		-1,
						255,	0,		0,		-1,		-1,
						255,	0,		0,		-1,		-1,
						-1,		-1,		-1,		-1,		-1,
						-1,		-1,		-1,		-1,		-1		};

#endif /*MOSAICTESTSUITETESTDATA_H_*/
