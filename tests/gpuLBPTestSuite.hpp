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

#ifndef _gpu_LBP_TEST_SUITE_H_
#define _gpu_LBP_TEST_SUITE_H_

#include <cxxtest/TestSuite.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <chrono>
#include <numeric>


#ifdef HAVE_CGI
    #include "../src/base/cvTileConversion.hpp"
#endif

#include "../src/base/cvTile.hpp"
#include "../src/base/Tiler.hpp"
#include <boost/filesystem.hpp>


#include "../src/gpu/drivers/GpuLBP.hpp"
#include "../src/gpu/drivers/GpuWindowFilterAlgorithm.hpp"


#define SHOW_OUTPUT 0

class gpuLBPTestSuite : public CxxTest::TestSuite
{
    public:

        void setUp()
        {;}

        void tearDown()
        {;}

        void testFlatSignalChar () {
            std::cout << "\nGPU LBP VERIFICATION TESTS SUITE\n" << std::endl;
            constexpr unsigned short cudaDeviceId = 0;
            constexpr unsigned short windowRadius = 1;
            //Not a magic number, worked out by hand.
            //Result of LBP on a flat signal.
            constexpr unsigned char expectedResult = 255;

            cv::Size2i roi(2,2);
            cv::Size2i tileSize(roi.width + windowRadius * 2, roi.height + windowRadius * 2);
            vector<unsigned char> data;
            data.resize(tileSize.area());

            for (auto i = 0; i<tileSize.area(); ++i) {
                data[i] = 1;
            }

            cvt::cvTile<unsigned char> inTile(data.data(), tileSize, 1);
            cvt::cvTile<unsigned char>* outTile;

            //set ROI of input tile.
            inTile.setROI(cv::Rect(windowRadius, windowRadius, roi.width, roi.height));

            cvt::gpu::GpuLBP<unsigned char, 1, unsigned char, 1> lbp(cudaDeviceId, roi.width, roi.height, windowRadius);

            //init device
            cvt::ErrorCode last = lbp.initializeDevice(cvt::gpu::SPARSE_RING);
            if (last != 0)
            {
                std::cout << "Could not init device" << std::endl;
            }
            //std::cout << std::endl << lbp.getRelativeOffsetString();

            cudaError cuer = cudaGetLastError();
            if (cuer != cudaSuccess) {
                std::cerr << "dead!" << std::endl;
            }

            lbp(inTile, (const cvt::cvTile<unsigned char>**)&outTile);
            //lbp(inTile, (const cvt::cvTile<short>**)&outTile, 16, 16);

            //Make sure out tile is not null
            TS_ASSERT_EQUALS(0, (outTile == NULL));

            unsigned char result = (*outTile)[0].at<unsigned char>(1,1);
            TS_ASSERT_EQUALS(expectedResult, result);
            /*unsigned char test = 1;
            //print the bits...
            for (unsigned long i=0; i<sizeof(unsigned char) * 8; ++i)
            {
                if (test & result)
                {
                    printf("%d ", 1);
                } else {
                    printf("%d ", 0);
                }
                test <<= 1;
                //printf(" - test %d -", test);
            }
            printf("\n");
            */

        }

        void testFlatSignalShort () {
            constexpr unsigned short cudaDeviceId = 0;
            constexpr unsigned short windowRadius = 2;
            //Not a magic number, worked out by hand.
            //Result of LBP on a flat signal.
            constexpr short expectedResult = -1;

            cv::Size2i roi(5,5);
            cv::Size2i tileSize(roi.width + windowRadius * 2, roi.height + windowRadius * 2);
            vector<short> data;
            data.resize(tileSize.area());

            for (auto i = 0; i<tileSize.area(); ++i) {
                data[i] = 1;
            }

            cvt::cvTile<short> inTile(data.data(), tileSize, 1);
            cvt::cvTile<short>* outTile;

            //set ROI of input tile.
            inTile.setROI(cv::Rect(windowRadius, windowRadius, roi.width, roi.height));

            cvt::gpu::GpuLBP<short, 1, short, 1> lbp(cudaDeviceId, roi.width, roi.height, windowRadius);

            //init device
            cvt::ErrorCode last = lbp.initializeDevice(cvt::gpu::SPARSE_RING);
            if (last != 0)
            {
            std::cout << "Could not init device" << std::endl;
            }
            //std::cout << std::endl << lbp.getRelativeOffsetString();

            cudaError cuer = cudaGetLastError();
            if (cuer != cudaSuccess) {
                std::cerr << "dead!" << std::endl;
            }

            lbp(inTile, (const cvt::cvTile<short>**)&outTile);
            //lbp(inTile, (const cvt::cvTile<short>**)&outTile, 16, 16);

            //Make sure out tile is not null
            TS_ASSERT_EQUALS(0, (outTile == NULL));

            //std::cout << "output of [2,2] (middle): " << (*outTile)[0].at<short>(2,2) << std::endl;

            short result = (*outTile)[0].at<short>(2,2);
            TS_ASSERT_EQUALS(expectedResult, result);
            //short test = 1;
            //printf(result);
            //print the bits...
            /*for (unsigned long i=0; i<sizeof(short) * 8; ++i)
            {
                if (test & result)
                {
                    printf("%d ", 1);
                } else {
                    printf("%d ", 0);
                }
                test <<= 1;
                //printf(" - test %d -", test);
            }
            printf("\n");
            */
        }

        void testIncreasingSignalShort () {
            constexpr unsigned short cudaDeviceId = 0;
            constexpr unsigned short windowRadius = 2;
            //Not a magic number, worked out by hand.
            //Result of LBP on an increasing signal.
            constexpr short expectedResult = -8161;

            cv::Size2i roi(5,5);
            cv::Size2i tileSize(roi.width + windowRadius * 2, roi.height + windowRadius * 2);
            vector<short> data;
            data.resize(tileSize.area());

            for (auto i = 0; i<tileSize.area(); ++i) {
                data[i] = i;
            }

            cvt::cvTile<short> inTile(data.data(), tileSize, 1);
            cvt::cvTile<short>* outTile;

            //set ROI of input tile.
            inTile.setROI(cv::Rect(windowRadius, windowRadius, roi.width, roi.height));

            cvt::gpu::GpuLBP<short, 1, short, 1> lbp(cudaDeviceId, roi.width, roi.height, windowRadius);

            //init device
            cvt::ErrorCode last = lbp.initializeDevice(cvt::gpu::SPARSE_RING);
            if (last != 0)
            {
            std::cout << "Could not init device" << std::endl;
            }
            //std::cout << std::endl << lbp.getRelativeOffsetString();

            cudaError cuer = cudaGetLastError();
            if (cuer != cudaSuccess) {
                std::cerr << "dead!" << std::endl;
            }

            lbp(inTile, (const cvt::cvTile<short>**)&outTile);
            //lbp(inTile, (const cvt::cvTile<short>**)&outTile, 16, 16);

            //Make sure out tile is not null
            TS_ASSERT_EQUALS(0, (outTile == NULL));

            //std::cout << "output of [2,2] (middle): " << (*outTile)[0].at<short>(2,2) << std::endl;

            short result = (*outTile)[0].at<short>(2,2);
            TS_ASSERT_EQUALS(expectedResult, result);

            /*short test = 1;
            //printf(result);
            //print the bits...
            for (unsigned long i=0; i<sizeof(short) * 8; ++i)
            {
                if (test & result)
                {
                    printf("%d ", 1);
                } else {
                    printf("%d ", 0);
                }
                test <<= 1;
                //printf(" - test %d -", test);
            }
            printf("\n");
            */
        }

        void testRelativeOffsetsUnsignedChar () {
            constexpr unsigned int cudaDeviceId = 0;
            unsigned int windowRadius = 1;
            cv::Size2i roi(1,1);
            cv::Size2i tileSize(roi.width + windowRadius * 2, roi.height + windowRadius * 2);

            cvt::gpu::GpuLBP<unsigned char, 1, unsigned char, 1> lbp(cudaDeviceId, roi.width, roi.height, windowRadius);

            //init device
            cvt::ErrorCode last = lbp.initializeDevice(cvt::gpu::SPARSE_RING);
            if (last != 0)
            {
                std::cout << "Could not init device" << std::endl;
            }

            std::string expected {"Relative Offsets: "
                         "Size: 8 Data: [(0,1),(1,1),(1,0),(1,-1)"
                         ",(0,-1),(-1,-1),(-1,0),(-1,1)]"};

            TS_ASSERT(not lbp.getRelativeOffsetString().compare(expected));
            //std::cout << std::endl << lbp.getRelativeOffsetString();


        }

        void testRelativeOffsetsShort () {
            constexpr unsigned int cudaDeviceId = 0;
            unsigned int windowRadius = 2;
            cv::Size2i roi(1,1);
            cv::Size2i tileSize(roi.width + windowRadius * 2, roi.height + windowRadius * 2);

            cvt::gpu::GpuLBP<short, 1, short, 1> lbp(cudaDeviceId, roi.width, roi.height, windowRadius);

            //init device
            cvt::ErrorCode last = lbp.initializeDevice(cvt::gpu::SPARSE_RING);
            if (last != 0)
            {
                std::cout << "Could not init device" << std::endl;
            }

            std::string expected {"Relative Offsets: "
                                  "Size: 16 "
                                  "Data: [(0,2),(1,2),(1,1),(2,1),(2,0)"
                                  ",(2,-1),(1,-1),(1,-2),(0,-2),(-1,-2)"
                                  ",(-1,-1),(-2,-1),(-2,0),(-2,1),(-1,1)"
                                  ",(-1,2)]"};

            TS_ASSERT(not lbp.getRelativeOffsetString().compare(expected));

            //std::cout << std::endl << lbp.getRelativeOffsetString();

        }

        void testRelativeOffsetsUCharRad2 () {
            constexpr unsigned int cudaDeviceId = 0;
            unsigned int windowRadius = 2;
            cv::Size2i roi(1,1);
            cv::Size2i tileSize(roi.width + windowRadius * 2, roi.height + windowRadius * 2);

            cvt::gpu::GpuLBP<unsigned char, 1, unsigned char, 1> lbp(cudaDeviceId, roi.width, roi.height, windowRadius);

            //init device
            cvt::ErrorCode last = lbp.initializeDevice(cvt::gpu::SPARSE_RING);
            if (last != 0)
            {
                std::cout << "Could not init device" << std::endl;
            }

            //This is not the cleanest, but exposing the attr. publicly is worse.
            std::string expected {"Relative Offsets: "
                                  "Size: 8 Data: [(0,2),(1,1),(2,0),(1,-1)"
                                  ",(0,-2),(-1,-1),(-2,0),(-1,1)]"};

            TS_ASSERT(not lbp.getRelativeOffsetString().compare(expected));

            //std::cout << std::endl << lbp.getRelativeOffsetString();

        }

};

#endif
