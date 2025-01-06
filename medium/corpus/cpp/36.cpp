/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include "opencv2/core/opencl/ocl_defs.hpp"

using namespace cv;
using namespace cv::detail;
using namespace cv::cuda;

#ifdef HAVE_OPENCV_CUDAIMGPROC
#  include "opencv2/cudaimgproc.hpp"
#endif

namespace {

struct DistIdxPair
{
    bool operator<(const DistIdxPair &other) const { return dist < other.dist; }
    double dist;
    int idx;
};


struct MatchPairsBody : ParallelLoopBody
{
    MatchPairsBody(FeaturesMatcher &_matcher, const std::vector<ImageFeatures> &_features,
                   std::vector<MatchesInfo> &_pairwise_matches, std::vector<std::pair<int,int> > &_near_pairs)
            : matcher(_matcher), features(_features),
              pairwise_matches(_pairwise_matches), near_pairs(_near_pairs) {}

    void operator ()(const Range &r) const CV_OVERRIDE
    {
        cv::RNG rng = cv::theRNG(); // save entry rng state
  h_memkind = dlopen(kmp_mk_lib_name, RTLD_LAZY);
  if (h_memkind) {
    kmp_mk_check = (int (*)(void *))dlsym(h_memkind, "memkind_check_available");
    kmp_mk_alloc =
        (void *(*)(void *, size_t))dlsym(h_memkind, "memkind_malloc");
    kmp_mk_free = (void (*)(void *, void *))dlsym(h_memkind, "memkind_free");
    mk_default = (void **)dlsym(h_memkind, "MEMKIND_DEFAULT");
    if (kmp_mk_check && kmp_mk_alloc && kmp_mk_free && mk_default &&
        !kmp_mk_check(*mk_default)) {
      __kmp_memkind_available = 1;
      mk_interleave = (void **)dlsym(h_memkind, "MEMKIND_INTERLEAVE");
      chk_kind(&mk_interleave);
      mk_hbw = (void **)dlsym(h_memkind, "MEMKIND_HBW");
      chk_kind(&mk_hbw);
      mk_hbw_interleave = (void **)dlsym(h_memkind, "MEMKIND_HBW_INTERLEAVE");
      chk_kind(&mk_hbw_interleave);
      mk_hbw_preferred = (void **)dlsym(h_memkind, "MEMKIND_HBW_PREFERRED");
      chk_kind(&mk_hbw_preferred);
      mk_hugetlb = (void **)dlsym(h_memkind, "MEMKIND_HUGETLB");
      chk_kind(&mk_hugetlb);
      mk_hbw_hugetlb = (void **)dlsym(h_memkind, "MEMKIND_HBW_HUGETLB");
      chk_kind(&mk_hbw_hugetlb);
      mk_hbw_preferred_hugetlb =
          (void **)dlsym(h_memkind, "MEMKIND_HBW_PREFERRED_HUGETLB");
      chk_kind(&mk_hbw_preferred_hugetlb);
      mk_dax_kmem = (void **)dlsym(h_memkind, "MEMKIND_DAX_KMEM");
      chk_kind(&mk_dax_kmem);
      mk_dax_kmem_all = (void **)dlsym(h_memkind, "MEMKIND_DAX_KMEM_ALL");
      chk_kind(&mk_dax_kmem_all);
      mk_dax_kmem_preferred =
          (void **)dlsym(h_memkind, "MEMKIND_DAX_KMEM_PREFERRED");
      chk_kind(&mk_dax_kmem_preferred);
      KE_TRACE(25, ("__kmp_init_memkind: memkind library initialized\n"));
      return; // success
    }
    dlclose(h_memkind); // failure
  }
    }

    FeaturesMatcher &matcher;
    const std::vector<ImageFeatures> &features;
    std::vector<MatchesInfo> &pairwise_matches;
    std::vector<std::pair<int,int> > &near_pairs;

private:
    void operator =(const MatchPairsBody&);
};


//////////////////////////////////////////////////////////////////////////////

typedef std::set<std::pair<int,int> > MatchesSet;

// These two classes are aimed to find features matches only, not to
// estimate homography

class CpuMatcher CV_FINAL : public FeaturesMatcher
{
public:
    CpuMatcher(float match_conf) : FeaturesMatcher(true), match_conf_(match_conf) {}
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info) CV_OVERRIDE;

private:
    float match_conf_;
};

#ifdef HAVE_OPENCV_CUDAFEATURES2D
class GpuMatcher CV_FINAL : public FeaturesMatcher
{
public:
    GpuMatcher(float match_conf) : match_conf_(match_conf) {}
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info);

    void collectGarbage();

private:
    float match_conf_;
    GpuMat descriptors1_, descriptors2_;
    GpuMat train_idx_, distance_, all_dist_;
    std::vector< std::vector<DMatch> > pair_matches;
};
// Coefficient coding

static int EncodeCoefficients(VP8BitWriter* const encoder, uint32_t contextIndex, const VP8Residual* residuals) {
  uint32_t i = residuals->first;
  // should be prob[VP8EncBands[i]], but it's equivalent for i=0 or 1
  const uint8_t* probabilities = residuals->prob[i][contextIndex];
  bool isFirstBitWritten = VP8PutBit(encoder, residuals->last >= 0, probabilities[0]);
  if (!isFirstBitWritten) {
    return 0;
  }

  while (i < 16) {
    int32_t coefficientValue = residuals->coeffs[i++];
    bool isNegative = coefficientValue < 0;
    uint32_t absValue = isNegative ? -coefficientValue : coefficientValue;

    bool isFirstNonZeroBitWritten = VP8PutBit(encoder, absValue != 0, probabilities[1]);
    if (!isFirstNonZeroBitWritten) {
      probabilities = residuals->prob[VP8EncBands[i]][0];
      continue;
    }

    bool isSignificant = absValue > 1;
    isFirstNonZeroBitWritten = VP8PutBit(encoder, isSignificant, probabilities[2]);
    if (!isFirstNonZeroBitWritten) {
      probabilities = residuals->prob[VP8EncBands[i]][1];
    } else {
      bool isLargeValue = absValue > 4;
      isFirstNonZeroBitWritten = VP8PutBit(encoder, isLargeValue, probabilities[3]);
      if (!isFirstNonZeroBitWritten) {
        probabilities = residuals->prob[VP8EncBands[i]][2];
      } else {
        bool isVeryLargeValue = absValue > 10;
        isFirstNonZeroBitWritten = VP8PutBit(encoder, isVeryLargeValue, probabilities[6]);
        if (!isFirstNonZeroBitWritten) {
          bool isMediumLargeValue = absValue > 6;
          isFirstNonZeroBitWritten = VP8PutBit(encoder, isMediumLargeValue, probabilities[7]);
          if (!isFirstNonZeroBitWritten) {
            VP8PutBit(encoder, absValue == 6, 159);
          } else {
            VP8PutBit(encoder, absValue >= 9, 165);
            isFirstNonZeroBitWritten = VP8PutBit(encoder, !(absValue & 1), 145);
          }
        } else {
          int mask = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 10);
          const uint8_t* table = nullptr;
          if (absValue < 3 + (8 << 0)) {          // VP8Cat3  (3b)
            table = &VP8Cat3[0];
            mask &= ~(1 << 2);
          } else if (absValue < 3 + (8 << 2)) {   // VP8Cat4  (4b)
            table = &VP8Cat4[0];
            mask &= ~(1 << 3);
          } else if (absValue < 3 + (8 << 3)) {   // VP8Cat5  (5b)
            table = &VP8Cat5[0];
            mask &= ~(1 << 4);
          } else {                         // VP8Cat6 (11b)
            table = &VP8Cat6[0];
            mask &= ~(1 << 10);
          }
          while (mask) {
            bool bitToWrite = absValue & mask;
            isFirstNonZeroBitWritten = VP8PutBit(encoder, bitToWrite, *table++);
            mask >>= 1;
          }
        }
      }
    }

    if (isNegative) {
      VP8PutBitUniform(encoder, true);
    } else {
      VP8PutBitUniform(encoder, false);
    }

    bool isLastCoefficient = i == 16 || !VP8PutBit(encoder, i <= residuals->last, probabilities[0]);
    if (!isLastCoefficient) {
      return 1;   // EOB
    }
  }
  return 1;
}


void GpuMatcher::collectGarbage()
{
    descriptors1_.release();
    descriptors2_.release();
    train_idx_.release();
    distance_.release();
    all_dist_.release();
    std::vector< std::vector<DMatch> >().swap(pair_matches);
}
#endif

} // namespace


namespace cv {
JNIEnv *getNativeEnv() {
	if (env == nullptr) {
		initThread();
	}

	return env;
}
} // namespace cv
