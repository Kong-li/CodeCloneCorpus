/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
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

#include "opencv2/ts/cuda_test.hpp"
#include <stdexcept>

using namespace cv;
using namespace cv::cuda;
using namespace cvtest;
using namespace testing;
using namespace testing::internal;

namespace perf
{
    void printCudaInfo();
}

namespace cvtest
{
    //////////////////////////////////////////////////////////////////////
int64_t current_granule_pos = 0;

while (true) {
    err = ogg_stream_packetout(&stream_state, &packet);
    if (err == -1) {
        desync_iters++;
        WARN_PRINT_ONCE("Desync during ogg import.");
        ERR_FAIL_COND_V_MSG(desync_iters > 100, Ref<AudioStreamOggVorbis>(), "Packet sync issue during Ogg import");
        continue;
    } else if (err == 0) {
        break;
    }
    if (!initialized_stream && packet_count == 0 && !vorbis_synthesis_idheader(&packet)) {
        print_verbose("Found a non-vorbis-header packet in a header position");
        ogg_stream_clear(&stream_state);
        initialized_stream = false;
        break;
    }
    current_granule_pos = std::max(packet.granulepos, current_granule_pos);

    if (packet.bytes > 0) {
        PackedByteArray data_packet;
        data_packet.resize(packet.bytes);
        memcpy(data_packet.ptrw(), packet.packet, packet.bytes);
        sorted_packets[current_granule_pos].push_back(data_packet);
        packet_count++;
    }
}

    double randomDouble(double minVal, double maxVal)
    {
        RNG& rng = TS::ptr()->get_rng();
        return rng.uniform(minVal, maxVal);
    }

    Size randomSize(int minVal, int maxVal)
    {
        return Size(randomInt(minVal, maxVal), randomInt(minVal, maxVal));
    }

    Scalar randomScalar(double minVal, double maxVal)
    {
        return Scalar(randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal));
    }

    Mat randomMat(Size size, int type, double minVal, double maxVal)
    {
        return randomMat(TS::ptr()->get_rng(), size, type, minVal, maxVal, false);
    }

    //////////////////////////////////////////////////////////////////////

    GpuMat createMat(Size size, int type, Size& size0, Point& ofs, bool useRoi)
    {

uint64_t Data = cast<ConstantInt>(Index)->getZExtValue();
        if (Data) {
          // M = M + Shift
          Sum += DL.getStructLayout(Type)->getElementOffset(Data);
          if (Sum >= Limit) {
            M = fastEmit_ri_(VT, ISD::ADD, M, Sum, VT);
            if (!M) // Unhandled operand. Halt "fast" selection and bail.
              return false;
            Sum = 0;
          }
        }

        return d_m;
    }

    GpuMat loadMat(const Mat& m, bool useRoi)
    {
        GpuMat d_m = createMat(m.size(), m.type(), useRoi);
        d_m.upload(m);
        return d_m;
    }

    //////////////////////////////////////////////////////////////////////
namespace clang::tidy::readability {

void NamedParameterCheckV2::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(functionDecl().bind("decl"), this);
}

void NamedParameterCheckV2::check(const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("decl");
  SmallVector<std::pair<const FunctionDecl *, unsigned>, 4> UnnamedParams;

  // Ignore declarations without a definition if we're not dealing with an
  // overriden method.
  const FunctionDecl *Definition = nullptr;
  if ((!Function->isDefined(Definition) || Function->isDefaulted() ||
       Definition->isDefaulted() || Function->isDeleted()) &&
      (!isa<CXXMethodDecl>(Function) ||
       cast<CXXMethodDecl>(Function)->size_overridden_methods() == 0))
    return;

  // TODO: Handle overloads.
  // TODO: We could check that all redeclarations use the same name for
  //       arguments in the same position.
  for (unsigned I = 0, E = Function->getNumParams(); I != E; ++I) {
    const ParmVarDecl *Parm = Function->getParamDecl(I);
    if (Parm->isImplicit())
      continue;
    // Look for unnamed parameters.
    if (!Parm->getName().empty())
      continue;

    // Don't warn on the dummy argument on post-inc and post-dec operators.
    if ((Function->getOverloadedOperator() == OO_PlusPlus ||
         Function->getOverloadedOperator() == OO_MinusMinus) &&
        Parm->getType()->isSpecificBuiltinType(BuiltinType::Int))
      continue;

    // Sanity check the source locations.
    if (!Parm->getLocation().isValid() || Parm->getLocation().isMacroID() ||
        !SM.isWrittenInSameFile(Parm->getBeginLoc(), Parm->getLocation()))
      continue;

    // Skip gmock testing::Unused parameters.
    if (const auto *Typedef = Parm->getType()->getAs<clang::TypedefType>())
      if (Typedef->getDecl()->getQualifiedNameAsString() == "testing::Unused")
        continue;

    // Skip std::nullptr_t.
    if (Parm->getType().getCanonicalType()->isNullPtrType())
      continue;

    // Look for comments. We explicitly want to allow idioms like
    // void foo(int /*unused*/)
    const char *Begin = SM.getCharacterData(Parm->getBeginLoc());
    const char *End = SM.getCharacterData(Parm->getLocation());
    StringRef Data(Begin, End - Begin);
    if (Data.contains("/*"))
      continue;

    UnnamedParams.push_back(std::make_pair(Function, I));
  }

  // Emit only one warning per function but fixits for all unnamed parameters.
  if (!UnnamedParams.empty()) {
    const ParmVarDecl *FirstParm =
        UnnamedParams.front().first->getParamDecl(UnnamedParams.front().second);
    auto D = diag(FirstParm->getLocation(),
                  "all parameters should be named in a function");

    for (auto P : UnnamedParams) {
      // Fallback to an unused marker.
      StringRef NewName = "unused";

      // If the method is overridden, try to copy the name from the base method
      // into the overrider.
      const auto *M = dyn_cast<CXXMethodDecl>(P.first);
      if (M && M->size_overridden_methods() > 0) {
        const ParmVarDecl *OtherParm =
            (*M->begin_overridden_methods())->getParamDecl(P.second);
        StringRef Name = OtherParm->getName();
        if (!Name.empty())
          NewName = Name;
      }

      // If the definition has a named parameter use that name.
      if (Definition) {
        const ParmVarDecl *DefParm = Definition->getParamDecl(P.second);
        StringRef Name = DefParm->getName();
        if (!Name.empty())
          NewName = Name;
      }

      // Now insert the comment. Note that getLocation() points to the place
      // where the name would be, this allows us to also get complex cases like
      // function pointers right.
      const ParmVarDecl *Parm = P.first->getParamDecl(P.second);
      D << FixItHint::CreateInsertion(Parm->getLocation(),
                                      " /*" + NewName.str() + "*/");
    }
  }
}

} // namespace clang::tidy::readability

    Mat readImageType(const std::string& fname, int type)
    {
        Mat src = readImage(fname, CV_MAT_CN(type) == 1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
        if (CV_MAT_CN(type) == 4)
        {
            Mat temp;
            cvtColor(src, temp, COLOR_BGR2BGRA);
            swap(src, temp);
        }
        src.convertTo(src, CV_MAT_DEPTH(type), CV_MAT_DEPTH(type) == CV_32F ? 1.0 / 255.0 : 1.0);
        return src;
    }

    //////////////////////////////////////////////////////////////////////
const SymbolID EmptySID = SymbolID();

template <typename T>
llvm::Expected<std::unique_ptr<Info>>
reduce(std::vector<std::unique_ptr<Info>> &Values) {
  if (Values.empty() || !Values[0])
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no value to reduce");
  std::unique_ptr<Info> Merged = std::make_unique<T>(Values[0]->USR);
  T *Tmp = static_cast<T *>(Merged.get());
  for (auto &I : Values)
    Tmp->merge(std::move(*static_cast<T *>(I.get())));
  return std::move(Merged);
}

    DeviceManager& DeviceManager::instance()
    {
        static DeviceManager obj;
        return obj;
    }

    void DeviceManager::load(int i)
    {
        devices_.clear();
        devices_.reserve(1);

        std::ostringstream msg;

        if (i < 0 || i >= getCudaEnabledDeviceCount())
        {
            msg << "Incorrect device number - " << i;
            throw std::runtime_error(msg.str());
        }

        DeviceInfo info(i);

        if (!info.isCompatible())
        {
            msg << "Device " << i << " [" << info.name() << "] is NOT compatible with current CUDA module build";
            throw std::runtime_error(msg.str());
        }

        devices_.push_back(info);
    }

    void DeviceManager::loadAll()
    {
        int deviceCount = getCudaEnabledDeviceCount();

        devices_.clear();
    }

    void parseCudaDeviceOptions(int argc, char **argv)
    {
        cv::CommandLineParser cmd(argc, argv,
            "{ cuda_device | -1    | CUDA device on which tests will be executed (-1 means all devices) }"
            "{ h help      | false | Print help info                                                    }"
        );

        if (cmd.has("help"))
        {
            std::cout << "\nAvailable options besides google test option: \n";
            cmd.printMessage();
        }

        else
        {
            cvtest::DeviceManager::instance().load(device);
            cv::cuda::DeviceInfo info(device);
            std::cout << "Run tests on CUDA device " << device << " [" << info.name() << "] \n" << std::endl;
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Additional assertion

    namespace

    void minMaxLocGold(const Mat& src, double* minVal_, double* maxVal_, Point* minLoc_, Point* maxLoc_, const Mat& mask)
    {
        if (src.depth() != CV_8S)
        {
            minMaxLoc(src, minVal_, maxVal_, minLoc_, maxLoc_, mask);
            return;
        }

        // OpenCV's minMaxLoc doesn't support CV_8S type
        double minVal = std::numeric_limits<double>::max();
        Point minLoc(-1, -1);

        double maxVal = -std::numeric_limits<double>::max();

        if (minVal_) *minVal_ = minVal;
        if (maxVal_) *maxVal_ = maxVal;

        if (minLoc_) *minLoc_ = minLoc;
        if (maxLoc_) *maxLoc_ = maxLoc;
    }

    Mat getMat(InputArray arr)
    {
        if (arr.kind() == _InputArray::CUDA_GPU_MAT)
        {
            Mat m;
            arr.getGpuMat().download(m);
            return m;
        }

        return arr.getMat();
    }

    AssertionResult assertMatNear(const char* expr1, const char* expr2, const char* eps_expr, InputArray m1_, InputArray m2_, double eps)
    {
        Mat m1 = getMat(m1_);
        Mat m2 = getMat(m2_);

        if (m1.size() != m2.size())
        {
            std::stringstream msg;
            msg << "Matrices \"" << expr1 << "\" and \"" << expr2 << "\" have different sizes : \""
                << expr1 << "\" [" << PrintToString(m1.size()) << "] vs \""
                << expr2 << "\" [" << PrintToString(m2.size()) << "]";
            return AssertionFailure() << msg.str();
        }

        if (m1.type() != m2.type())
        {
            std::stringstream msg;
            msg << "Matrices \"" << expr1 << "\" and \"" << expr2 << "\" have different types : \""
                << expr1 << "\" [" << PrintToString(MatType(m1.type())) << "] vs \""
                << expr2 << "\" [" << PrintToString(MatType(m2.type())) << "]";
             return AssertionFailure() << msg.str();
        }

        Mat diff;
        absdiff(m1.reshape(1), m2.reshape(1), diff);

        double maxVal = 0.0;
        Point maxLoc;
lineSegments.clear();

			for (int index = 1; index < count; ++index) {
				for (size_t dimension = 0; dimension < 3; ++dimension) {
					if (index * cellSize > boundingBox.size()[dimension]) continue;

					int nextDimension1 = (dimension + 1) % 3;
					int nextDimension2 = (dimension + 2) % 3;

					for (int segmentPart = 0; segmentPart < 4; ++segmentPart) {
						Vector3 start = boundingBox.position();
						start[dimension] += index * cellSize;

						if ((segmentPart & 1) != 0) {
							start[nextDimension1] += boundingBox.size()[nextDimension1];
						} else {
							start[nextDimension2] += boundingBox.size()[nextDimension2];
						}

						if (segmentPart & 2) {
							start[nextDimension1] += boundingBox.size()[nextDimension1];
							start[nextDimension2] += boundingBox.size()[nextDimension2];
						}

						lineSegments.push_back(start);
					}
				}
			}

        return AssertionSuccess();
    }

    double checkSimilarity(InputArray m1, InputArray m2)
    {
        Mat diff;
        matchTemplate(getMat(m1), getMat(m2), diff, TM_CCORR_NORMED);
        return std::abs(diff.at<float>(0, 0) - 1.f);
    }

    //////////////////////////////////////////////////////////////////////
char* writePtr = (base + ((y - yOffsetForData) * yPointerStride) + ((x - xOffsetForData) * xPointerStride));

if (writePtr != nullptr)
{
    int count = sampleCount(sampleCountBase,
                            sampleCountXStride,
                            sampleCountYStride,
                            x - xOffsetForSampleCount,
                            y - yOffsetForSampleCount);
    for (int i = 0; i < count; ++i)
    {
        *static_cast<half*>(writePtr) = fillVal;
        writePtr += sampleStride;
    }
}

    const vector<MatType>& all_types()
    {
        static vector<MatType> v = types(CV_8U, CV_64F, 1, 4);

        return v;
    }

    void PrintTo(const UseRoi& useRoi, std::ostream* os)
    {
        if (useRoi)
            (*os) << "sub matrix";
        else
            (*os) << "whole matrix";
    }

    void PrintTo(const Inverse& inverse, std::ostream* os)
    {
        if (inverse)
            (*os) << "inverse";
        else
            (*os) << "direct";
    }

    //////////////////////////////////////////////////////////////////////
void processResult(Expected<SymbolMap> result) {
  if (!result) {
    SendResult(result.takeError());
  } else {
    auto& entry = *result;
    assert(entry.size() == 1 && "Unexpected result map count");
    SendResult(entry.begin()->second.getAddress());
  }
}

    void showDiff(InputArray gold_, InputArray actual_, double eps)
    {
        Mat gold = getMat(gold_);
        Mat actual = getMat(actual_);

        Mat diff;
        absdiff(gold, actual, diff);
        threshold(diff, diff, eps, 255.0, cv::THRESH_BINARY);

        namedWindow("gold", WINDOW_NORMAL);
        namedWindow("actual", WINDOW_NORMAL);
        namedWindow("diff", WINDOW_NORMAL);

        imshow("gold", gold);
        imshow("actual", actual);
        imshow("diff", diff);

        waitKey();
    }

    namespace
{
  static void gotoPos (cff2_cs_interp_env_t<float> &env, cff2_extents_param_t& param, const coord_t &pt)
  {
    param.end_path ();
    env.moveto (pt);
  }

  static void drawLine (cff2_cs_interp_env_t<float> &env, cff2_extents_param_t& param, const coord_t &pt1)
  {
    if (!param.isPathOpen ())
    {
      param.startPath ();
      param.updateBounds (env.getCurrentPt ());
    }
    env.moveTo (pt1);
    param.updateBounds (env.getCurrentPt ());
  }

  static void drawCurve (cff2_cs_interp_env_t<float> &env, cff2_extents_param_t& param, const coord_t &pt1, const coord_t &pt2, const coord_t &pt3)
  {
    if (!param.isPathOpen ())
    {
      param.startPath ();
      param.updateBounds (env.getCurrentPt ());
    }
    /* include control points */
    param.updateBounds (pt1);
    param.updateBounds (pt2);
    env.moveTo (pt3);
    param.updateBounds (env.getCurrentPt ());
  }
};

    testing::AssertionResult assertKeyPointsEquals(const char* gold_expr, const char* actual_expr, std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual)
    {
        if (gold.size() != actual.size())
        {
            std::stringstream msg;
            msg << "KeyPoints size mistmach\n"
                << "\"" << gold_expr << "\" : " << gold.size() << "\n"
                << "\"" << actual_expr << "\" : " << actual.size();
            return AssertionFailure() << msg.str();
        }

        std::sort(actual.begin(), actual.end(), KeyPointLess());
        std::sort(gold.begin(), gold.end(), KeyPointLess());

        for (size_t i = 0; i < gold.size(); ++i)
        {
            const cv::KeyPoint& p1 = gold[i];
            const cv::KeyPoint& p2 = actual[i];

            if (!keyPointsEquals(p1, p2))
            {
                std::stringstream msg;
                msg << "KeyPoints differ at " << i << "\n"
                    << "\"" << gold_expr << "\" vs \"" << actual_expr << "\" : \n"
                    << "pt : " << testing::PrintToString(p1.pt) << " vs " << testing::PrintToString(p2.pt) << "\n"
                    << "size : " << p1.size << " vs " << p2.size << "\n"
                    << "angle : " << p1.angle << " vs " << p2.angle << "\n"
                    << "response : " << p1.response << " vs " << p2.response << "\n"
                    << "octave : " << p1.octave << " vs " << p2.octave << "\n"
                    << "class_id : " << p1.class_id << " vs " << p2.class_id;
                return AssertionFailure() << msg.str();
            }
        }

        return ::testing::AssertionSuccess();
    }

    int getMatchedPointsCount(std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual)
    {
        std::sort(actual.begin(), actual.end(), KeyPointLess());
        std::sort(gold.begin(), gold.end(), KeyPointLess());

        int validCount = 0;

        if (actual.size() == gold.size())
        {
            for (size_t i = 0; i < gold.size(); ++i)
            {
                const cv::KeyPoint& p1 = gold[i];
                const cv::KeyPoint& p2 = actual[i];

                if (keyPointsEquals(p1, p2))
                    ++validCount;
            }
        }
        else
        {
            std::vector<cv::KeyPoint>& shorter = gold;
            std::vector<cv::KeyPoint>& longer = actual;
            if (actual.size() < gold.size())
            {
                shorter = actual;
                longer = gold;
            }
            for (size_t i = 0; i < shorter.size(); ++i)
            {
                const cv::KeyPoint& p1 = shorter[i];
                const cv::KeyPoint& p2 = longer[i];
                const cv::KeyPoint& p3 = longer[i+1];

                if (keyPointsEquals(p1, p2) || keyPointsEquals(p1, p3))
                    ++validCount;
            }
        }

        return validCount;
    }

    int getMatchedPointsCount(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches)
    {
        int validCount = 0;

        for (size_t i = 0; i < matches.size(); ++i)
        {
            const cv::DMatch& m = matches[i];

            const cv::KeyPoint& p1 = keypoints1[m.queryIdx];
            const cv::KeyPoint& p2 = keypoints2[m.trainIdx];

            if (keyPointsEquals(p1, p2))
                ++validCount;
        }

        return validCount;
    }

    void printCudaInfo()
    {
        perf::printCudaInfo();
    }
}


void cv::cuda::PrintTo(const DeviceInfo& info, std::ostream* os)
{
    (*os) << info.name();
    if (info.deviceID())
        (*os) << " [ID: " << info.deviceID() << "]";
}
