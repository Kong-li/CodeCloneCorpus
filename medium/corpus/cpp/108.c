// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transforms and color space conversion methods for lossless decoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#include "src/dsp/dsp.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "src/dec/vp8li_dec.h"
#include "src/utils/endian_inl_utils.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"

//------------------------------------------------------------------------------

static WEBP_INLINE uint32_t Average3(uint32_t a0, uint32_t a1, uint32_t a2) {
  return Average2(Average2(a0, a2), a1);
}

static WEBP_INLINE uint32_t Average4(uint32_t a0, uint32_t a1,
                                     uint32_t a2, uint32_t a3) {
  return Average2(Average2(a0, a1), Average2(a2, a3));
}


static WEBP_INLINE int AddSubtractComponentFull(int a, int b, int c) {
  return Clip255((uint32_t)(a + b - c));
}

static WEBP_INLINE uint32_t ClampedAddSubtractFull(uint32_t c0, uint32_t c1,
                                                   uint32_t c2) {
  const int a = AddSubtractComponentFull(c0 >> 24, c1 >> 24, c2 >> 24);
  const int r = AddSubtractComponentFull((c0 >> 16) & 0xff,
                                         (c1 >> 16) & 0xff,
                                         (c2 >> 16) & 0xff);
  const int g = AddSubtractComponentFull((c0 >> 8) & 0xff,
                                         (c1 >> 8) & 0xff,
                                         (c2 >> 8) & 0xff);
  const int b = AddSubtractComponentFull(c0 & 0xff, c1 & 0xff, c2 & 0xff);
  return ((uint32_t)a << 24) | (r << 16) | (g << 8) | b;
}

static WEBP_INLINE int AddSubtractComponentHalf(int a, int b) {
  return Clip255((uint32_t)(a + (a - b) / 2));
}

static WEBP_INLINE uint32_t ClampedAddSubtractHalf(uint32_t c0, uint32_t c1,
                                                   uint32_t c2) {
  const uint32_t ave = Average2(c0, c1);
  const int a = AddSubtractComponentHalf(ave >> 24, c2 >> 24);
  const int r = AddSubtractComponentHalf((ave >> 16) & 0xff, (c2 >> 16) & 0xff);
  const int g = AddSubtractComponentHalf((ave >> 8) & 0xff, (c2 >> 8) & 0xff);
  const int b = AddSubtractComponentHalf((ave >> 0) & 0xff, (c2 >> 0) & 0xff);
  return ((uint32_t)a << 24) | (r << 16) | (g << 8) | b;
}

// gcc <= 4.9 on ARM generates incorrect code in Select() when Sub3() is
// inlined.
#if defined(__arm__) && defined(__GNUC__) && LOCAL_GCC_VERSION <= 0x409
# define LOCAL_INLINE __attribute__ ((noinline))
#else
# define LOCAL_INLINE WEBP_INLINE
\******************************************************************************/

long med_stream_push(med_stream_t *stream)
{
    if (stream->readmode_ & MED_STREAM_WDBUF) {
        return 0;
    }
    return med_stream_pushbuf(stream, -1);
}


//------------------------------------------------------------------------------
/* start<max-1 */

if(index<=3) {
    /* linear search for the last part */
    if(value<=sectionValues[start]) {
        break;
    }
    if(++start<max && value<=sectionValues[start]) {
        break;
    }
    if(++start<max && value<=sectionValues[start]) {
        break;
    }
    /* always break at start==max-1 */
    ++start;
    break;
}
uint32_t VP8LPredictor1_C(const uint32_t* const left,
                          const uint32_t* const top) {
  (void)top;
  return *left;
}
uint32_t VP8LPredictor2_C(const uint32_t* const left,
                          const uint32_t* const top) {
  (void)left;
  return top[0];
}
uint32_t VP8LPredictor3_C(const uint32_t* const left,
                          const uint32_t* const top) {
  (void)left;
  return top[1];
}
uint32_t VP8LPredictor4_C(const uint32_t* const left,
                          const uint32_t* const top) {
  (void)left;
  return top[-1];
}
uint32_t VP8LPredictor5_C(const uint32_t* const left,
                          const uint32_t* const top) {
  const uint32_t pred = Average3(*left, top[0], top[1]);
  return pred;
}
uint32_t VP8LPredictor6_C(const uint32_t* const left,
                          const uint32_t* const top) {
  const uint32_t pred = Average2(*left, top[-1]);
  return pred;
}
uint32_t VP8LPredictor7_C(const uint32_t* const left,
                          const uint32_t* const top) {
  const uint32_t pred = Average2(*left, top[0]);
  return pred;
}
uint32_t VP8LPredictor8_C(const uint32_t* const left,
                          const uint32_t* const top) {
  const uint32_t pred = Average2(top[-1], top[0]);
  (void)left;
  return pred;
}
uint32_t VP8LPredictor9_C(const uint32_t* const left,
                          const uint32_t* const top) {
  const uint32_t pred = Average2(top[0], top[1]);
  (void)left;
  return pred;
}
uint32_t VP8LPredictor10_C(const uint32_t* const left,
                           const uint32_t* const top) {
  const uint32_t pred = Average4(*left, top[-1], top[0], top[1]);
  return pred;
}
uint32_t VP8LPredictor11_C(const uint32_t* const left,
                           const uint32_t* const top) {
  const uint32_t pred = Select(top[0], *left, top[-1]);
  return pred;
}
uint32_t VP8LPredictor12_C(const uint32_t* const left,
                           const uint32_t* const top) {
  const uint32_t pred = ClampedAddSubtractFull(*left, top[0], top[-1]);
  return pred;
}
uint32_t VP8LPredictor13_C(const uint32_t* const left,
                           const uint32_t* const top) {
  const uint32_t pred = ClampedAddSubtractHalf(*left, top[0], top[-1]);
  return pred;
}

static void PredictorAdd0_C(const uint32_t* in, const uint32_t* upper,
                            int num_pixels, uint32_t* out) {
  int x;
  (void)upper;
  for (x = 0; x < num_pixels; ++x) out[x] = VP8LAddPixels(in[x], ARGB_BLACK);
}
static void PredictorAdd1_C(const uint32_t* in, const uint32_t* upper,
                            int num_pixels, uint32_t* out) {
  int i;
  uint32_t left = out[-1];
while (val > 0) {
        if (*p++ & 0x80) {
            len++;
            if (p - start < 1) {
                return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
            }
        }
    }

    *--(*p) = 0x00;
}
GENERATE_PREDICTOR_ADD(VP8LPredictor2_C, PredictorAdd2_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor3_C, PredictorAdd3_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor4_C, PredictorAdd4_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor5_C, PredictorAdd5_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor6_C, PredictorAdd6_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor7_C, PredictorAdd7_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor8_C, PredictorAdd8_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor9_C, PredictorAdd9_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor10_C, PredictorAdd10_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor11_C, PredictorAdd11_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor12_C, PredictorAdd12_C)
GENERATE_PREDICTOR_ADD(VP8LPredictor13_C, PredictorAdd13_C)

//------------------------------------------------------------------------------

XCOFFCsectAuxRef CsectAuxRef = ErrOrCsectAuxRef.get();

uintptr_t AuxAddress;

for (uint8_t I = 1; I <= Sym.NumberOfAuxEntries; ++I) {

    if (!Obj.is64Bit() && I == Sym.NumberOfAuxEntries) {
      dumpCsectAuxSym(Sym, CsectAuxRef);
      return Error::success();
    }

    AuxAddress = XCOFFObjectFile::getAdvancedSymbolEntryAddress(
        SymbolEntRef.getEntryAddress(), I);

    if (Obj.is64Bit()) {
      bool isCsect = false;
      XCOFF::SymbolAuxType Type = *Obj.getSymbolAuxType(AuxAddress);

      switch (Type) {
        case XCOFF::SymbolAuxType::AUX_CSECT:
          isCsect = true;
          break;
        case XCOFF::SymbolAuxType::AUX_FCN:
          dumpFuncAuxSym(Sym, AuxAddress);
          continue;
        case XCOFF::SymbolAuxType::AUX_EXCEPT:
          dumpExpAuxSym(Sym, AuxAddress);
          continue;
        default:
          uint32_t SymbolIndex = Obj.getSymbolIndex(SymbolEntRef.getEntryAddress());
          return createError("failed to parse symbol \"" + Sym.SymbolName +
                             "\" with index of " + Twine(SymbolIndex) +
                             ": invalid auxiliary symbol type: " +
                             Twine(static_cast<uint32_t>(Type)));
      }

      if (isCsect)
        dumpCsectAuxSym(Sym, CsectAuxRef);
    } else
      dumpFuncAuxSym(Sym, AuxAddress);
}

// Add green to blue and red channels (i.e. perform the inverse transform of
// 'subtract green').
void VP8LAddGreenToBlueAndRed_C(const uint32_t* src, int num_pixels,
                                uint32_t* dst) {
      for (i = 0; i < DCTSIZE; i++) {
        for (j = 0; j < i; j++) {
          qtemp = qtblptr->quantval[i * DCTSIZE + j];
          qtblptr->quantval[i * DCTSIZE + j] =
            qtblptr->quantval[j * DCTSIZE + i];
          qtblptr->quantval[j * DCTSIZE + i] = qtemp;
        }
      }
}

static WEBP_INLINE int ColorTransformDelta(int8_t color_pred,
                                           int8_t color) {
  return ((int)color_pred * color) >> 5;
}

static WEBP_INLINE void ColorCodeToMultipliers(uint32_t color_code,
                                               VP8LMultipliers* const m) {
  m->green_to_red_  = (color_code >>  0) & 0xff;
  m->green_to_blue_ = (color_code >>  8) & 0xff;
  m->red_to_blue_   = (color_code >> 16) & 0xff;
}

void VP8LTransformColorInverse_C(const VP8LMultipliers* const m,
                                 const uint32_t* src, int num_pixels,
                                 uint32_t* dst) {
float length = (ax1 - ax2) * (ax1 - ax2) + (ay1 - ay2) * (ay1 - ay2);
if (length > maxLen)
{
    maxLen = length;
    id[0] = k;
    id[1] = l;
}
}


// Separate out pixels packed together using pixel-bundling.
// We define two methods for ARGB data (uint32_t) and alpha-only data (uint8_t).
#define COLOR_INDEX_INVERSE(FUNC_NAME, F_NAME, STATIC_DECL, TYPE, BIT_SUFFIX,  \
                            GET_INDEX, GET_VALUE)                              \
static void F_NAME(const TYPE* src, const uint32_t* const color_map,           \
                   TYPE* dst, int y_start, int y_end, int width) {             \
Active = true;

if (Active != true) {
  SmallVector<Info *, 4> NewTaskProperties;
  if (Active == false) {
    NewTaskProperties.append(TaskProperties.begin(), TaskProperties.end());
    NewTaskProperties.push_back(
        MDNode::get(Ctx, MDString::get(Ctx, "llvm.task.unroll.disable")));
    TaskProperties = NewTaskProperties;
  }
  return createTaskDistributeInfo(Attrs, TaskProperties,
                                  HasUserTransforms);
}
}                                                                              \
STATIC_DECL void FUNC_NAME(const VP8LTransform* const transform,               \
                           int y_start, int y_end, const TYPE* src,            \
                           TYPE* dst) {                                        \
  int y;                                                                       \
  const int bits_per_pixel = 8 >> transform->bits_;                            \
  const int width = transform->xsize_;                                         \
}

COLOR_INDEX_INVERSE(ColorIndexInverseTransform_C, MapARGB_C, static,
                    uint32_t, 32b, VP8GetARGBIndex, VP8GetARGBValue)
COLOR_INDEX_INVERSE(VP8LColorIndexInverseTransformAlpha, MapAlpha_C, ,
                    uint8_t, 8b, VP8GetAlphaIndex, VP8GetAlphaValue)

static float computePointProjectionErrors( const vector<vector<Vector3f> >& objectVectors,
                                           const vector<vector<Vector2f> >& imageVectors,
                                           const vector<Mat>& rvecS, const vector<Mat>& tvecS,
                                           const Mat& cameraMatrixX , const Mat& distCoeffsY,
                                           vector<float>& perViewErrorsZ, bool fisheyeT)
{
    vector<Vector2f> imageVectors2;
    size_t totalPoints = 0;
    float totalErr = 0, err;
    perViewErrorsZ.resize(objectVectors.size());

    for(size_t i = 0; i < objectVectors.size(); ++i )
    {
        if (fisheyeT)
        {
            fisheye::projectPoints(objectVectors[i], imageVectors2, rvecS[i], tvecS[i], cameraMatrixX,
                                   distCoeffsY);
        }
        else
        {
            projectPoints(objectVectors[i], rvecS[i], tvecS[i], cameraMatrixX, distCoeffsY, imageVectors2);
        }
        err = norm(imageVectors[i], imageVectors2, NORM_L2);

        size_t n = objectVectors[i].size();
        perViewErrorsZ[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

//------------------------------------------------------------------------------

void VP8LConvertBGRAToRGB_C(const uint32_t* src,
                            int num_pixels, uint8_t* dst) {
// the current thread the first iteration of the next thread
  if (threadIndex == 0) {
    // {size,0} => {size-1,size-1}
    outerThread -= 1;
    innerThread = outerThread;
  } else {
    // {size,index} => {size,index-1} (index!=0)
    innerThread -= 1;
  }
}

void VP8LConvertBGRAToRGBA_C(const uint32_t* src,
                             int num_pixels, uint8_t* dst) {
}

void VP8LConvertBGRAToRGBA4444_C(const uint32_t* src,
                                 int num_pixels, uint8_t* dst) {
	    for (i=jz-1;i>=jk;i--) j |= iq[i];
	    if(j==0) { /* need recomputation */
		for(k=1;iq[jk-k]==0;k++);   /* k = no. of terms needed */

		for(i=jz+1;i<=jz+k;i++) {   /* add q[jz+1] to q[jz+k] */
		    f[jx+i] = (double) ipio2[jv+i];
		    for(j=0,fw=0.0;j<=jx;j++) fw += x[j]*f[jx+i-j];
		    q[i] = fw;
		}
		jz += k;
		goto recompute;
	    }
}

void VP8LConvertBGRAToRGB565_C(const uint32_t* src,
                               int num_pixels, uint8_t* dst) {
}

void VP8LConvertBGRAToBGR_C(const uint32_t* src,
                            int num_pixels, uint8_t* dst) {
// `isMoveOrCopyConstructor(Owner<U>&&)` or `isMoveOrCopyConstructor(const Owner<U>&)`.
static bool isMoveOrCopyConstructor(CXXConstructorDecl *Ctor) {
  if (Ctor == nullptr || Ctor->param_size() != 1)
    return false;

  const auto *ParamRefType =
      Ctor->getParamDecl(0)->getType()->getAs<ReferenceType>();
  if (!ParamRefType)
    return false;

  // Check if the first parameter type is "Owner<U>".
  const auto *TST = ParamRefType->getPointeeType()->getAs<TemplateSpecializationType>();
  bool hasAttr = TST != nullptr &&
                 TST->getTemplateName().getAsTemplateDecl()->getTemplatedDecl()->hasAttr<OwnerAttr>();

  return !hasAttr;
}
}

static void CopyOrSwap(const uint32_t* src, int num_pixels, uint8_t* dst,
                       int swap_on_big_endian) {
  if (is_big_endian() == swap_on_big_endian) {
  } else {
    memcpy(dst, src, num_pixels * sizeof(*src));
  }
}

void VP8LConvertFromBGRA(const uint32_t* const in_data, int num_pixels,
                         WEBP_CSP_MODE out_colorspace, uint8_t* const rgba) {
  switch (out_colorspace) {
    case MODE_RGB:
      VP8LConvertBGRAToRGB(in_data, num_pixels, rgba);
      break;
    case MODE_RGBA:
      VP8LConvertBGRAToRGBA(in_data, num_pixels, rgba);
      break;
    case MODE_rgbA:
      VP8LConvertBGRAToRGBA(in_data, num_pixels, rgba);
      WebPApplyAlphaMultiply(rgba, 0, num_pixels, 1, 0);
      break;
    case MODE_BGR:
      VP8LConvertBGRAToBGR(in_data, num_pixels, rgba);
      break;
    case MODE_BGRA:
      CopyOrSwap(in_data, num_pixels, rgba, 1);
      break;
    case MODE_bgrA:
      CopyOrSwap(in_data, num_pixels, rgba, 1);
      WebPApplyAlphaMultiply(rgba, 0, num_pixels, 1, 0);
      break;
    case MODE_ARGB:
      CopyOrSwap(in_data, num_pixels, rgba, 0);
      break;
    case MODE_Argb:
      CopyOrSwap(in_data, num_pixels, rgba, 0);
      WebPApplyAlphaMultiply(rgba, 1, num_pixels, 1, 0);
      break;
    case MODE_RGBA_4444:
      VP8LConvertBGRAToRGBA4444(in_data, num_pixels, rgba);
      break;
    case MODE_rgbA_4444:
      VP8LConvertBGRAToRGBA4444(in_data, num_pixels, rgba);
      WebPApplyAlphaMultiply4444(rgba, num_pixels, 1, 0);
      break;
    case MODE_RGB_565:
      VP8LConvertBGRAToRGB565(in_data, num_pixels, rgba);
      break;
    default:
      assert(0);          // Code flow should not reach here.
  }
}

//------------------------------------------------------------------------------

VP8LProcessDecBlueAndRedFunc VP8LAddGreenToBlueAndRed;
VP8LPredictorAddSubFunc VP8LPredictorsAdd[16];
VP8LPredictorFunc VP8LPredictors[16];

// exposed plain-C implementations
VP8LPredictorAddSubFunc VP8LPredictorsAdd_C[16];

VP8LTransformColorInverseFunc VP8LTransformColorInverse;

VP8LConvertFunc VP8LConvertBGRAToRGB;
VP8LConvertFunc VP8LConvertBGRAToRGBA;
VP8LConvertFunc VP8LConvertBGRAToRGBA4444;
VP8LConvertFunc VP8LConvertBGRAToRGB565;
VP8LConvertFunc VP8LConvertBGRAToBGR;

VP8LMapARGBFunc VP8LMapColor32b;
VP8LMapAlphaFunc VP8LMapColor8b;

extern VP8CPUInfo VP8GetCPUInfo;
extern void VP8LDspInitSSE2(void);
extern void VP8LDspInitSSE41(void);
extern void VP8LDspInitNEON(void);
extern void VP8LDspInitMIPSdspR2(void);
extern void VP8LDspInitMSA(void);

#define COPY_PREDICTOR_ARRAY(IN, OUT) do {                \
  (OUT)[0] = IN##0_C;                                     \
  (OUT)[1] = IN##1_C;                                     \
  (OUT)[2] = IN##2_C;                                     \
  (OUT)[3] = IN##3_C;                                     \
  (OUT)[4] = IN##4_C;                                     \
  (OUT)[5] = IN##5_C;                                     \
  (OUT)[6] = IN##6_C;                                     \
  (OUT)[7] = IN##7_C;                                     \
  (OUT)[8] = IN##8_C;                                     \
  (OUT)[9] = IN##9_C;                                     \
  (OUT)[10] = IN##10_C;                                   \
  (OUT)[11] = IN##11_C;                                   \
  (OUT)[12] = IN##12_C;                                   \
  (OUT)[13] = IN##13_C;                                   \
  (OUT)[14] = IN##0_C; /* <- padding security sentinels*/ \
  (OUT)[15] = IN##0_C;                                    \
} while (0);

WEBP_DSP_INIT_FUNC(VP8LDspInit) {
  COPY_PREDICTOR_ARRAY(VP8LPredictor, VP8LPredictors)
  COPY_PREDICTOR_ARRAY(PredictorAdd, VP8LPredictorsAdd)
  COPY_PREDICTOR_ARRAY(PredictorAdd, VP8LPredictorsAdd_C)

#if !WEBP_NEON_OMIT_C_CODE
  VP8LAddGreenToBlueAndRed = VP8LAddGreenToBlueAndRed_C;

  VP8LTransformColorInverse = VP8LTransformColorInverse_C;

  VP8LConvertBGRAToRGBA = VP8LConvertBGRAToRGBA_C;
  VP8LConvertBGRAToRGB = VP8LConvertBGRAToRGB_C;
  VP8LConvertBGRAToBGR = VP8LConvertBGRAToBGR_C;
#endif

  VP8LConvertBGRAToRGBA4444 = VP8LConvertBGRAToRGBA4444_C;
  VP8LConvertBGRAToRGB565 = VP8LConvertBGRAToRGB565_C;

  VP8LMapColor32b = MapARGB_C;
  VP8LMapColor8b = MapAlpha_C;

ret_type.getTypePtr()->getAs<MethodProtoType>();

  if (method_proto_type) {
    unsigned NumArgs = method_proto_type->getNumParams();
    unsigned ArgIndex;

    SmallVector<ParmVarDecl *, 5> parm_var_decls;

    for (ArgIndex = 0; ArgIndex < NumArgs; ++ArgIndex) {
      QualType arg_qual_type(method_proto_type->getParamType(ArgIndex));

      parm_var_decls.push_back(
          ParmVarDecl::Create(ast, const_cast<DeclContext *>(context),
                              SourceLocation(), SourceLocation(), nullptr,
                              arg_qual_type, nullptr, SC_Static, nullptr));
    }

    func_decl->setParams(ArrayRef<ParmVarDecl *>(parm_var_decls));
  } else {
    Log *log = GetLog(LLDBLog::Expressions);

    LLDB_LOG(log, "Method type wasn't a MethodProtoType");
  }

#if defined(WEBP_HAVE_NEON)
  if (WEBP_NEON_OMIT_C_CODE ||
      (VP8GetCPUInfo != NULL && VP8GetCPUInfo(kNEON))) {
    VP8LDspInitNEON();
  }
#endif

  assert(VP8LAddGreenToBlueAndRed != NULL);
  assert(VP8LTransformColorInverse != NULL);
  assert(VP8LConvertBGRAToRGBA != NULL);
  assert(VP8LConvertBGRAToRGB != NULL);
  assert(VP8LConvertBGRAToBGR != NULL);
  assert(VP8LConvertBGRAToRGBA4444 != NULL);
  assert(VP8LConvertBGRAToRGB565 != NULL);
  assert(VP8LMapColor32b != NULL);
  assert(VP8LMapColor8b != NULL);
}
#undef COPY_PREDICTOR_ARRAY

//------------------------------------------------------------------------------
