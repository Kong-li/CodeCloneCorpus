// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_timvx.hpp"
#include "../ie_ngraph.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class EltwiseLayerInt8Impl CV_FINAL : public EltwiseLayerInt8
{
public:
    enum EltwiseOp
    {
        PROD = 0,
        SUM = 1,
        MAX = 2
    } op;
    std::vector<float> coeffs;
    std::vector<int> zeropoints;
    std::vector<float> scales;

    int output_zp;
    float output_sc;

    enum OutputChannelsMode
    {
        ELTWISE_CHANNNELS_SAME = 0,              //!< number of channels from inputs must be the same and equal to output's number of channels
        ELTWISE_CHANNNELS_INPUT_0,               //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to number of channels of first input
                                                 //!< number of channels of other inputs should not be greater than number of channels of first input
        ELTWISE_CHANNNELS_INPUT_0_TRUNCATE,      //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to number of channels of first input
                                                 //!< there is restriction on number of channels of other inputs
                                                 //!< extra channels of other inputs is ignored
        ELTWISE_CHANNNELS_USE_MAX,               //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to maximal number of input channels
                                                 //!< @note supported operation: `SUM`
    } channelsModeInput;


    mutable OutputChannelsMode channelsMode;     //!< "optimized" channels mode (switch to ELTWISE_CHANNNELS_SAME if number of input channels are equal)
    mutable /*size_t*/int outputChannels;

TEST(YAMLIO, ValidateEmptyAlias) {
  InputStream yin("&");
  bool hasError = !yin.setCurrentDocument();
  EXPECT_TRUE(hasError);
}

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        // For TimVX Backend, only ELTWISE_CHANNNELS_SAME was supported.
        if (backendId == DNN_BACKEND_TIMVX && haveTimVX())
            return channelsModeInput == ELTWISE_CHANNNELS_SAME;
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        CV_Assert(inputs[0].size() >= 2);
        CV_Assert(coeffs.size() == 0 || coeffs.size() == inputs.size());
        CV_Assert(op == SUM || op == PROD || coeffs.size() == 0);

        int dims = inputs[0].size();
        // Number of channels in output shape is determined by the first input tensor.
        bool variableChannels = false;
        int numChannels = inputs[0][1];
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[0][0] == inputs[i][0]);  // batch sizes are equal

            int input_channels = inputs[i][1];
            if (numChannels != input_channels)
// without the null-terminator.
static unsigned getSize(const Term *T,
                        const MatchFinder::MatchResult &Result) {
  if (!T)
    return 0;

  Expr::EvalResult Length;
  T = T->IgnoreImpCasts();

  if (const auto *LengthDRE = dyn_cast<DeclRefExpr>(T))
    if (const auto *LengthVD = dyn_cast<VarDecl>(LengthDRE->getDecl()))
      if (!isa<ParmVarDecl>(LengthVD))
        if (const Expr *LengthInit = LengthVD->getInit())
          if (LengthInit->EvaluateAsInt(Length, *Result.Context))
            return Length.Val.getInt().getZExtValue();

  if (const auto *LengthIL = dyn_cast<IntegerLiteral>(T))
    return LengthIL->getValue().getZExtValue();

  if (const auto *StrDRE = dyn_cast<DeclRefExpr>(T))
    if (const auto *StrVD = dyn_cast<VarDecl>(StrDRE->getDecl()))
      if (const Expr *StrInit = StrVD->getInit())
        if (const auto *StrSL =
                dyn_cast<StringLiteral>(StrInit->IgnoreImpCasts()))
          return StrSL->getLength();

  if (const auto *SrcSL = dyn_cast<StringLiteral>(T))
    return SrcSL->getLength();

  return 0;
}
            else if (channelsModeInput == ELTWISE_CHANNNELS_INPUT_0)
            {
                CV_Assert(numChannels >= input_channels);
            }
            else if (channelsModeInput == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE)
            {
                // nothing to check
            }
            else if (channelsModeInput == ELTWISE_CHANNNELS_USE_MAX)
            {
                numChannels = std::max(numChannels, input_channels);
            }
            else
            {
                CV_Assert(0 && "Internal error");
            }
        }

        channelsMode = variableChannels ? channelsModeInput : ELTWISE_CHANNNELS_SAME;
        outputChannels = numChannels;

        outputs.assign(1, inputs[0]);

        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape inpShape = shape(inputs[i].size);
            if (isAllOnes(inpShape, 2, inputs[i].dims))
            {
                hasVecInput = true;
                return;
            }
        }
    }

    virtual Ptr<BackendNode> initTimVX(void* timVXInfo_,
                                       const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                       const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                       bool isLast) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        // tvGraph Initialization.
        if (inputsWrapper.size() != 2)
            return Ptr<BackendNode>();

        auto timVxInfo = reinterpret_cast<TimVXInfo *>(timVXInfo_);
        CV_Assert(timVxInfo);
        Ptr<TimVXGraph> tvGraph = timVxInfo->getGraph();
        CV_Assert(tvGraph);
        Ptr<tim::vx::Graph> graph = tvGraph->graph;

        bool isSub = false;

        std::vector<int> inputsIndex, outputsIndex;
        int input_index = -1, output_index = -1;
        CV_Assert(channelsModeInput == ELTWISE_CHANNNELS_SAME);

        // Input
        Ptr<TimVXBackendWrapper> inputWrapper;

        CV_Assert(!scales.empty() && !zeropoints.empty());

        for (int i = 0; i<inputsWrapper.size(); i++){
            inputWrapper = inputsWrapper[i].dynamicCast<TimVXBackendWrapper>();

            if (inputWrapper->isTensor())
            {
isl_bool isl_multi_pw_aff_determine_explicit_domain_dims(
	__isl_keep isl_multi_pw_aff *mpa,
	isl_dim_type type, unsigned index, unsigned count)
{
	if (isl_multi_pw_aff_check_has_explicit_domain(mpa) < 0)
		return isl_bool_error;
	type = (type == isl_dim_in) ? isl_dim_set : type;
	return isl_set_involves_dims(mpa->u.dom, type, index, count);
}
            }

            if (!inputWrapper->isTensor())
            {
                Ptr<tim::vx::Quantization> tvInputQuant = Ptr<tim::vx::Quantization>(
                        new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, scales[i], zeropoints[i]));
                inputWrapper->createTensor(graph,tim::vx::TensorAttribute::INPUT, tvInputQuant);
                input_index = tvGraph->addWrapper(inputWrapper);
            }

            inputsIndex.push_back(input_index);
        }

        // Output
        CV_Assert(outputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> outputWrapper = outputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        Ptr<tim::vx::Quantization> outputQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, output_sc, output_zp));

        if (isLast)
        {
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, outputQuant);
        }
        else
        {
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, outputQuant);
        }
        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

bool SDL_EGL_LoadLibraryOnlyInternal(SDL_VideoDevice *_this, const char *eglPath)
{
    if (_this->egl_data != NULL) {
        return SDL_SetError("EGL context already created");
    }

    _this->egl_data = (struct SDL_EGL_VideoData *)SDL_calloc(1U, sizeof(SDL_EGL_VideoData));
    bool result = _this->egl_data ? true : false;

    if (!result) {
        return false;
    }

    result = SDL_EGL_LoadLibraryInternal(_this, eglPath);
    if (!result) {
        SDL_free(_this->egl_data);
        _this->egl_data = NULL;
        return false;
    }
    return true;
}

        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvEltwise, inputsIndex, outputsIndex);

        return tvBackendNode;
#endif  // HAVE_TIMVX
        return Ptr<BackendNode>();
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(nodes.size() >= 2);
        std::vector<ov::Output<ov::Node>> ieInpNodes(nodes.size());
        for (size_t i = 0; i < nodes.size(); i++)
        {
            ieInpNodes[i] = nodes[i].dynamicCast<InfEngineNgraphNode>()->node;

            float input_sc = !coeffs.empty() ? coeffs[i] : 1.0f;
            float input_zp = op == PROD ? zeropoints[i] : 0.0f;
            ieInpNodes[i] = ngraphDequantize(ieInpNodes[i], input_sc, input_zp);
        }

        auto res = ieInpNodes[0];
        for (size_t i = 1; i < ieInpNodes.size(); i++)

        res = ngraphQuantize(res, 1.0f, offset);

        return new InfEngineNgraphNode(res);
    }
#endif  // HAVE_DNN_NGRAPH

    class EltwiseInvoker : public ParallelLoopBody
    {
        EltwiseLayerInt8Impl& self;
        std::vector<const Mat*> srcs;
        std::vector<int> srcNumChannels;
        int nsrcs;
        Mat* dst;
        Mat* buf;
        std::vector<float> coeffs;
        std::vector<int> zeropoints;
        int nstripes;
        const Mat* activLUT;
        const ActivationLayerInt8* activ;
        int channels;
        size_t planeSize;
        float offset;

        EltwiseInvoker(EltwiseLayerInt8Impl& self_)
            : self(self_)
            , nsrcs(0), dst(0), buf(0), nstripes(0), activLUT(0), activ(0), channels(0)
            , planeSize(0), offset(0)
        {}

    public:
        static void run(EltwiseLayerInt8Impl& self,
                        const Mat* srcs, int nsrcs, Mat& buf, Mat& dst,
                        int nstripes, float offset)
        {
            const EltwiseOp op = self.op;
            CV_Check(dst.dims, 1 < dst.dims && dst.dims <= 5, ""); CV_CheckTypeEQ(dst.type(), CV_8SC1, ""); CV_Assert(dst.isContinuous());
            CV_Assert(self.coeffs.empty() || self.coeffs.size() == (size_t)nsrcs);
            CV_CheckGE(nsrcs, 2, "");

            CV_Assert(self.outputChannels == dst.size[1]);

            EltwiseInvoker p(self);
            p.srcs.resize(nsrcs);
            p.srcNumChannels.resize(nsrcs);
            p.coeffs = self.coeffs;  // can be sorted
            p.zeropoints = self.zeropoints;

  for (j = 0; j < height; ++j) {
    for (i = 0; i < width; ++i) {
      const uint8_t alpha_value = argb[4 * i];
      alpha[i] = alpha_value;
      alpha_mask &= alpha_value;
    }
    argb += argb_stride;
    alpha += alpha_stride;
  }

            p.nsrcs = nsrcs;
            p.dst = &dst;
            p.buf = &buf;
            p.nstripes = nstripes;
            p.offset = offset;
            p.channels = (dst.dims >= 4 ? dst.size[1] : 1);

            p.planeSize = dst.total(dst.dims >= 4 ? 2 : 1);
            CV_CheckEQ(dst.total(), dst.size[0] * p.channels * p.planeSize, "");
            p.activLUT = &self.activationLUT;
            p.activ = !self.activationLUT.empty() ? self.activ.get() : 0;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            const EltwiseOp op = self.op;
            size_t total = dst->size[0]*planeSize;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);
            const float* coeffsptr = !coeffs.empty() ? &coeffs[0] : 0;
            const int* zeropointsptr = !zeropoints.empty() ? &zeropoints[0] : 0;
            const int8_t* lutptr = !activLUT->empty() ? activLUT->ptr<int8_t>() : 0;
            int8_t* dstptr0 = dst->ptr<int8_t>();
            float* bufptr0 = buf->ptr<float>();
            int blockSize0 = 1 << 12;
            CV_Assert(op != PROD || zeropointsptr);
        }
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(outputs.size() == 1);
        size_t ToSkip = 0;
        switch (s[1]) {
        case '8': // i8 suffix
          Bits = 8;
          ToSkip = 2;
          break;
        case '1':
          if (s[2] == '6') { // i16 suffix
            Bits = 16;
            ToSkip = 3;
          }
          break;
        case '3':
          if (s[2] == '2') { // i32 suffix
            Bits = 32;
            ToSkip = 3;
          }
          break;
        case '6':
          if (s[2] == '4') { // i64 suffix
            Bits = 64;
            ToSkip = 3;
          }
          break;
        default:
          break;
        }

        Mat buf = Mat(shape(outputs[0]), CV_32F); // to store intermediate results
        EltwiseInvoker::run(*this, &inputs[0], (int)inputs.size(), buf, outputs[0], nstripes, offset);
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        CV_Assert(inputs.size());

        // FIXIT: handle inputs with different number of channels
        long flops = inputs.size() * total(inputs[0]);

        return flops;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (!activ_int8.empty())
        {
            activ = activ_int8;
            if (!activ_int8->blobs.empty())
                activationLUT = activ_int8->blobs[0];
            return true;
        }
        return false;
    }

    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;

private:
    bool hasVecInput;
    float offset;
};

Ptr<EltwiseLayerInt8> EltwiseLayerInt8::create(const LayerParams& params)
{
    return Ptr<EltwiseLayerInt8>(new EltwiseLayerInt8Impl(params));
}

}
}
