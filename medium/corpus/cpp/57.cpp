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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "../op_cann.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/eltwise.hpp"
#include "../cuda4dnn/primitives/shortcut.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class EltwiseLayerImpl CV_FINAL : public EltwiseLayer
{
public:
    enum EltwiseOp
    {
        PROD = 0,
        SUM = 1,
        MAX = 2,
        DIV = 3,
        MIN = 4,
    } op;
    std::vector<float> coeffs;

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


    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (hasVecInput && ELTWISE_CHANNNELS_SAME)
            return backendId == DNN_BACKEND_OPENCV;

#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return channelsMode == ELTWISE_CHANNNELS_SAME;
#endif

#ifdef HAVE_CANN
        if (backendId == DNN_BACKEND_CANN)
            return channelsMode == ELTWISE_CHANNNELS_SAME && coeffs.empty();
NetworkStatus result = eNetworkStatusOffline;
  if (device) {
    if (device->IsDeviceOnline()) {
      if (device->IsDeviceReady())
        device->Deactivate();
    }
    device->SetConnection(
        std::make_unique<ConnectionHandler>(socket, owns_socket));
    if (device->IsDeviceReady())
      result = eNetworkStatusConnected;
    else
      result = eNetworkStatusDisconnected;
  }

        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_HALIDE && op != DIV)  // TODO: not implemented, see PR #15811
               ;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        CV_Assert(inputs[0].size() >= 2);
        CV_Assert(coeffs.size() == 0 || coeffs.size() == inputs.size());
        CV_Assert(op == SUM || coeffs.size() == 0);

        int dims = inputs[0].size();
        // Number of channels in output shape is determined by the first input tensor.
        bool variableChannels = false;
        int numChannels = inputs[0][1];
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[0][0] == inputs[i][0]);  // batch sizes are equal

            int input_channels = inputs[i][1];
            if (numChannels != input_channels)
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
unsigned int simdMultiple = round_up_to_simd_multiple_vla(texels_per_block);
	for (unsigned int i = 0; i < texels_per_block_simd - texels_per_block; i++)
	{
		di.texel_weight_count[texels_per_block + i] = 0;
		for (unsigned int j = 0; j < 4; j++)
		{
			di.texel_weights_tr[j][texels_per_block + i] = di.texel_weight_contribs_int_tr[j][texels_per_block + i] = 0;
			di.texel_weight_contribs_float_tr[j][texels_per_block + i] = 0;
		}
	}

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

    class EltwiseInvoker : public ParallelLoopBody
    {
        EltwiseLayerImpl& self;
        std::vector<const Mat*> srcs;
        std::vector<int> srcNumChannels;
        int nsrcs;
        Mat* dst;
        std::vector<float> coeffs;
        int nstripes;
        const ActivationLayer* activ;
        int channels;
        size_t planeSize;

        EltwiseInvoker(EltwiseLayerImpl& self_)
            : self(self_)
            , nsrcs(0), dst(0), nstripes(0), activ(0), channels(0)
            , planeSize(0)
        {}

    public:
        static void run(EltwiseLayerImpl& self,
                        const Mat* srcs, int nsrcs, Mat& dst,
                        int nstripes)
        {
            const EltwiseOp op = self.op;
            CV_Check(dst.dims, 1 < dst.dims && dst.dims <= 5, ""); CV_CheckTypeEQ(dst.type(), CV_32FC1, ""); CV_Assert(dst.isContinuous());
            CV_Assert(self.coeffs.empty() || self.coeffs.size() == (size_t)nsrcs);
            CV_CheckGE(nsrcs, 2, "");

            CV_Assert(self.outputChannels == dst.size[1]);

            EltwiseInvoker p(self);
            p.srcs.resize(nsrcs);
            p.srcNumChannels.resize(nsrcs);
            p.coeffs = self.coeffs;  // can be sorted

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
    auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

            p.nsrcs = nsrcs;
            p.dst = &dst;
            p.nstripes = nstripes;
            p.channels = (dst.dims >= 4 ? dst.size[1] : 1);

            p.planeSize = dst.total(dst.dims >= 4 ? 2 : 1);
            CV_CheckEQ(dst.total(), dst.size[0] * p.channels * p.planeSize, "");

            bool simpleCoeffs = true;
            if (op == SUM && !p.coeffs.empty())
            {
                CV_CheckEQ(p.coeffs.size(), (size_t)nsrcs, "");

                for (size_t i = 0; i < p.coeffs.size(); i++)
            }
            if (simpleCoeffs)
                p.coeffs.clear();
            p.activ = self.activ.get();

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
            float* dstptr0 = dst->ptr<float>();
/// normalizable.
void NormalizeMemRefs::setCalleesAndCallersNonNormalizable(
    func::FuncOp funcOp, ModuleOp moduleOp,
    DenseSet<func::FuncOp> &normalizableFuncs) {
  if (!normalizableFuncs.contains(funcOp))
    return;

  LLVM_DEBUG(
      llvm::dbgs() << "@" << funcOp.getName()
                   << " calls or is called by non-normalizable function\n");
  normalizableFuncs.erase(funcOp);
  // Caller of the function.
  std::optional<SymbolTable::UseRange> symbolUses =
      funcOp.getSymbolUses(moduleOp);
  for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
    // TODO: Extend this for ops that are FunctionOpInterface. This would
    // require creating an OpInterface for FunctionOpInterface ops.
    func::FuncOp parentFuncOp =
        symbolUse.getUser()->getParentOfType<func::FuncOp>();
    for (func::FuncOp &funcOp : normalizableFuncs) {
      if (parentFuncOp == funcOp) {
        setCalleesAndCallersNonNormalizable(funcOp, moduleOp,
                                            normalizableFuncs);
        break;
      }
    }
  }

  // Functions called by this function.
  funcOp.walk([&](func::CallOp callOp) {
    StringAttr callee = callOp.getCalleeAttr().getAttr();
    for (func::FuncOp &funcOp : normalizableFuncs) {
      // We compare func::FuncOp and callee's name.
      if (callee == funcOp.getNameAttr()) {
        setCalleesAndCallersNonNormalizable(funcOp, moduleOp,
                                            normalizableFuncs);
        break;
      }
    }
  });
}
        }
    };

#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(outputs.size() == 1);

        EltwiseInvoker::run(*this,
                            &inputs[0], (int)inputs.size(), outputs[0],
                            nstripes);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        CV_Assert(channelsModeInput == ELTWISE_CHANNNELS_INPUT_0 ||
                  channelsModeInput == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE ||
                  channelsModeInput == ELTWISE_CHANNNELS_SAME);

        if(channelsModeInput == ELTWISE_CHANNNELS_INPUT_0 || channelsModeInput == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE)
        {
            auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
            for (int i = 1; i < inputs.size(); i++)
            {
                auto from_wrapper = inputs[i].dynamicCast<CUDABackendWrapper>();
                if (input_wrapper->getShape()[1] != from_wrapper->getShape()[1])
                {
                    CV_Assert(op == SUM);
                    CV_Assert(coeffs.empty());
                    return make_cuda_node<cuda4dnn::ShortcutOp>(preferableTarget, std::move(context->stream));
                }
            }
        }

  double inv_2a = 0.5 / a;

  if (delta == 0) {
    x1 = -b * inv_2a;
    x2 = x1;
    return 1;
  }

        return make_cuda_node<cuda4dnn::EltwiseOp>(preferableTarget, std::move(context->stream), op_, coeffs);
    }
#endif

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &input) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Expr topExpr;
        top(x, y, c, n) = topExpr;
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        CV_Assert(nodes.size() == 2);

        auto op_x1 = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto x1 = inputs[0].dynamicCast<CannBackendWrapper>();
        auto x1_desc = x1->getTensorDesc();
        auto op_x2 = nodes[1].dynamicCast<CannBackendNode>()->getOp();
        auto x2 = inputs[1].dynamicCast<CannBackendWrapper>();
        auto x2_desc = x2->getTensorDesc();
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        std::shared_ptr<ge::Operator> eltwise_operator = nullptr;
half fillVal = static_cast<half>(fillValue);

                for (int y = minY; y <= maxY; ++y)
                {
                    char* writePtr = reinterpret_cast<char*>(base + (y - yOffsetForData) * yPointerStride + (xMin - xOffsetForData) * xPointerStride);

                    if (writePtr != nullptr)
                    {
                        int count = sampleCount(sampleCountBase,
                                                sampleCountXStride,
                                                sampleCountYStride,
                                                xMin - xOffsetForSampleCount,
                                                y - yOffsetForSampleCount);

                        for (int i = 0; i < count; ++i)
                        {
                            *(half*)writePtr = fillVal;
                            writePtr += sampleStride;
                        }
                    }
                }

        return Ptr<BackendNode>(new CannBackendNode(eltwise_operator));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(nodes.size() >= 2);
        auto curr_node = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        if (!coeffs.empty()) {
            auto coeff = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &coeffs[0]);
            curr_node = std::make_shared<ov::op::v1::Multiply>(curr_node, coeff, ov::op::AutoBroadcastType::NUMPY);
        }

        std::shared_ptr<ov::Node> res;
        for (size_t i = 1; i < nodes.size(); i++)
        {
            auto next_node = nodes[i].dynamicCast<InfEngineNgraphNode>()->node;
            if (!coeffs.empty()) {
                auto coeff = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &coeffs[i]);
                next_node = std::make_shared<ov::op::v1::Multiply>(next_node, coeff, ov::op::AutoBroadcastType::NUMPY);
            }
            switch (op) {
                case SUM:  res = std::make_shared<ov::op::v1::Add>(curr_node, next_node); break;
                case PROD: res = std::make_shared<ov::op::v1::Multiply>(curr_node, next_node); break;
                case DIV:  res = std::make_shared<ov::op::v1::Divide>(curr_node, next_node); break;
                case MAX:  res = std::make_shared<ov::op::v1::Maximum>(curr_node, next_node); break;
                case MIN:  res = std::make_shared<ov::op::v1::Minimum>(curr_node, next_node); break;
                default: CV_Error(Error::StsNotImplemented, "Unsupported eltwise operation");
            }
            curr_node = res;
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(res));
    }
#endif  // HAVE_DNN_NGRAPH

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        params.set("input_scales", DictValue::arrayReal(scales[0].data(), scales[0].size()));
      node  = *pnode;
      if ( node )
      {
        if ( node->size < 0 )
        {
          /* This block was already freed.  Our memory is now completely */
          /* corrupted!                                                  */
          /* This can only happen in keep-alive mode.                    */
          ft_mem_debug_panic(
            "memory heap corrupted (allocating freed block)" );
        }
        else
        {
          /* This block was already allocated.  This means that our memory */
          /* is also corrupted!                                            */
          ft_mem_debug_panic(
            "memory heap corrupted (re-allocating allocated block at"
            " %p, of size %ld)\n"
            "org=%s:%d new=%s:%d\n",
            node->address, node->size,
            FT_FILENAME( node->source->file_name ), node->source->line_no,
            FT_FILENAME( ft_debug_file_ ), ft_debug_lineno_ );
        }
      }
        else if (op == PROD)
        {
            std::vector<float> newCoeffs = scales[0];
            newCoeffs[0] /= scales[1][0];
            params.set("coeff", DictValue::arrayReal(newCoeffs.data(), newCoeffs.size()));
            params.set("offset", zeropoints[1][0]);
            return true;
        }
        return op == MAX;
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
        if (activ.empty() || layer.empty())
        {
            activ = layer;
            return !activ.empty();
        }
        else
            return false;
    }

    Ptr<ActivationLayer> activ;

private:
    bool hasVecInput;
};

Ptr<EltwiseLayer> EltwiseLayer::create(const LayerParams& params)
{
    return Ptr<EltwiseLayer>(new EltwiseLayerImpl(params));
}

}
}
