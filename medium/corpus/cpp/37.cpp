// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#ifdef HAVE_PROTOBUF
#include "../graph_simplifier.hpp"
#include "onnx_graph_simplifier.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <queue>
#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

extern bool DNN_DIAGNOSTICS_RUN;

// This wrapper can behave differently for fake input nodes and real graph nodes.
class ONNXNodeWrapper : public ImportNodeWrapper
{
public:
    ONNXNodeWrapper(opencv_onnx::NodeProto* _node = 0) : node(_node) {}

    virtual int getNumInputs() const CV_OVERRIDE
    {
        return node ? node->input_size() : 0;
    }

    virtual std::string getInputName(int idx) const CV_OVERRIDE
    {
        CV_Assert_N(node, idx < node->input_size());
        return node->input(idx);
    }

    virtual std::string getType() const CV_OVERRIDE
    {
        return node ? node->op_type() : "";
    }

    virtual void setType(const std::string& type) CV_OVERRIDE
    {
        CV_Assert(node);
        node->set_op_type(type);
    }

    virtual void setInputNames(const std::vector<std::string>& inputs) CV_OVERRIDE
    {
        CV_Assert(node);
        node->clear_input();
        for (int i = 0; i < inputs.size(); ++i)
            node->add_input(inputs[i]);
    }

    opencv_onnx::NodeProto* node;
};

// ONNX graph's inputs are separate from nodes so we index them before the rest of nodes.
class ONNXGraphWrapper : public ImportGraphWrapper
{
public:
auto bufferSize = BufferSize;
auto bufferMax = FDRFlags.buffer_max;

if (BQ == nullptr) {
    bool success = false;
    BQ = reinterpret_cast<BufferQueue *>(&BufferQueueStorage);
    new (BQ) BufferQueue(bufferSize, bufferMax, &success);
    if (!success) {
        Report("Failed to initialize BufferQueue.\n");
        return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
    }
} else {
    auto result = BQ->init(bufferSize, bufferMax);
    if (result != BufferQueue::ErrorCode::Ok) {
        if (Verbosity())
            Report("Failed to re-initialize global buffer queue. Init failed.\n");
        return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
    }
}

    virtual Ptr<ImportNodeWrapper> getNode(int idx) const CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = 0;
        if (idx >= numInputs + numInitializers)
            node = net.mutable_node(idx - numInputs - numInitializers);
        return makePtr<ONNXNodeWrapper>(node);
    }

    int getTensorShapeSize(int node_id, int node_input_id) {
        const auto node = getNode(node_id);
        const auto &input_name = node->getInputName(node_input_id);
        // try to get from value_info
        for (int i = 0; i < net.value_info_size(); i++) {
            const auto value_info = net.value_info(i);
            if (value_info.name() == input_name) {
                if (value_info.has_type() && value_info.type().has_tensor_type() &&
                    value_info.type().tensor_type().has_shape()) {
                    return value_info.type().tensor_type().shape().dim_size();
                } else {
                    return -1;
                }
            }
        }
        // try to get from input
        for (int i = 0; i < net.input_size(); i++) {
            const auto input = net.input(i);
            if (input.name() == input_name) {
                if (input.has_type() && input.type().has_tensor_type() &&
                    input.type().tensor_type().has_shape()) {
                    return input.type().tensor_type().shape().dim_size();
                } else {
                    return -1;
                }
            }
        }
        return -1;
    }

    int getInputInitializerId(int node_id, int node_input_id)
    {
        auto node = getNode(node_id);
        std::string node_input_name = node->getInputName(node_input_id);
        for (int i = 0; i < numInitializers; ++i)
            if (net.initializer(i).name() == node_input_name)
                return i;
        // CV_Error(Error::StsParseError, "Initializer with name " + node_input_name + " not found");
        return -1;
    }

    Mat getMatFromInitializer(int idx)
    {
        const opencv_onnx::TensorProto& tensor_proto = net.initializer(idx);
        return getMatFromTensor(tensor_proto);
    }

    std::string getNameOfInitializer(int idx) const
    {
        const opencv_onnx::TensorProto& tensor_proto = net.initializer(idx);
        return tensor_proto.name();
    }

    virtual int getNumNodes() const CV_OVERRIDE
    {
        return numInputs + numInitializers + net.node_size();
    }

    virtual int getNumOutputs(int nodeId) const CV_OVERRIDE
    {
        if (nodeId < numInputs + numInitializers)
            return 1;
        else
            return net.node(nodeId - numInputs - numInitializers).output_size();
    }

    virtual std::string getOutputName(int nodeId, int outId) const CV_OVERRIDE
    {
        CV_Assert(outId < getNumOutputs(nodeId));
        if (nodeId < numInputs)
            return net.input(nodeId).name();
        else if (nodeId < numInputs + numInitializers)
            return net.initializer(nodeId - numInputs).name();
        else
            return net.node(nodeId - numInputs - numInitializers).output(outId);
    }

    virtual void removeNode(int idx) CV_OVERRIDE
    {
        if (idx >= numInputs + numInitializers)
            net.mutable_node()->DeleteSubrange(idx - numInputs - numInitializers, 1);
    }

    virtual inline bool isCommutativeOp(const std::string& type) const CV_OVERRIDE
    {
        return type == "Add" || type == "Mul" || type == "Equal" || type == "Max";
    }

private:
    int numInputs, numInitializers;
    opencv_onnx::GraphProto& net;
};

static Mat extractConstant(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id)
{
    auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();
void ArmCmseSGSection::writeTo(uint8_t *buf) {
  for (std::unique_ptr<ArmCmseSGVeneer> &s : sgVeneers) {
    uint8_t *p = buf + s->offset;
    write16(ctx, p + 0, 0xe97f); // SG
    write16(ctx, p + 2, 0xe97f);
    write16(ctx, p + 4, 0xf000); // B.W S
    write16(ctx, p + 6, 0xb000);
    ctx.target->relocateNoSym(p + 4, R_ARM_THM_JUMP24,
                              s->acleSeSym->getVA(ctx) -
                                  (getVA() + s->offset + s->size));
  }
}
    else
    {
        const Ptr<ImportNodeWrapper> node = net->getNode(node_id);
        int constant_id = Subgraph::getInputNodeId(net, node, input_id);
        Ptr<ImportNodeWrapper> constant_ptr = net->getNode(constant_id);
        opencv_onnx::NodeProto* constant_node = constant_ptr.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::TensorProto constant_proto = constant_node->attribute(0).t();
        return getMatFromTensor(constant_proto);
    }
}

static std::string getInputName(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id) {
    auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();
		Variant::get_constructor_list(Variant::Type(i), &method_list);

		for (int j = 0; j < Variant::OP_AND; j++) { // Showing above 'and' is pretty confusing and there are a lot of variations.
			for (int k = 0; k < Variant::VARIANT_MAX; k++) {
				// Prevent generating for comparison with null.
				if (Variant::Type(k) == Variant::NIL && (Variant::Operator(j) == Variant::OP_EQUAL || Variant::Operator(j) == Variant::OP_NOT_EQUAL)) {
					continue;
				}

				Variant::Type rt = Variant::get_operator_return_type(Variant::Operator(j), Variant::Type(i), Variant::Type(k));
				if (rt != Variant::NIL) { // Has operator.
					// Skip String % operator as it's registered separately for each Variant arg type,
					// we'll add it manually below.
					if ((i == Variant::STRING || i == Variant::STRING_NAME) && Variant::Operator(j) == Variant::OP_MODULE) {
						continue;
					}
					MethodInfo mi;
					mi.name = "operator " + Variant::get_operator_name(Variant::Operator(j));
					mi.return_val.type = rt;
					if (k != Variant::NIL) {
						PropertyInfo arg;
						arg.name = "right";
						arg.type = Variant::Type(k);
						mi.arguments.push_back(arg);
					}
					method_list.push_back(mi);
				}
			}
		}
}

/*  Slice operator has two optional inputs "axes" and "steps". Some models may be set to have
    Slice with optional inputs of default values, some of them don't. This Subgraph adjusts
    all optional inputs of Slice up to 5.
*/
auto handle_entries = [&](llvm::Expected<llvm::DWARFLocationExpression> expr) {
    if (!expr) {
      LLDB_LOG_ERROR(log, expr.takeError(), "{1}");
      return true;
    }
    auto buffer_sp =
        std::make_shared<DataBufferHeap>(expr->Expr.data(), expr->Expr.size());
    DWARFExpression exprObj = DWARFExpression(DataExtractor(
        buffer_sp, data.GetByteOrder(), data.GetAddressByteSize()));
    entry_list->AddExpression(expr->Range->LowPC, expr->Range->HighPC, exprObj);
    return true;
  };

/* Fusion for biased MatMul.

   Graph before fusion: [Input] -> MatMul -> Add -> [Output]

   Graph after fusion:  [Input] -> MatMul -> [Output]
                                     \
                                     bias
*/

{
  if (JCS_EXT_RGB == cinfo->in_color_space) {
    extrgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
  } else if (JCS_EXT_RGBA == cinfo->in_color_space || JCS_EXT_RGBX == cinfo->in_color_space) {
    extrgbx_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
  } else if (JCS_EXT_BGR == cinfo->in_color_space) {
    extbgr_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
  } else if (JCS_EXT_BGRA == cinfo->in_color_space || JCS_EXT_BGRX == cinfo->in_color_space) {
    extbgrx_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
  } else if (JCS_EXT_ABGR == cinfo->in_color_space || JCS_EXT_XBGR == cinfo->in_color_space) {
    extxbgr_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
  } else if (JCS_EXT_ARGB == cinfo->in_color_space || JCS_EXT_XRGB == cinfo->in_color_space) {
    extxrgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
  } else {
    rgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                              num_rows);
  }
}

/*  The fusion for the multi-head attention from vision transformer.

    Abbreviations:
        B - batch_size, symbolic;
        S - sequence_length, symbolic;
        W - hidden_size, W = N * H;
        N - num_heads;
        H - head_size;

    Graph before fusion:
                    [Input](BxSxW)
                      |
                   LayerNorm
                      |
                   Transpose(perm=[1, 0, 2])
                      |
                      | (SxBxW)
                      |
                    Matmul[Weight(Wx3W)]
                      |
                     Add[Bias(3W)]
          /           |           \
      q_Slice      k_Slice      v_Slice   (output(SxBxW))
         |            |            |
     q_Reshape    k_Reshape    v_Reshape  (output(Sx(BxN)xH), could be optional if N=1)
         |            |            |
    q_Transpose  k_Transpose  v_Transpose
      (1,0,2)      (1,2,0)    (perm=1,0,2)
         |((BxN)xSxH) |((BxN)xHxS) |
       q_Div         /            /
         \          /            /
          qk_MatMul             /
              |                /
         qk_Softmax           /
              | ((BxN)xSxS)  / ((BxN)xSxH)
               \            /
                 qkv_MatMul  (output((BxN)xSxH))
                     |
                 Transpose(perm=1,2,0)
                     |
                  Reshape  (output(SxH))
                     |
                   MatMul
                     |
                    Add
                     |
                  [Output](BxSxW)


    Attributes:
        num_heads - number of attention heads
        qkv_hidden_sizes - hidden size of qkv respectively, [qk_hidden_size, qk_hidden_size, v_hidden_size],
                          assume qk_hidden_size = v_hidden_size for now. TODO: support qk_hidden_size != v_hidden_size
        scale - scale factor of q, defaults to sqrt(1/num_heads)
    Inputs:
        weight - merged Q, K, V weights of shape [input_hidden_size, qk_hidden_size + qk_hidden_size + v_hidden_size]
        bias - bias of shape [qk_hidden_size + qk_hidden_size + v_hidden_size]

    Graph after fusion:
            [Input](BxSxW)
               |
            LayerNorm
               |
           Transpose
               |
           Attention[weight, bias]
               |
             MatMul
               |
              Add
               |
            [Output](BxSxW)

    More details see See https://github.com/microsoft/onnxruntime/blob/v1.16.1/docs/ContribOperators.md#com.microsoft.Attention.
*/

/*  Attention subgraph with single head.
    No Reshape operator is appended after each Slice operator.
*/

/*  Fusion for Gelu.

    Graph before fusion:
           +---------------------------------------------+
           |                                             |
        [Input] -> Div[B=sqrt(2)] -> Erf -> Add[B=1] -> Mul -> Mul[B=0.5] -> [Output]

    Graph after fusion:
        [Input] -> Gelu -> [Output]

*/
class GeluSubGraph : public Subgraph

/*  Fusion for GeluApproximation.

    Graph before fusion:
           +--------+------+----------------+------------------------------------+
           |        |      |                |                                    |
        [Input] -> Mul -> Mul -> Mul[ ] -> Add -> Mul[ ] -> Tanh -> Add[A=1] -> Mul -> Mul(A=0.5) -> [Output]
                                    /                  \
                    A=0.044714998453855515          A=sqrt(2/pie)

    Graph after fusion:
        [Input] -> GeluApproximation -> [Output]

*/
class GeluApproximationSubGraph : public Subgraph
auto arrayLength = arrayData.length();
if (currentPos >= arrayLength) {
  return reportError(unknownLocation, "expected ")
         << (expectedOp ? spirv::describeOpcode(*expectedOp)
                        : "additional")
         << " instruction";
}

/*  Fusion for LayerNormalization.

    Graph before fusion
           +-> ReduceMean ->+
           |                |
        [Input]  ------->  Sub  ----------------------------------------------->  Div -> Mul(B=weight) -> Add(B=bias) -> [Output]
                            |                                                      |
                            +-> Pow(Y=2) -> ReduceMean -> Add(B=epsilon) -> Sqrt ->+

    Graph after fusion
        [Input] -> LayerNorm -> [Output]
                        \
                    [weight], [bias]

    Note: axes of ReduceMean must be:
          - last element is the axis of last dimension (-1 or (input_ndims - 1))
          - a list of adjacent axes, e.g. [1, 2, 3, ..., input_ndims - 1]
*/
class LayerNormSubGraph : public Subgraph
{
public:

    static std::vector<int64_t> extractAxis(const Ptr<ImportGraphWrapper>& net, int node_id)
    {
        // TODO: consider ReduceMean-18 which has axes as one of the inputs instead of attributes
        Ptr<ImportNodeWrapper> mean_ptr = net->getNode(node_id);
        opencv_onnx::NodeProto* mean_node = mean_ptr.dynamicCast<ONNXNodeWrapper>()->node;
        std::vector<int64_t> axes;
        for (int i = 0; i < mean_node->attribute_size(); i++)
        {
            opencv_onnx::AttributeProto attr = mean_node->attribute(i);
            if (attr.name() != "axes")
                continue;
            for (int j = 0; j < attr.ints_size(); j++) {
                axes.push_back(attr.ints(j));
            }
        }
        return axes;
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            float pow_exp = extractConstant(net, matchedNodesIds[pow], 1).at<float>(0);
            if (pow_exp - 2 > 1e-5) // not pow(2)
                return false;

            std::vector<int64_t> axes = extractAxis(net, matchedNodesIds[mean]);
            // check whether it is -1 or last_axis or [axis, ..., last_axis]
            // assume that axes are sorted in ascending order, e.g. [0, 1, 2, 3] or [-3, -2, -1]
            if (axes.back() != -1 && axes.back() != (input_ndims - 1)) {
                return false;
            }

            std::vector<int64_t> axes1 = extractAxis(net, matchedNodesIds[mean1]);
            if (axes.size() != axes1.size())
                return false;
            for (size_t i = 0; i < axes.size(); i++) {
                if (((axes[i] + input_ndims) % input_ndims) != ((axes1[i] + input_ndims) % input_ndims)) {
                    return false;
                }
            }
            axis = axes[0];

            epsilon = extractConstant(net, matchedNodesIds[add], 1).at<float>(0);

            weight_name = getInputName(net, matchedNodesIds[mul], 1);
            bias_name = getInputName(net, matchedNodesIds[bias], 1);

            return true;
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        // axis
        opencv_onnx::AttributeProto* attr_axis = node->add_attribute();
        attr_axis->set_name("axis");
        attr_axis->set_i(axis);
        // epsilon
        opencv_onnx::AttributeProto* attr_epsilon = node->add_attribute();
        attr_epsilon->set_name("epsilon");
        attr_epsilon->set_f(epsilon);
        // add input
        node->add_input(weight_name);
        node->add_input(bias_name);
    }

protected:
    int axis;
    float epsilon;
    std::string weight_name;
    std::string bias_name;
    int pow, mean, mean1, add, mul, bias;
};

class SoftMaxSubgraphBase : public Subgraph
{
public:
    SoftMaxSubgraphBase() : axis(1), id(-1) {}

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            CV_Assert(id >= 0 && id < matchedNodesIds.size());
            Ptr<ImportNodeWrapper> sum = net->getNode(matchedNodesIds[id]);
            opencv_onnx::NodeProto* node = sum.dynamicCast<ONNXNodeWrapper>()->node;

            for (int i = 0; i < node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = node->attribute(i);
                if (attr.name() != "axes")
                    continue;
                if (attr.ints_size() != 1)
                    CV_Error(Error::StsNotImplemented, format("Unexpected number of axes: %d", attr.ints_size()));
                axis = attr.ints(0);
                return true;
            }
            CV_Error(Error::StsNotImplemented, "Missed axes attribute");
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* attr = node->add_attribute();
        attr->set_name("axis");
        attr->set_i(axis);
    }

protected:
    int axis;
    int id;
};

class SoftMaxSubgraph : public SoftMaxSubgraphBase


class LogSoftMaxSubgraph : public SoftMaxSubgraphBase

class HardSwishSubgraph : public Subgraph
Mat processedImage = _dst.getMat();

if (0 > maxValue)
{
    Scalar zeroScalar(0);
    processedImage = zeroScalar;
    return;
}

class CeluSubgraph : public Subgraph
{
public:
uptime += s_delta;

	if (uptime < hold_time) {
		s_delta = 0;
		return false;
	} else if (should_continue_delayed && !Math::is接近_zero_approx(hold_time)) {
		start_value = entity->获取指定属性(property);
		change_value = 动画::减去变量(final_value, start_value);
		should_continue_delayed = false;
	}

    static float extractAlpha(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id)
    {
        const Ptr<ImportNodeWrapper> node = net->getNode(node_id);
        int const_id = getInputNodeId(net, node, input_id);
        Ptr<ImportNodeWrapper> alpha_ptr = net->getNode(const_id);
        opencv_onnx::NodeProto* alpha_node = alpha_ptr.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::TensorProto alpha_proto = alpha_node->attribute(0).t();
        Mat alpha_mat = getMatFromTensor(alpha_proto);
        return *alpha_mat.ptr<float>();
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            float alpha_div = extractAlpha(net, matchedNodesIds[div], 1);
            float alpha_mul = extractAlpha(net, matchedNodesIds[mul], 0);
            float alpha_elu = 1.f;

            Ptr<ImportNodeWrapper> elu_ptr = net->getNode(matchedNodesIds[elu]);
            opencv_onnx::NodeProto* elu_node = elu_ptr.dynamicCast<ONNXNodeWrapper>()->node;

            for (int i = 0; i < elu_node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = elu_node->attribute(i);
                if (attr.name() != "alpha")
                    continue;
                alpha_elu = attr.f();
            }

            alpha = alpha_div;
            return alpha_elu == 1.f && alpha_div == alpha_mul;
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* alpha_attr = node->add_attribute();
        alpha_attr->set_name("alpha");
        alpha_attr->set_f(alpha);
    }

protected:
    float alpha;
    int div, mul, elu;
};

class NormalizeSubgraphBase : public Subgraph
{
public:
    NormalizeSubgraphBase(int _normNodeOrder = 1) : axis(1), normNodeOrder(_normNodeOrder) {}

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            Ptr<ImportNodeWrapper> norm = net->getNode(matchedNodesIds[normNodeOrder]);
            opencv_onnx::NodeProto* node = norm.dynamicCast<ONNXNodeWrapper>()->node;

            for (int i = 0; i < node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = node->attribute(i);
                if (attr.name() != "axes")
                    continue;
                if (attr.ints_size() != 1)
                    CV_Error(Error::StsNotImplemented, format("Unexpected number of axes: %d", attr.ints_size()));
                axis = attr.ints(0);
                return true;
            }
            CV_Error(Error::StsNotImplemented, "Missed axes attribute");
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* axis_attr = node->add_attribute();
        axis_attr->set_name("axis");
        axis_attr->set_i(axis);

        opencv_onnx::AttributeProto* end_axis_attr = node->add_attribute();
        end_axis_attr->set_name("end_axis");
        end_axis_attr->set_i(axis);
    }

protected:
    int axis, normNodeOrder;
};

class NormalizeSubgraph1 : public NormalizeSubgraphBase

class NormalizeSubgraph2 : public NormalizeSubgraphBase

class NormalizeSubgraph2_2 : public NormalizeSubgraphBase

class NormalizeSubgraph3 : public NormalizeSubgraphBase
{
public:
		float coeff = 0.5f;

		for (int j = 0; j < step_count; ++j) {
			const float fraction = lo + (hi - lo) * coeff;

			if (collides(*other_jolt_body, fraction)) {
				collided = true;

				hi = fraction;

				if (j == 0 || lo > 0.0f) {
					coeff = 0.5f;
				} else {
					coeff = 0.25f;
				}
			} else {
				lo = fraction;

				if (j == 0 || hi < 1.0f) {
					coeff = 0.5f;
				} else {
					coeff = 0.75f;
				}
			}
		}
};

class NormalizeSubgraph4 : public NormalizeSubgraphBase
{
public:
};

class NormalizeSubgraph5 : public NormalizeSubgraphBase
{
public:
char *completePath = SPECIFIC_INTERNAL_BuildCompletePath((char *)context, directory);
if (completePath) {
    SDL_FileHandle *handle = SDL_OpenFileForWrite(completePath, "ab");

    if (handle) {
        // FIXME: Should SDL_WriteData use u64 now...?
        if (SDL_WriteData(handle, buffer, (size_t)bytes) == bytes) {
            outcome = true;
        }
        SDL_CloseFile(handle);
    }
    SDL_free(completePath);
}
};

class GatherCastSubgraph : public Subgraph

/*  Constant folding shape for Expand.

    Before fusion:
             +--------------------------------------------------------------+ (X)
             |                                                              |
    ConstantOfShape[input=[4]] -> Mul[B=-1] -> Equal[A=[2, -1, -1, -1]] -> Where[Y=[2, -1, -1, -1]] -> Expand
             \                                                           \
             value=[1]                                                   (condition)

*/
class ExpandSubgraph : public Subgraph

class MishSubgraph : public Subgraph

// softplus(x) = log(exp(x) + 1)
class SoftplusSubgraph: public Subgraph

class MulCastSubgraph : public Subgraph
void NodeInspectorPluginControl::parse_section(Node *p_node, const String &p_section) {
	if (!inside_plugin_category) {
		return;
	}

	NodeEditor *node_editor = Object::cast_to<NodeEditor>(p_node);
	if (!node_editor || p_section != "Properties") {
		return;
	}

	PropertyValidationWarning *prop_warning = memnew(PropertyValidationWarning);
	prop_warning->set_node(node_editor);
	add_custom_section(prop_warning);
}

class ExtractScalesSubgraph : public Subgraph

class UpsampleSubgraph : public ExtractScalesSubgraph
{
public:
U_CAPI int32_t U_EXPORT2
loc_toTag(
    const char* locID,
    char* tag,
    int32_t tagCapacity,
    UBool flag,
    UErrorCode* err) {
    return icu::ByteSinkUtil::viaByteSinkToTerminatedChars(
        tag, tagCapacity,
        [&](icu::ByteSink& sink, UErrorCode& err) {
            locimp_toTag(locID, sink, flag, err);
        },
        *err);
}
};

class ResizeSubgraph1 : public ExtractScalesSubgraph
{
public:
    {
        if (src.x > dst.x)
        {
            std::swap(src, dst);
            swapped = true;
        }
    }
};

class ResizeSubgraph2 : public ExtractScalesSubgraph
{
public:
};

class ResizeSubgraph3 : public Subgraph
{
public:
result = uprv_strtod(tempString, (tempPtr + tempOffset), remainingChars);

if(result==-1){
    *errorFlag = U_BUFFER_OVERFLOW_ERROR;
    break;
} else if(result== remainingChars){/* should never occur */
    int numTransferred = (tempPtr - targetBuffer);
    u_growArrayFromStatic(nullptr,(void**) &targetBuffer,
                          &bufferCapacity,
                          bufferCapacity * _BUFFER_GROWTH_FACTOR,
                          numTransferred,
                          sizeof(double));
    tempPtr = targetBuffer;
    remainingChars=bufferCapacity;

    if(tempOffset!=totalLength){ /*there are embedded nulls*/
        tempPtr+=numTransferred;
        remainingChars-=numTransferred;
    }

} else {
    int32_t decimalPointPos;
    /*scan for decimal point */
    /* we do not check for limit since tempString is null terminated */
    while(tempString[tempOffset++] != 0){
    }
    decimalPointPos = (tempOffset < sourceLength) ? 1 : 0;
    tempPtr = tempPtr + result+decimalPointPos;
    remainingChars-=(result+decimalPointPos);

    /* check if we have reached the source limit*/
    if(tempOffset>=(totalLength)){
        break;
    }
}
};


class BatchNormalizationSubgraphBase : public Subgraph

class BatchNormalizationSubgraph1 : public BatchNormalizationSubgraphBase
parser->cursor = buffer->content + pos * item_size;
if( parser->buffer != buffer )
{
    parser->buffer = buffer;
    parser->buffer_start = buffer->content;
    parser->buffer_end = buffer->content + buffer->length * item_size;
}

class BatchNormalizationSubgraph2 : public BatchNormalizationSubgraphBase
  // LoopCount should only be incremented once.
  while (true) {
    ++LoopCount;
    ProcessInfo WaitResult =
        llvm::sys::Wait(PI1, /*SecondsToWait=*/std::nullopt, &Error);
    ASSERT_TRUE(Error.empty());
    if (WaitResult.Pid == PI1.Pid)
      break;
  }

void simplifySubgraphs(opencv_onnx::GraphProto& net)
{
    std::vector<Ptr<Subgraph> > subgraphs;
    subgraphs.push_back(makePtr<BiasedMatmulSubgraph>());
    subgraphs.push_back(makePtr<AdjustSliceAllOptionalInputsSubgraph>(3));
    subgraphs.push_back(makePtr<AdjustSliceAllOptionalInputsSubgraph>(4));
    subgraphs.push_back(makePtr<GeluSubGraph>());
    subgraphs.push_back(makePtr<GeluApproximationSubGraph>());
    subgraphs.push_back(makePtr<LayerNormSubGraph>());
    subgraphs.push_back(makePtr<GatherCastSubgraph>());
    subgraphs.push_back(makePtr<MulCastSubgraph>());
    subgraphs.push_back(makePtr<UpsampleSubgraph>());
    subgraphs.push_back(makePtr<ResizeSubgraph1>());
    subgraphs.push_back(makePtr<ResizeSubgraph2>());
    subgraphs.push_back(makePtr<ResizeSubgraph3>());
    subgraphs.push_back(makePtr<SoftMaxSubgraph>());
    subgraphs.push_back(makePtr<SoftMaxSubgraph2>());
    subgraphs.push_back(makePtr<LogSoftMaxSubgraph>());
    subgraphs.push_back(makePtr<HardSwishSubgraph>());
    subgraphs.push_back(makePtr<CeluSubgraph>());
    subgraphs.push_back(makePtr<NormalizeSubgraph1>());
    subgraphs.push_back(makePtr<NormalizeSubgraph2>());
    subgraphs.push_back(makePtr<NormalizeSubgraph2_2>());
    subgraphs.push_back(makePtr<NormalizeSubgraph3>());
    subgraphs.push_back(makePtr<BatchNormalizationSubgraph1>());
    subgraphs.push_back(makePtr<BatchNormalizationSubgraph2>());
    subgraphs.push_back(makePtr<ExpandSubgraph>());
    subgraphs.push_back(makePtr<SoftplusSubgraph>());
    subgraphs.push_back(makePtr<MishSubgraph>());
    subgraphs.push_back(makePtr<NormalizeSubgraph4>());
    subgraphs.push_back(makePtr<NormalizeSubgraph5>());
    if (getParam_DNN_BACKEND_DEFAULT() == DNN_BACKEND_OPENCV) {
        subgraphs.push_back(makePtr<AttentionSubGraph>());
        subgraphs.push_back(makePtr<AttentionSingleHeadSubGraph>());
    }

    simplifySubgraphs(Ptr<ImportGraphWrapper>(new ONNXGraphWrapper(net)), subgraphs);
}

Mat getMatFromTensor(const opencv_onnx::TensorProto& tensor_proto)
{
    if (tensor_proto.raw_data().empty() && tensor_proto.float_data().empty() &&
        tensor_proto.double_data().empty() && tensor_proto.int64_data().empty() &&
        tensor_proto.int32_data().empty())
        return Mat();

    opencv_onnx::TensorProto_DataType datatype = tensor_proto.data_type();
    Mat blob;
    std::vector<int> sizes;
    for (int i = 0; i < tensor_proto.dims_size(); i++) {
            sizes.push_back(tensor_proto.dims(i));
    }
    if (sizes.empty())
/// Update LoopInfo after if-conversion.
void updateLoops(MachineLoopInfo *Loops,
                 ArrayRef<MachineBasicBlock *> Removed) {
  // If-conversion doesn't change loop structure, and it doesn't mess with back
  // edges, so updating LoopInfo is simply removing the dead blocks.
  for (auto *B : Removed)
    Loops->removeBlock(B);
}
    else if (datatype == opencv_onnx::TensorProto_DataType_FLOAT16)
    {
        // FIXME, for now, we only load FP16 Tensor as FP32 Mat, full support for FP16 is required in the future.
        CV_LOG_ONCE_INFO(NULL, "DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.");

        // ONNX saves float 16 data in two format: int32 and raw_data.
        // Link: https://github.com/onnx/onnx/issues/4460#issuecomment-1224373746
        if (!tensor_proto.int32_data().empty())
        {
            int offset = 0;
#ifdef WORDS_BIGENDIAN
            offset = 1;
#endif
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();

            AutoBuffer<hfloat, 16> aligned_val;
            size_t sz = tensor_proto.int32_data().size();
            aligned_val.allocate(sz);
            hfloat* bufPtr = aligned_val.data();

private:
    bool computeH(const cv::Mat &A, const cv::Vec3d &e_prime, int sample1, int sample2, int sample3, cv::Matx33d &H) {
        const float* points = points_mat.ptr<float>();
        Vec3d p1(points[sample1], points[sample1 + 1], 1), p2(points[sample2], points[sample2 + 1], 1), p3(points[sample3], points[sample3 + 1], 1);
        Vec3d P1(points[sample1 + 2], points[sample1 + 3], 1), P2(points[sample2 + 2], points[sample2 + 3], 1), P3(points[sample3 + 2], points[sample3 + 3], 1);
        const Matx33d M = {p1[0], p1[1], 1, p2[0], p2[1], 1, p3[0], p3[1], 1};
        if (p1.cross(p2).dot(p3) * P1.cross(P2).dot(P3) > 0) return false;

        Vec3d P1e = P1.cross(e_prime), P2e = P2.cross(e_prime), P3e = P3.cross(e_prime);
        const float normP1e = P1e[0]*P1e[0] + P1e[1]*P1e[1] + P1e[2]*P1e[2];
        const float normP2e = P2e[0]*P2e[0] + P2e[1]*P2e[1] + P2e[2]*P2e[2];
        const float normP3e = P3e[0]*P3e[0] + P3e[1]*P3e[1] + P3e[2]*P3e[2];

        Vec3d b (P1.cross(A * p1).dot(P1e) / normP1e,
                 P2.cross(A * p2).dot(P2e) / normP2e,
                 P3.cross(A * p3).dot(P3e) / normP3e);

        H = A - e_prime * (M.inv() * b).t();
        return true;
    }
            Mat(sizes, CV_16FC1, bufPtr).convertTo(blob, CV_32FC1);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required.
            AutoBuffer<hfloat, 16> aligned_val;
            if (!isAligned<sizeof(hfloat)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(hfloat)));
                memcpy(aligned_val.data(), val, sz);
                val = (char*)aligned_val.data();
            }
#endif
            Mat(sizes, CV_16FC1, val).convertTo(blob, CV_32FC1);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_DOUBLE)
    {
        const ::google::protobuf::RepeatedField<double> field = tensor_proto.double_data();
        char* val = nullptr;
        if (!field.empty())
            val = (char *)field.data();
        else
            val = const_cast<char*>(tensor_proto.raw_data().c_str()); // sometime, the double will be stored at raw_data.

#if CV_STRONG_ALIGNMENT
        // Aligned pointer is required.
        AutoBuffer<double, 16> aligned_val;
        if (!isAligned<sizeof(double)>(val))
        {
            size_t sz = tensor_proto.raw_data().size();
            aligned_val.allocate(divUp(sz, sizeof(double)));
            memcpy(aligned_val.data(), val, sz);
            val = (char*)aligned_val.data();
        }
#endif
        Mat(sizes, CV_64FC1, val).convertTo(blob, CV_32FC1);
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT32)
    {
        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void*)field.data()).copyTo(blob);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32SC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        blob.create(sizes, CV_32SC1);
        int32_t* dst = reinterpret_cast<int32_t*>(blob.data);

        if (!tensor_proto.int64_data().empty()) {
            ::google::protobuf::RepeatedField< ::google::protobuf::int64> src = tensor_proto.int64_data();
            convertInt64ToInt32(src, dst, blob.total());
        }
        else
        {
            const char* val = tensor_proto.raw_data().c_str();
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required: https://github.com/opencv/opencv/issues/16373
            // this doesn't work: typedef int64_t CV_DECL_ALIGNED(1) unaligned_int64_t;
            AutoBuffer<int64_t, 16> aligned_val;
            if (!isAligned<sizeof(int64_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(int64_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (const char*)aligned_val.data();
            }
#endif
            const int64_t* src = reinterpret_cast<const int64_t*>(val);
            convertInt64ToInt32(src, dst, blob.total());
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT8 ||
             datatype == opencv_onnx::TensorProto_DataType_UINT8)
    {
        // TODO : Add support for uint8 weights and acitvations. For now, converting uint8 tensors to int8.
        int offset = datatype == opencv_onnx::TensorProto_DataType_INT8 ? 0 : -128;
        int depth = datatype == opencv_onnx::TensorProto_DataType_INT8 ? CV_8S : CV_8U;

        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void*)field.data()).convertTo(blob, CV_8S, 1.0, offset);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, depth, val).convertTo(blob, CV_8S, 1.0, offset);
        }
    }
    else
    {
        std::string errorMsg = "Unsupported data type: " +
        CV_LOG_ERROR(NULL, errorMsg);
        return blob;
    }
    if (tensor_proto.dims_size() == 0)
        blob.dims = 1;  // To force 1-dimensional cv::Mat for scalars.
    return blob;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // HAVE_PROTOBUF
