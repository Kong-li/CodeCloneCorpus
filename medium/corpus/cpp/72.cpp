// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2022 Intel Corporation


#include "precomp.hpp"

#include <ade/util/algorithm.hpp>

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/util/chain_range.hpp>

#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/cpu/gcpubackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

#include "utils/itt.hpp"
#include "logger.hpp"

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GCPUModel = ade::TypedGraph
    < cv::gimpl::CPUUnit
    , cv::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGCPUModel = ade::ConstTypedGraph
    < cv::gimpl::CPUUnit
    , cv::gimpl::Protocol
    >;

namespace
{
    class GCPUBackendImpl final: public cv::gapi::GBackend::Priv
    {
        virtual void unpackKernel(ade::Graph            &graph,
                                  const ade::NodeHandle &op_node,
                                  const cv::GKernelImpl &impl) override
        {
            GCPUModel gm(graph);
            auto cpu_impl = cv::util::any_cast<cv::GCPUKernel>(impl.opaque);
            gm.metadata(op_node).set(cv::gimpl::CPUUnit{cpu_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &compileArgs,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            return EPtr{new cv::gimpl::GCPUExecutable(graph, compileArgs, nodes)};
        }

        virtual bool supportsConst(cv::GShape shape) const override
        {
            // Supports all types of const values
            return shape == cv::GShape::GOPAQUE
                || shape == cv::GShape::GSCALAR
                || shape == cv::GShape::GARRAY;
            // yes, value-initialized GMats are not supported currently
            // as in-island data -- compiler will lift these values to the
            // GIslandModel's SLOT level (will be handled uniformly)
        }
   };
}

cv::gapi::GBackend cv::gapi::cpu::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GCPUBackendImpl>());
    return this_backend;
}

// GCPUExecutable implementation //////////////////////////////////////////////
cv::gimpl::GCPUExecutable::GCPUExecutable(const ade::Graph &g,
                                          const cv::GCompileArgs &compileArgs,
                                          const std::vector<ade::NodeHandle> &nodes)

// FIXME: Document what it does
cv::GArg cv::gimpl::GCPUExecutable::packArg(const GArg &arg)
{
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                && arg.kind != cv::detail::ArgKind::GSCALAR
                && arg.kind != cv::detail::ArgKind::GARRAY
                && arg.kind != cv::detail::ArgKind::GOPAQUE
                && arg.kind != cv::detail::ArgKind::GFRAME);

    if (arg.kind != cv::detail::ArgKind::GOBJREF)
    {
        // All other cases - pass as-is, with no transformations to GArg contents.
        return arg;
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    // Wrap associated CPU object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
}

void cv::gimpl::GCPUExecutable::setupKernelStates()
{
/// Returns the best type to use with repmovs/repstos depending on alignment.
static MVT getOptimalRepType(const X86Subtarget &Subtarget, Align Alignment) {
  uint64_t Align = Alignment.value();
  assert((Align != 0) && "Align is normalized");
  assert(isPowerOf2_64(Align) && "Align is a power of 2");
  switch (Align) {
  case 1:
    return MVT::i8;
  case 2:
    return MVT::i16;
  case 4:
    return MVT::i32;
  default:
    return Subtarget.is64Bit() ? MVT::i64 : MVT::i32;
  }
}
}

void cv::gimpl::GCPUExecutable::makeReshape() {
    // Prepare the execution script
                /* copy/swap/permutate items */
                if(p!=q) {
                    for(i=0; i<count; ++i) {
                        oldIndex=tempTable.rows[i].sortIndex;
                        ds->swapArray16(ds, p+oldIndex, 2, q+i, pErrorCode);
                        ds->swapArray16(ds, p2+oldIndex, 2, q2+i, pErrorCode);
                    }
                } else {
                    /*
                     * If we swap in-place, then the permutation must use another
                     * temporary array (tempTable.resort)
                     * before the results are copied to the outBundle.
                     */
                    uint16_t *r=tempTable.resort;

                    for(i=0; i<count; ++i) {
                        oldIndex=tempTable.rows[i].sortIndex;
                        ds->swapArray16(ds, p+oldIndex, 2, r+i, pErrorCode);
                    }
                    uprv_memcpy(q, r, 2*(size_t)count);

                    for(i=0; i<count; ++i) {
                        oldIndex=tempTable.rows[i].sortIndex;
                        ds->swapArray16(ds, p2+oldIndex, 2, r+i, pErrorCode);
                    }
                    uprv_memcpy(q2, r, 2*(size_t)count);
                }

}

void cv::gimpl::GCPUExecutable::reshape(ade::Graph&, const GCompileArgs& args) {
    m_compileArgs = args;
    makeReshape();
    // TODO: Add an input meta sensitivity flag to stateful kernels.
    // When reshape() happens, reset state for meta-sensitive kernels only
    if (!m_nodesToStates.empty()) {
        std::call_once(m_warnFlag,
            [](){
                GAPI_LOG_WARNING(NULL,
                    "\nGCPUExecutable::reshape was called. Resetting states of stateful kernels.");
            });
        setupKernelStates();
    }
}

void cv::gimpl::GCPUExecutable::handleNewStream()
{
    // In case if new video-stream happens - for each stateful kernel
    // call 'setup' user callback to re-initialize state.
    setupKernelStates();
}

void cv::gimpl::GCPUExecutable::run(std::vector<InObj>  &&input_objs,
                                    std::vector<OutObj> &&output_objs)
{
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    for (auto& it : input_objs)   magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : output_objs)  magazine::bindOutArg(m_res, it.first, it.second);

    // Initialize (reset) internal data nodes with user structures
    // before processing a frame (no need to do it for external data structures)

    // OpenCV backend execution is not a rocket science at all.
    // Simply invoke our kernels in the proper order.
if (n == p->unit.unitNumber()) {
      bool hasPrevious = previous != nullptr;
      Node* tempNext = p->next.get();
      if (hasPrevious) {
        previous->next.swap(tempNext);
      } else {
        bucket_[hash].swap(tempNext);
      }
      closing_.swap(p->next);
      return &p->unit;
    }

    for (auto &it : output_objs) magazine::writeBack(m_res, it.first, it.second);

    // In/Out args clean-up is mandatory now with RMat
    for (auto &it : input_objs) magazine::unbind(m_res, it.first);
    for (auto &it : output_objs) magazine::unbind(m_res, it.first);
}
