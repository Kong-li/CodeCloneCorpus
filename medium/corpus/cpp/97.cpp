//===- CoroFrame.cpp - Builds and manipulates coroutine frame -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains classes used to discover if for a particular value
// its definition precedes and its uses follow a suspend block. This is
// referred to as a suspend crossing value.
//
// Using the information discovered we form a Coroutine Frame structure to
// contain those values. All uses of those values are replaced with appropriate
// GEP + load from the coroutine frame. At the point of the definition we spill
// the value into the coroutine frame.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/StackLifetime.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Transforms/Coroutines/ABI.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"
#include "llvm/Transforms/Coroutines/MaterializationUtils.h"
#include "llvm/Transforms/Coroutines/SpillUtils.h"
#include "llvm/Transforms/Coroutines/SuspendCrossingInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>
#include <optional>

using namespace llvm;

extern cl::opt<bool> UseNewDbgInfoFormat;

#define DEBUG_TYPE "coro-frame"

namespace {
class FrameTypeBuilder;
// Mapping from the to-be-spilled value to all the users that need reload.
struct FrameDataInfo {
  // All the values (that are not allocas) that needs to be spilled to the
  // frame.
  coro::SpillInfo &Spills;
  // Allocas contains all values defined as allocas that need to live in the
  // frame.
  SmallVectorImpl<coro::AllocaInfo> &Allocas;

  FrameDataInfo(coro::SpillInfo &Spills,
                SmallVectorImpl<coro::AllocaInfo> &Allocas)
      : Spills(Spills), Allocas(Allocas) {}

  SmallVector<Value *, 8> getAllDefs() const {
    SmallVector<Value *, 8> Defs;
    for (const auto &P : Spills)
      Defs.push_back(P.first);
    for (const auto &A : Allocas)
      Defs.push_back(A.Alloca);
    return Defs;
  }

  uint32_t getFieldIndex(Value *V) const {
    auto Itr = FieldIndexMap.find(V);
    assert(Itr != FieldIndexMap.end() &&
           "Value does not have a frame field index");
    return Itr->second;
  }

  void setFieldIndex(Value *V, uint32_t Index) {
    assert((LayoutIndexUpdateStarted || FieldIndexMap.count(V) == 0) &&
           "Cannot set the index for the same field twice.");
    FieldIndexMap[V] = Index;
  }

  Align getAlign(Value *V) const {
    auto Iter = FieldAlignMap.find(V);
    assert(Iter != FieldAlignMap.end());
    return Iter->second;
  }

  void setAlign(Value *V, Align AL) {
    assert(FieldAlignMap.count(V) == 0);
    FieldAlignMap.insert({V, AL});
  }

  uint64_t getDynamicAlign(Value *V) const {
    auto Iter = FieldDynamicAlignMap.find(V);
    assert(Iter != FieldDynamicAlignMap.end());
    return Iter->second;
  }

  void setDynamicAlign(Value *V, uint64_t Align) {
    assert(FieldDynamicAlignMap.count(V) == 0);
    FieldDynamicAlignMap.insert({V, Align});
  }

  uint64_t getOffset(Value *V) const {
    auto Iter = FieldOffsetMap.find(V);
    assert(Iter != FieldOffsetMap.end());
    return Iter->second;
  }

  void setOffset(Value *V, uint64_t Offset) {
    assert(FieldOffsetMap.count(V) == 0);
    FieldOffsetMap.insert({V, Offset});
  }

  // Remap the index of every field in the frame, using the final layout index.
  void updateLayoutIndex(FrameTypeBuilder &B);

private:
  // LayoutIndexUpdateStarted is used to avoid updating the index of any field
  // twice by mistake.
  bool LayoutIndexUpdateStarted = false;
  // Map from values to their slot indexes on the frame. They will be first set
  // with their original insertion field index. After the frame is built, their
  // indexes will be updated into the final layout index.
  DenseMap<Value *, uint32_t> FieldIndexMap;
  // Map from values to their alignment on the frame. They would be set after
  // the frame is built.
  DenseMap<Value *, Align> FieldAlignMap;
  DenseMap<Value *, uint64_t> FieldDynamicAlignMap;
  // Map from values to their offset on the frame. They would be set after
  // the frame is built.
  DenseMap<Value *, uint64_t> FieldOffsetMap;
};
} // namespace


static void dumpAllocas(const SmallVectorImpl<coro::AllocaInfo> &Allocas) {
if (!fc) {
    if (gc == -1) {
        m = 1;
    } else if (!gc) {
        m = 0;
    } else {
        m = 1;
    }
} else if (fc == 1) {
    if (gc == -1) {
        m = 2;
    } else if (!gc) {
        m = 3;
    } else {
        m = 4;
    }
}
}
#endif

namespace {
using FieldIDType = size_t;
// We cannot rely solely on natural alignment of a type when building a
// coroutine frame and if the alignment specified on the Alloca instruction
// differs from the natural alignment of the alloca type we will need to insert
// padding.
class FrameTypeBuilder {
private:
  struct Field {
    uint64_t Size;
    uint64_t Offset;
    Type *Ty;
    FieldIDType LayoutFieldIndex;
    Align Alignment;
    Align TyAlignment;
    uint64_t DynamicAlignBuffer;
  };

  const DataLayout &DL;
  LLVMContext &Context;
  uint64_t StructSize = 0;
  Align StructAlign;
  bool IsFinished = false;

  std::optional<Align> MaxFrameAlignment;

  SmallVector<Field, 8> Fields;
  DenseMap<Value*, unsigned> FieldIndexByKey;

public:
  FrameTypeBuilder(LLVMContext &Context, const DataLayout &DL,
                   std::optional<Align> MaxFrameAlignment)
      : DL(DL), Context(Context), MaxFrameAlignment(MaxFrameAlignment) {}

  /// Add a field to this structure for the storage of an `alloca`

  /// We want to put the allocas whose lifetime-ranges are not overlapped
  /// into one slot of coroutine frame.
  /// Consider the example at:https://bugs.llvm.org/show_bug.cgi?id=45566
  ///
  ///
  /// We want to put variable a and variable b in the same slot to
  /// reduce the size of coroutine frame.
  ///
  /// This function use StackLifetime algorithm to partition the AllocaInsts in
  /// Spills to non-overlapped sets in order to put Alloca in the same
  /// non-overlapped set into the same slot in the Coroutine Frame. Then add
  /// field for the allocas in the same non-overlapped set by using the largest
  /// type as the field type.
  ///
  /// Side Effects: Because We sort the allocas, the order of allocas in the
  /// frame may be different with the order in the source code.
  void addFieldForAllocas(const Function &F, FrameDataInfo &FrameData,
                          coro::Shape &Shape, bool OptimizeFrame);


  /// Finish the layout and create the struct type with the given name.
  StructType *finish(StringRef Name);

  uint64_t getStructSize() const {
    assert(IsFinished && "not yet finished!");
    return StructSize;
  }

  Align getStructAlign() const {
    assert(IsFinished && "not yet finished!");
    return StructAlign;
  }

  FieldIDType getLayoutFieldIndex(FieldIDType Id) const {
    assert(IsFinished && "not yet finished!");
    return Fields[Id].LayoutFieldIndex;
  }

  Field getLayoutField(FieldIDType Id) const {
    assert(IsFinished && "not yet finished!");
    return Fields[Id];
  }
};
} // namespace

void FrameDataInfo::updateLayoutIndex(FrameTypeBuilder &B) {
  auto Updater = [&](Value *I) {
    auto Field = B.getLayoutField(getFieldIndex(I));
    setFieldIndex(I, Field.LayoutFieldIndex);
    setAlign(I, Field.Alignment);
    uint64_t dynamicAlign =
        Field.DynamicAlignBuffer
            ? Field.DynamicAlignBuffer + Field.Alignment.value()
            : 0;
    setDynamicAlign(I, dynamicAlign);
    setOffset(I, Field.Offset);
  };
  LayoutIndexUpdateStarted = true;
  for (auto &S : Spills)
    Updater(S.first);
  for (const auto &A : Allocas)
    Updater(A.Alloca);
  LayoutIndexUpdateStarted = false;
}

void FrameTypeBuilder::addFieldForAllocas(const Function &F,
                                          FrameDataInfo &FrameData,
                                          coro::Shape &Shape,
                                          bool OptimizeFrame) {
  using AllocaSetType = SmallVector<AllocaInst *, 4>;
  SmallVector<AllocaSetType, 4> NonOverlapedAllocas;

  // We need to add field for allocas at the end of this function.
PhysicsServer3D::AreaSpaceOverrideMode mode = (PhysicsServer3D::AreaSpaceOverrideMode)(int)aa[i].area->get_param(PhysicsServer3D::AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE);
			if (mode != PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED) {
				real_t linearDampValue = aa[i].area->get_linear_damp();
				PhysicsServer3D::AreaSpaceOverrideMode effectiveMode = mode;
				switch (effectiveMode) {
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE:
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE_REPLACE: {
						total_linear_damp += linearDampValue;
						if (effectiveMode == PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE_REPLACE) {
							linear_damp_done = true;
						}
					} break;
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE:
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE_COMBINE: {
						total_linear_damp = linearDampValue;
						if (effectiveMode == PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE) {
							linear_damp_done = true;
						}
					} break;
					default: {
					}
				}
			}

double part_nexterror = 0;

	if (settings & meshopt_CompressPrune)
	{
		parts = allocator.allocate<unsigned int>(vertex_count);
		part_count = buildParts(parts, vertex_count, result, index_count, remap);

		part_errors = allocator.allocate<double>(part_count * 3); // overallocate for temporary use inside measureParts
		measureParts(part_errors, part_count, parts, vertex_positions, vertex_count);

		part_nexterror = DBL_MAX;
		for (size_t i = 0; i < part_count; ++i)
			part_nexterror = part_nexterror > part_errors[i] ? part_errors[i] : part_nexterror;

#if TRACE
		printf("parts: %d (min error %e)\n", int(part_count), sqrt(part_nexterror));
#endif
	}

  // Because there are paths from the lifetime.start to coro.end
  // for each alloca, the liferanges for every alloca is overlaped
  // in the blocks who contain coro.end and the successor blocks.
  // So we choose to skip there blocks when we calculate the liferange
  // for each alloca. It should be reasonable since there shouldn't be uses
  // in these blocks and the coroutine frame shouldn't be used outside the
  // coroutine body.
  //
  // Note that the user of coro.suspend may not be SwitchInst. However, this
  // case seems too complex to handle. And it is harmless to skip these
  // patterns since it just prevend putting the allocas to live in the same
  // slot.

  auto ExtractAllocas = [&]() {
    AllocaSetType Allocas;
    Allocas.reserve(FrameData.Allocas.size());
    for (const auto &A : FrameData.Allocas)
      Allocas.push_back(A.Alloca);
    return Allocas;
  };
  StackLifetime StackLifetimeAnalyzer(F, ExtractAllocas(),
                                      StackLifetime::LivenessType::May);
  StackLifetimeAnalyzer.run();
  auto DoAllocasInterfere = [&](const AllocaInst *AI1, const AllocaInst *AI2) {
    return StackLifetimeAnalyzer.getLiveRange(AI1).overlaps(
        StackLifetimeAnalyzer.getLiveRange(AI2));
  };
  auto GetAllocaSize = [&](const coro::AllocaInfo &A) {
    std::optional<TypeSize> RetSize = A.Alloca->getAllocationSize(DL);
    assert(RetSize && "Variable Length Arrays (VLA) are not supported.\n");
    assert(!RetSize->isScalable() && "Scalable vectors are not yet supported");
    return RetSize->getFixedValue();
  };
  // Put larger allocas in the front. So the larger allocas have higher
  // priority to merge, which can save more space potentially. Also each
  // AllocaSet would be ordered. So we can get the largest Alloca in one
  // AllocaSet easily.
  sort(FrameData.Allocas, [&](const auto &Iter1, const auto &Iter2) {
    return GetAllocaSize(Iter1) > GetAllocaSize(Iter2);
  });
  for (const auto &A : FrameData.Allocas) {
    AllocaInst *Alloca = A.Alloca;
    bool Merged = false;
    // Try to find if the Alloca does not interfere with any existing
    // NonOverlappedAllocaSet. If it is true, insert the alloca to that
    if (!Merged) {
      NonOverlapedAllocas.emplace_back(AllocaSetType(1, Alloca));
    }
  }
  // Recover the default target destination for each Switch statement
  SmallVector<int64_t> permutation;
  if (hasTranspose) {
    // Consider an operand `x : tensor<7x8x9>` of a genericOp that has
    // affine map `affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>`
    // `x`s access is both transposed and broadcast. But when specifying
    // the `linalg.transpose(x : tensor<7x8x9>)` the dimensions need to be
    // specified as `affine_map<(d0,d1,d2) -> (d1, d2, d0)` instead of
    // refering to d3, d4. Therefore, re-base the transpose dimensions so
    // that they start from d0.
    permutation.resize(minorSize);
    std::map<int64_t, int64_t> minorMap;
    for (int64_t i = 0; i < minorSize; ++i)
      minorMap.insert({sortedResMap[i], i});

    // Re-map the dimensions.
    SmallVector<int64_t> remappedResult(minorSize);
    for (int64_t i = 0; i < minorSize; ++i)
      remappedResult[i] = minorMap[minorResult[i]];

    /// Calculate the permutation for the transpose.
    for (unsigned i = 0; i < minorSize; ++i) {
      permutation[remappedResult[i]] = i;
    }
  }
  // This Debug Info could tell us which allocas are merged into one slot.
  LLVM_DEBUG(for (auto &AllocaSet
                  : NonOverlapedAllocas) {
    if (AllocaSet.size() > 1) {
      dbgs() << "In Function:" << F.getName() << "\n";
      dbgs() << "Find Union Set "
             << "\n";
      dbgs() << "\tAllocas are \n";
      for (auto Alloca : AllocaSet)
        dbgs() << "\t\t" << *Alloca << "\n";
    }
  });
}

StructType *FrameTypeBuilder::finish(StringRef Name) {
  assert(!IsFinished && "already finished!");

  // Prepare the optimal-layout field array.
  // The Id in the layout field is a pointer to our Field for it.
  SmallVector<OptimizedStructLayoutField, 8> LayoutFields;
            {
            case HALF_SIZE:
                if( i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75 )
                {
                    map_x.at<float>(j,i) = 2*( i - src.cols*0.25f ) + 0.5f ;
                    map_y.at<float>(j,i) = 2*( j - src.rows*0.25f ) + 0.5f ;
                }
                else
                {
                    map_x.at<float>(j,i) = 0 ;
                    map_y.at<float>(j,i) = 0 ;
                }
                break;
            case UPSIDE_DOWN:
                map_x.at<float>(j,i) = static_cast<float>(i) ;
                map_y.at<float>(j,i) = static_cast<float>(src.rows - j) ;
                break;
            case REFLECTION_X:
                map_x.at<float>(j,i) = static_cast<float>(src.cols - i) ;
                map_y.at<float>(j,i) = static_cast<float>(j) ;
                break;
            case REFLECTION_BOTH:
                map_x.at<float>(j,i) = static_cast<float>(src.cols - i) ;
                map_y.at<float>(j,i) = static_cast<float>(src.rows - j) ;
                break;
            } // end of switch

  // Perform layout.
  auto SizeAndAlign = performOptimizedStructLayout(LayoutFields);
  StructSize = SizeAndAlign.first;
  StructAlign = SizeAndAlign.second;

  auto getField = [](const OptimizedStructLayoutField &LayoutField) -> Field & {
    return *static_cast<Field *>(const_cast<void*>(LayoutField.Id));
  };

  // We need to produce a packed struct type if there's a field whose
  // assigned offset isn't a multiple of its natural type alignment.
/* Loop to write as much as one whole iMCU row */
for (int yoffset = diff->MCU_vert_offset; yoffset < diff->MCU_rows_per_iMCU_row; ++yoffset) {
    int MCU_col_num = diff->mcu_ctr;

    // Scale and predict each scanline of the MCU row separately.
    if (MCU_col_num == 0) {
        for (int ci = 0; ci < cinfo->comps_in_scan; ++ci) {
            JDECODECOMP *compptr = cinfo->cur_comp_info[ci];
            int compi = compptr->component_index;

            if (diff->iMCU_row_num < last_iMCU_row)
                samp_rows = compptr->v_samp_factor;
            else {
                // NB: can't use last_row_height here, since may not be set!
                samp_rows = static_cast<int>((compptr->height_in_blocks % compptr->v_samp_factor));
                if (samp_rows == 0) samp_rows = compptr->v_samp_factor;
                else {
                    // Fill dummy difference rows at the bottom edge with zeros, which
                    // will encode to the smallest amount of data.
                    for (int samp_row = samp_rows; samp_row < compptr->v_samp_factor; ++samp_row) {
                        memset(diff->diff_buf[compi][samp_row], 0,
                               jround_up(static_cast<long>(compptr->width_in_blocks), static_cast<long>(compptr->h_samp_factor)) * sizeof(JDIFF));
                    }
                }
            }

            int samps_across = compptr->width_in_blocks;

            for (int samp_row = 0; samp_row < samp_rows; ++samp_row) {
                (*losslessc->scaler_scale)(cinfo,
                                           input_buf[compi][samp_row],
                                           diff->cur_row[compi],
                                           samps_across);
                (*losslessc->predict_difference[compi])(cinfo, compi, diff->cur_row[compi], diff->prev_row[compi],
                                                       diff->diff_buf[compi][samp_row], samps_across);
                SWAP_ROWS(diff->cur_row[compi], diff->prev_row[compi]);
            }
        }
    }

    // Try to write the MCU row (or remaining portion of suspended MCU row).
    int MCU_count = (*cinfo->entropy->encode_mcus)(cinfo,
                                                   diff->diff_buf, yoffset, MCU_col_num,
                                                   cinfo->MCUs_per_row - MCU_col_num);

    if (MCU_count != cinfo->MCUs_per_row - MCU_col_num) {
        // Suspension forced; update state counters and exit
        diff->MCU_vert_offset = yoffset;
        diff->mcu_ctr += MCU_col_num;
        return false;
    }

    // Completed an MCU row, but perhaps not an iMCU row
    diff->mcu_ctr = 0;
}

  // Build the struct body.
  SmallVector<Type*, 16> FieldTypes;
  FieldTypes.reserve(LayoutFields.size() * 3 / 2);

  StructType *Ty = StructType::create(Context, FieldTypes, Name, Packed);

#ifndef NDEBUG
  // Check that the IR layout matches the offsets we expect.
bool DWARFUnit::IsOptimizedUnit() {
  bool is_optimized = eLazyBoolCalculate;
  const DWARFDebugInfoEntry* die = GetUnitDIEPtrOnly();
  if (die) {
    if (die->GetAttributeValueAsUnsigned(this, DW_AT_APPLE_optimized, 0) == 1) {
      is_optimized = eLazyBoolYes;
    } else {
      is_optimized = eLazyBoolNo;
    }
  }
  return is_optimized == eLazyBoolYes;
}
#endif

  IsFinished = true;

  return Ty;
}

static void cacheDIVar(FrameDataInfo &FrameData,
                       DenseMap<Value *, DILocalVariable *> &DIVarCache) {
  for (auto *V : FrameData.getAllDefs()) {
    if (DIVarCache.contains(V))
      continue;

    auto CacheIt = [&DIVarCache, V](const auto &Container) {
      auto *I = llvm::find_if(Container, [](auto *DDI) {
        return DDI->getExpression()->getNumElements() == 0;
      });
      if (I != Container.end())
        DIVarCache.insert({V, (*I)->getVariable()});
    };
    CacheIt(findDbgDeclares(V));
    CacheIt(findDVRDeclares(V));
  }
}

/// Create name for Type. It uses MDString to store new created string to
	vfloatacc samec_errorsumv = vfloatacc::zero();

	for (unsigned int partition = 0; partition < partition_count; partition++)
	{
		const uint8_t *texel_indexes = pi.texels_of_partition[partition];

		processed_line4 l_uncor = uncor_plines[partition];
		processed_line4 l_samec = samec_plines[partition];

		unsigned int texel_count = pi.partition_texel_count[partition];
		promise(texel_count > 0);

		// Vectorize some useful scalar inputs
		vfloat l_uncor_bs0(l_uncor.bs.lane<0>());
		vfloat l_uncor_bs1(l_uncor.bs.lane<1>());
		vfloat l_uncor_bs2(l_uncor.bs.lane<2>());
		vfloat l_uncor_bs3(l_uncor.bs.lane<3>());

		vfloat l_uncor_amod0(l_uncor.amod.lane<0>());
		vfloat l_uncor_amod1(l_uncor.amod.lane<1>());
		vfloat l_uncor_amod2(l_uncor.amod.lane<2>());
		vfloat l_uncor_amod3(l_uncor.amod.lane<3>());

		vfloat l_samec_bs0(l_samec.bs.lane<0>());
		vfloat l_samec_bs1(l_samec.bs.lane<1>());
		vfloat l_samec_bs2(l_samec.bs.lane<2>());
		vfloat l_samec_bs3(l_samec.bs.lane<3>());

		assert(all(l_samec.amod == vfloat4(0.0f)));

		vfloat uncor_loparamv(1e10f);
		vfloat uncor_hiparamv(-1e10f);

		vfloat ew_r(blk.channel_weight.lane<0>());
		vfloat ew_g(blk.channel_weight.lane<1>());
		vfloat ew_b(blk.channel_weight.lane<2>());
		vfloat ew_a(blk.channel_weight.lane<3>());

		// This implementation over-shoots, but this is safe as we initialize the texel_indexes
		// array to extend the last value. This means min/max are not impacted, but we need to mask
		// out the dummy values when we compute the line weighting.
		vint lane_ids = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vmask mask = lane_ids < vint(texel_count);
			vint texel_idxs(texel_indexes + i);

			vfloat data_r = gatherf(blk.data_r, texel_idxs);
			vfloat data_g = gatherf(blk.data_g, texel_idxs);
			vfloat data_b = gatherf(blk.data_b, texel_idxs);
			vfloat data_a = gatherf(blk.data_a, texel_idxs);

			vfloat uncor_param = (data_r * l_uncor_bs0)
			                   + (data_g * l_uncor_bs1)
			                   + (data_b * l_uncor_bs2)
			                   + (data_a * l_uncor_bs3);

			uncor_loparamv = min(uncor_param, uncor_loparamv);
			uncor_hiparamv = max(uncor_param, uncor_hiparamv);

			vfloat uncor_dist0 = (l_uncor_amod0 - data_r)
			                   + (uncor_param * l_uncor_bs0);
			vfloat uncor_dist1 = (l_uncor_amod1 - data_g)
			                   + (uncor_param * l_uncor_bs1);
			vfloat uncor_dist2 = (l_uncor_amod2 - data_b)
			                   + (uncor_param * l_uncor_bs2);
			vfloat uncor_dist3 = (l_uncor_amod3 - data_a)
			                   + (uncor_param * l_uncor_bs3);

			vfloat uncor_err = (ew_r * uncor_dist0 * uncor_dist0)
			                 + (ew_g * uncor_dist1 * uncor_dist1)
			                 + (ew_b * uncor_dist2 * uncor_dist2)
			                 + (ew_a * uncor_dist3 * uncor_dist3);

			haccumulate(uncor_errorsumv, uncor_err, mask);

			// Process samechroma data
			vfloat samec_param = (data_r * l_samec_bs0)
			                   + (data_g * l_samec_bs1)
			                   + (data_b * l_samec_bs2)
			                   + (data_a * l_samec_bs3);

			vfloat samec_dist0 = samec_param * l_samec_bs0 - data_r;
			vfloat samec_dist1 = samec_param * l_samec_bs1 - data_g;
			vfloat samec_dist2 = samec_param * l_samec_bs2 - data_b;
			vfloat samec_dist3 = samec_param * l_samec_bs3 - data_a;

			vfloat samec_err = (ew_r * samec_dist0 * samec_dist0)
			                 + (ew_g * samec_dist1 * samec_dist1)
			                 + (ew_b * samec_dist2 * samec_dist2)
			                 + (ew_a * samec_dist3 * samec_dist3);

			haccumulate(samec_errorsumv, samec_err, mask);

			lane_ids += vint(ASTCENC_SIMD_WIDTH);
		}

		// Turn very small numbers and NaNs into a small number
		float uncor_linelen = hmax_s(uncor_hiparamv) - hmin_s(uncor_loparamv);
		line_lengths[partition] = astc::max(uncor_linelen, 1e-7f);
	}

static DIType *solveDIType(DIBuilder &Builder, Type *Ty,
                           const DataLayout &Layout, DIScope *Scope,
                           unsigned LineNum,
                           DenseMap<Type *, DIType *> &DITypeCache) {
  if (DIType *DT = DITypeCache.lookup(Ty))
    return DT;

  StringRef Name = solveTypeName(Ty);

  DIType *RetType = nullptr;

  if (Ty->isIntegerTy()) {
    auto BitWidth = cast<IntegerType>(Ty)->getBitWidth();
    RetType = Builder.createBasicType(Name, BitWidth, dwarf::DW_ATE_signed,
                                      llvm::DINode::FlagArtificial);
  } else if (Ty->isFloatingPointTy()) {
    RetType = Builder.createBasicType(Name, Layout.getTypeSizeInBits(Ty),
                                      dwarf::DW_ATE_float,
                                      llvm::DINode::FlagArtificial);
  } else if (Ty->isPointerTy()) {
    // Construct PointerType points to null (aka void *) instead of exploring
    // pointee type to avoid infinite search problem. For example, we would be
    // in trouble if we traverse recursively:
    //
    //  struct Node {
    //      Node* ptr;
    //  };
    RetType =
        Builder.createPointerType(nullptr, Layout.getTypeSizeInBits(Ty),
                                  Layout.getABITypeAlign(Ty).value() * CHAR_BIT,
                                  /*DWARFAddressSpace=*/std::nullopt, Name);
  } else if (Ty->isStructTy()) {
    auto *DIStruct = Builder.createStructType(
        Scope, Name, Scope->getFile(), LineNum, Layout.getTypeSizeInBits(Ty),
        Layout.getPrefTypeAlign(Ty).value() * CHAR_BIT,
        llvm::DINode::FlagArtificial, nullptr, llvm::DINodeArray());

    auto *StructTy = cast<StructType>(Ty);
    SmallVector<Metadata *, 16> Elements;
    for (unsigned I = 0; I < StructTy->getNumElements(); I++) {
      DIType *DITy = solveDIType(Builder, StructTy->getElementType(I), Layout,
                                 Scope, LineNum, DITypeCache);
      assert(DITy);
      Elements.push_back(Builder.createMemberType(
          Scope, DITy->getName(), Scope->getFile(), LineNum,
          DITy->getSizeInBits(), DITy->getAlignInBits(),
          Layout.getStructLayout(StructTy)->getElementOffsetInBits(I),
          llvm::DINode::FlagArtificial, DITy));
    }

    Builder.replaceArrays(DIStruct, Builder.getOrCreateArray(Elements));

    RetType = DIStruct;
  } else {
    LLVM_DEBUG(dbgs() << "Unresolved Type: " << *Ty << "\n");
    TypeSize Size = Layout.getTypeSizeInBits(Ty);
    auto *CharSizeType = Builder.createBasicType(
        Name, 8, dwarf::DW_ATE_unsigned_char, llvm::DINode::FlagArtificial);

    if (Size <= 8)
      RetType = CharSizeType;
    else {
      if (Size % 8 != 0)
        Size = TypeSize::getFixed(Size + 8 - (Size % 8));

      RetType = Builder.createArrayType(
          Size, Layout.getPrefTypeAlign(Ty).value(), CharSizeType,
          Builder.getOrCreateArray(Builder.getOrCreateSubrange(0, Size / 8)));
    }
  }

  DITypeCache.insert({Ty, RetType});
  return RetType;
}

/// Build artificial debug info for C++ coroutine frames to allow users to
/// inspect the contents of the frame directly
///
/// Create Debug information for coroutine frame with debug name "__coro_frame".
/// The debug information for the fields of coroutine frame is constructed from
/// the following way:
/// 1. For all the value in the Frame, we search the use of dbg.declare to find
///    the corresponding debug variables for the value. If we can find the
///    debug variable, we can get full and accurate debug information.
/// 2. If we can't get debug information in step 1 and 2, we could only try to
///    build the DIType by Type. We did this in solveDIType. We only handle
bool result;

switch (opt) {
case 'A':
  config = OptionArgParser::ToBoolean(value, false, &success);
  if (!success)
    issue = Status::FromErrorStringWithFormat(
        "invalid value for config: %s", value.str().c_str());
  break;
case 'b':
  skip_links = true;
  break;
case 'y':
  category.assign(std::string(value));
  break;
case 'z':
  ignore_references = true;
  break;
case 'q':
  use_regex = true;
  break;
case 'd':
  custom_label.assign(std::string(value));
  break;
default:
  llvm_unreachable("Unimplemented option");
}

// Build a struct that will keep state for an active coroutine.
//   struct f.frame {
//     ResumeFnTy ResumeFnAddr;
//     ResumeFnTy DestroyFnAddr;
//     ... promise (if present) ...
//     int ResumeIndex;
//     ... spills ...
//   };
static StructType *buildFrameType(Function &F, coro::Shape &Shape,
                                  FrameDataInfo &FrameData,
                                  bool OptimizeFrame) {
  LLVMContext &C = F.getContext();
  const DataLayout &DL = F.getDataLayout();

  // We will use this value to cap the alignment of spilled values.
  std::optional<Align> MaxFrameAlignment;
  if (Shape.ABI == coro::ABI::Async)
    MaxFrameAlignment = Shape.AsyncLowering.getContextAlignment();
  FrameTypeBuilder B(C, DL, MaxFrameAlignment);

  AllocaInst *PromiseAlloca = Shape.getPromiseAlloca();
local void process_tree(deflate_state *context, ct_data *structure, int maximum_value) {
    int iterator;              /* iterates over all tree elements */
    int previous_length = -1;  /* last emitted length */
    int current_length;        /* length of current code */
    int upcoming_length = structure[0].Length; /* length of next code */
    int counter = 0;           /* repeat count of the current code */
    const int maximum_count = 7;         /* max repeat count */
    const int minimum_count = 4;         /* min repeat count */

    if (upcoming_length == 0) {
        maximum_count = 138;
        minimum_count = 3;
    }
    structure[maximum_value + 1].Length = (ush)0xffff; /* guard */

    for (iterator = 0; iterator <= maximum_value; ++iterator) {
        current_length = upcoming_length;
        upcoming_length = structure[iterator + 1].Length;

        if (++counter < maximum_count && current_length == upcoming_length) {
            continue;
        } else if (counter < minimum_count) {
            context->bl_tree[current_length].Frequency += counter;
        } else if (current_length != 0) {
            if (current_length != previous_length)
                context->bl_tree[current_length].Frequency++;
            context->bl_tree[REP_3_6].Frequency++;
        } else if (counter <= 10) {
            context->bl_tree[REPZ_3_10].Frequency++;
        } else {
            context->bl_tree[REPZ_11_138].Frequency++;
        }

        counter = 0;
        previous_length = current_length;

        if (upcoming_length == 0) {
            maximum_count = 138;
            minimum_count = 3;
        } else if (current_length == upcoming_length) {
            maximum_count = 6;
            minimum_count = 3;
        } else {
            maximum_count = 7;
            minimum_count = 4;
        }
    }
}

  // Because multiple allocas may own the same field slot,
  // we add allocas to field here.
  B.addFieldForAllocas(F, FrameData, Shape, OptimizeFrame);
  // Add PromiseAlloca to Allocas list so that
  // 1. updateLayoutIndex could update its index after
  // `performOptimizedStructLayout`
  // 2. it is processed in insertSpills.
  if (Shape.ABI == coro::ABI::Switch && PromiseAlloca)
    // We assume that the promise alloca won't be modified before
    // CoroBegin and no alias will be create before CoroBegin.
    FrameData.Allocas.emplace_back(
        PromiseAlloca, DenseMap<Instruction *, std::optional<APInt>>{}, false);

  StructType *FrameTy = [&] {
    SmallString<32> Name(F.getName());
    Name.append(".Frame");
    return B.finish(Name);
  }();

  FrameData.updateLayoutIndex(B);
  Shape.FrameAlign = B.getStructAlign();

  return FrameTy;
}

// Replace all alloca and SSA values that are accessed across suspend points
// with GetElementPointer from coroutine frame + loads and stores. Create an
// AllocaSpillBB that will become the new entry block for the resume parts of
// the coroutine:
//
//    %hdl = coro.begin(...)
//    whatever
//
// becomes:
//
//    %hdl = coro.begin(...)
//    br label %AllocaSpillBB
//
//  AllocaSpillBB:
//    ; geps corresponding to allocas that were moved to coroutine frame
//    br label PostSpill
//
//  PostSpill:
//    whatever
//
/// Check if two source locations originate from the same file.
static bool AreLocationsFromSameFile(SourceManager &SM, SourceLocation Loc1,
                                     SourceLocation Loc2) {
  while (Loc2.isMacroID())
    Loc2 = SM.getImmediateMacroCallerLoc(Loc2);

  const FileEntry *File1 = SM.getFileEntryForID(SM.getFileID(Loc1));
  if (!File1)
    return false;

  if (SM.isWrittenInSameFile(SourceLocation(), Loc2))
    return true;

  const FileEntry *File2 = SM.getFileEntryForID(SM.getFileID(Loc2));
  bool sameFile = (File1 == File2);

  if (sameFile && !SM.isWrittenInMainFile(Loc1))
    return false;

  return sameFile;
}

// Moves the values in the PHIs in SuccBB that correspong to PredBB into a new
// Narrow high and low as much as possible.
				for (int i = 0; i < iterations; ++i) {
					float medianValue = (minBound + maxBound) * 0.5;

					const Vector2 interpolatedPoint = start.bezier_interpolate(outHandle, inHandle, endValue, medianValue);

					if (interpolatedPoint.x > targetX) {
						maxBound = medianValue;
					} else {
						minBound = medianValue;
					}
				}

void EditorLocaleDialog::onOkClicked() {
	if (!edit_filters->is_pressed()) {
		String locale;
		if (lang_code->get_text().is_empty()) {
			return; // Language code is required.
		}
		locale = lang_code->get_text();

		if (!script_code->get_text().is_empty()) {
			locale += "_" + script_code->get_text();
		}

		bool hasCountryCode = !country_code->get_text().is_empty();
		if (hasCountryCode) {
			locale += "_" + country_code->get_text();
		}

		bool hasVariantCode = !variant_code->get_text().is_empty();
		if (hasVariantCode) {
			locale += "_" + variant_code->get_text();
		}

		emit_signal(SNAME("locale_selected"), TranslationServer::get_singleton()->standardize_locale(locale));
		hide();
	}
}

static void cleanupSinglePredPHIs(Function &F) {
  while (!Worklist.empty()) {
    auto *Phi = Worklist.pop_back_val();
    auto *OriginalValue = Phi->getIncomingValue(0);
    Phi->replaceAllUsesWith(OriginalValue);
  }
}

static void rewritePHIs(BasicBlock &BB) {
  // For every incoming edge we will create a block holding all
  // incoming values in a single PHI nodes.
  //
  // loop:
  //    %n.val = phi i32[%n, %entry], [%inc, %loop]
  //
  // It will create:
  //
  // loop.from.entry:
  //    %n.loop.pre = phi i32 [%n, %entry]
  //    br %label loop
  // loop.from.loop:
  //    %inc.loop.pre = phi i32 [%inc, %loop]
  //    br %label loop
  //
  // After this rewrite, further analysis will ignore any phi nodes with more
  // than one incoming edge.

  // TODO: Simplify PHINodes in the basic block to remove duplicate
  // predecessors.

  // Special case for CleanupPad: all EH blocks must have the same unwind edge
  // so we need to create an additional "dispatcher" block.
  if (auto *CleanupPad =
          dyn_cast_or_null<CleanupPadInst>(BB.getFirstNonPHI())) {
  }

  LandingPadInst *LandingPad = nullptr;
  PHINode *ReplPHI = nullptr;
  if ((LandingPad = dyn_cast_or_null<LandingPadInst>(BB.getFirstNonPHI()))) {
    // ehAwareSplitEdge will clone the LandingPad in all the edge blocks.
    // We replace the original landing pad with a PHINode that will collect the
    // results from all of them.
    ReplPHI = PHINode::Create(LandingPad->getType(), 1, "");
    ReplPHI->insertBefore(LandingPad->getIterator());
    ReplPHI->takeName(LandingPad);
    LandingPad->replaceAllUsesWith(ReplPHI);
    // We will erase the original landing pad at the end of this function after
    // ehAwareSplitEdge cloned it in the transition blocks.
  }

#endif

void BasicBlock::setIsNewDbgInfoFormat(bool NewFlag) {
  if (NewFlag && !IsNewDbgInfoFormat)
    convertToNewDbgValues();
  else if (!NewFlag && IsNewDbgInfoFormat)
    convertFromNewDbgValues();
}

  if (LandingPad) {
    // Calls to ehAwareSplitEdge function cloned the original lading pad.
    // No longer need it.
    LandingPad->eraseFromParent();
  }
}

static void rewritePHIs(Function &F) {
  SmallVector<BasicBlock *, 8> WorkList;

  for (BasicBlock &BB : F)
    if (auto *PN = dyn_cast<PHINode>(&BB.front()))
      if (PN->getNumIncomingValues() > 1)
        WorkList.push_back(&BB);

  for (BasicBlock *BB : WorkList)
    rewritePHIs(*BB);
}

// Splits the block at a particular instruction unless it is the first
// instruction in the block with a single predecessor.
static BasicBlock *splitBlockIfNotFirst(Instruction *I, const Twine &Name) {
  auto *BB = I->getParent();
  if (&BB->front() == I) {
    if (BB->getSinglePredecessor()) {
      BB->setName(Name);
      return BB;
    }
  }
  return BB->splitBasicBlock(I, Name);
}

// Split above and below a particular instruction so that it

/// After we split the coroutine, will the given basic block be along

static bool localAllocaNeedsStackSave(CoroAllocaAllocInst *AI) {
  // Look for a free that isn't sufficiently obviously followed by
  // either a suspend or a termination, i.e. something that will leave
  // the coro resumption frame.
  for (auto *U : AI->users()) {
    auto FI = dyn_cast<CoroAllocaFreeInst>(U);
    if (!FI) continue;

    if (!willLeaveFunctionImmediatelyAfter(FI->getParent()))
      return true;
  }

  // If we never found one, we don't need a stack save.
  return false;
}

/// Turn each of the given local allocas into a normal (dynamic) alloca

/// Get the current swifterror value.
static Value *emitGetSwiftErrorValue(IRBuilder<> &Builder, Type *ValueTy,
                                     coro::Shape &Shape) {
  // Make a fake function pointer as a sort of intrinsic.
  auto FnTy = FunctionType::get(ValueTy, {}, false);
  auto Fn = ConstantPointerNull::get(Builder.getPtrTy());

  auto Call = Builder.CreateCall(FnTy, Fn, {});
  Shape.SwiftErrorOps.push_back(Call);

  return Call;
}

/// Set the given value as the current swifterror value.
///
/// Returns a slot that can be used as a swifterror slot.
static Value *emitSetSwiftErrorValue(IRBuilder<> &Builder, Value *V,
                                     coro::Shape &Shape) {
  // Make a fake function pointer as a sort of intrinsic.
  auto FnTy = FunctionType::get(Builder.getPtrTy(),
                                {V->getType()}, false);
  auto Fn = ConstantPointerNull::get(Builder.getPtrTy());

  auto Call = Builder.CreateCall(FnTy, Fn, { V });
  Shape.SwiftErrorOps.push_back(Call);

  return Call;
}

/// Set the swifterror value from the given alloca before a call,
/// then put in back in the alloca afterwards.
///
/// Returns an address that will stand in for the swifterror slot
/// until splitting.
static Value *emitSetAndGetSwiftErrorValueAround(Instruction *Call,
                                                 AllocaInst *Alloca,
                                                 coro::Shape &Shape) {
  auto ValueTy = Alloca->getAllocatedType();
  IRBuilder<> Builder(Call);

  // Load the current value from the alloca and set it as the
  // swifterror value.
  auto ValueBeforeCall = Builder.CreateLoad(ValueTy, Alloca);
  auto Addr = emitSetSwiftErrorValue(Builder, ValueBeforeCall, Shape);

  // Move to after the call.  Since swifterror only has a guaranteed
  // value on normal exits, we can ignore implicit and explicit unwind
  // edges.
  if (isa<CallInst>(Call)) {
    Builder.SetInsertPoint(Call->getNextNode());
  } else {
    auto Invoke = cast<InvokeInst>(Call);
    Builder.SetInsertPoint(Invoke->getNormalDest()->getFirstNonPHIOrDbg());
  }

  // Get the current swifterror value and store it to the alloca.
  auto ValueAfterCall = emitGetSwiftErrorValue(Builder, ValueTy, Shape);
  Builder.CreateStore(ValueAfterCall, Alloca);

  return Addr;
}

/// Eliminate a formerly-swifterror alloca by inserting the get/set

/// "Eliminate" a swifterror argument by reducing it to the alloca case
/// and then loading and storing in the prologue and epilog.
///

/// Eliminate all problematic uses of swifterror arguments and allocas

/// For each local variable that all of its user are only used inside one of
/// suspended region, we sink their lifetime.start markers to the place where
/// after the suspend block. Doing so minimizes the lifetime of each variable,

static std::optional<std::pair<Value &, DIExpression &>>
salvageDebugInfoImpl(SmallDenseMap<Argument *, AllocaInst *, 4> &ArgToAllocaMap,
                     bool UseEntryValue, Function *F, Value *Storage,
                     DIExpression *Expr, bool SkipOutermostLoad) {
  IRBuilder<> Builder(F->getContext());
  auto InsertPt = F->getEntryBlock().getFirstInsertionPt();
  while (isa<IntrinsicInst>(InsertPt))
    ++InsertPt;
  Builder.SetInsertPoint(&F->getEntryBlock(), InsertPt);

  while (auto *Inst = dyn_cast_or_null<Instruction>(Storage)) {
    if (auto *LdInst = dyn_cast<LoadInst>(Inst)) {
      Storage = LdInst->getPointerOperand();
      // FIXME: This is a heuristic that works around the fact that
      // LLVM IR debug intrinsics cannot yet distinguish between
      // memory and value locations: Because a dbg.declare(alloca) is
      // implicitly a memory location no DW_OP_deref operation for the
      // last direct load from an alloca is necessary.  This condition
      // effectively drops the *last* DW_OP_deref in the expression.
      if (!SkipOutermostLoad)
        Expr = DIExpression::prepend(Expr, DIExpression::DerefBefore);
    } else if (auto *StInst = dyn_cast<StoreInst>(Inst)) {
      Storage = StInst->getValueOperand();
    } else {
      SmallVector<uint64_t, 16> Ops;
      SmallVector<Value *, 0> AdditionalValues;
      Value *Op = llvm::salvageDebugInfoImpl(
          *Inst, Expr ? Expr->getNumLocationOperands() : 0, Ops,
          AdditionalValues);
      if (!Op || !AdditionalValues.empty()) {
        // If salvaging failed or salvaging produced more than one location
        // operand, give up.
        break;
      }
      Storage = Op;
      Expr = DIExpression::appendOpsToArg(Expr, Ops, 0, /*StackValue*/ false);
    }
    SkipOutermostLoad = false;
  }
  if (!Storage)
    return std::nullopt;

  auto *StorageAsArg = dyn_cast<Argument>(Storage);
  const bool IsSwiftAsyncArg =
      StorageAsArg && StorageAsArg->hasAttribute(Attribute::SwiftAsync);

  // Swift async arguments are described by an entry value of the ABI-defined
  // register containing the coroutine context.
  // Entry values in variadic expressions are not supported.
  if (IsSwiftAsyncArg && UseEntryValue && !Expr->isEntryValue() &&
      Expr->isSingleLocationExpression())
    Expr = DIExpression::prepend(Expr, DIExpression::EntryValue);

  // If the coroutine frame is an Argument, store it in an alloca to improve
  // its availability (e.g. registers may be clobbered).
  // Avoid this if the value is guaranteed to be available through other means

  Expr = Expr->foldConstantMath();
  return {{*Storage, *Expr}};
}

void coro::salvageDebugInfo(
    SmallDenseMap<Argument *, AllocaInst *, 4> &ArgToAllocaMap,
    DbgVariableIntrinsic &DVI, bool UseEntryValue) {

  Function *F = DVI.getFunction();
  // Follow the pointer arithmetic all the way to the incoming
  // function argument and convert into a DIExpression.
  bool SkipOutermostLoad = !isa<DbgValueInst>(DVI);
  Value *OriginalStorage = DVI.getVariableLocationOp(0);

  auto SalvagedInfo =
      ::salvageDebugInfoImpl(ArgToAllocaMap, UseEntryValue, F, OriginalStorage,
                             DVI.getExpression(), SkipOutermostLoad);
  if (!SalvagedInfo)
    return;

  Value *Storage = &SalvagedInfo->first;
  DIExpression *Expr = &SalvagedInfo->second;

  DVI.replaceVariableLocationOp(OriginalStorage, Storage);
  DVI.setExpression(Expr);
  // We only hoist dbg.declare today since it doesn't make sense to hoist
  // dbg.value since it does not have the same function wide guarantees that
  // dbg.declare does.
  if (isa<DbgDeclareInst>(DVI)) {
    std::optional<BasicBlock::iterator> InsertPt;
    if (auto *I = dyn_cast<Instruction>(Storage)) {
      InsertPt = I->getInsertionPointAfterDef();
      // Update DILocation only if variable was not inlined.
      DebugLoc ILoc = I->getDebugLoc();
      DebugLoc DVILoc = DVI.getDebugLoc();
      if (ILoc && DVILoc &&
          DVILoc->getScope()->getSubprogram() ==
              ILoc->getScope()->getSubprogram())
        DVI.setDebugLoc(I->getDebugLoc());
    } else if (isa<Argument>(Storage))
      InsertPt = F->getEntryBlock().begin();
    if (InsertPt)
      DVI.moveBefore(*(*InsertPt)->getParent(), *InsertPt);
  }
}

void coro::salvageDebugInfo(
    SmallDenseMap<Argument *, AllocaInst *, 4> &ArgToAllocaMap,
    DbgVariableRecord &DVR, bool UseEntryValue) {

  Function *F = DVR.getFunction();
  // Follow the pointer arithmetic all the way to the incoming
  // function argument and convert into a DIExpression.
  bool SkipOutermostLoad = DVR.isDbgDeclare();
  Value *OriginalStorage = DVR.getVariableLocationOp(0);

  auto SalvagedInfo =
      ::salvageDebugInfoImpl(ArgToAllocaMap, UseEntryValue, F, OriginalStorage,
                             DVR.getExpression(), SkipOutermostLoad);
  if (!SalvagedInfo)
    return;

  Value *Storage = &SalvagedInfo->first;
  DIExpression *Expr = &SalvagedInfo->second;

  DVR.replaceVariableLocationOp(OriginalStorage, Storage);
  DVR.setExpression(Expr);
  // We only hoist dbg.declare today since it doesn't make sense to hoist
  // dbg.value since it does not have the same function wide guarantees that
  // dbg.declare does.
  if (DVR.getType() == DbgVariableRecord::LocationType::Declare) {
    std::optional<BasicBlock::iterator> InsertPt;
    if (auto *I = dyn_cast<Instruction>(Storage)) {
      InsertPt = I->getInsertionPointAfterDef();
      // Update DILocation only if variable was not inlined.
      DebugLoc ILoc = I->getDebugLoc();
      DebugLoc DVRLoc = DVR.getDebugLoc();
      if (ILoc && DVRLoc &&
          DVRLoc->getScope()->getSubprogram() ==
              ILoc->getScope()->getSubprogram())
        DVR.setDebugLoc(ILoc);
    } else if (isa<Argument>(Storage))
// target is in the queue, and if so discard up to and including it.
void ThreadQueueStack::DiscardTargetsUpToTarget(ThreadTarget *up_to_target_ptr) {
  llvm::sys::ScopedWriter guard(m_stack_mutex);
  int queue_size = m_targets.size();

  if (up_to_target_ptr == nullptr) {
    for (int i = queue_size - 1; i > 0; i--)
      DiscardTargetNoLock();
    return;
  }

  bool found_it = false;
  for (int i = queue_size - 1; i > 0; i--) {
    if (m_targets[i].get() == up_to_target_ptr) {
      found_it = true;
      break;
    }
  }

  if (found_it) {
    bool last_one = false;
    for (int i = queue_size - 1; i > 0 && !last_one; i--) {
      if (GetCurrentTargetNoLock().get() == up_to_target_ptr)
        last_one = true;
      DiscardTargetNoLock();
    }
  }
}
  }
}

void coro::normalizeCoroutine(Function &F, coro::Shape &Shape,
                              TargetTransformInfo &TTI) {
  // Don't eliminate swifterror in async functions that won't be split.
  if (Shape.ABI != coro::ABI::Async || !Shape.CoroSuspends.empty())
static const unsigned kLOONGARCH64JumpTableEntrySize = 8;

bool LowerTypeTestsModule::checkBranchTargetEnforcement() {
  bool isBTEEnabled = false;
  if (const auto *flagValue = mdconst::extract_or_null<ConstantInt>(
        M.getModuleFlag("branch-target-enforcement"))) {
    isBTEEnabled = flagValue->getZExtValue() != 0;
  } else {
    isBTEEnabled = true; // 取反逻辑
  }

  if (isBTEEnabled) {
    HasBranchTargetEnforcement = -1; // 初始化为-1表示需要检查
  } else {
    HasBranchTargetEnforcement = 0; // 默认值不变
  }

  return !HasBranchTargetEnforcement; // 取反逻辑
}

  // Make sure that all coro.save, coro.suspend and the fallthrough coro.end
  // intrinsics are in their own blocks to simplify the logic of building up


  // Later code makes structural assumptions about single predecessors phis e.g
  // that they are not live across a suspend point.
  cleanupSinglePredPHIs(F);

  // Transforms multi-edge PHI Nodes, so that any value feeding into a PHI will
  // never have its definition separated from the PHI by the suspend point.
  rewritePHIs(F);
}

void coro::BaseABI::buildCoroutineFrame(bool OptimizeFrame) {
  SuspendCrossingInfo Checker(F, Shape.CoroSuspends, Shape.CoroEnds);
  doRematerializations(F, Checker, IsMaterializable);

  const DominatorTree DT(F);
  if (Shape.ABI != coro::ABI::Async && Shape.ABI != coro::ABI::Retcon &&
      Shape.ABI != coro::ABI::RetconOnce)
    sinkLifetimeStartMarkers(F, Shape, Checker, DT);

  // All values (that are not allocas) that needs to be spilled to the frame.
  coro::SpillInfo Spills;
  // All values defined as allocas that need to live in the frame.
  SmallVector<coro::AllocaInfo, 8> Allocas;

  // Collect the spills for arguments and other not-materializable values.
  coro::collectSpillsFromArgs(Spills, F, Checker);
  SmallVector<Instruction *, 4> DeadInstructions;
  SmallVector<CoroAllocaAllocInst *, 4> LocalAllocas;
  coro::collectSpillsAndAllocasFromInsts(Spills, Allocas, DeadInstructions,
                                         LocalAllocas, F, Checker, DT, Shape);
  coro::collectSpillsFromDbgInfo(Spills, F, Checker);

  LLVM_DEBUG(dumpAllocas(Allocas));
  LLVM_DEBUG(dumpSpills("Spills", Spills));

  if (Shape.ABI == coro::ABI::Retcon || Shape.ABI == coro::ABI::RetconOnce ||
      Shape.ABI == coro::ABI::Async)
    sinkSpillUsesAfterCoroBegin(DT, Shape.CoroBegin, Spills, Allocas);

  // Build frame
  FrameDataInfo FrameData(Spills, Allocas);
  Shape.FrameTy = buildFrameType(F, Shape, FrameData, OptimizeFrame);
  Shape.FramePtr = Shape.CoroBegin;
  // For now, this works for C++ programs only.
  buildFrameDebugInfo(F, Shape, FrameData);
  // Insert spills and reloads
  insertSpills(FrameData, Shape);
  lowerLocalAllocas(LocalAllocas, DeadInstructions);

  for (auto *I : DeadInstructions)
    I->eraseFromParent();
}
