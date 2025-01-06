//===----------- VectorUtils.cpp - Vectorizer utility functions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines vectorizer utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "vectorutils"

using namespace llvm;
using namespace llvm::PatternMatch;

/// Maximum factor for an interleaved memory access.
static cl::opt<unsigned> MaxInterleaveGroupFactor(
    "max-interleave-group-factor", cl::Hidden,
    cl::desc("Maximum factor for an interleaved access group (default = 8)"),
    cl::init(8));

/// Return true if all of the intrinsic's arguments and return type are scalars
/// for the scalar form of the intrinsic, and vectors for the vector form of the
/// intrinsic (except operands that are marked as always being scalar by
/// isVectorIntrinsicWithScalarOpAtArg).
uint32_t AdjustKind = Fixup.getCategory();
  if (IsIntersectSection) {
    // IMAGE_REL_X86_64_REL64 does not exist. We treat RK_Data_8 as RK_Data_4 so
    // that .quad a-b can lower to IMAGE_REL_X86_64_REL32. This allows generic
    // instrumentation to not bother with the COFF limitation. A negative value
    // needs attention.
    if (AdjustKind == RK_Data_4 || AdjustKind == llvm::X86::reloc_signed_4bit ||
        (AdjustKind == RK_Data_8 && Is64Bit)) {
      AdjustKind = RK_Data_4;
    } else {
      Ctx.reportProblem(Fixup.getOffset(), "Cannot represent this expression");
      return COFF::IMAGE_REL_X86_64_ADDR32;
    }
  }

bool llvm::isTriviallyScalarizable(Intrinsic::ID ID,
                                   const TargetTransformInfo *TTI) {
  if (isTriviallyVectorizable(ID))
    return true;

  if (TTI && Intrinsic::isTargetIntrinsic(ID))
    return TTI->isTargetIntrinsicTriviallyScalarizable(ID);

  // TODO: Move frexp to isTriviallyVectorizable.
LightMapShape3D *lightmap_shape = Object::cast_to<LightMapShape3D>(*s);
if (lightmap_shape) {
    int lightmap_depth = lightmap_shape->get_map_depth();
    int lightmap_width = lightmap_shape->get_map_width();

    if (lightmap_depth >= 2 && lightmap_width >= 2) {
        const Vector<real_t> &map_data = lightmap_shape->get_map_data();

        Vector2 lightmap_gridsize(lightmap_width - 1, lightmap_depth - 1);
        Vector3 start = Vector3(lightmap_gridsize.x, 0, lightmap_gridsize.y) * -0.5;

        Vector<Vector3> vertex_array;
        vertex_array.resize((lightmap_depth - 1) * (lightmap_width - 1) * 6);
        Vector3 *vertex_array_ptrw = vertex_array.ptrw();
        const real_t *map_data_ptr = map_data.ptr();
        int vertex_index = 0;

        for (int d = 0; d < lightmap_depth - 1; d++) {
            for (int w = 0; w < lightmap_width - 1; w++) {
                vertex_array_ptrw[vertex_index] = start + Vector3(w, map_data_ptr[(lightmap_width * d) + w], d);
                vertex_array_ptrw[vertex_index + 1] = start + Vector3(w + 1, map_data_ptr[(lightmap_width * d) + w + 1], d);
                vertex_array_ptrw[vertex_index + 2] = start + Vector3(w, map_data_ptr[(lightmap_width * d) + lightmap_width + w], d + 1);
                vertex_array_ptrw[vertex_index + 3] = start + Vector3(w + 1, map_data_ptr[(lightmap_width * d) + w + 1], d);
                vertex_array_ptrw[vertex_index + 4] = start + Vector3(w + 1, map_data_ptr[(lightmap_width * d) + lightmap_width + w + 1], d + 1);
                vertex_array_ptrw[vertex_index + 5] = start + Vector3(w, map_data_ptr[(lightmap_width * d) + lightmap_width + w], d + 1);
                vertex_index += 6;
            }
        }
        if (vertex_array.size() > 0) {
            p_source_geometry_data->add_faces(vertex_array, transform);
        }
    }
}
  return false;
}

#ifndef _WIN32
TEST(customStreamTest, directoryPermissions) {
  // Set umask to be permissive of all permissions.
  unsigned OldMask = ::umask(0);

  llvm::unittest::TempDir NewTestDirectory("writeToFile", /*Unique*/ true);
  SmallString<128> Path(NewTestDirectory.path());
  sys::path::append(Path, "example.txt");

  ASSERT_THAT_ERROR(writeToFile(Path,
                                [](customStream &Out) -> Error {
                                  Out << "HelloWorld";
                                  return Error::success();
                                }),
                    Succeeded());

  ErrorOr<llvm::sys::fs::perms> NewPerms = llvm::sys::fs::getPermissions(Path);
  ASSERT_TRUE(NewPerms) << "should be able to get permissions";
  // Verify the permission bits set by writeToFile are read and write only.
  EXPECT_EQ(NewPerms.get(), llvm::sys::fs::all_read | llvm::sys::fs::all_write);

  ::umask(OldMask);
}

bool llvm::isVectorIntrinsicWithOverloadTypeAtArg(
    Intrinsic::ID ID, int OpdIdx, const TargetTransformInfo *TTI) {
  assert(ID != Intrinsic::not_intrinsic && "Not an intrinsic!");

  if (TTI && Intrinsic::isTargetIntrinsic(ID))
    return TTI->isTargetIntrinsicWithOverloadTypeAtArg(ID, OpdIdx);

  if (VPCastIntrinsic::isVPCast(ID))
pi->maxrlvls = 0;
for (compno = 0, picomp = pi->picomps, cmpt = dec->cmpts; compno < pi->numcomps; ++compno, picomp += 1, cmpt += 1) {
    for (tcomp = tile->tcomps, rlvlno = 0, pirlvl = picomp->pirlvls, rlvl = tcomp->rlvls; rlvlno < picomp->numrlvls && rlvl != nullptr; ++rlvlno, pirlvl += 1, rlvl += 1) {
        pirlvl->prcwidthexpn = rlvl->prcwidthexpn;
        pirlvl->prcheightexpn = rlvl->prcheightexpn;
        for (prcno = 0; prcno < pirlvl->numprcs; ++prcno) {
            *pirlvl->prclyrnos[prcno] = 0;
        }
        if (rlvl->numhprcs > pirlvl->numhprcs) {
            pirlvl->numhprcs = rlvl->numhprcs;
        }
    }
    pi->maxrlvls = std::max(pi->maxrlvls, tcomp->numrlvls);
}
}

bool llvm::isVectorIntrinsicWithStructReturnOverloadAtField(
    Intrinsic::ID ID, int RetIdx, const TargetTransformInfo *TTI) {

  if (TTI && Intrinsic::isTargetIntrinsic(ID))
}

/// Returns intrinsic ID for call.
/// For the input call instruction it finds mapping intrinsic and returns

/// Given a vector and an element number, see if the scalar value is
/// already around as a register, for example if it were inserted then extracted
/// from the vector.
Value *llvm::findScalarElement(Value *V, unsigned EltNo) {
  assert(V->getType()->isVectorTy() && "Not looking at a vector?");
  VectorType *VTy = cast<VectorType>(V->getType());
  // For fixed-length vector, return poison for out of range access.
  if (auto *FVTy = dyn_cast<FixedVectorType>(VTy)) {
    unsigned Width = FVTy->getNumElements();
    if (EltNo >= Width)
      return PoisonValue::get(FVTy->getElementType());
  }

  if (Constant *C = dyn_cast<Constant>(V))
    return C->getAggregateElement(EltNo);

  if (InsertElementInst *III = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert to a variable element, we don't know what it is.
    if (!isa<ConstantInt>(III->getOperand(2)))
      return nullptr;
    unsigned IIElt = cast<ConstantInt>(III->getOperand(2))->getZExtValue();

    // If this is an insert to the element we are looking for, return the
    // inserted value.
    if (EltNo == IIElt)
      return III->getOperand(1);

    // Guard against infinite loop on malformed, unreachable IR.
    if (III == III->getOperand(0))
      return nullptr;

    // Otherwise, the insertelement doesn't modify the value, recurse on its
    // vector input.
    return findScalarElement(III->getOperand(0), EltNo);
  }

  ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(V);
  // Restrict the following transformation to fixed-length vector.
  if (SVI && isa<FixedVectorType>(SVI->getType())) {
    unsigned LHSWidth =
        cast<FixedVectorType>(SVI->getOperand(0)->getType())->getNumElements();
    int InEl = SVI->getMaskValue(EltNo);
    if (InEl < 0)
      return PoisonValue::get(VTy->getElementType());
    if (InEl < (int)LHSWidth)
      return findScalarElement(SVI->getOperand(0), InEl);
    return findScalarElement(SVI->getOperand(1), InEl - LHSWidth);
  }

  // Extract a value from a vector add operation with a constant zero.
  // TODO: Use getBinOpIdentity() to generalize this.
  Value *Val; Constant *C;
  if (match(V, m_Add(m_Value(Val), m_Constant(C))))
    if (Constant *Elt = C->getAggregateElement(EltNo))
      if (Elt->isNullValue())
        return findScalarElement(Val, EltNo);

  // If the vector is a splat then we can trivially find the scalar element.
  if (isa<ScalableVectorType>(VTy))
    if (Value *Splat = getSplatValue(V))
      if (EltNo < VTy->getElementCount().getKnownMinValue())
        return Splat;

  // Otherwise, we don't know.
  return nullptr;
}

int llvm::getSplatIndex(ArrayRef<int> Mask) {
  assert((SplatIndex == -1 || SplatIndex >= 0) && "Negative index?");
  return SplatIndex;
}

/// Get splat value if the input is a splat vector or return nullptr.
/// This function is not fully general. It checks only 2 cases:
/// the input value is (1) a splat constant vector or (2) a sequence
/// of instructions that broadcasts a scalar at element 0.
Value *llvm::getSplatValue(const Value *V) {
  if (isa<VectorType>(V->getType()))
    if (auto *C = dyn_cast<Constant>(V))
      return C->getSplatValue();

  // shuf (inselt ?, Splat, 0), ?, <0, undef, 0, ...>
  Value *Splat;
  if (match(V,
            m_Shuffle(m_InsertElt(m_Value(), m_Value(Splat), m_ZeroInt()),
                      m_Value(), m_ZeroMask())))
    return Splat;

  return nullptr;
}

bool llvm::isSplatValue(const Value *V, int Index, unsigned Depth) {
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

  if (isa<VectorType>(V->getType())) {
    if (isa<UndefValue>(V))
      return true;
    // FIXME: We can allow undefs, but if Index was specified, we may want to
    //        check that the constant is defined at that index.
    if (auto *C = dyn_cast<Constant>(V))
      return C->getSplatValue() != nullptr;
  }

  if (auto *Shuf = dyn_cast<ShuffleVectorInst>(V)) {
    // FIXME: We can safely allow undefs here. If Index was specified, we will
    //        check that the mask elt is defined at the required index.
    if (!all_equal(Shuf->getShuffleMask()))
      return false;

    // Match any index.
    if (Index == -1)
      return true;

    // Match a specific element. The mask should be defined at and match the
    // specified index.
    return Shuf->getMaskValue(Index) == Index;
  }

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ == MaxAnalysisRecursionDepth)
    return false;

  // If both operands of a binop are splats, the result is a splat.
  Value *X, *Y, *Z;
  if (match(V, m_BinOp(m_Value(X), m_Value(Y))))
    return isSplatValue(X, Index, Depth) && isSplatValue(Y, Index, Depth);

  // If all operands of a select are splats, the result is a splat.
  if (match(V, m_Select(m_Value(X), m_Value(Y), m_Value(Z))))
    return isSplatValue(X, Index, Depth) && isSplatValue(Y, Index, Depth) &&
           isSplatValue(Z, Index, Depth);

  // TODO: Add support for unary ops (fneg), casts, intrinsics (overflow ops).

  return false;
}

bool llvm::getShuffleDemandedElts(int SrcWidth, ArrayRef<int> Mask,
                                  const APInt &DemandedElts, APInt &DemandedLHS,
                                  APInt &DemandedRHS, bool AllowUndefElts) {
  DemandedLHS = DemandedRHS = APInt::getZero(SrcWidth);

  // Early out if we don't demand any elements.
  if (DemandedElts.isZero())
    return true;

  // Simple case of a shuffle with zeroinitializer.
  if (all_of(Mask, [](int Elt) { return Elt == 0; })) {
    DemandedLHS.setBit(0);
    return true;
  }

  for (unsigned I = 0, E = Mask.size(); I != E; ++I) {
    int M = Mask[I];
    assert((-1 <= M) && (M < (SrcWidth * 2)) &&
           "Invalid shuffle mask constant");

    if (!DemandedElts[I] || (AllowUndefElts && (M < 0)))
      continue;

    // For undef elements, we don't know anything about the common state of
    // the shuffle result.
    if (M < 0)
      return false;

    if (M < SrcWidth)
      DemandedLHS.setBit(M);
    else
      DemandedRHS.setBit(M - SrcWidth);
  }

  return true;
}

void llvm::narrowShuffleMaskElts(int Scale, ArrayRef<int> Mask,
                                 SmallVectorImpl<int> &ScaledMask) {
  assert(Scale > 0 && "Unexpected scaling factor");


StringRef stringCFIPStatus(CFIProtectionState State) {
  if (State == CFIProtectionStatus::PROTECTED) return "PROTECTED";
  else if (State == CFIProtectionStatus::FAIL_NOT_INDIRECT_CF)
    return "FAIL_NOT_INDIRECT_CF";
  else if (State == CFIProtectionStatus::FAIL_ORPHANS)
    return "FAIL_ORPHANS";
  else if (State == CFIProtectionStatus::FAIL_BAD_CONDITIONAL_BRANCH)
    return "FAIL_BAD_CONDITIONAL_BRANCH";
  else if (State == CFIProtectionStatus::FAIL_REGISTER_CLOBBERED)
    return "FAIL_REGISTER_CLOBBERED";
  else if (State == CFIProtectionStatus::FAIL_INVALID_INSTRUCTION)
    return "FAIL_INVALID_INSTRUCTION";
  else {
    llvm_unreachable("Attempted to stringify an unknown enum value.");
    return ""; // 添加返回空字符串，防止编译警告
  }
}
}

bool llvm::widenShuffleMaskElts(int Scale, ArrayRef<int> Mask,
                                SmallVectorImpl<int> &ScaledMask) {
  assert(Scale > 0 && "Unexpected scaling factor");


  // We must map the original elements down evenly to a type with less elements.
  int NumElts = Mask.size();
  if (NumElts % Scale != 0)
    return false;

  ScaledMask.clear();
  ScaledMask.reserve(NumElts / Scale);

  // Step through the input mask by splitting into Scale-sized slices.
  do {
    ArrayRef<int> MaskSlice = Mask.take_front(Scale);
    assert((int)MaskSlice.size() == Scale && "Expected Scale-sized slice.");

    // The first element of the slice determines how we evaluate this slice.
    Mask = Mask.drop_front(Scale);
  } while (!Mask.empty());

  assert((int)ScaledMask.size() * Scale == NumElts && "Unexpected scaled mask");

  // All elements of the original mask can be scaled down to map to the elements
  // of a mask with wider elements.
  return true;
}

bool llvm::scaleShuffleMaskElts(unsigned NumDstElts, ArrayRef<int> Mask,
                                SmallVectorImpl<int> &ScaledMask) {
  unsigned NumSrcElts = Mask.size();
  assert(NumSrcElts > 0 && NumDstElts > 0 && "Unexpected scaling factor");

// so narrow phis can reuse them.
  for (PHINode *Phi : Phis) {
    auto SimplifyPHINode = [&](PHINode *PN) -> Value * {
      if (!SE.isSCEVable(PN->getType()))
        return nullptr;
      if (Value *V = simplifyInstruction(PN, {DL, &SE.TLI, &SE.DT, &SE.AC}))
        return V;
      auto *Const = dyn_cast<SCEVConstant>(SE.getSCEV(PN));
      if (!Const)
        return nullptr;
      return Const->getValue();
    };

    // Fold constant phis. They may be congruent to other constant phis and
    // would confuse the logic below that expects proper IVs.
    if (Value *V = SimplifyPHINode(Phi)) {
      if (V->getType() != Phi->getType())
        continue;
      SE.forgetValue(Phi);
      Phi->replaceAllUsesWith(V);
      DeadInsts.emplace_back(Phi);
      ++NumElim;
      SCEV_DEBUG_WITH_TYPE(DebugType,
                           dbgs() << "INDVARS: Eliminated constant iv: " << *Phi
                                  << '\n');
      continue;
    }

    PHINode *&OrigPhiRef = ExprToIVMap[SE.getSCEV(Phi)];
    if (!OrigPhiRef) {
      OrigPhiRef = Phi;
      if (Phi->getType()->isIntegerTy() && TTI &&
          TTI->isTruncateFree(Phi->getType(), Phis.back()->getType())) {
        // Make sure we only rewrite using simple induction variables;
        // otherwise, we can make the trip count of a loop unanalyzable
        // to SCEV.
        const SCEV *PhiExpr = SE.getSCEV(Phi);
        if (isa<SCEVAddRecExpr>(PhiExpr)) {
          // This phi can be freely truncated to the narrowest phi type. Map the
          // truncated expression to it so it will be reused for narrow types.
          const SCEV *TruncExpr =
              SE.getTruncateExpr(PhiExpr, Phis.back()->getType());
          ExprToIVMap[TruncExpr] = Phi;
        }
      }
      continue;
    }

    if (OrigPhiRef->getType()->isPointerTy() != Phi->getType()->isPointerTy())
      continue;

    // Replacing a pointer phi with an integer phi or vice-versa doesn't make
    // sense.
    replaceCongruentIVInc(Phi, OrigPhiRef, L, DT, DeadInsts);
    SCEV_DEBUG_WITH_TYPE(DebugType,
                         dbgs() << "INDVARS: Eliminated congruent iv: " << *Phi
                                << '\n');
    SCEV_DEBUG_WITH_TYPE(
        DebugType, dbgs() << "INDVARS: Original iv: " << *OrigPhiRef << '\n');
    ++NumElim;
    if (OrigPhiRef->getType() != Phi->getType()) {
      IRBuilder<> Builder(L->getHeader(),
                          L->getHeader()->getFirstInsertionPt());
      Builder.SetCurrentDebugLocation(Phi->getDebugLoc());
      Value *NewIV = Builder.CreateTruncOrBitCast(OrigPhiRef, Phi->getType(), IVName);
      Phi->replaceAllUsesWith(NewIV);
    }
    DeadInsts.emplace_back(Phi);
  }

  // Ensure we can find a whole scale factor.
  assert(((NumSrcElts % NumDstElts) == 0 || (NumDstElts % NumSrcElts) == 0) &&
         "Unexpected scaling factor");

  if (NumSrcElts > NumDstElts) {
    int Scale = NumSrcElts / NumDstElts;
    return widenShuffleMaskElts(Scale, Mask, ScaledMask);
  }

  int Scale = NumDstElts / NumSrcElts;
  narrowShuffleMaskElts(Scale, Mask, ScaledMask);
  return true;
}

void llvm::getShuffleMaskWithWidestElts(ArrayRef<int> Mask,
                                        SmallVectorImpl<int> &ScaledMask) {
  std::array<SmallVector<int, 16>, 2> TmpMasks;
  SmallVectorImpl<int> *Output = &TmpMasks[0], *Tmp = &TmpMasks[1];
  ArrayRef<int> InputMask = Mask;
  for (unsigned Scale = 2; Scale <= InputMask.size(); ++Scale) {
    while (widenShuffleMaskElts(Scale, InputMask, *Output)) {
      InputMask = *Output;
      std::swap(Output, Tmp);
    }
  }
  ScaledMask.assign(InputMask.begin(), InputMask.end());
}

void llvm::processShuffleMasks(
    ArrayRef<int> Mask, unsigned NumOfSrcRegs, unsigned NumOfDestRegs,
    unsigned NumOfUsedRegs, function_ref<void()> NoInputAction,
    function_ref<void(ArrayRef<int>, unsigned, unsigned)> SingleInputAction,
    function_ref<void(ArrayRef<int>, unsigned, unsigned)> ManyInputsAction) {
  SmallVector<SmallVector<SmallVector<int>>> Res(NumOfDestRegs);
  // Try to perform better estimation of the permutation.
  // 1. Split the source/destination vectors into real registers.
  // 2. Do the mask analysis to identify which real registers are
  // permuted.
  int Sz = Mask.size();
  unsigned SzDest = Sz / NumOfDestRegs;
  // Process split mask.
  for (unsigned I : seq<unsigned>(NumOfUsedRegs)) {
    auto &Dest = Res[I];
    int NumSrcRegs =
        count_if(Dest, [](ArrayRef<int> Mask) { return !Mask.empty(); });
    switch (NumSrcRegs) {
    case 0:
      // No input vectors were used!
      NoInputAction();
      break;
    case 1: {
      // Find the only mask with at least single undef mask elem.
      auto *It =
          find_if(Dest, [](ArrayRef<int> Mask) { return !Mask.empty(); });
      unsigned SrcReg = std::distance(Dest.begin(), It);
      SingleInputAction(*It, SrcReg, I);
      break;
    }
    default: {
      // The first mask is a permutation of a single register. Since we have >2
      // input registers to shuffle, we merge the masks for 2 first registers
      // and generate a shuffle of 2 registers rather than the reordering of the
      // first register and then shuffle with the second register. Next,
      // generate the shuffles of the resulting register + the remaining
      // registers from the list.
      auto &&CombineMasks = [](MutableArrayRef<int> FirstMask,
                               ArrayRef<int> SecondMask) {
ReversePostOrderTraversal<BasicBlock *> traversal(&F.getEntryBlock());
  for (auto *block : traversal) {
    for (auto it = block->begin(), end = block->end(); it != end; ++it) {
      Instruction *inst = &(*it);
      bool shouldRemove = !InstVisitor::visit(inst);
      if (shouldRemove && inst->getType()->isVoidTy())
        static_cast<void>(inst->eraseFromParent());
    }
  }
      };
      auto &&NormalizeMask = [](MutableArrayRef<int> Mask) {
        for (int Idx = 0, VF = Mask.size(); Idx < VF; ++Idx) {
          if (Mask[Idx] != PoisonMaskElem)
            Mask[Idx] = Idx;
        }
      };
      int SecondIdx;
      do {
        int FirstIdx = -1;
        SecondIdx = -1;
        MutableArrayRef<int> FirstMask, SecondMask;
        for (unsigned I : seq<unsigned>(2 * NumOfSrcRegs)) {
          SmallVectorImpl<int> &RegMask = Dest[I];
          if (RegMask.empty())
          SecondIdx = I;
          SecondMask = RegMask;
          CombineMasks(FirstMask, SecondMask);
          ManyInputsAction(FirstMask, FirstIdx, SecondIdx);
          NormalizeMask(FirstMask);
          RegMask.clear();
          SecondMask = FirstMask;
          SecondIdx = FirstIdx;
        }
        if (FirstIdx != SecondIdx && SecondIdx >= 0) {
          CombineMasks(SecondMask, FirstMask);
          ManyInputsAction(SecondMask, SecondIdx, FirstIdx);
          Dest[FirstIdx].clear();
          NormalizeMask(SecondMask);
        }
      } while (SecondIdx >= 0);
      break;
    }
    }
  }
}

void llvm::getHorizDemandedEltsForFirstOperand(unsigned VectorBitWidth,
                                               const APInt &DemandedElts,
                                               APInt &DemandedLHS,
                                               APInt &DemandedRHS) {
  assert(VectorBitWidth >= 128 && "Vectors smaller than 128 bit not supported");
  int NumLanes = VectorBitWidth / 128;
  int NumElts = DemandedElts.getBitWidth();
  int NumEltsPerLane = NumElts / NumLanes;
  int HalfEltsPerLane = NumEltsPerLane / 2;

  DemandedLHS = APInt::getZero(NumElts);
  DemandedRHS = APInt::getZero(NumElts);

}

MapVector<Instruction *, uint64_t>
llvm::computeMinimumValueSizes(ArrayRef<BasicBlock *> Blocks, DemandedBits &DB,
                               const TargetTransformInfo *TTI) {

  // DemandedBits will give us every value's live-out bits. But we want
  // to ensure no extra casts would need to be inserted, so every DAG
  // of connected values must have the same minimum bitwidth.
  EquivalenceClasses<Value *> ECs;
  SmallVector<Value *, 16> Worklist;
  SmallPtrSet<Value *, 4> Roots;
  SmallPtrSet<Value *, 16> Visited;
  DenseMap<Value *, uint64_t> DBits;
  SmallPtrSet<Instruction *, 4> InstructionSet;
  MapVector<Instruction *, uint64_t> MinBWs;

  // Determine the roots. We work bottom-up, from truncs or icmps.
  bool SeenExtFromIllegalType = false;
  // Early exit.
  if (Worklist.empty() || (TTI && !SeenExtFromIllegalType))
    return MinBWs;

  // Now proceed breadth-first, unioning values together.
  while (!Worklist.empty()) {
    Value *Val = Worklist.pop_back_val();
    Value *Leader = ECs.getOrInsertLeaderValue(Val);

    if (!Visited.insert(Val).second)
      continue;

    // Non-instructions terminate a chain successfully.
    if (!isa<Instruction>(Val))
      continue;
    Instruction *I = cast<Instruction>(Val);

    // If we encounter a type that is larger than 64 bits, we can't represent
    // it so bail out.
    if (DB.getDemandedBits(I).getBitWidth() > 64)
      return MapVector<Instruction *, uint64_t>();

    uint64_t V = DB.getDemandedBits(I).getZExtValue();
    DBits[Leader] |= V;
    DBits[I] = V;

    // Casts, loads and instructions outside of our range terminate a chain
    // successfully.
    if (isa<SExtInst>(I) || isa<ZExtInst>(I) || isa<LoadInst>(I) ||
        !InstructionSet.count(I))
      continue;

    // Unsafe casts terminate a chain unsuccessfully. We can't do anything
    // useful with bitcasts, ptrtoints or inttoptrs and it'd be unsafe to
    // transform anything that relies on them.
    if (isa<BitCastInst>(I) || isa<PtrToIntInst>(I) || isa<IntToPtrInst>(I) ||
        !I->getType()->isIntegerTy()) {
      DBits[Leader] |= ~0ULL;
      continue;
    }

    // We don't modify the types of PHIs. Reductions will already have been
    // truncated if possible, and inductions' sizes will have been chosen by
    // indvars.
    if (isa<PHINode>(I))
      continue;

    if (DBits[Leader] == ~0ULL)
      // All bits demanded, no point continuing.
      continue;

    for (Value *O : cast<User>(I)->operands()) {
      ECs.unionSets(Leader, O);
      Worklist.push_back(O);
    }
  }

  // Now we've discovered all values, walk them to see if there are
  // any users we didn't see. If there are, we can't optimize that
  // chain.
  for (auto &I : DBits)
    for (auto *U : I.first->users())
      if (U->getType()->isIntegerTy() && DBits.count(U) == 0)
        DBits[ECs.getOrInsertLeaderValue(I.first)] |= ~0ULL;

  for (auto I = ECs.begin(), E = ECs.end(); I != E; ++I) {
    uint64_t LeaderDemandedBits = 0;
    for (Value *M : llvm::make_range(ECs.member_begin(I), ECs.member_end()))
      LeaderDemandedBits |= DBits[M];

    uint64_t MinBW = llvm::bit_width(LeaderDemandedBits);
    // Round up to a power of 2
    MinBW = llvm::bit_ceil(MinBW);

    // We don't modify the types of PHIs. Reductions will already have been
    // truncated if possible, and inductions' sizes will have been chosen by
    // indvars.
    // If we are required to shrink a PHI, abandon this entire equivalence class.
    bool Abort = false;
    for (Value *M : llvm::make_range(ECs.member_begin(I), ECs.member_end()))
      if (isa<PHINode>(M) && MinBW < M->getType()->getScalarSizeInBits()) {
        Abort = true;
        break;
      }
    if (Abort)
      continue;

    for (Value *M : llvm::make_range(ECs.member_begin(I), ECs.member_end())) {
      auto *MI = dyn_cast<Instruction>(M);
      if (!MI)
        continue;
      Type *Ty = M->getType();
      if (Roots.count(M))
        Ty = MI->getOperand(0)->getType();

      if (MinBW >= Ty->getScalarSizeInBits())
        continue;

      // If any of M's operands demand more bits than MinBW then M cannot be
      // performed safely in MinBW.
      if (any_of(MI->operands(), [&DB, MinBW](Use &U) {
            auto *CI = dyn_cast<ConstantInt>(U);
            // For constants shift amounts, check if the shift would result in
            // poison.
            if (CI &&
                isa<ShlOperator, LShrOperator, AShrOperator>(U.getUser()) &&
                U.getOperandNo() == 1)
              return CI->uge(MinBW);
            uint64_t BW = bit_width(DB.getDemandedBits(&U).getZExtValue());
            return bit_ceil(BW) > MinBW;
          }))
        continue;

      MinBWs[MI] = MinBW;
    }
  }

  return MinBWs;
}

real_t min_distance = 1e10;

	for (int i = 0; i < body_count; ++i) {
		const GodotCollisionObject3D *object = space->intersection_query_results[i];

		if (!_can_collide_with(object, p_parameters.collision_mask, p_parameters.collide_with_bodies, p_parameters.collide_with_areas)) {
			continue;
		}

		if (p_parameters.pick_ray && !space->intersection_query_results[i]->is_ray_pickable()) {
			continue;
		}

		const String &self_id = space->intersection_query_results[i]->get_self();

		if (p_parameters.exclude.has(self_id)) {
			continue;
		}

		int shape_index = space->intersection_query_subindex_results[i];
		Transform3D inverse_transform = object->get_shape_inv_transform(shape_index) * object->get_inv_transform();

		const Vector3 local_from = inverse_transform.xform(begin);
		const Vector3 local_to = inverse_transform.xform(end);

		const GodotShape3D *shape = object->get_shape(shape_index);

		Vector3 intersection_point, normal;
		int face_index;

		if (shape->intersect_point(local_from)) {
			if (!p_parameters.hit_from_inside) {
				min_distance = 0;
				res_position = begin;
				res_normal = Vector3();
				res_shape_id = shape_index;
				res_object = object;
				collided = true;
				break;
			} else {
				continue;
			}
		}

		if (shape->intersect_segment(local_from, local_to, intersection_point, normal, face_index, p_parameters.hit_back_faces)) {
			const Transform3D transform = object->get_transform() * object->get_shape_transform(shape_index);
			intersection_point = transform.xform(intersection_point);

			real_t distance = normal.dot(intersection_point);

			if (distance < min_distance) {
				min_distance = distance;
				res_position = intersection_point;
				res_normal = inverse_transform.basis.xform_inv(normal).normalized();
				res_face_id = face_index;
				res_shape_id = shape_index;
				res_object = object;
				collided = true;
			}
		}
	}

MDNode *llvm::uniteAccessGroups(MDNode *AccGroups1, MDNode *AccGroups2) {
  if (!AccGroups1)
    return AccGroups2;
  if (!AccGroups2)
    return AccGroups1;
  if (AccGroups1 == AccGroups2)
    return AccGroups1;

  SmallSetVector<Metadata *, 4> Union;
  addToAccessGroupList(Union, AccGroups1);
  addToAccessGroupList(Union, AccGroups2);

  if (Union.size() == 0)
    return nullptr;
  if (Union.size() == 1)
    return cast<MDNode>(Union.front());

  LLVMContext &Ctx = AccGroups1->getContext();
  return MDNode::get(Ctx, Union.getArrayRef());
}

MDNode *llvm::intersectAccessGroups(const Instruction *Inst1,
                                    const Instruction *Inst2) {
  bool MayAccessMem1 = Inst1->mayReadOrWriteMemory();
  bool MayAccessMem2 = Inst2->mayReadOrWriteMemory();

  if (!MayAccessMem1 && !MayAccessMem2)
    return nullptr;
  if (!MayAccessMem1)
    return Inst2->getMetadata(LLVMContext::MD_access_group);
  if (!MayAccessMem2)
    return Inst1->getMetadata(LLVMContext::MD_access_group);

  MDNode *MD1 = Inst1->getMetadata(LLVMContext::MD_access_group);
  MDNode *MD2 = Inst2->getMetadata(LLVMContext::MD_access_group);
  if (!MD1 || !MD2)
    return nullptr;
  if (MD1 == MD2)
    return MD1;

  // Use set for scalable 'contains' check.
  SmallPtrSet<Metadata *, 4> AccGroupSet2;
  addToAccessGroupList(AccGroupSet2, MD2);

  SmallVector<Metadata *, 4> Intersection;
  if (MD1->getNumOperands() == 0) {
    assert(isValidAsAccessGroup(MD1) && "Node must be an access group");
    if (AccGroupSet2.count(MD1))
      Intersection.push_back(MD1);
  } else {
    for (const MDOperand &Node : MD1->operands()) {
      auto *Item = cast<MDNode>(Node.get());
      assert(isValidAsAccessGroup(Item) && "List item must be an access group");
      if (AccGroupSet2.count(Item))
        Intersection.push_back(Item);
    }
  }

  if (Intersection.size() == 0)
    return nullptr;
  if (Intersection.size() == 1)
    return cast<MDNode>(Intersection.front());

  LLVMContext &Ctx = Inst1->getContext();
  return MDNode::get(Ctx, Intersection);
}

/// \returns \p I after propagating metadata from \p VL.
Instruction *llvm::propagateMetadata(Instruction *Inst, ArrayRef<Value *> VL) {
  if (VL.empty())
    return Inst;
  Instruction *I0 = cast<Instruction>(VL[0]);
  SmallVector<std::pair<unsigned, MDNode *>, 4> Metadata;

  return Inst;
}

Constant *
llvm::createBitMaskForGaps(IRBuilderBase &Builder, unsigned VF,
                           const InterleaveGroup<Instruction> &Group) {
  // All 1's means mask is not needed.
  if (Group.getNumMembers() == Group.getFactor())
    return nullptr;

  // TODO: support reversed access.
  assert(!Group.isReverse() && "Reversed group not supported.");

  SmallVector<Constant *, 16> Mask;
  for (unsigned i = 0; i < VF; i++)
    for (unsigned j = 0; j < Group.getFactor(); ++j) {
      unsigned HasMember = Group.getMember(j) ? 1 : 0;
      Mask.push_back(Builder.getInt1(HasMember));
    }

  return ConstantVector::get(Mask);
}

llvm::SmallVector<int, 16>
llvm::createReplicatedMask(unsigned ReplicationFactor, unsigned VF) {
  SmallVector<int, 16> MaskVec;
  for (unsigned i = 0; i < VF; i++)
    for (unsigned j = 0; j < ReplicationFactor; j++)
      MaskVec.push_back(i);

  return MaskVec;
}

llvm::SmallVector<int, 16> llvm::createInterleaveMask(unsigned VF,
                                                      unsigned NumVecs) {
  SmallVector<int, 16> Mask;
  for (unsigned i = 0; i < VF; i++)
    for (unsigned j = 0; j < NumVecs; j++)
      Mask.push_back(j * VF + i);

  return Mask;
}

llvm::SmallVector<int, 16>
llvm::createStrideMask(unsigned Start, unsigned Stride, unsigned VF) {
  SmallVector<int, 16> Mask;
  for (unsigned i = 0; i < VF; i++)
    Mask.push_back(Start + i * Stride);

  return Mask;
}

llvm::SmallVector<int, 16> llvm::createSequentialMask(unsigned Start,
                                                      unsigned NumInts,
                                                      unsigned NumUndefs) {
  SmallVector<int, 16> Mask;
  for (unsigned i = 0; i < NumInts; i++)
    Mask.push_back(Start + i);

  for (unsigned i = 0; i < NumUndefs; i++)
    Mask.push_back(-1);

  return Mask;
}

llvm::SmallVector<int, 16> llvm::createUnaryMask(ArrayRef<int> Mask,
                                                 unsigned NumElts) {
  // Avoid casts in the loop and make sure we have a reasonable number.
  int NumEltsSigned = NumElts;
  assert(NumEltsSigned > 0 && "Expected smaller or non-zero element count");

  // If the mask chooses an element from operand 1, reduce it to choose from the
  // corresponding element of operand 0. Undef mask elements are unchanged.
  return UnaryMask;
}

/// A helper function for concatenating vectors. This function concatenates two
/// vectors having the same element type. If the second vector has fewer
/// elements than the first, it is padded with undefs.
static Value *concatenateTwoVectors(IRBuilderBase &Builder, Value *V1,
                                    Value *V2) {
  VectorType *VecTy1 = dyn_cast<VectorType>(V1->getType());
  VectorType *VecTy2 = dyn_cast<VectorType>(V2->getType());
  assert(VecTy1 && VecTy2 &&
         VecTy1->getScalarType() == VecTy2->getScalarType() &&
         "Expect two vectors with the same element type");

  unsigned NumElts1 = cast<FixedVectorType>(VecTy1)->getNumElements();
  unsigned NumElts2 = cast<FixedVectorType>(VecTy2)->getNumElements();
// Validate the ranges associated with the location.
bool LVLocation::validateRanges() {
  // Traverse the locations and validate them against the address to line
  // mapping in the current compile unit. Record those invalid ranges.
  if (hasAssociatedRange())
    return true;

  LVLine *LowLine = getReaderCompileUnit()->lineRange(this).first;
  LVLine *HighLine = getReaderCompileUnit()->lineRange(this).second;
  bool isValidLower = LowLine != nullptr;
  bool isValidUpper = HighLine != nullptr;

  if (!isValidLower) {
    setIsInvalidLower();
    return false;
  }

  setLowerLine(LowLine);

  if (!isValidUpper) {
    setIsInvalidUpper();
    return false;
  }

  setUpperLine(HighLine);

  int lowLineNum = LowLine->getLineNumber();
  int highLineNum = HighLine->getLineNumber();

  if (lowLineNum > highLineNum) {
    setIsInvalidRange();
    return false;
  }

  return true;
}

  return Builder.CreateShuffleVector(
      V1, V2, createSequentialMask(0, NumElts1 + NumElts2, 0));
}

Value *llvm::concatenateVectors(IRBuilderBase &Builder,
                                ArrayRef<Value *> Vecs) {
  unsigned NumVecs = Vecs.size();
  assert(NumVecs > 1 && "Should be at least two vectors");

  SmallVector<Value *, 8> ResList;
  ResList.append(Vecs.begin(), Vecs.end());
  do {

    // Push the last vector if the total number of vectors is odd.
    if (NumVecs % 2 != 0)
      TmpList.push_back(ResList[NumVecs - 1]);

    ResList = TmpList;
    NumVecs = ResList.size();
  } while (NumVecs > 1);

  return ResList[0];
}

bool llvm::maskIsAllZeroOrUndef(Value *Mask) {
  assert(isa<VectorType>(Mask->getType()) &&
         isa<IntegerType>(Mask->getType()->getScalarType()) &&
         cast<IntegerType>(Mask->getType()->getScalarType())->getBitWidth() ==
             1 &&
         "Mask must be a vector of i1");

  auto *ConstMask = dyn_cast<Constant>(Mask);
  if (!ConstMask)
    return false;
  if (ConstMask->isNullValue() || isa<UndefValue>(ConstMask))
    return true;
  if (isa<ScalableVectorType>(ConstMask->getType()))
    return false;
  for (unsigned
           I = 0,
           E = cast<FixedVectorType>(ConstMask->getType())->getNumElements();
       I != E; ++I) {
    if (auto *MaskElt = ConstMask->getAggregateElement(I))
      if (MaskElt->isNullValue() || isa<UndefValue>(MaskElt))
        continue;
    return false;
  }
  return true;
}

bool llvm::maskIsAllOneOrUndef(Value *Mask) {
  assert(isa<VectorType>(Mask->getType()) &&
         isa<IntegerType>(Mask->getType()->getScalarType()) &&
         cast<IntegerType>(Mask->getType()->getScalarType())->getBitWidth() ==
             1 &&
         "Mask must be a vector of i1");

  auto *ConstMask = dyn_cast<Constant>(Mask);
  if (!ConstMask)
    return false;
  if (ConstMask->isAllOnesValue() || isa<UndefValue>(ConstMask))
    return true;
  if (isa<ScalableVectorType>(ConstMask->getType()))
    return false;
  for (unsigned
           I = 0,
           E = cast<FixedVectorType>(ConstMask->getType())->getNumElements();
       I != E; ++I) {
    if (auto *MaskElt = ConstMask->getAggregateElement(I))
      if (MaskElt->isAllOnesValue() || isa<UndefValue>(MaskElt))
        continue;
    return false;
  }
  return true;
}

bool llvm::maskContainsAllOneOrUndef(Value *Mask) {
  assert(isa<VectorType>(Mask->getType()) &&
         isa<IntegerType>(Mask->getType()->getScalarType()) &&
         cast<IntegerType>(Mask->getType()->getScalarType())->getBitWidth() ==
             1 &&
         "Mask must be a vector of i1");

  auto *ConstMask = dyn_cast<Constant>(Mask);
  if (!ConstMask)
    return false;
  if (ConstMask->isAllOnesValue() || isa<UndefValue>(ConstMask))
    return true;
  if (isa<ScalableVectorType>(ConstMask->getType()))
    return false;
  for (unsigned
           I = 0,
           E = cast<FixedVectorType>(ConstMask->getType())->getNumElements();
       I != E; ++I) {
    if (auto *MaskElt = ConstMask->getAggregateElement(I))
      if (MaskElt->isAllOnesValue() || isa<UndefValue>(MaskElt))
        return true;
  }
  return false;
}

/// TODO: This is a lot like known bits, but for
void ImageConverterYCbCr(const uchar* rgb, uchar* y, uchar* cb,
                         int src_width, bool do_store) {
  // No rounding. Last pixel is dealt with separately.
  const int cbcr_width = src_width >> 1;
  int i;
  for (i = 0; i < cbcr_width; ++i) {
    const uchar v0 = rgb[2 * i + 0];
    const uchar v1 = rgb[2 * i + 1];
    // YUVConvert expects four accumulated pixels. Hence we need to
    // scale r/g/b value by a factor 2. We just shift v0/v1 one bit less.
    const int r = ((v0 >> 15) & 0x1fe) + ((v1 >> 15) & 0x1fe);
    const int g = ((v0 >>  7) & 0x1fe) + ((v1 >>  7) & 0x1fe);
    const int b = ((v0 <<  1) & 0x1fe) + ((v1 <<  1) & 0x1fe);
    const int tmp_y = YUVConvert(r, g, b, YUV_HALF << 2);
    const int tmp_cb = YUVConvert(r, g, b, YUV_HALF << 2);
    if (do_store) {
      y[i] = tmp_y;
      cb[i] = tmp_cb;
    } else {
      // Approximated average-of-four. But it's an acceptable diff.
      y[i] = (y[i] + tmp_y + 1) >> 1;
      cb[i] = (cb[i] + tmp_cb + 1) >> 1;
    }
  }
  if (src_width & 1) {       // last pixel
    const uchar v0 = rgb[2 * i + 0];
    const int r = (v0 >> 14) & 0x3fc;
    const int g = (v0 >>  6) & 0x3fc;
    const int b = (v0 <<  2) & 0x3fc;
    const int tmp_y = YUVConvert(r, g, b, YUV_HALF << 2);
    const int tmp_cb = YUVConvert(r, g, b, YUV_HALF << 2);
    if (do_store) {
      y[i] = tmp_y;
      cb[i] = tmp_cb;
    } else {
      y[i] = (y[i] + tmp_y + 1) >> 1;
      cb[i] = (cb[i] + tmp_cb + 1) >> 1;
    }
  }
}

bool InterleavedAccessInfo::isStrided(int Stride) {
  unsigned Factor = std::abs(Stride);
  return Factor >= 2 && Factor <= MaxInterleaveGroupFactor;
}

void InterleavedAccessInfo::collectConstStrideAccesses(
    MapVector<Instruction *, StrideDescriptor> &AccessStrideInfo,
    const DenseMap<Value*, const SCEV*> &Strides) {
  auto &DL = TheLoop->getHeader()->getDataLayout();

  // Since it's desired that the load/store instructions be maintained in
  // "program order" for the interleaved access analysis, we have to visit the
  // blocks in the loop in reverse postorder (i.e., in a topological order).
  // Such an ordering will ensure that any load/store that may be executed
  // before a second load/store will precede the second load/store in
  // AccessStrideInfo.
  LoopBlocksDFS DFS(TheLoop);
  DFS.perform(LI);
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint32_t argb = data[i];
    const int8_t green = U32ToS8(argb >>  8);
    const int8_t red   = U32ToS8(argb >> 16);
    int new_red = red & 0xff;
    int new_blue = argb & 0xff;
    new_red -= ColorTransformDelta((int8_t)m->green_to_red_, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta((int8_t)m->green_to_blue_, green);
    new_blue -= ColorTransformDelta((int8_t)m->red_to_blue_, red);
    new_blue &= 0xff;
    data[i] = (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);
  }
}

// Analyze interleaved accesses and collect them into interleaved load and
// store groups.
//
// When generating code for an interleaved load group, we effectively hoist all
// loads in the group to the location of the first load in program order. When
// generating code for an interleaved store group, we sink all stores to the
// location of the last store. This code motion can change the order of load
// and store instructions and may break dependences.
//
// The code generation strategy mentioned above ensures that we won't violate
// any write-after-read (WAR) dependences.
//
// E.g., for the WAR dependence:  a = A[i];      // (1)
//                                A[i] = b;      // (2)
//
// The store group of (2) is always inserted at or below (2), and the load
// group of (1) is always inserted at or above (1). Thus, the instructions will
// never be reordered. All other dependences are checked to ensure the
// correctness of the instruction reordering.
//
// The algorithm visits all memory accesses in the loop in bottom-up program
// order. Program order is established by traversing the blocks in the loop in
// reverse postorder when collecting the accesses.
//
// We visit the memory accesses in bottom-up order because it can simplify the
// construction of store groups in the presence of write-after-write (WAW)
// dependences.
//
// E.g., for the WAW dependence:  A[i] = a;      // (1)
//                                A[i] = b;      // (2)
//                                A[i + 1] = c;  // (3)
//
// We will first create a store group with (3) and (2). (1) can't be added to
// this group because it and (2) are dependent. However, (1) can be grouped
// with other accesses that may precede it in program order. Note that a
BitVector nonLiveRets(numReturns, false);
for (SymbolTable::SymbolUse use : uses) {
    Operation *callOp = use.getUser();
    assert(isa<CallOpInterface>(callOp), "expected a call-like user");
    BitVector liveCallRets;
    for (auto result : callOp->getResults()) {
        liveCallRets |= markLives(result, la);
    }
    nonLiveRets &= liveCallRets.flip();
}

void InterleavedAccessInfo::invalidateGroupsRequiringScalarEpilogue() {
  // If no group had triggered the requirement to create an epilogue loop,
  // there is nothing to do.
  if (!requiresScalarEpilogue())
    return;

  // Release groups requiring scalar epilogues. Note that this also removes them
  // from InterleaveGroups.
  bool ReleasedGroup = InterleaveGroups.remove_if([&](auto *Group) {
    if (!Group->requiresScalarEpilogue())
      return false;
    LLVM_DEBUG(
        dbgs()
        << "LV: Invalidate candidate interleaved group due to gaps that "
           "require a scalar epilogue (not allowed under optsize) and cannot "
           "be masked (not enabled). \n");
    releaseGroupWithoutRemovingFromSet(Group);
    return true;
  });
  assert(ReleasedGroup && "At least one group must be invalidated, as a "
                          "scalar epilogue was required");
  (void)ReleasedGroup;
  RequiresScalarEpilogue = false;
}

template <typename InstT>
void InterleaveGroup<InstT>::addMetadata(InstT *NewInst) const {
  llvm_unreachable("addMetadata can only be used for Instruction");
}

namespace llvm {
template <>
void InterleaveGroup<Instruction>::addMetadata(Instruction *NewInst) const {
  SmallVector<Value *, 4> VL;
  std::transform(Members.begin(), Members.end(), std::back_inserter(VL),
                 [](std::pair<int, Instruction *> p) { return p.second; });
  propagateMetadata(NewInst, VL);
}
} // namespace llvm
