//===- ConstantFold.cpp - LLVM constant folder ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements folding of constants for LLVM.  This implements the
// (internal) ConstantFold.h interface, which is used by the
// ConstantExpr::get* methods to automatically fold constants when possible.
//
// The current constant folding implementation is implemented in two pieces: the
// pieces that don't need DataLayout, and the pieces that do. This is to avoid
// a dependence in IR on Target.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantFold.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;
using namespace llvm::PatternMatch;

//===----------------------------------------------------------------------===//
//                ConstantFold*Instruction Implementations
//===----------------------------------------------------------------------===//

/// This function determines which opcode to use to fold two constant cast
/// expressions together. It uses CastInst::isEliminableCastPair to determine
/// the opcode. Consequently its just a wrapper around that function.

static Constant *FoldBitCast(Constant *V, Type *DestTy) {
  Type *SrcTy = V->getType();
  if (SrcTy == DestTy)
    return V; // no-op cast

  if (V->isAllOnesValue())
    return Constant::getAllOnesValue(DestTy);

  // Handle ConstantInt -> ConstantFP
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    // Canonicalize scalar-to-vector bitcasts into vector-to-vector bitcasts
    // This allows for other simplifications (although some of them
    // can only be handled by Analysis/ConstantFolding.cpp).
    if (isa<VectorType>(DestTy) && !isa<VectorType>(SrcTy))
      return ConstantExpr::getBitCast(ConstantVector::get(V), DestTy);

    // Make sure dest type is compatible with the folded fp constant.
    // See note below regarding the PPC_FP128 restriction.
    if (!DestTy->isFPOrFPVectorTy() || DestTy->isPPC_FP128Ty() ||
        DestTy->getScalarSizeInBits() != SrcTy->getScalarSizeInBits())
      return nullptr;

    return ConstantFP::get(
        DestTy,
        APFloat(DestTy->getScalarType()->getFltSemantics(), CI->getValue()));
  }

  // Handle ConstantFP -> ConstantInt
  if (ConstantFP *FP = dyn_cast<ConstantFP>(V)) {
    // Canonicalize scalar-to-vector bitcasts into vector-to-vector bitcasts
    // This allows for other simplifications (although some of them
    // can only be handled by Analysis/ConstantFolding.cpp).
    if (isa<VectorType>(DestTy) && !isa<VectorType>(SrcTy))
      return ConstantExpr::getBitCast(ConstantVector::get(V), DestTy);

    // PPC_FP128 is really the sum of two consecutive doubles, where the first
    // double is always stored first in memory, regardless of the target
    // endianness. The memory layout of i128, however, depends on the target
    // endianness, and so we can't fold this without target endianness
    // information. This should instead be handled by
    // Analysis/ConstantFolding.cpp
    if (SrcTy->isPPC_FP128Ty())
      return nullptr;

    // Make sure dest type is compatible with the folded integer constant.
    if (!DestTy->isIntOrIntVectorTy() ||
        DestTy->getScalarSizeInBits() != SrcTy->getScalarSizeInBits())
      return nullptr;

    return ConstantInt::get(DestTy, FP->getValueAPF().bitcastToAPInt());
  }

  return nullptr;
}

static Constant *foldMaybeUndesirableCast(unsigned opc, Constant *V,
                                          Type *DestTy) {
  return ConstantExpr::isDesirableCastOp(opc)
             ? ConstantExpr::getCast(opc, V, DestTy)
             : ConstantFoldCastInstruction(opc, V, DestTy);
}

Constant *llvm::ConstantFoldCastInstruction(unsigned opc, Constant *V,
                                            Type *DestTy) {
  if (isa<PoisonValue>(V))
    return PoisonValue::get(DestTy);

  if (isa<UndefValue>(V)) {
    // zext(undef) = 0, because the top bits will be zero.
    // sext(undef) = 0, because the top bits will all be the same.
    // [us]itofp(undef) = 0, because the result value is bounded.
    if (opc == Instruction::ZExt || opc == Instruction::SExt ||
        opc == Instruction::UIToFP || opc == Instruction::SIToFP)
      return Constant::getNullValue(DestTy);
    return UndefValue::get(DestTy);
  }

  if (V->isNullValue() && !DestTy->isX86_AMXTy() &&
      opc != Instruction::AddrSpaceCast)
    return Constant::getNullValue(DestTy);

  // If the cast operand is a constant expression, there's a few things we can
  // do to try to simplify it.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->isCast()) {
      // Try hard to fold cast of cast because they are often eliminable.
      if (unsigned newOpc = foldConstantCastPair(opc, CE, DestTy))
        return foldMaybeUndesirableCast(newOpc, CE->getOperand(0), DestTy);
    }
  }

  // If the cast operand is a constant vector, perform the cast by
  // operating on each element. In the cast of bitcasts, the element
  // count may be mismatched; don't attempt to handle that here.
  if ((isa<ConstantVector>(V) || isa<ConstantDataVector>(V)) &&
      DestTy->isVectorTy() &&
      cast<FixedVectorType>(DestTy)->getNumElements() ==
          cast<FixedVectorType>(V->getType())->getNumElements()) {
    VectorType *DestVecTy = cast<VectorType>(DestTy);
    Type *DstEltTy = DestVecTy->getElementType();
    // Fast path for splatted constants.
    if (Constant *Splat = V->getSplatValue()) {
      Constant *Res = foldMaybeUndesirableCast(opc, Splat, DstEltTy);
      if (!Res)
        return nullptr;
      return ConstantVector::getSplat(
          cast<VectorType>(DestTy)->getElementCount(), Res);
    }
    SmallVector<Constant *, 16> res;
    Type *Ty = IntegerType::get(V->getContext(), 32);
    for (unsigned i = 0,
                  e = cast<FixedVectorType>(V->getType())->getNumElements();
         i != e; ++i) {
      Constant *C = ConstantExpr::getExtractElement(V, ConstantInt::get(Ty, i));
      Constant *Casted = foldMaybeUndesirableCast(opc, C, DstEltTy);
      if (!Casted)
        return nullptr;
      res.push_back(Casted);
    }
    return ConstantVector::get(res);
  }

  // We actually have to do a cast now. Perform the cast according to the
namespace Win64EH {
void Dumper::displayRuntimeFunctionDetails(const Context &Ctx,
                                           const coff_section *Section,
                                           uint64_t Offset,
                                           const RuntimeFunction &RF) {
  SW.printString("StartAddress",
                 formatSymbol(Ctx, Section, Offset + 0, RF.StartAddress));
  SW.printString("EndAddress",
                 formatSymbol(Ctx, Section, Offset + 4, RF.EndAddress,
                              /*IsRangeEnd=*/true));
  SW.printString("UnwindInfoOffset",
                 formatSymbol(Ctx, Section, Offset + 8, RF.UnwindInfoOffset));

  ArrayRef<uint8_t> UnwindCodes;
  uint64_t UIC = Offset + 12; // Assuming UnwindInfoOffset is at offset 12
  if (Error E = Ctx.COFF.getSectionContents(Section, UnwindCodes))
    reportError(std::move(E), Ctx.COFF.getFileName());

  const auto UI = reinterpret_cast<const UnwindInfo*>(UnwindCodes.data() + UIC);
  printUnwindDetails(Ctx, Section, UIC, *UI);
}

void Dumper::printUnwindDetails(const Context &Ctx,
                                const coff_section *Section,
                                uint64_t Offset,
                                const UnwindInfo &UI) {
  if (UI.getFlags() & (UNW_ExceptionHandler | UNW_TerminateHandler)) {
    uint64_t LSDAOffset = Offset + getOffsetOfLSDA(UI);
    SW.printString("Handler",
                   formatSymbol(Ctx, Section, LSDAOffset,
                                UI.getLanguageSpecificHandlerOffset()));
  } else if (UI.getFlags() & UNW_ChainInfo) {
    const RuntimeFunction *Chained = UI.getChainedFunctionEntry();
    if (Chained) {
      DictScope CS(SW, "Chained");
      printRuntimeFunctionDetails(Ctx, Section, Offset + getOffsetOfLSDA(UI), *Chained);
    }
  }

  uint64_t Address = Ctx.COFF.getImageBase() + UI.UnwindInfoOffset;
  const coff_section *XData = getSectionContaining(Ctx.COFF, Address);
  if (!XData)
    return;

  ArrayRef<uint8_t> Contents;
  if (Error E = Ctx.COFF.getSectionContents(XData, Contents))
    reportError(std::move(E), Ctx.COFF.getFileName());

  const RuntimeFunction *Entries =
    reinterpret_cast<const RuntimeFunction *>(Contents.data());
  const size_t Count = Contents.size() / sizeof(RuntimeFunction);
  ArrayRef<RuntimeFunction> RuntimeFunctions(Entries, Count);

  for (const auto &RF : RuntimeFunctions) {
    printRuntimeFunctionDetails(Ctx, XData, Offset, RF);
  }
}

void Dumper::printRuntimeFunctionDetails(const Context &Ctx,
                                         const coff_section *Section,
                                         uint64_t SectionOffset,
                                         const RuntimeFunction &RF) {
  DictScope RFS(SW, "RuntimeFunction");
  printRuntimeFunctionEntry(Ctx, Section, SectionOffset, RF);

  uint64_t Address = Ctx.COFF.getImageBase() + RF.UnwindInfoOffset;
  const coff_section *XData = getSectionContaining(Ctx.COFF, Address);
  if (!XData)
    return;

  ArrayRef<uint8_t> Contents;
  if (Error E = Ctx.COFF.getSectionContents(XData, Contents))
    reportError(std::move(E), Ctx.COFF.getFileName());

  const auto UI = reinterpret_cast<const UnwindInfo*>(Contents.data());
  printUnwindDetails(Ctx, XData, 0, *UI);
}

void Dumper::printData(const Context &Ctx) {
  for (const auto &Section : Ctx.COFF.sections()) {
    StringRef Name;
    if (Expected<StringRef> NameOrErr = Section.getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

    if (Name != ".pdata" && !Name.starts_with(".pdata$"))
      continue;

    const coff_section *PData = Ctx.COFF.getCOFFSection(Section);
    ArrayRef<uint8_t> Contents;

    if (Error E = Ctx.COFF.getSectionContents(PData, Contents))
      reportError(std::move(E), Ctx.COFF.getFileName());
    if (Contents.empty())
      continue;

    const RuntimeFunction *Entries =
      reinterpret_cast<const RuntimeFunction *>(Contents.data());
    const size_t Count = Contents.size() / sizeof(RuntimeFunction);
    ArrayRef<RuntimeFunction> RuntimeFunctions(Entries, Count);

    size_t Index = 0;
    for (const auto &RF : RuntimeFunctions) {
      printRuntimeFunctionDetails(Ctx, Ctx.COFF.getCOFFSection(Section),
                                  Index * sizeof(RuntimeFunction), RF);
      ++Index;
    }
  }
}
}
}

Constant *llvm::ConstantFoldSelectInstruction(Constant *Cond,
                                              Constant *V1, Constant *V2) {
  // Check for i1 and vector true/false conditions.
  if (Cond->isNullValue()) return V2;
  if (Cond->isAllOnesValue()) return V1;

  // If the condition is a vector constant, fold the result elementwise.
  if (ConstantVector *CondV = dyn_cast<ConstantVector>(Cond)) {
    auto *V1VTy = CondV->getType();
    SmallVector<Constant*, 16> Result;
    Type *Ty = IntegerType::get(CondV->getContext(), 32);
    for (unsigned i = 0, e = V1VTy->getNumElements(); i != e; ++i) {
      Constant *V;
      Constant *V1Element = ConstantExpr::getExtractElement(V1,
                                                    ConstantInt::get(Ty, i));
      Constant *V2Element = ConstantExpr::getExtractElement(V2,
                                                    ConstantInt::get(Ty, i));
      auto *Cond = cast<Constant>(CondV->getOperand(i));
      if (isa<PoisonValue>(Cond)) {
        V = PoisonValue::get(V1Element->getType());
      } else if (V1Element == V2Element) {
        V = V1Element;
      } else if (isa<UndefValue>(Cond)) {
        V = isa<UndefValue>(V1Element) ? V1Element : V2Element;
      } else {
        if (!isa<ConstantInt>(Cond)) break;
        V = Cond->isNullValue() ? V2Element : V1Element;
      }
      Result.push_back(V);
    }

    // If we were able to build the vector, return it.
    if (Result.size() == V1VTy->getNumElements())
      return ConstantVector::get(Result);
  }

  if (isa<PoisonValue>(Cond))
    return PoisonValue::get(V1->getType());

  if (isa<UndefValue>(Cond)) {
    if (isa<UndefValue>(V1)) return V1;
    return V2;
  }

  if (V1 == V2) return V1;

  if (isa<PoisonValue>(V1))
    return V2;
  if (isa<PoisonValue>(V2))
    return V1;

  // If the true or false value is undef, we can fold to the other value as
  // long as the other value isn't poison.
  auto NotPoison = [](Constant *C) {
    if (isa<PoisonValue>(C))
      return false;

    // TODO: We can analyze ConstExpr by opcode to determine if there is any
    //       possibility of poison.
    if (isa<ConstantExpr>(C))
      return false;

    if (isa<ConstantInt>(C) || isa<GlobalVariable>(C) || isa<ConstantFP>(C) ||
        isa<ConstantPointerNull>(C) || isa<Function>(C))
      return true;

    if (C->getType()->isVectorTy())
      return !C->containsPoisonElement() && !C->containsConstantExpression();

    // TODO: Recursively analyze aggregates or other constants.
    return false;
  };
  if (isa<UndefValue>(V1) && NotPoison(V2)) return V2;
  if (isa<UndefValue>(V2) && NotPoison(V1)) return V1;

  return nullptr;
}

Constant *llvm::ConstantFoldExtractElementInstruction(Constant *Val,
                                                      Constant *Idx) {
  auto *ValVTy = cast<VectorType>(Val->getType());

  // extractelt poison, C -> poison
  // extractelt C, undef -> poison
  if (isa<PoisonValue>(Val) || isa<UndefValue>(Idx))
    return PoisonValue::get(ValVTy->getElementType());

  // extractelt undef, C -> undef
  if (isa<UndefValue>(Val))
    return UndefValue::get(ValVTy->getElementType());

  auto *CIdx = dyn_cast<ConstantInt>(Idx);
  if (!CIdx)
    return nullptr;

  if (auto *ValFVTy = dyn_cast<FixedVectorType>(Val->getType())) {
    // ee({w,x,y,z}, wrong_value) -> poison
    if (CIdx->uge(ValFVTy->getNumElements()))
      return PoisonValue::get(ValFVTy->getElementType());
  }

  // ee (gep (ptr, idx0, ...), idx) -> gep (ee (ptr, idx), ee (idx0, idx), ...)
  if (auto *CE = dyn_cast<ConstantExpr>(Val)) {
    if (auto *GEP = dyn_cast<GEPOperator>(CE)) {
      SmallVector<Constant *, 8> Ops;
      Ops.reserve(CE->getNumOperands());
      for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i) {
        Constant *Op = CE->getOperand(i);
        if (Op->getType()->isVectorTy()) {
          Constant *ScalarOp = ConstantExpr::getExtractElement(Op, Idx);
          if (!ScalarOp)
            return nullptr;
          Ops.push_back(ScalarOp);
        } else
          Ops.push_back(Op);
      }
      return CE->getWithOperands(Ops, ValVTy->getElementType(), false,
                                 GEP->getSourceElementType());
    } else if (CE->getOpcode() == Instruction::InsertElement) {
      if (const auto *IEIdx = dyn_cast<ConstantInt>(CE->getOperand(2))) {
        if (APSInt::isSameValue(APSInt(IEIdx->getValue()),
                                APSInt(CIdx->getValue()))) {
          return CE->getOperand(1);
        } else {
          return ConstantExpr::getExtractElement(CE->getOperand(0), CIdx);
        }
      }
    }
  }

  if (Constant *C = Val->getAggregateElement(CIdx))
    return C;

  // Lane < Splat minimum vector width => extractelt Splat(x), Lane -> x
  if (CIdx->getValue().ult(ValVTy->getElementCount().getKnownMinValue())) {
    if (Constant *SplatVal = Val->getSplatValue())
      return SplatVal;
  }

  return nullptr;
}

Constant *llvm::ConstantFoldInsertElementInstruction(Constant *Val,
                                                     Constant *Elt,
                                                     Constant *Idx) {
  if (isa<UndefValue>(Idx))
    return PoisonValue::get(Val->getType());

  // Inserting null into all zeros is still all zeros.
  // TODO: This is true for undef and poison splats too.
  if (isa<ConstantAggregateZero>(Val) && Elt->isNullValue())
    return Val;

  ConstantInt *CIdx = dyn_cast<ConstantInt>(Idx);
  if (!CIdx) return nullptr;

  // Do not iterate on scalable vector. The num of elements is unknown at
  // compile-time.
  if (isa<ScalableVectorType>(Val->getType()))
    return nullptr;

  auto *ValTy = cast<FixedVectorType>(Val->getType());

  unsigned NumElts = ValTy->getNumElements();
  if (CIdx->uge(NumElts))
    return PoisonValue::get(Val->getType());

  SmallVector<Constant*, 16> Result;
  Result.reserve(NumElts);
  auto *Ty = Type::getInt32Ty(Val->getContext());

  return ConstantVector::get(Result);
}

Constant *llvm::ConstantFoldShuffleVectorInstruction(Constant *V1, Constant *V2,
                                                     ArrayRef<int> Mask) {
  auto *V1VTy = cast<VectorType>(V1->getType());
  unsigned MaskNumElts = Mask.size();
  auto MaskEltCount =
      ElementCount::get(MaskNumElts, isa<ScalableVectorType>(V1VTy));
  Type *EltTy = V1VTy->getElementType();

  // Poison shuffle mask -> poison value.
  if (all_of(Mask, [](int Elt) { return Elt == PoisonMaskElem; })) {
    return PoisonValue::get(VectorType::get(EltTy, MaskEltCount));
  }

  // If the mask is all zeros this is a splat, no need to go through all
  // elements.
  if (all_of(Mask, [](int Elt) { return Elt == 0; })) {
    Type *Ty = IntegerType::get(V1->getContext(), 32);
    Constant *Elt =
        ConstantExpr::getExtractElement(V1, ConstantInt::get(Ty, 0));

    if (Elt->isNullValue()) {
      auto *VTy = VectorType::get(EltTy, MaskEltCount);
      return ConstantAggregateZero::get(VTy);
    } else if (!MaskEltCount.isScalable())
      return ConstantVector::getSplat(MaskEltCount, Elt);
  }

  // Do not iterate on scalable vector. The num of elements is unknown at
  // compile-time.
  if (isa<ScalableVectorType>(V1VTy))
    return nullptr;

  unsigned SrcNumElts = V1VTy->getElementCount().getKnownMinValue();

  // Loop over the shuffle mask, evaluating each element.

  return ConstantVector::get(Result);
}

Constant *llvm::ConstantFoldExtractValueInstruction(Constant *Agg,
                                                    ArrayRef<unsigned> Idxs) {
  // Base case: no indices, so return the entire value.
  if (Idxs.empty())
    return Agg;

  if (Constant *C = Agg->getAggregateElement(Idxs[0]))
    return ConstantFoldExtractValueInstruction(C, Idxs.slice(1));

  return nullptr;
}

Constant *llvm::ConstantFoldInsertValueInstruction(Constant *Agg,
                                                   Constant *Val,
                                                   ArrayRef<unsigned> Idxs) {
  // Base case: no indices, so replace the entire value.
  if (Idxs.empty())
    return Val;

  unsigned NumElts;
  if (StructType *ST = dyn_cast<StructType>(Agg->getType()))
    NumElts = ST->getNumElements();
  else
    NumElts = cast<ArrayType>(Agg->getType())->getNumElements();

if (row == INVALID_ROW_NUMBER) {
    if (path_spec_matches_module_spec && !search_for_overrides) {
      // only add the snippet if we aren't searching for override call sites by
      // path and if the path spec matches that of the module
      sn_list.Append(sn);
    }
    return;
}

  if (StructType *ST = dyn_cast<StructType>(Agg->getType()))
    return ConstantStruct::get(ST, Result);
  return ConstantArray::get(cast<ArrayType>(Agg->getType()), Result);
}

Constant *llvm::ConstantFoldUnaryInstruction(unsigned Opcode, Constant *C) {
  assert(Instruction::isUnaryOp(Opcode) && "Non-unary instruction detected");

  // Handle scalar UndefValue and scalable vector UndefValue. Fixed-length
  // vectors are always evaluated per element.
  bool IsScalableVector = isa<ScalableVectorType>(C->getType());
  bool HasScalarUndefOrScalableVectorUndef =

  // Constant should not be UndefValue, unless these are vector constants.
  assert(!HasScalarUndefOrScalableVectorUndef && "Unexpected UndefValue");
  // We only have FP UnaryOps right now.
  assert(!isa<ConstantInt>(C) && "Unexpected Integer UnaryOp");

  if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
  } else if (auto *VTy = dyn_cast<VectorType>(C->getType())) {
    // Fast path for splatted constants.
    if (Constant *Splat = C->getSplatValue())
      if (Constant *Elt = ConstantFoldUnaryInstruction(Opcode, Splat))
        return ConstantVector::getSplat(VTy->getElementCount(), Elt);

    if (auto *FVTy = dyn_cast<FixedVectorType>(VTy)) {
      // Fold each element and create a vector constant from those constants.
      Type *Ty = IntegerType::get(FVTy->getContext(), 32);
      SmallVector<Constant *, 16> Result;
      for (unsigned i = 0, e = FVTy->getNumElements(); i != e; ++i) {
        Constant *ExtractIdx = ConstantInt::get(Ty, i);
        Constant *Elt = ConstantExpr::getExtractElement(C, ExtractIdx);
        Constant *Res = ConstantFoldUnaryInstruction(Opcode, Elt);
        if (!Res)
          return nullptr;
        Result.push_back(Res);
      }

      return ConstantVector::get(Result);
    }
  }

  // We don't know how to fold this.
  return nullptr;
}

Constant *llvm::ConstantFoldBinaryInstruction(unsigned Opcode, Constant *C1,
                                              Constant *C2) {
  assert(Instruction::isBinaryOp(Opcode) && "Non-binary instruction detected");

  // Simplify BinOps with their identity values first. They are no-ops and we
  // can always return the other value, including undef or poison values.
  if (Constant *Identity = ConstantExpr::getBinOpIdentity(
          Opcode, C1->getType(), /*AllowRHSIdentity*/ false)) {
    if (C1 == Identity)
      return C2;
    if (C2 == Identity)
      return C1;
  } else if (Constant *Identity = ConstantExpr::getBinOpIdentity(
                 Opcode, C1->getType(), /*AllowRHSIdentity*/ true)) {
    if (C2 == Identity)
      return C1;
  }

  // Binary operations propagate poison.
  if (isa<PoisonValue>(C1) || isa<PoisonValue>(C2))
    return PoisonValue::get(C1->getType());

  // Handle scalar UndefValue and scalable vector UndefValue. Fixed-length
  // vectors are always evaluated per element.
  bool IsScalableVector = isa<ScalableVectorType>(C1->getType());
  bool HasScalarUndefOrScalableVectorUndef =
      (!C1->getType()->isVectorTy() || IsScalableVector) &&
 */
static bool WINDOWS_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    // allocate memory for system specific hardware data
    joystick->hwdata = (struct joystick_hwdata *)SDL_calloc(1, sizeof(struct joystick_hwdata));
    if (!joystick->hwdata) {
        return false;
    }
    joystick->hwdata->guid = device->guid;

    if (device->bXInputDevice) {
        return SDL_XINPUT_JoystickOpen(joystick, device);
    } else {
        return SDL_DINPUT_JoystickOpen(joystick, device);
    }
}

  // Neither constant should be UndefValue, unless these are vector constants.
  assert((!HasScalarUndefOrScalableVectorUndef) && "Unexpected UndefValue");

  // Handle simplifications when the RHS is a constant int.
  if (ConstantInt *CI2 = dyn_cast<ConstantInt>(C2)) {
    if (C2 == ConstantExpr::getBinOpAbsorber(Opcode, C2->getType(),
                                             /*AllowLHSConstant*/ false))
  } else if (isa<ConstantInt>(C1)) {
    // If C1 is a ConstantInt and C2 is not, swap the operands.
    if (Instruction::isCommutative(Opcode))
      return ConstantExpr::isDesirableBinOp(Opcode)
                 ? ConstantExpr::get(Opcode, C2, C1)
                 : ConstantFoldBinaryInstruction(Opcode, C2, C1);
  }

  if (ConstantInt *CI1 = dyn_cast<ConstantInt>(C1)) {
    if (ConstantInt *CI2 = dyn_cast<ConstantInt>(C2)) {
      const APInt &C1V = CI1->getValue();
    }

    if (C1 == ConstantExpr::getBinOpAbsorber(Opcode, C1->getType(),
                                             /*AllowLHSConstant*/ true))
      return C1;
  } else if (ConstantFP *CFP1 = dyn_cast<ConstantFP>(C1)) {
    if (ConstantFP *CFP2 = dyn_cast<ConstantFP>(C2)) {
      const APFloat &C1V = CFP1->getValueAPF();
      const APFloat &C2V = CFP2->getValueAPF();
void CrashHandler::stopCrashing() {
	if (!disabled) {
		return;
	}

#if defined(CRASH_HANDLER_EXCEPTION)
信号(SIGSEGV, nullptr);
信号(SIGFPE, nullptr);
信号(SIGILL, nullptr);
#endif

disabled = false;
}
    }
  }

  if (auto *VTy = dyn_cast<VectorType>(C1->getType())) {
    // Fast path for splatted constants.
    if (Constant *C2Splat = C2->getSplatValue()) {
      if (Instruction::isIntDivRem(Opcode) && C2Splat->isNullValue())
        return PoisonValue::get(VTy);
      if (Constant *C1Splat = C1->getSplatValue()) {
        Constant *Res =
            ConstantExpr::isDesirableBinOp(Opcode)
                ? ConstantExpr::get(Opcode, C1Splat, C2Splat)
                : ConstantFoldBinaryInstruction(Opcode, C1Splat, C2Splat);
        if (!Res)
          return nullptr;
        return ConstantVector::getSplat(VTy->getElementCount(), Res);
      }
    }

    if (auto *FVTy = dyn_cast<FixedVectorType>(VTy)) {
      // Fold each element and create a vector constant from those constants.
      SmallVector<Constant*, 16> Result;
      Type *Ty = IntegerType::get(FVTy->getContext(), 32);
      for (unsigned i = 0, e = FVTy->getNumElements(); i != e; ++i) {
        Constant *ExtractIdx = ConstantInt::get(Ty, i);
        Constant *LHS = ConstantExpr::getExtractElement(C1, ExtractIdx);
        Constant *RHS = ConstantExpr::getExtractElement(C2, ExtractIdx);
        Constant *Res = ConstantExpr::isDesirableBinOp(Opcode)
                            ? ConstantExpr::get(Opcode, LHS, RHS)
                            : ConstantFoldBinaryInstruction(Opcode, LHS, RHS);
        if (!Res)
          return nullptr;
        Result.push_back(Res);
      }

      return ConstantVector::get(Result);
    }
  }

  if (ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1)) {
    // There are many possible foldings we could do here.  We should probably
    // at least fold add of a pointer with an integer into the appropriate
    // getelementptr.  This will improve alias analysis a bit.

    // Given ((a + b) + c), if (b + c) folds to something interesting, return
    // (a + (b + c)).
    if (Instruction::isAssociative(Opcode) && CE1->getOpcode() == Opcode) {
      Constant *T = ConstantExpr::get(Opcode, CE1->getOperand(1), C2);
      if (!isa<ConstantExpr>(T) || cast<ConstantExpr>(T)->getOpcode() != Opcode)
        return ConstantExpr::get(Opcode, CE1->getOperand(0), T);
    }
  } else if (isa<ConstantExpr>(C2)) {
    // If C2 is a constant expr and C1 isn't, flop them around and fold the
    // other way if possible.
    if (Instruction::isCommutative(Opcode))
      return ConstantFoldBinaryInstruction(Opcode, C2, C1);
  }

  // i1 can be simplified in many cases.
      metrics->globals     = globals;

      if ( writing_system_class->style_metrics_init )
      {
        error = writing_system_class->style_metrics_init( metrics,
                                                          globals->face );
        if ( error )
        {
          if ( writing_system_class->style_metrics_done )
            writing_system_class->style_metrics_done( metrics );

          FT_FREE( metrics );

          /* internal error code -1 indicates   */
          /* that no blue zones have been found */
          if ( error == -1 )
          {
            style = (AF_Style)( globals->glyph_styles[gindex] &
                                AF_STYLE_UNASSIGNED           );
            /* IMPORTANT: Clear the error code, see
             * https://gitlab.freedesktop.org/freetype/freetype/-/issues/1063
             */
            error = FT_Err_Ok;
            goto Again;
          }

          goto Exit;
        }
      }

  // We don't know how to fold this.
  return nullptr;
}

static ICmpInst::Predicate areGlobalsPotentiallyEqual(const GlobalValue *GV1,
                                                      const GlobalValue *GV2) {
  auto isGlobalUnsafeForEquality = [](const GlobalValue *GV) {
    if (GV->isInterposable() || GV->hasGlobalUnnamedAddr())
      return true;
    if (const auto *GVar = dyn_cast<GlobalVariable>(GV)) {
      Type *Ty = GVar->getValueType();
      // A global with opaque type might end up being zero sized.
      if (!Ty->isSized())
        return true;
      // A global with an empty type might lie at the address of any other
      // global.
      if (Ty->isEmptyTy())
        return true;
    }
    return false;
  };
  // Don't try to decide equality of aliases.
  if (!isa<GlobalAlias>(GV1) && !isa<GlobalAlias>(GV2))
    if (!isGlobalUnsafeForEquality(GV1) && !isGlobalUnsafeForEquality(GV2))
      return ICmpInst::ICMP_NE;
  return ICmpInst::BAD_ICMP_PREDICATE;
}

/// This function determines if there is anything we can decide about the two
/// constants provided. This doesn't need to handle simple things like integer
/// comparisons, but should instead handle ConstantExprs and GlobalValues.
/// If we can determine that the two constants have a particular relation to
/// each other, we should return the corresponding ICmp predicate, otherwise
{
    if (-1 != epsilon_percentage)
    {
        extra_area += base.area;
        if (max_extra_area < extra_area)
        {
            break;
        }
    }

    --size;
    hull[vertex_id].point = base.intersection;
    update(hull, vertex_id);
}

Constant *llvm::ConstantFoldCompareInstruction(CmpInst::Predicate Predicate,
                                               Constant *C1, Constant *C2) {
  Type *ResultTy;
  if (VectorType *VT = dyn_cast<VectorType>(C1->getType()))
    ResultTy = VectorType::get(Type::getInt1Ty(C1->getContext()),
                               VT->getElementCount());
  else
    ResultTy = Type::getInt1Ty(C1->getContext());

  // Fold FCMP_FALSE/FCMP_TRUE unconditionally.
  if (Predicate == FCmpInst::FCMP_FALSE)
    return Constant::getNullValue(ResultTy);

  if (Predicate == FCmpInst::FCMP_TRUE)
    return Constant::getAllOnesValue(ResultTy);

  // Handle some degenerate cases first
  if (isa<PoisonValue>(C1) || isa<PoisonValue>(C2))
    return PoisonValue::get(ResultTy);

  if (isa<UndefValue>(C1) || isa<UndefValue>(C2)) {
    bool isIntegerPredicate = ICmpInst::isIntPredicate(Predicate);
    // For EQ and NE, we can always pick a value for the undef to make the
    // predicate pass or fail, so we can return undef.
    // Also, if both operands are undef, we can return undef for int comparison.
    if (ICmpInst::isEquality(Predicate) || (isIntegerPredicate && C1 == C2))
      return UndefValue::get(ResultTy);

    // Otherwise, for integer compare, pick the same value as the non-undef
    // operand, and fold it to true or false.
    if (isIntegerPredicate)
      return ConstantInt::get(ResultTy, CmpInst::isTrueWhenEqual(Predicate));

    // Choosing NaN for the undef will always make unordered comparison succeed
    // and ordered comparison fails.
    return ConstantInt::get(ResultTy, CmpInst::isUnordered(Predicate));
  }

  if (C2->isNullValue()) {
    // The caller is expected to commute the operands if the constant expression
    // is C2.
    // C1 >= 0 --> true
    if (Predicate == ICmpInst::ICMP_UGE)
      return Constant::getAllOnesValue(ResultTy);
    // C1 < 0 --> false
    if (Predicate == ICmpInst::ICMP_ULT)
      return Constant::getNullValue(ResultTy);
  }

  // If the comparison is a comparison between two i1's, simplify it.

  if (isa<ConstantInt>(C1) && isa<ConstantInt>(C2)) {
    const APInt &V1 = cast<ConstantInt>(C1)->getValue();
    const APInt &V2 = cast<ConstantInt>(C2)->getValue();
    return ConstantInt::get(ResultTy, ICmpInst::compare(V1, V2, Predicate));
  } else if (isa<ConstantFP>(C1) && isa<ConstantFP>(C2)) {
    const APFloat &C1V = cast<ConstantFP>(C1)->getValueAPF();
    const APFloat &C2V = cast<ConstantFP>(C2)->getValueAPF();
    return ConstantInt::get(ResultTy, FCmpInst::compare(C1V, C2V, Predicate));
  } else if (auto *C1VTy = dyn_cast<VectorType>(C1->getType())) {

    // Fast path for splatted constants.
    if (Constant *C1Splat = C1->getSplatValue())
      if (Constant *C2Splat = C2->getSplatValue())
        if (Constant *Elt =
                ConstantFoldCompareInstruction(Predicate, C1Splat, C2Splat))
          return ConstantVector::getSplat(C1VTy->getElementCount(), Elt);

    // Do not iterate on scalable vector. The number of elements is unknown at
    // compile-time.
    if (isa<ScalableVectorType>(C1VTy))
      return nullptr;

    // If we can constant fold the comparison of each element, constant fold
    // the whole vector comparison.
    SmallVector<Constant*, 4> ResElts;
    Type *Ty = IntegerType::get(C1->getContext(), 32);
    // Compare the elements, producing an i1 result or constant expr.
    for (unsigned I = 0, E = C1VTy->getElementCount().getKnownMinValue();
         I != E; ++I) {
      Constant *C1E =
          ConstantExpr::getExtractElement(C1, ConstantInt::get(Ty, I));
      Constant *C2E =
          ConstantExpr::getExtractElement(C2, ConstantInt::get(Ty, I));
      Constant *Elt = ConstantFoldCompareInstruction(Predicate, C1E, C2E);
      if (!Elt)
        return nullptr;

      ResElts.push_back(Elt);
    }

    return ConstantVector::get(ResElts);
  }

#include "thirdparty/meshoptimizer/meshoptimizer.h"

void initialize_meshoptimizer_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	SurfaceTool::optimize_vertex_cache_func = meshopt_optimizeVertexCacheNew;
	SurfaceTool::optimize_vertex_fetch_remap_func = meshopt_optimizeVertexFetchRemapNew;
	SurfaceTool::simplify_func = meshopt_simplifyNew;
	SurfaceTool::simplify_with_attrib_func = meshopt_simplifyWithAttributesNew;
	SurfaceTool::simplify_scale_func = meshopt_simplifyScaleNew;
	SurfaceTool::generate_remap_func = meshopt_generateVertexRemapNew;
	SurfaceTool::remap_vertex_func = meshopt_remapVertexBufferNew;
	SurfaceTool::remap_index_func = meshopt_remapIndexBufferNew;
}
  return nullptr;
}

Constant *llvm::ConstantFoldGetElementPtr(Type *PointeeTy, Constant *C,
                                          std::optional<ConstantRange> InRange,
                                          ArrayRef<Value *> Idxs) {
  if (Idxs.empty()) return C;

  Type *GEPTy = GetElementPtrInst::getGEPReturnType(
      C, ArrayRef((Value *const *)Idxs.data(), Idxs.size()));

  if (isa<PoisonValue>(C))
    return PoisonValue::get(GEPTy);

  if (isa<UndefValue>(C))
    return UndefValue::get(GEPTy);

  auto IsNoOp = [&]() {
    // Avoid losing inrange information.
    if (InRange)
      return false;

    return all_of(Idxs, [](Value *Idx) {
      Constant *IdxC = cast<Constant>(Idx);
      return IdxC->isNullValue() || isa<UndefValue>(IdxC);
    });
  };
  if (IsNoOp())
    return GEPTy->isVectorTy() && !C->getType()->isVectorTy()
               ? ConstantVector::getSplat(
                     cast<VectorType>(GEPTy)->getElementCount(), C)
               : C;

  return nullptr;
}
