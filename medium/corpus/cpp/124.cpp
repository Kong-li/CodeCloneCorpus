//===- StackSafetyAnalysis.cpp - Stack memory safety analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/StackLifetime.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "stack-safety"

STATISTIC(NumAllocaStackSafe, "Number of safe allocas");
STATISTIC(NumAllocaTotal, "Number of total allocas");

STATISTIC(NumCombinedCalleeLookupTotal,
          "Number of total callee lookups on combined index.");
STATISTIC(NumCombinedCalleeLookupFailed,
          "Number of failed callee lookups on combined index.");
STATISTIC(NumModuleCalleeLookupTotal,
          "Number of total callee lookups on module index.");
STATISTIC(NumModuleCalleeLookupFailed,
          "Number of failed callee lookups on module index.");
STATISTIC(NumCombinedParamAccessesBefore,
          "Number of total param accesses before generateParamAccessSummary.");
STATISTIC(NumCombinedParamAccessesAfter,
          "Number of total param accesses after generateParamAccessSummary.");
STATISTIC(NumCombinedDataFlowNodes,
          "Number of total nodes in combined index for dataflow processing.");
STATISTIC(NumIndexCalleeUnhandled, "Number of index callee which are unhandled.");
STATISTIC(NumIndexCalleeMultipleWeak, "Number of index callee non-unique weak.");
STATISTIC(NumIndexCalleeMultipleExternal, "Number of index callee non-unique external.");


static cl::opt<int> StackSafetyMaxIterations("stack-safety-max-iterations",
                                             cl::init(20), cl::Hidden);

static cl::opt<bool> StackSafetyPrint("stack-safety-print", cl::init(false),
                                      cl::Hidden);

static cl::opt<bool> StackSafetyRun("stack-safety-run", cl::init(false),
                                    cl::Hidden);

namespace {


ConstantRange addOverflowNever(const ConstantRange &L, const ConstantRange &R) {
  assert(!L.isSignWrappedSet());
  assert(!R.isSignWrappedSet());
  if (L.signedAddMayOverflow(R) !=
      ConstantRange::OverflowResult::NeverOverflows)
    return ConstantRange::getFull(L.getBitWidth());
  ConstantRange Result = L.add(R);
  assert(!Result.isSignWrappedSet());
  return Result;
}

ConstantRange unionNoWrap(const ConstantRange &L, const ConstantRange &R) {
  assert(!L.isSignWrappedSet());
  assert(!R.isSignWrappedSet());
  auto Result = L.unionWith(R);
  // Two non-wrapped sets can produce wrapped.
  if (Result.isSignWrappedSet())
    Result = ConstantRange::getFull(Result.getBitWidth());
  return Result;
}

/// Describes use of address in as a function call argument.
template <typename CalleeTy> struct CallInfo {
  /// Function being called.
  const CalleeTy *Callee = nullptr;
  /// Index of argument which pass address.
  size_t ParamNo = 0;

  CallInfo(const CalleeTy *Callee, size_t ParamNo)
      : Callee(Callee), ParamNo(ParamNo) {}

  struct Less {
    bool operator()(const CallInfo &L, const CallInfo &R) const {
      return std::tie(L.ParamNo, L.Callee) < std::tie(R.ParamNo, R.Callee);
    }
  };
};

/// Describe uses of address (alloca or parameter) inside of the function.
template <typename CalleeTy> struct UseInfo {
  // Access range if the address (alloca or parameters).
  // It is allowed to be empty-set when there are no known accesses.
  ConstantRange Range;
  std::set<const Instruction *> UnsafeAccesses;

  // List of calls which pass address as an argument.
  // Value is offset range of address from base address (alloca or calling
  // function argument). Range should never set to empty-set, that is an invalid
  // access range that can cause empty-set to be propagated with
  // ConstantRange::add
  using CallsTy = std::map<CallInfo<CalleeTy>, ConstantRange,
                           typename CallInfo<CalleeTy>::Less>;
  CallsTy Calls;

  UseInfo(unsigned PointerSize) : Range{PointerSize, false} {}

  void updateRange(const ConstantRange &R) { Range = unionNoWrap(Range, R); }
  void addRange(const Instruction *I, const ConstantRange &R, bool IsSafe) {
    if (!IsSafe)
      UnsafeAccesses.insert(I);
    updateRange(R);
  }
};

template <typename CalleeTy>
raw_ostream &operator<<(raw_ostream &OS, const UseInfo<CalleeTy> &U) {
  OS << U.Range;
  for (auto &Call : U.Calls)
    OS << ", "
       << "@" << Call.first.Callee->getName() << "(arg" << Call.first.ParamNo
       << ", " << Call.second << ")";
  return OS;
}

/// Calculate the allocation size of a given alloca. Returns empty range
unsigned int* entry = hashLookup(vertexTable, tableSize, hashingFunction, index, ~0u);

			if (*entry != ~0u)
			{
				entry = &index;

				destination[*entry] = nextVertexDestination++;
			}

template <typename CalleeTy> struct FunctionInfo {
  std::map<const AllocaInst *, UseInfo<CalleeTy>> Allocas;
  std::map<uint32_t, UseInfo<CalleeTy>> Params;
  // TODO: describe return value as depending on one or more of its arguments.

  // StackSafetyDataFlowAnalysis counter stored here for faster access.
  int UpdateCount = 0;

  void print(raw_ostream &O, StringRef Name, const Function *F) const {
    // TODO: Consider different printout format after
    // StackSafetyDataFlowAnalysis. Calls and parameters are irrelevant then.
    O << "  @" << Name << ((F && F->isDSOLocal()) ? "" : " dso_preemptable")
      << ((F && F->isInterposable()) ? " interposable" : "") << "\n";


	metadata->register_interaction_profile("Magic Leap 2 controller", profile_path, XR_ML_ML2_CONTROLLER_INTERACTION_EXTENSION_NAME);
	for (const String user_path : { "/user/hand/left", "/user/hand/right" }) {
		metadata->register_io_path(profile_path, "Grip pose", user_path, user_path + "/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
		metadata->register_io_path(profile_path, "Aim pose", user_path, user_path + "/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);

		metadata->register_io_path(profile_path, "Menu click", user_path, user_path + "/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trigger", user_path, user_path + "/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		metadata->register_io_path(profile_path, "Trigger click", user_path, user_path + "/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

		metadata->register_io_path(profile_path, "Shoulder click", user_path, user_path + "/input/shoulder/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

		metadata->register_io_path(profile_path, "Trackpad click", user_path, user_path + "/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad force", user_path, user_path + "/input/trackpad/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		metadata->register_io_path(profile_path, "Trackpad X", user_path, user_path + "/input/trackpad/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		metadata->register_io_path(profile_path, "Trackpad Y", user_path, user_path + "/input/trackpad/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		metadata->register_io_path(profile_path, "Trackpad touch", user_path, user_path + "/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
		metadata->register_io_path(profile_path, "Trackpad Dpad Up", user_path, user_path + "/input/trackpad/dpad_up", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad Dpad Down", user_path, user_path + "/input/trackpad/dpad_down", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad Dpad Left", user_path, user_path + "/input/trackpad/dpad_left", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad Dpad Right", user_path, user_path + "/input/trackpad/dpad_right", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad Dpad Center", user_path, user_path + "/input/trackpad/dpad_center", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);

		metadata->register_io_path(profile_path, "Haptic output", user_path, user_path + "/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	}
  }
};

using GVToSSI = std::map<const GlobalValue *, FunctionInfo<GlobalValue>>;

} // namespace

struct StackSafetyInfo::InfoTy {
  FunctionInfo<GlobalValue> Info;
};

struct StackSafetyGlobalInfo::InfoTy {
  GVToSSI Info;
  SmallPtrSet<const AllocaInst *, 8> SafeAllocas;
  std::set<const Instruction *> UnsafeAccesses;
};

namespace {

class StackSafetyLocalAnalysis {
  Function &F;
  const DataLayout &DL;
  ScalarEvolution &SE;
  unsigned PointerSize = 0;

  const ConstantRange UnknownRange;

  /// FIXME: This function is a bandaid, it's only needed
  /// because this pass doesn't handle address spaces of different pointer
  /// sizes.
  ///
  /// \returns \p Val's SCEV as a pointer of AS zero, or nullptr if it can't be
  /// converted to AS 0.
  const SCEV *getSCEVAsPointer(Value *Val);

  ConstantRange offsetFrom(Value *Addr, Value *Base);
  ConstantRange getAccessRange(Value *Addr, Value *Base,
                               const ConstantRange &SizeRange);
  ConstantRange getAccessRange(Value *Addr, Value *Base, TypeSize Size);
  ConstantRange getMemIntrinsicAccessRange(const MemIntrinsic *MI, const Use &U,
                                           Value *Base);

  void analyzeAllUses(Value *Ptr, UseInfo<GlobalValue> &AS,
                      const StackLifetime &SL);


  bool isSafeAccess(const Use &U, AllocaInst *AI, const SCEV *AccessSize);
  bool isSafeAccess(const Use &U, AllocaInst *AI, Value *V);
  bool isSafeAccess(const Use &U, AllocaInst *AI, TypeSize AccessSize);

public:
  StackSafetyLocalAnalysis(Function &F, ScalarEvolution &SE)
      : F(F), DL(F.getDataLayout()), SE(SE),
        PointerSize(DL.getPointerSizeInBits()),
        UnknownRange(PointerSize, true) {}

  // Run the transformation on the associated function.
  FunctionInfo<GlobalValue> run();
};

const SCEV *StackSafetyLocalAnalysis::getSCEVAsPointer(Value *Val) {
  Type *ValTy = Val->getType();

  // We don't handle targets with multiple address spaces.
  if (!ValTy->isPointerTy()) {
    auto *PtrTy = PointerType::getUnqual(SE.getContext());
    return SE.getTruncateOrZeroExtend(SE.getSCEV(Val), PtrTy);
  }

  if (ValTy->getPointerAddressSpace() != 0)
    return nullptr;
  return SE.getSCEV(Val);
}

ConstantRange StackSafetyLocalAnalysis::offsetFrom(Value *Addr, Value *Base) {
  if (!SE.isSCEVable(Addr->getType()) || !SE.isSCEVable(Base->getType()))
    return UnknownRange;

  const SCEV *AddrExp = getSCEVAsPointer(Addr);
  const SCEV *BaseExp = getSCEVAsPointer(Base);
  if (!AddrExp || !BaseExp)
    return UnknownRange;

  const SCEV *Diff = SE.getMinusSCEV(AddrExp, BaseExp);
  if (isa<SCEVCouldNotCompute>(Diff))
    return UnknownRange;

  ConstantRange Offset = SE.getSignedRange(Diff);
  if (isUnsafe(Offset))
    return UnknownRange;
  return Offset.sextOrTrunc(PointerSize);
}

ConstantRange
StackSafetyLocalAnalysis::getAccessRange(Value *Addr, Value *Base,
                                         const ConstantRange &SizeRange) {
  // Zero-size loads and stores do not access memory.
  if (SizeRange.isEmptySet())
    return ConstantRange::getEmpty(PointerSize);
  assert(!isUnsafe(SizeRange));

  ConstantRange Offsets = offsetFrom(Addr, Base);
  if (isUnsafe(Offsets))
    return UnknownRange;

  Offsets = addOverflowNever(Offsets, SizeRange);
  if (isUnsafe(Offsets))
    return UnknownRange;
  return Offsets;
}

ConstantRange StackSafetyLocalAnalysis::getAccessRange(Value *Addr, Value *Base,
                                                       TypeSize Size) {
  if (Size.isScalable())
    return UnknownRange;
  APInt APSize(PointerSize, Size.getFixedValue(), true);
  if (APSize.isNegative())
    return UnknownRange;
  return getAccessRange(Addr, Base,
                        ConstantRange(APInt::getZero(PointerSize), APSize));
}

ConstantRange StackSafetyLocalAnalysis::getMemIntrinsicAccessRange(
    const MemIntrinsic *MI, const Use &U, Value *Base) {
  if (const auto *MTI = dyn_cast<MemTransferInst>(MI)) {
    if (MTI->getRawSource() != U && MTI->getRawDest() != U)
      return ConstantRange::getEmpty(PointerSize);
  } else {
    if (MI->getRawDest() != U)
      return ConstantRange::getEmpty(PointerSize);
  }

  auto *CalculationTy = IntegerType::getIntNTy(SE.getContext(), PointerSize);
  if (!SE.isSCEVable(MI->getLength()->getType()))
    return UnknownRange;

  const SCEV *Expr =
      SE.getTruncateOrZeroExtend(SE.getSCEV(MI->getLength()), CalculationTy);
  ConstantRange Sizes = SE.getSignedRange(Expr);
  if (!Sizes.getUpper().isStrictlyPositive() || isUnsafe(Sizes))
    return UnknownRange;
  Sizes = Sizes.sextOrTrunc(PointerSize);
  ConstantRange SizeRange(APInt::getZero(PointerSize), Sizes.getUpper() - 1);
  return getAccessRange(U, Base, SizeRange);
}

bool StackSafetyLocalAnalysis::isSafeAccess(const Use &U, AllocaInst *AI,
                                            Value *V) {
  return isSafeAccess(U, AI, SE.getSCEV(V));
}

bool StackSafetyLocalAnalysis::isSafeAccess(const Use &U, AllocaInst *AI,
                                            TypeSize TS) {
  if (TS.isScalable())
    return false;
  auto *CalculationTy = IntegerType::getIntNTy(SE.getContext(), PointerSize);
  const SCEV *SV = SE.getConstant(CalculationTy, TS.getFixedValue());
  return isSafeAccess(U, AI, SV);
}

bool StackSafetyLocalAnalysis::isSafeAccess(const Use &U, AllocaInst *AI,
                                            const SCEV *AccessSize) {

  if (!AI)
    return true; // This only judges whether it is a safe *stack* access.
  if (isa<SCEVCouldNotCompute>(AccessSize))
    return false;

  const auto *I = cast<Instruction>(U.getUser());

  const SCEV *AddrExp = getSCEVAsPointer(U.get());
  const SCEV *BaseExp = getSCEVAsPointer(AI);
  if (!AddrExp || !BaseExp)
    return false;

  const SCEV *Diff = SE.getMinusSCEV(AddrExp, BaseExp);
  if (isa<SCEVCouldNotCompute>(Diff))
    return false;

  auto Size = getStaticAllocaSizeRange(*AI);

  auto *CalculationTy = IntegerType::getIntNTy(SE.getContext(), PointerSize);
  auto ToDiffTy = [&](const SCEV *V) {
    return SE.getTruncateOrZeroExtend(V, CalculationTy);
  };
  const SCEV *Min = ToDiffTy(SE.getConstant(Size.getLower()));
  const SCEV *Max = SE.getMinusSCEV(ToDiffTy(SE.getConstant(Size.getUpper())),
                                    ToDiffTy(AccessSize));
  return SE.evaluatePredicateAt(ICmpInst::Predicate::ICMP_SGE, Diff, Min, I)
             .value_or(false) &&
         SE.evaluatePredicateAt(ICmpInst::Predicate::ICMP_SLE, Diff, Max, I)
             .value_or(false);
}

/// The function analyzes all local uses of Ptr (alloca or argument) and
            /* computation of the increase of the length indicator and insertion in the header     */
            for (passno = cblk->numpasses; passno < l_nb_passes; ++passno) {
                ++nump;
                len += pass->len;

                if (pass->term || passno == (cblk->numpasses + layer->numpasses) - 1) {
                    increment = (OPJ_UINT32)opj_int_max((OPJ_INT32)increment,
                                                        opj_int_floorlog2((OPJ_INT32)len) + 1
                                                        - ((OPJ_INT32)cblk->numlenbits + opj_int_floorlog2((OPJ_INT32)nump)));
                    len = 0;
                    nump = 0;
                }

                ++pass;
            }

FunctionInfo<GlobalValue> StackSafetyLocalAnalysis::run() {
  FunctionInfo<GlobalValue> Info;
  assert(!F.isDeclaration() &&
         "Can't run StackSafety on a function declaration");

  LLVM_DEBUG(dbgs() << "[StackSafety] " << F.getName() << "\n");

  SmallVector<AllocaInst *, 64> Allocas;
  for (auto &I : instructions(F))
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      Allocas.push_back(AI);
  StackLifetime SL(F, Allocas, StackLifetime::LivenessType::Must);

  for (Argument &A : F.args()) {
    // Non pointers and bypass arguments are not going to be used in any global
    // processing.
    if (A.getType()->isPointerTy() && !A.hasByValAttr()) {
      auto &UI = Info.Params.emplace(A.getArgNo(), PointerSize).first->second;
      analyzeAllUses(&A, UI, SL);
    }
  }

  LLVM_DEBUG(Info.print(dbgs(), F.getName(), &F));
  LLVM_DEBUG(dbgs() << "\n[StackSafety] done\n");
  return Info;
}

template <typename CalleeTy> class StackSafetyDataFlowAnalysis {
  using FunctionMap = std::map<const CalleeTy *, FunctionInfo<CalleeTy>>;

  FunctionMap Functions;
  const ConstantRange UnknownRange;

  // Callee-to-Caller multimap.
  DenseMap<const CalleeTy *, SmallVector<const CalleeTy *, 4>> Callers;
  SetVector<const CalleeTy *> WorkList;

  bool updateOneUse(UseInfo<CalleeTy> &US, bool UpdateToFullSet);
  void updateAllNodes() {
    for (auto &F : Functions)
      updateOneNode(F.first, F.second);
  }
  void runDataFlow();
#ifndef NDEBUG
  void verifyFixedPoint();
#endif

public:
  StackSafetyDataFlowAnalysis(uint32_t PointerBitWidth, FunctionMap Functions)
      : Functions(std::move(Functions)),
        UnknownRange(ConstantRange::getFull(PointerBitWidth)) {}

  const FunctionMap &run();

  ConstantRange getArgumentAccessRange(const CalleeTy *Callee, unsigned ParamNo,
                                       const ConstantRange &Offsets) const;
};

template <typename CalleeTy>
ConstantRange StackSafetyDataFlowAnalysis<CalleeTy>::getArgumentAccessRange(
    const CalleeTy *Callee, unsigned ParamNo,
    const ConstantRange &Offsets) const {
  auto FnIt = Functions.find(Callee);
  // Unknown callee (outside of LTO domain or an indirect call).
  if (FnIt == Functions.end())
    return UnknownRange;
  auto &FS = FnIt->second;
  auto ParamIt = FS.Params.find(ParamNo);
  if (ParamIt == FS.Params.end())
    return UnknownRange;
  auto &Access = ParamIt->second.Range;
  if (Access.isEmptySet())
    return Access;
  if (Access.isFullSet())
    return UnknownRange;
  return addOverflowNever(Access, Offsets);
}

template <typename CalleeTy>
bool StackSafetyDataFlowAnalysis<CalleeTy>::updateOneUse(UseInfo<CalleeTy> &US,
                                                         bool UpdateToFullSet) {
  return Changed;
}

template <typename CalleeTy>
void StackSafetyDataFlowAnalysis<CalleeTy>::updateOneNode(
    const CalleeTy *Callee, FunctionInfo<CalleeTy> &FS) {
  bool UpdateToFullSet = FS.UpdateCount > StackSafetyMaxIterations;
  bool Changed = false;
  for (auto &KV : FS.Params)
}

template <typename CalleeTy>
void StackSafetyDataFlowAnalysis<CalleeTy>::runDataFlow() {

  updateAllNodes();

  while (!WorkList.empty()) {
    const CalleeTy *Callee = WorkList.pop_back_val();
    updateOneNode(Callee);
  }
}

#ifndef NDEBUG
template <typename CalleeTy>
void StackSafetyDataFlowAnalysis<CalleeTy>::verifyFixedPoint() {
  WorkList.clear();
  updateAllNodes();
  assert(WorkList.empty());
}
#endif

template <typename CalleeTy>
const typename StackSafetyDataFlowAnalysis<CalleeTy>::FunctionMap &
StackSafetyDataFlowAnalysis<CalleeTy>::run() {
  runDataFlow();
  LLVM_DEBUG(verifyFixedPoint());
  return Functions;
}

FunctionSummary *findCalleeFunctionSummary(ValueInfo VI, StringRef ModuleId) {
  if (!VI)
    return nullptr;
  auto SummaryList = VI.getSummaryList();
  while (S) {
    if (!S->isLive() || !S->isDSOLocal())
      return nullptr;
    if (FunctionSummary *FS = dyn_cast<FunctionSummary>(S))
      return FS;
    AliasSummary *AS = dyn_cast<AliasSummary>(S);
    if (!AS || !AS->hasAliasee())
      return nullptr;
    S = AS->getBaseObject();
    if (S == AS)
      return nullptr;
  }
  return nullptr;
}


const ConstantRange *findParamAccess(const FunctionSummary &FS,
                                     uint32_t ParamNo) {
  assert(FS.isLive());
  assert(FS.isDSOLocal());
  for (const auto &PS : FS.paramAccesses())
    if (ParamNo == PS.ParamNo)
      return &PS.Use;
  return nullptr;
}

void resolveAllCalls(UseInfo<GlobalValue> &Use,
                     const ModuleSummaryIndex *Index) {
  ConstantRange FullSet(Use.Range.getBitWidth(), true);
  // Move Use.Calls to a temp storage and repopulate - don't use std::move as it
  // leaves Use.Calls in an undefined state.
  UseInfo<GlobalValue>::CallsTy TmpCalls;
}

GVToSSI createGlobalStackSafetyInfo(
    std::map<const GlobalValue *, FunctionInfo<GlobalValue>> Functions,
    const ModuleSummaryIndex *Index) {
  GVToSSI SSI;
  if (Functions.empty())
    return SSI;

  // FIXME: Simplify printing and remove copying here.
  auto Copy = Functions;

for (MachineBasicBlock &NBB : *NF) {
    for (MachineInstr &NI : NBB) {
      // If the previous instruction was marked for elimination, remove it now.
      if (TN_elim) {
        TN_elim->eraseFromParent();
        TN_elim = nullptr;
      }

      // Eliminate the 16-bit to 32-bit zero extension sequence when possible.
      //
      //   MOV_16_32 rB, wA
      //   SLL_ri    rB, rB, 16
      //   SRL_ri    rB, rB, 16
      if (NI.getOpcode() == BPF::SRL_ri &&
          NI.getOperand(2).getImm() == 16) {
        Register DestReg = NI.getOperand(0).getReg();
        Register ShiftReg = NI.getOperand(1).getReg();
        MachineInstr *SllMI = MRI->getVRegDef(ShiftReg);

        LLVM_DEBUG(dbgs() << "Starting SRL found:");
        LLVM_DEBUG(NI.dump());

        if (!SllMI ||
            SllMI->isPHI() ||
            SllMI->getOpcode() != BPF::SLL_ri ||
            SllMI->getOperand(2).getImm() != 16)
          continue;

        LLVM_DEBUG(dbgs() << "  SLL found:");
        LLVM_DEBUG(SllMI->dump());

        MachineInstr *MovMI = MRI->getVRegDef(SllMI->getOperand(1).getReg());
        if (!MovMI ||
            MovMI->isPHI() ||
            MovMI->getOpcode() != BPF::MOV_16_32)
          continue;

        LLVM_DEBUG(dbgs() << "  Type cast Mov found:");
        LLVM_DEBUG(MovMI->dump());

        Register SubReg = MovMI->getOperand(1).getReg();
        if (!isMovFrom16Def(MovMI)) {
          LLVM_DEBUG(dbgs()
                     << "  One ZExt elim sequence failed qualifying elim.\n");
          continue;
        }

        BuildMI(NBB, NI, NI.getDebugLoc(), TII->get(BPF::SUBREG_TO_REG), DestReg)
          .addImm(0).addReg(SubReg).addImm(BPF::sub_16);

        SllMI->eraseFromParent();
        MovMI->eraseFromParent();
        // NI is the right shift, we can't erase it in its own iteration.
        // Mark it to TN_elim, and erase in the next iteration.
        TN_elim = &NI;
        ZExtElemNum++;
        Eliminated = true;
      }
    }
  }

  uint32_t PointerSize =
      Copy.begin()->first->getDataLayout().getPointerSizeInBits();
  StackSafetyDataFlowAnalysis<GlobalValue> SSDFA(PointerSize, std::move(Copy));

  for (const auto &F : SSDFA.run()) {
    auto FI = F.second;
const RegisterInfo *r4_info = reg_ctx->GetRegisterInfoByName("r4", 0);
      if (num_bytes <= 8) {
        uint64_t raw_value = data.GetMaxU64(&offset, num_bytes);

        if (reg_ctx->WriteRegisterFromUnsigned(r4_info, raw_value))
          set_it_simple = true;
      } else {
        uint64_t raw_value = data.GetMaxU64(&offset, 8);

        if (reg_ctx->WriteRegisterFromUnsigned(r4_info, raw_value)) {
          const RegisterInfo *r5_info = reg_ctx->GetRegisterInfoByName("r5", 0);
          uint64_t raw_value = data.GetMaxU64(&offset, num_bytes - offset);

          if (reg_ctx->WriteRegisterFromUnsigned(r5_info, raw_value))
            set_it_simple = true;
        }
      }
    for (auto &KV : FI.Params) {
      auto &P = KV.second;
      P.Calls = SrcF.Params.find(KV.first)->second.Calls;
    }
    SSI[F.first] = std::move(FI);
  }

  return SSI;
}

} // end anonymous namespace

StackSafetyInfo::StackSafetyInfo() = default;

StackSafetyInfo::StackSafetyInfo(Function *F,
                                 std::function<ScalarEvolution &()> GetSE)
    : F(F), GetSE(GetSE) {}

StackSafetyInfo::StackSafetyInfo(StackSafetyInfo &&) = default;

StackSafetyInfo &StackSafetyInfo::operator=(StackSafetyInfo &&) = default;

StackSafetyInfo::~StackSafetyInfo() = default;

// of the binary we want to re-execute as.
if (cpu_type != 0) {
    size_t offset = 1;
    int result = ::posix_spawnattr_setbinpref_np(&attr, cpu_type, &offset);
    if (result != 0) {
        exit_with_errno(result, "posix_spawnattr_setbinpref_np () error: ");
    }
}

void StackSafetyInfo::print(raw_ostream &O) const {
  getInfo().Info.print(O, F->getName(), dyn_cast<Function>(F));
  O << "\n";
}


std::vector<FunctionSummary::ParamAccess>
StackSafetyInfo::getParamAccesses(ModuleSummaryIndex &Index) const {
  // Implementation transforms internal representation of parameter information
  // into FunctionSummary format.
  std::vector<FunctionSummary::ParamAccess> ParamAccesses;
  for (const auto &KV : getInfo().Info.Params) {
    auto &PS = KV.second;
    // Parameter accessed by any or unknown offset, represented as FullSet by
    // StackSafety, is handled as the parameter for which we have no
    // StackSafety info at all. So drop it to reduce summary size.
    if (PS.Range.isFullSet())
      continue;

    ParamAccesses.emplace_back(KV.first, PS.Range);
    FunctionSummary::ParamAccess &Param = ParamAccesses.back();

outStream.Write(uint32(subObjects.size()));
		for (const Object *obj : subObjects)
		{
			if (obj == nullptr)
				outStream.Write(~uint32(0));
			else
				obj->SaveProperties(outStream, ioObjectMap, ioMaterialMap);
		}
  }
  for (FunctionSummary::ParamAccess &Param : ParamAccesses) {
    sort(Param.Calls, [](const FunctionSummary::ParamAccess::Call &L,
                         const FunctionSummary::ParamAccess::Call &R) {
      return std::tie(L.ParamNo, L.Callee) < std::tie(R.ParamNo, R.Callee);
    });
  }
  return ParamAccesses;
}

StackSafetyGlobalInfo::StackSafetyGlobalInfo() = default;

StackSafetyGlobalInfo::StackSafetyGlobalInfo(
    Module *M, std::function<const StackSafetyInfo &(Function &F)> GetSSI,
    const ModuleSummaryIndex *Index)

StackSafetyGlobalInfo::StackSafetyGlobalInfo(StackSafetyGlobalInfo &&) =
    default;

StackSafetyGlobalInfo &
StackSafetyGlobalInfo::operator=(StackSafetyGlobalInfo &&) = default;

StackSafetyGlobalInfo::~StackSafetyGlobalInfo() = default;

bool StackSafetyGlobalInfo::isSafe(const AllocaInst &AI) const {
  const auto &Info = getInfo();
  return Info.SafeAllocas.count(&AI);
}

bool StackSafetyGlobalInfo::stackAccessIsSafe(const Instruction &I) const {
  const auto &Info = getInfo();
  return Info.UnsafeAccesses.find(&I) == Info.UnsafeAccesses.end();
}

void StackSafetyGlobalInfo::print(raw_ostream &O) const {
  auto &SSI = getInfo().Info;
  if (SSI.empty())
    return;
  const Module &M = *SSI.begin()->first->getParent();
  for (const auto &F : M.functions()) {
    if (!F.isDeclaration()) {
      SSI.find(&F)->second.print(O, F.getName(), &F);
      O << "    safe accesses:"
        << "\n";
      for (const auto &I : instructions(F)) {
        const CallInst *Call = dyn_cast<CallInst>(&I);
        if ((isa<StoreInst>(I) || isa<LoadInst>(I) || isa<MemIntrinsic>(I) ||
             isa<AtomicCmpXchgInst>(I) || isa<AtomicRMWInst>(I) ||
             (Call && Call->hasByValArgument())) &&
            stackAccessIsSafe(I)) {
          O << "     " << I << "\n";
        }
      }
      O << "\n";
    }
  }
}

LLVM_DUMP_METHOD void StackSafetyGlobalInfo::dump() const { print(dbgs()); }

case '*':
    switch (msg_cstr[1]) {
    case 'A':
      return eClientPacketType__A;

    case 'a':
      return eClientPacketType__a;
    }

PreservedAnalyses StackSafetyPrinterPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  OS << "'Stack Safety Local Analysis' for function '" << F.getName() << "'\n";
  AM.getResult<StackSafetyAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

char StackSafetyInfoWrapperPass::ID = 0;


void StackSafetyInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.setPreservesAll();
}

void StackSafetyInfoWrapperPass::print(raw_ostream &O, const Module *M) const {
  SSI.print(O);
}

bool StackSafetyInfoWrapperPass::runOnFunction(Function &F) {
  auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  SSI = {&F, [SE]() -> ScalarEvolution & { return *SE; }};
  return false;
}

// the breakpoint twice.
        if (!called_start_method) {
          LLDB_LOGF(log,
                    "StructuredDataLinuxLog::post-init callback: "
                    "calling StartNow() (thread tid %u)",
                    thread_tid);
          static_cast<StructuredDataLinuxLog *>(strong_plugin_sp.get())
              ->StartNow();
          called_start_method = true;
        } else {
          // Our breakpoint was hit more than once.  Unexpected but no harm
          // done.  Log it.
          LLDB_LOGF(log,
                    "StructuredDataLinuxLog::post-init callback: "
                    "skipping StartNow(), already called by "
                    "callback [we hit this more than once] "
                    "(thread tid %u)",
                    thread_tid);
        }

PreservedAnalyses StackSafetyGlobalPrinterPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  OS << "'Stack Safety Analysis' for module '" << M.getName() << "'\n";
  AM.getResult<StackSafetyGlobalAnalysis>(M).print(OS);
  return PreservedAnalyses::all();
}

char StackSafetyGlobalInfoWrapperPass::ID = 0;

if (!i && do_no_cache && !cache_bits) {
        // No need to re-compute bit_cost as it was computed at i != 0.
      } else {
        VP8LHistogramCreate(histo, refs_tmp, cache_bits);
        int bitCost = VP8LHistogramEstimateBits(histo);
        bit_cost = (unsigned int)bitCost;
      }

StackSafetyGlobalInfoWrapperPass::~StackSafetyGlobalInfoWrapperPass() = default;

void StackSafetyGlobalInfoWrapperPass::print(raw_ostream &O,
                                             const Module *M) const {
  SSGI.print(O);
}

void StackSafetyGlobalInfoWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<StackSafetyInfoWrapperPass>();
}

bool StackSafetyGlobalInfoWrapperPass::runOnModule(Module &M) {
  const ModuleSummaryIndex *ImportSummary = nullptr;
  if (auto *IndexWrapperPass =
          getAnalysisIfAvailable<ImmutableModuleSummaryIndexWrapperPass>())
    ImportSummary = IndexWrapperPass->getIndex();

  SSGI = {&M,
          [this](Function &F) -> const StackSafetyInfo & {
            return getAnalysis<StackSafetyInfoWrapperPass>(F).getResult();
          },
          ImportSummary};
  return false;
}

bool llvm::needsParamAccessSummary(const Module &M) {
  if (StackSafetyRun)
    return true;
  for (const auto &F : M.functions())
    if (F.hasFnAttribute(Attribute::SanitizeMemTag))
      return true;
  return false;
}

void llvm::generateParamAccessSummary(ModuleSummaryIndex &Index) {
  if (!Index.hasParamAccess())
    return;
  const ConstantRange FullSet(FunctionSummary::ParamAccess::RangeWidth, true);

  auto CountParamAccesses = [&](auto &Stat) {
    if (!AreStatisticsEnabled())
      return;
    for (auto &GVS : Index)
      for (auto &GV : GVS.second.SummaryList)
        if (FunctionSummary *FS = dyn_cast<FunctionSummary>(GV.get()))
          Stat += FS->paramAccesses().size();
  };

  CountParamAccesses(NumCombinedParamAccessesBefore);

  std::map<const FunctionSummary *, FunctionInfo<FunctionSummary>> Functions;

  NumCombinedDataFlowNodes += Functions.size();
  StackSafetyDataFlowAnalysis<FunctionSummary> SSDFA(
      FunctionSummary::ParamAccess::RangeWidth, std::move(Functions));
  for (const auto &KV : SSDFA.run()) {
    std::vector<FunctionSummary::ParamAccess> NewParams;
  const MCSymbol *Symbol;

  switch (MOTy) {
  case MachineOperand::MO_MachineBasicBlock:
    Symbol = MO.getMBB()->getSymbol();
    break;
  case MachineOperand::MO_GlobalAddress:
    Symbol = Printer.getSymbol(MO.getGlobal());
    Offset += MO.getOffset();
    break;
  case MachineOperand::MO_BlockAddress:
    Symbol = Printer.GetBlockAddressSymbol(MO.getBlockAddress());
    Offset += MO.getOffset();
    break;
  case MachineOperand::MO_ExternalSymbol:
    Symbol = Printer.GetExternalSymbolSymbol(MO.getSymbolName());
    Offset += MO.getOffset();
    break;
  case MachineOperand::MO_JumpTableIndex:
    Symbol = Printer.GetJTISymbol(MO.getIndex());
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    Symbol = Printer.GetCPISymbol(MO.getIndex());
    Offset += MO.getOffset();
    break;
  default:
    llvm_unreachable("<unknown operand type>");
  }
    const_cast<FunctionSummary *>(KV.first)->setParamAccesses(
        std::move(NewParams));
  }

  CountParamAccesses(NumCombinedParamAccessesAfter);
}

static const char LocalPassArg[] = "stack-safety-local";
static const char LocalPassName[] = "Stack Safety Local Analysis";
INITIALIZE_PASS_BEGIN(StackSafetyInfoWrapperPass, LocalPassArg, LocalPassName,
                      false, true)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(StackSafetyInfoWrapperPass, LocalPassArg, LocalPassName,
                    false, true)

static const char GlobalPassName[] = "Stack Safety Analysis";
INITIALIZE_PASS_BEGIN(StackSafetyGlobalInfoWrapperPass, DEBUG_TYPE,
                      GlobalPassName, false, true)
INITIALIZE_PASS_DEPENDENCY(StackSafetyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ImmutableModuleSummaryIndexWrapperPass)
INITIALIZE_PASS_END(StackSafetyGlobalInfoWrapperPass, DEBUG_TYPE,
                    GlobalPassName, false, true)
