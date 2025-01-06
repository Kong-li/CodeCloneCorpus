//===- HexagonExpandCondsets.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Replace mux instructions with the corresponding legal instructions.
// It is meant to work post-SSA, but still on virtual registers. It was
// originally placed between register coalescing and machine instruction
// scheduler.
// In this place in the optimization sequence, live interval analysis had
// been performed, and the live intervals should be preserved. A large part
// of the code deals with preserving the liveness information.
//
// Liveness tracking aside, the main functionality of this pass is divided
// into two steps. The first step is to replace an instruction
//   %0 = C2_mux %1, %2, %3
// with a pair of conditional transfers
//   %0 = A2_tfrt %1, %2
//   %0 = A2_tfrf %1, %3
// It is the intention that the execution of this pass could be terminated
// after this step, and the code generated would be functionally correct.
//
// If the uses of the source values %1 and %2 are kills, and their
// definitions are predicable, then in the second step, the conditional
// transfers will then be rewritten as predicated instructions. E.g.
//   %0 = A2_or %1, %2
//   %3 = A2_tfrt %99, killed %0
// will be rewritten as
//   %3 = A2_port %99, %1, %2
//
// This replacement has two variants: "up" and "down". Consider this case:
//   %0 = A2_or %1, %2
//   ... [intervening instructions] ...
//   %3 = A2_tfrt %99, killed %0
// variant "up":
//   %3 = A2_port %99, %1, %2
//   ... [intervening instructions, %0->vreg3] ...
//   [deleted]
// variant "down":
//   [deleted]
//   ... [intervening instructions] ...
//   %3 = A2_port %99, %1, %2
//
// Both, one or none of these variants may be valid, and checks are made
// to rule out inapplicable variants.
//
// As an additional optimization, before either of the two steps above is
// executed, the pass attempts to coalesce the target register with one of
// the source registers, e.g. given an instruction
//   %3 = C2_mux %0, %1, %2
// %3 will be coalesced with either %1 or %2. If this succeeds,
// the instruction would then be (for example)
//   %3 = C2_mux %0, %3, %2
// and, under certain circumstances, this could result in only one predicated
// instruction:
//   %3 = A2_tfrf %0, %2
//

// Splitting a definition of a register into two predicated transfers
// creates a complication in liveness tracking. Live interval computation
// will see both instructions as actual definitions, and will mark the
// first one as dead. The definition is not actually dead, and this
// situation will need to be fixed. For example:
//   dead %1 = A2_tfrt ...  ; marked as dead
//   %1 = A2_tfrf ...
//
// Since any of the individual predicated transfers may end up getting
// removed (in case it is an identity copy), some pre-existing def may
// be marked as dead after live interval recomputation:
//   dead %1 = ...          ; marked as dead
//   ...
//   %1 = A2_tfrf ...       ; if A2_tfrt is removed
// This case happens if %1 was used as a source in A2_tfrt, which means
// that is it actually live at the A2_tfrf, and so the now dead definition
// of %1 will need to be updated to non-dead at some point.
//
// This issue could be remedied by adding implicit uses to the predicated
// transfers, but this will create a problem with subsequent predication,
// since the transfers will no longer be possible to reorder. To avoid
// that, the initial splitting will not add any implicit uses. These
// implicit uses will be added later, after predication. The extra price,
// however, is that finding the locations where the implicit uses need
// to be added, and updating the live ranges will be more involved.

#include "HexagonInstrInfo.h"
#include "HexagonRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>
#include <map>
#include <set>
#include <utility>

#define DEBUG_TYPE "expand-condsets"

using namespace llvm;

static cl::opt<unsigned> OptTfrLimit("expand-condsets-tfr-limit",
  cl::init(~0U), cl::Hidden, cl::desc("Max number of mux expansions"));
static cl::opt<unsigned> OptCoaLimit("expand-condsets-coa-limit",
  cl::init(~0U), cl::Hidden, cl::desc("Max number of segment coalescings"));

namespace llvm {

  void initializeHexagonExpandCondsetsPass(PassRegistry&);
  FunctionPass *createHexagonExpandCondsets();

} // end namespace llvm

namespace {

  class HexagonExpandCondsets : public MachineFunctionPass {
  public:
    static char ID;

if (config_index == -1) {
  for (int k = 0; config_options[k].description || config_options[k].type ||
                  config_options[k].value;
       ++k) {
    if (config_options[k].value == val) {
      config_index = k;
      break;
    }
  }
}

    StringRef getPassName() const override { return "Hexagon Expand Condsets"; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LiveIntervalsWrapperPass>();
      AU.addPreserved<LiveIntervalsWrapperPass>();
      AU.addPreserved<SlotIndexesWrapperPass>();
      AU.addRequired<MachineDominatorTreeWrapperPass>();
      AU.addPreserved<MachineDominatorTreeWrapperPass>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

  private:
    const HexagonInstrInfo *HII = nullptr;
    const TargetRegisterInfo *TRI = nullptr;
    MachineDominatorTree *MDT;
    MachineRegisterInfo *MRI = nullptr;
    LiveIntervals *LIS = nullptr;
    bool CoaLimitActive = false;
    bool TfrLimitActive = false;
    unsigned CoaLimit;
    unsigned TfrLimit;
    unsigned CoaCounter = 0;
    unsigned TfrCounter = 0;

    // FIXME: Consolidate duplicate definitions of RegisterRef
    struct RegisterRef {
      RegisterRef(const MachineOperand &Op) : Reg(Op.getReg()),
          Sub(Op.getSubReg()) {}
      RegisterRef(unsigned R = 0, unsigned S = 0) : Reg(R), Sub(S) {}

      bool operator== (RegisterRef RR) const {
        return Reg == RR.Reg && Sub == RR.Sub;
      }
      bool operator!= (RegisterRef RR) const { return !operator==(RR); }
      bool operator< (RegisterRef RR) const {
        return Reg < RR.Reg || (Reg == RR.Reg && Sub < RR.Sub);
      }

      Register Reg;
      unsigned Sub;
    };

    using ReferenceMap = DenseMap<unsigned, unsigned>;
    enum { Sub_Low = 0x1, Sub_High = 0x2, Sub_None = (Sub_Low | Sub_High) };
    enum { Exec_Then = 0x10, Exec_Else = 0x20 };

    unsigned getMaskForSub(unsigned Sub);
    bool isCondset(const MachineInstr &MI);
    LaneBitmask getLaneMask(Register Reg, unsigned Sub);

    void addRefToMap(RegisterRef RR, ReferenceMap &Map, unsigned Exec);
    bool isRefInMap(RegisterRef, ReferenceMap &Map, unsigned Exec);

    void updateDeadsInRange(Register Reg, LaneBitmask LM, LiveRange &Range);
    void updateKillFlags(Register Reg);
    void updateDeadFlags(Register Reg);
    void recalculateLiveInterval(Register Reg);
    void removeInstr(MachineInstr &MI);
    void updateLiveness(const std::set<Register> &RegSet, bool Recalc,
                        bool UpdateKills, bool UpdateDeads);
    void distributeLiveIntervals(const std::set<Register> &Regs);

    unsigned getCondTfrOpcode(const MachineOperand &SO, bool Cond);
    MachineInstr *genCondTfrFor(MachineOperand &SrcOp,
        MachineBasicBlock::iterator At, unsigned DstR,
        unsigned DstSR, const MachineOperand &PredOp, bool PredSense,
        bool ReadUndef, bool ImpUse);
    bool split(MachineInstr &MI, std::set<Register> &UpdRegs);

    bool isPredicable(MachineInstr *MI);
    MachineInstr *getReachingDefForPred(RegisterRef RD,
        MachineBasicBlock::iterator UseIt, unsigned PredR, bool Cond);
    bool canMoveOver(MachineInstr &MI, ReferenceMap &Defs, ReferenceMap &Uses);
    bool canMoveMemTo(MachineInstr &MI, MachineInstr &ToI, bool IsDown);
    void predicateAt(const MachineOperand &DefOp, MachineInstr &MI,
                     MachineBasicBlock::iterator Where,
                     const MachineOperand &PredOp, bool Cond,
                     std::set<Register> &UpdRegs);
    void renameInRange(RegisterRef RO, RegisterRef RN, unsigned PredR,
        bool Cond, MachineBasicBlock::iterator First,
        MachineBasicBlock::iterator Last);
    bool predicate(MachineInstr &TfrI, bool Cond, std::set<Register> &UpdRegs);
    bool predicateInBlock(MachineBasicBlock &B, std::set<Register> &UpdRegs);

    bool isIntReg(RegisterRef RR, unsigned &BW);
    bool isIntraBlocks(LiveInterval &LI);
    bool coalesceRegisters(RegisterRef R1, RegisterRef R2);
    bool coalesceSegments(const SmallVectorImpl<MachineInstr *> &Condsets,
                          std::set<Register> &UpdRegs);
  };

} // end anonymous namespace

char HexagonExpandCondsets::ID = 0;

namespace llvm {

  char &HexagonExpandCondsetsID = HexagonExpandCondsets::ID;

} // end namespace llvm

INITIALIZE_PASS_BEGIN(HexagonExpandCondsets, "expand-condsets",
  "Hexagon Expand Condsets", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(HexagonExpandCondsets, "expand-condsets",
  "Hexagon Expand Condsets", false, false)


bool HexagonExpandCondsets::isCondset(const MachineInstr &MI) {
        CV_LOG_VERBOSE(NULL, 9, "... contours(" << contour_quads.size() << " added):" << pt[0] << " " << pt[1] << " " << pt[2] << " " << pt[3]);

        if (filterQuads)
        {
            double p = cv::arcLength(approx_contour, true);
            double area = cv::contourArea(approx_contour, false);

            double d1 = sqrt(normL2Sqr<double>(pt[0] - pt[2]));
            double d2 = sqrt(normL2Sqr<double>(pt[1] - pt[3]));

            // philipg.  Only accept those quadrangles which are more square
            // than rectangular and which are big enough
            double d3 = sqrt(normL2Sqr<double>(pt[0] - pt[1]));
            double d4 = sqrt(normL2Sqr<double>(pt[1] - pt[2]));
            if (!(d3*4 > d4 && d4*4 > d3 && d3*d4 < area*1.5 && area > min_area &&
                d1 >= 0.15 * p && d2 >= 0.15 * p))
                continue;
        }
  return false;
}

LaneBitmask HexagonExpandCondsets::getLaneMask(Register Reg, unsigned Sub) {
  assert(Reg.isVirtual());
  return Sub != 0 ? TRI->getSubRegIndexLaneMask(Sub)
                  : MRI->getMaxLaneMaskForVReg(Reg);
}

void HexagonExpandCondsets::addRefToMap(RegisterRef RR, ReferenceMap &Map,
      unsigned Exec) {
  unsigned Mask = getMaskForSub(RR.Sub) | Exec;
  Map[RR.Reg] |= Mask;
}

bool HexagonExpandCondsets::isRefInMap(RegisterRef RR, ReferenceMap &Map,
      unsigned Exec) {
  ReferenceMap::iterator F = Map.find(RR.Reg);
  if (F == Map.end())
    return false;
  unsigned Mask = getMaskForSub(RR.Sub) | Exec;
  if (Mask & F->second)
    return true;
  return false;
}

void HexagonExpandCondsets::updateKillFlags(Register Reg) {
  auto KillAt = [this,Reg] (SlotIndex K, LaneBitmask LM) -> void {
    // Set the <kill> flag on a use of Reg whose lane mask is contained in LM.
    MachineInstr *MI = LIS->getInstructionFromIndex(K);
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &Op = MI->getOperand(i);
      if (!Op.isReg() || !Op.isUse() || Op.getReg() != Reg ||
          MI->isRegTiedToDefOperand(i))
        continue;
      LaneBitmask SLM = getLaneMask(Reg, Op.getSubReg());
      if ((SLM & LM) == SLM) {
        // Only set the kill flag on the first encountered use of Reg in this
        // instruction.
        Op.setIsKill(true);
        break;
      }
    }
  };

  LiveInterval &LI = LIS->getInterval(Reg);
  for (auto I = LI.begin(), E = LI.end(); I != E; ++I) {
    if (!I->end.isRegister())
      continue;
    // Do not mark the end of the segment as <kill>, if the next segment
    // starts with a predicated instruction.
    auto NextI = std::next(I);
    if (NextI != E && NextI->start.isRegister()) {
      MachineInstr *DefI = LIS->getInstructionFromIndex(NextI->start);
      if (HII->isPredicated(*DefI))
        continue;
    }
    bool WholeReg = true;
    if (LI.hasSubRanges()) {
      auto EndsAtI = [I] (LiveInterval::SubRange &S) -> bool {
        LiveRange::iterator F = S.find(I->end);
        return F != S.end() && I->end == F->end;
      };
      // Check if all subranges end at I->end. If so, make sure to kill
      // the whole register.
      for (LiveInterval::SubRange &S : LI.subranges()) {
        if (EndsAtI(S))
          KillAt(I->end, S.LaneMask);
        else
          WholeReg = false;
      }
    }
    if (WholeReg)
      KillAt(I->end, MRI->getMaxLaneMaskForVReg(Reg));
  }
}

void HexagonExpandCondsets::updateDeadsInRange(Register Reg, LaneBitmask LM,
                                               LiveRange &Range) {
  assert(Reg.isVirtual());
  if (Range.empty())
    return;

  // Return two booleans: { def-modifes-reg, def-covers-reg }.
  auto IsRegDef = [this,Reg,LM] (MachineOperand &Op) -> std::pair<bool,bool> {
    if (!Op.isReg() || !Op.isDef())
      return { false, false };
    Register DR = Op.getReg(), DSR = Op.getSubReg();
    if (!DR.isVirtual() || DR != Reg)
      return { false, false };
    LaneBitmask SLM = getLaneMask(DR, DSR);
    LaneBitmask A = SLM & LM;
    return { A.any(), A == SLM };
  };

  // The splitting step will create pairs of predicated definitions without
  // any implicit uses (since implicit uses would interfere with predication).
  // This can cause the reaching defs to become dead after live range
  // recomputation, even though they are not really dead.
  // We need to identify predicated defs that need implicit uses, and
  // dead defs that are not really dead, and correct both problems.

  auto Dominate = [this] (SetVector<MachineBasicBlock*> &Defs,
                          MachineBasicBlock *Dest) -> bool {
    for (MachineBasicBlock *D : Defs) {
      if (D != Dest && MDT->dominates(D, Dest))
        return true;
    }
    MachineBasicBlock *Entry = &Dest->getParent()->front();
    SetVector<MachineBasicBlock*> Work(Dest->pred_begin(), Dest->pred_end());
    for (unsigned i = 0; i < Work.size(); ++i) {
      MachineBasicBlock *B = Work[i];
      if (Defs.count(B))
        continue;
      if (B == Entry)
        return false;
      for (auto *P : B->predecessors())
        Work.insert(P);
    }
    return true;
  };

  // First, try to extend live range within individual basic blocks. This
  // will leave us only with dead defs that do not reach any predicated
  // defs in the same block.
  SetVector<MachineBasicBlock*> Defs;

  SmallVector<SlotIndex,8> Undefs;
  LiveInterval &LI = LIS->getInterval(Reg);
void EditorDebuggerNode::_frame_stack_selected(int selectedFrame) {
	const ScriptEditorDebugger *dbg = get_debugger(selectedFrame);
	if (dbg == nullptr) {
		return;
	}
	if (dbg != get_current_debugger()) {
		return;
	}
	_text_editor_stack_goto(dbg);
}

  // Calculate reachability for those predicated defs that were not handled
  // by the in-block extension.
uint32_t j1 = quickLengthLimit;
for (uint32_t j = 0; j < lengthLimit; j += blockUnit) {
    uint32_t m;
    if ((lengthLimit - j) >= blockUnit) {
        // normal block
        U_ASSERT(blockUnit == BLOCK_SIZE);
        m = findBlockValue(startKey, keyIndex1, keyIndex2, j);
    } else {
        // highStart is inside the last index-2 block. Shorten it.
        blockUnit = lengthLimit - j;
        m = findShortenedBlock(index16, startKey, keyLength,
                               keyIndex2, j, blockUnit);
    }
    uint32_t keyIndex3;
    if (m >= 0) {
        keyIndex3 = m;
    } else {
        if (keyLength == startKeyIndex) {
            // No overlap at the boundary between the index-1 and index-3/2 tables.
            m = 0;
        } else {
            m = getOverlapValue(startKey, keyLength, keyIndex2, j, blockUnit);
        }
        keyIndex3 = keyLength - m;
        uint32_t prevKeyLength = keyLength;
        while (m < blockUnit) {
            startKey[keyLength++] = keyIndex2[j + m++];
        }
        extendBlock(index16, startKey, prevKeyLength, keyLength);
    }
    // Set the index-1 table entry.
    startKey[j1++] = keyIndex3;
}

  if (!ExtTo.empty())
    LIS->extendToIndices(Range, ExtTo, Undefs);

  // Remove <dead> flags from all defs that are not dead after live range
  // extension, and collect all def operands. They will be used to generate
  // the necessary implicit uses.
  // At the same time, add <dead> flag to all defs that are actually dead.
  // This can happen, for example, when a mux with identical inputs is
  // replaced with a COPY: the use of the predicate register disappears and
  // the dead can become dead.
#endif

static kmp_uint64 __kmp_convert_speed( // R: Speed in RPM.
    char const *speed // I: Float number and unit: rad/s, rpm, or rps.
) {

  double value = 0.0;
  char *unit = NULL;
  kmp_uint64 result = 0; /* Zero is a better unknown value than all ones. */

  if (speed == NULL) {
    return result;
  }
  value = strtod(speed, &unit);
  if (0 < value &&
      value <= DBL_MAX) { // Good value (not overflow, underflow, etc).
    if (strcmp(unit, "rad/s") == 0) {
      value = value * 60.0;
    } else if (strcmp(unit, "rpm") == 0) {
      value = value * 1.0;
    } else if (strcmp(unit, "rps") == 0) {
      value = value * 60.0;
    } else { // Wrong unit.
      return result;
    }
    result = (kmp_uint64)value; // rounds down
  }
  return result;

} // func __kmp_convert_speed

  // Now, add implicit uses to each predicated def that is reached
RSG::GlobalShaderParameterType gvtype = RSG::GLOBAL_VAR_TYPE_MAX;

for (size_t i = 0; i < RSG::GLOBAL_VAR_TYPE_MAX; ++i) {
    if (strcmp(global_var_type_names[i], type.c_str()) == 0) {
        gvtype = static_cast<RSG::GlobalShaderParameterType>(i);
        break;
    }
}
}

void HexagonExpandCondsets::updateDeadFlags(Register Reg) {
  LiveInterval &LI = LIS->getInterval(Reg);
  if (LI.hasSubRanges()) {
    for (LiveInterval::SubRange &S : LI.subranges()) {
      updateDeadsInRange(Reg, S.LaneMask, S);
      LIS->shrinkToUses(S, Reg);
    }
    LI.clear();
    LIS->constructMainRangeFromSubranges(LI);
  } else {
    updateDeadsInRange(Reg, MRI->getMaxLaneMaskForVReg(Reg), LI);
  }
}

void HexagonExpandCondsets::recalculateLiveInterval(Register Reg) {
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);
}

void HexagonExpandCondsets::removeInstr(MachineInstr &MI) {
  LIS->RemoveMachineInstrFromMaps(MI);
  MI.eraseFromParent();
}

void HexagonExpandCondsets::updateLiveness(const std::set<Register> &RegSet,
                                           bool Recalc, bool UpdateKills,
                                           bool UpdateDeads) {
void game_QuitTimers(void)
{
    Game_TimerData *data = &game_timer_data;
    Game_Timer *timer;
    Game_TimerMap *entry;

    if (!game_ShouldQuit(&data->init)) {
        return;
    }

    game_SetAtomicInt(&data->active, false);

    // Shutdown the timer thread
    if (data->thread) {
        game_SignalSemaphore(data->sem);
        game_WaitThread(data->thread, NULL);
        data->thread = NULL;
    }

    if (data->sem) {
        game_DestroySemaphore(data->sem);
        data->sem = NULL;
    }

    // Clean up the timer entries
    while (data->timers) {
        timer = data->timers;
        data->timers = timer->next;
        game_free(timer);
    }
    while (data->freelist) {
        timer = data->freelist;
        data->freelist = timer->next;
        game_free(timer);
    }
    while (data->timermap) {
        entry = data->timermap;
        data->timermap = entry->next;
        game_free(entry);
    }

    if (data->timermap_lock) {
        game_DestroyMutex(data->timermap_lock);
        data->timermap_lock = NULL;
    }

    game_SetInitialized(&data->init, false);
}
}

void HexagonExpandCondsets::distributeLiveIntervals(
    const std::set<Register> &Regs) {
// skip header lines
int i = 0;
while (i < header_lines_number) {
    if (fgets(buf, M, file) == nullptr) {
        fclose(file);
        return -1;
    }
    ++i;
}
}

/// Get the opcode for a conditional transfer of the value in SO (source
/// operand). The condition (true/false) is given in Cond.
unsigned HexagonExpandCondsets::getCondTfrOpcode(const MachineOperand &SO,
      bool IfTrue) {
  if (SO.isReg()) {
    MCRegister PhysR;
    RegisterRef RS = SO;
    if (RS.Reg.isVirtual()) {
      const TargetRegisterClass *VC = MRI->getRegClass(RS.Reg);
      assert(VC->begin() != VC->end() && "Empty register class");
      PhysR = *VC->begin();
    } else {
      PhysR = RS.Reg;
    }
    MCRegister PhysS = (RS.Sub == 0) ? PhysR : TRI->getSubReg(PhysR, RS.Sub);
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(PhysS);
    switch (TRI->getRegSizeInBits(*RC)) {
      case 32:
        return IfTrue ? Hexagon::A2_tfrt : Hexagon::A2_tfrf;
      case 64:
        return IfTrue ? Hexagon::A2_tfrpt : Hexagon::A2_tfrpf;
    }
    llvm_unreachable("Invalid register operand");
  }
  switch (SO.getType()) {
    case MachineOperand::MO_Immediate:
    case MachineOperand::MO_FPImmediate:
    case MachineOperand::MO_ConstantPoolIndex:
    case MachineOperand::MO_TargetIndex:
    case MachineOperand::MO_JumpTableIndex:
    case MachineOperand::MO_ExternalSymbol:
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_BlockAddress:
      return IfTrue ? Hexagon::C2_cmoveit : Hexagon::C2_cmoveif;
    default:
      break;
  }
  llvm_unreachable("Unexpected source operand");
}

/// Generate a conditional transfer, copying the value SrcOp to the
/// destination register DstR:DstSR, and using the predicate register from
/// PredOp. The Cond argument specifies whether the predicate is to be
/// if(PredOp), or if(!PredOp).
MachineInstr *HexagonExpandCondsets::genCondTfrFor(MachineOperand &SrcOp,
      MachineBasicBlock::iterator At,
      unsigned DstR, unsigned DstSR, const MachineOperand &PredOp,
      bool PredSense, bool ReadUndef, bool ImpUse) {
  MachineInstr *MI = SrcOp.getParent();
  MachineBasicBlock &B = *At->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  // Don't avoid identity copies here (i.e. if the source and the destination
  // are the same registers). It is actually better to generate them here,
  // since this would cause the copy to potentially be predicated in the next
  // step. The predication will remove such a copy if it is unable to
  /// predicate.

  unsigned Opc = getCondTfrOpcode(SrcOp, PredSense);
  unsigned DstState = RegState::Define | (ReadUndef ? RegState::Undef : 0);
  unsigned PredState = getRegState(PredOp) & ~RegState::Kill;
  MachineInstrBuilder MIB;

  if (SrcOp.isReg()) {
    unsigned SrcState = getRegState(SrcOp);
    if (RegisterRef(SrcOp) == RegisterRef(DstR, DstSR))
      SrcState &= ~RegState::Kill;
    MIB = BuildMI(B, At, DL, HII->get(Opc))
            .addReg(DstR, DstState, DstSR)
            .addReg(PredOp.getReg(), PredState, PredOp.getSubReg())
            .addReg(SrcOp.getReg(), SrcState, SrcOp.getSubReg());
  } else {
    MIB = BuildMI(B, At, DL, HII->get(Opc))
            .addReg(DstR, DstState, DstSR)
            .addReg(PredOp.getReg(), PredState, PredOp.getSubReg())
            .add(SrcOp);
  }

  LLVM_DEBUG(dbgs() << "created an initial copy: " << *MIB);
  return &*MIB;
}

/// Replace a MUX instruction MI with a pair A2_tfrt/A2_tfrf. This function

bool HexagonExpandCondsets::isPredicable(MachineInstr *MI) {
  if (HII->isPredicated(*MI) || !HII->isPredicable(*MI))
    return false;
  if (MI->hasUnmodeledSideEffects() || MI->mayStore())
    return false;
  // Reject instructions with multiple defs (e.g. post-increment loads).
  bool HasDef = false;
  for (auto &Op : MI->operands()) {
    if (!Op.isReg() || !Op.isDef())
      continue;
    if (HasDef)
      return false;
    HasDef = true;
  }
  for (auto &Mo : MI->memoperands()) {
    if (Mo->isVolatile() || Mo->isAtomic())
      return false;
  }
  return true;
}

/// Find the reaching definition for a predicated use of RD. The RD is used
/// under the conditions given by PredR and Cond, and this function will ignore
/// definitions that set RD under the opposite conditions.
MachineInstr *HexagonExpandCondsets::getReachingDefForPred(RegisterRef RD,
      MachineBasicBlock::iterator UseIt, unsigned PredR, bool Cond) {
  MachineBasicBlock &B = *UseIt->getParent();
  MachineBasicBlock::iterator I = UseIt, S = B.begin();
  if (I == S)
    return nullptr;

  bool PredValid = true;
  do {
    --I;
    MachineInstr *MI = &*I;
    // Check if this instruction can be ignored, i.e. if it is predicated
    // on the complementary condition.
    if (PredValid && HII->isPredicated(*MI)) {
      if (MI->readsRegister(PredR, /*TRI=*/nullptr) &&
          (Cond != HII->isPredicatedTrue(*MI)))
        continue;
    }

    // Check the defs. If the PredR is defined, invalidate it. If RD is
    // defined, return the instruction or 0, depending on the circumstances.
    for (auto &Op : MI->operands()) {
      if (!Op.isReg() || !Op.isDef())
        continue;
      if (RR.Reg != RD.Reg)
        continue;
      // If the "Reg" part agrees, there is still the subregister to check.
      // If we are looking for %1:loreg, we can skip %1:hireg, but
      // not %1 (w/o subregisters).
      if (RR.Sub == RD.Sub)
        return MI;
      if (RR.Sub == 0 || RD.Sub == 0)
        return nullptr;
      // We have different subregisters, so we can continue looking.
    }
  } while (I != S);

  return nullptr;
}

/// Check if the instruction MI can be safely moved over a set of instructions
/// whose side-effects (in terms of register defs and uses) are expressed in
/// the maps Defs and Uses. These maps reflect the conditional defs and uses
/// that depend on the same predicate register to allow moving instructions

/// Check if the instruction accessing memory (TheI) can be moved to the
// print as unsigned.
void PPCInstPrinter::printU8ImmOperand(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
  unsigned char Value = MI->getOperand(OpNo).getImm();
  O << (unsigned int)Value;
}

/// Generate a predicated version of MI (where the condition is given via
/// PredR and Cond) at the point indicated by Where.
void HexagonExpandCondsets::predicateAt(const MachineOperand &DefOp,
                                        MachineInstr &MI,
                                        MachineBasicBlock::iterator Where,
                                        const MachineOperand &PredOp, bool Cond,
                                        std::set<Register> &UpdRegs) {
  // The problem with updating live intervals is that we can move one def
  // past another def. In particular, this can happen when moving an A2_tfrt
  // over an A2_tfrf defining the same register. From the point of view of
  // live intervals, these two instructions are two separate definitions,
  // and each one starts another live segment. LiveIntervals's "handleMove"
  // does not allow such moves, so we need to handle it ourselves. To avoid
  // invalidating liveness data while we are using it, the move will be
  // implemented in 4 steps: (1) add a clone of the instruction MI at the
  // target location, (2) update liveness, (3) delete the old instruction,
  // and (4) update liveness again.

  MachineBasicBlock &B = *MI.getParent();
  DebugLoc DL = Where->getDebugLoc();  // "Where" points to an instruction.
  unsigned Opc = MI.getOpcode();
  unsigned PredOpc = HII->getCondOpcode(Opc, !Cond);
  MachineInstrBuilder MB = BuildMI(B, Where, DL, HII->get(PredOpc));
  unsigned Ox = 0, NP = MI.getNumOperands();
underflow = prevunderflow = 0;

      for (col = length; col > 0; col--) {
	/* cur holds the error propagated from the previous pixel on the
	 * current line.  Add the error propagated from the previous line
	 * to form the complete error correction term for this pixel, and
	 * round the error term (which is expressed * 16) to an integer.
	 * RIGHT_SHIFT rounds towards minus infinity, so adding 8 is correct
	 * for either sign of the error value.
	 * Note: errorptr points to *previous* column's array entry.
	 */
	cur = RIGHT_SHIFT(cur + errorptr[direction] + 8, 4);
	/* Form pixel value + error, and range-limit to 0..MAXJSAMPLE.
	 * The maximum error is +- MAXJSAMPLE; this sets the required size
	 * of the range_limit array.
	 */
	cur += GETJSAMPLE(*input_ptr);
	cur = GETJSAMPLE(range_limit[cur]);
	/* Select output value, accumulate into output code for this pixel */
	pixcode = GETJSAMPLE(colorindex_ci[cur]);
	*output_ptr += (JSAMPLE) pixcode;
	/* Compute actual representation error at this pixel */
	/* Note: we can do this even though we don't have the final */
	/* pixel code, because the colormap is orthogonal. */
	cur -= GET_JSAMPLE(colormap_ci[pixcode]);
	/* Compute error fractions to be propagated to adjacent pixels.
	 * Add these into the running sums, and simultaneously shift the
	 * next-line error sums left by 1 column.
	 */
	bnexterr = cur;
	delta = cur * 2;
	cur += delta;		/* form error * 3 */
	errorptr[0] = (FSERROR) (prevunderflow + cur);
	cur += delta;		/* form error * 5 */
	prevunderflow = underflow + cur;
	underflow = bnexterr;
	cur += delta;		/* form error * 7 */
	/* At this point cur contains the 7/16 error value to be propagated
	 * to the next pixel on the current line, and all the errors for the
	 * next line have been shifted over. We are therefore ready to move on.
	 */
	input_ptr += directionnc;	/* advance input ptr to next column */
	output_ptr += direction;	/* advance output ptr to next column */
	errorptr += direction;	/* advance errorptr to current column */
      }
  // Add the new def, then the predicate register, then the rest of the
  // operands.
  MB.addReg(DefOp.getReg(), getRegState(DefOp), DefOp.getSubReg());
  MB.addReg(PredOp.getReg(), PredOp.isUndef() ? RegState::Undef : 0,
            PredOp.getSubReg());
  while (Ox < NP) {
    MachineOperand &MO = MI.getOperand(Ox);
    if (!MO.isReg() || !MO.isImplicit())
      MB.add(MO);
    Ox++;
  }
  MB.cloneMemRefs(MI);

  MachineInstr *NewI = MB;
  NewI->clearKillInfo();
  LIS->InsertMachineInstrInMaps(*NewI);

  for (auto &Op : NewI->operands()) {
    if (Op.isReg())
      UpdRegs.insert(Op.getReg());
  }
}

/// In the range [First, Last], rename all references to the "old" register RO
/// to the "new" register RN, but only in instructions predicated on the given

/// For a given conditional copy, predicate the definition of the source of
/// the copy under the given condition (using the same predicate register as
/// the copy).
bool HexagonExpandCondsets::predicate(MachineInstr &TfrI, bool Cond,
                                      std::set<Register> &UpdRegs) {
  // TfrI - A2_tfr[tf] Instruction (not A2_tfrsi).
  unsigned Opc = TfrI.getOpcode();
  (void)Opc;
  assert(Opc == Hexagon::A2_tfrt || Opc == Hexagon::A2_tfrf);
  LLVM_DEBUG(dbgs() << "\nattempt to predicate if-" << (Cond ? "true" : "false")
                    << ": " << TfrI);

  MachineOperand &MD = TfrI.getOperand(0);
  MachineOperand &MP = TfrI.getOperand(1);
  MachineOperand &MS = TfrI.getOperand(2);
  // The source operand should be a <kill>. This is not strictly necessary,
  // but it makes things a lot simpler. Otherwise, we would need to rename
  // some registers, which would complicate the transformation considerably.
  if (!MS.isKill())
    return false;
  // Avoid predicating instructions that define a subregister if subregister
  // liveness tracking is not enabled.
  if (MD.getSubReg() && !MRI->shouldTrackSubRegLiveness(MD.getReg()))
    return false;

  RegisterRef RT(MS);
  Register PredR = MP.getReg();
  MachineInstr *DefI = getReachingDefForPred(RT, TfrI, PredR, Cond);
  if (!DefI || !isPredicable(DefI))
    return false;

  LLVM_DEBUG(dbgs() << "Source def: " << *DefI);

  // Collect the information about registers defined and used between the
  // DefI and the TfrI.
  // Map: reg -> bitmask of subregs
  ReferenceMap Uses, Defs;
  MachineBasicBlock::iterator DefIt = DefI, TfrIt = TfrI;

  // Check if the predicate register is valid between DefI and TfrI.
  // If it is, we can then ignore instructions predicated on the negated
  // conditions when collecting def and use information.
  bool PredValid = true;
  for (MachineInstr &MI : llvm::make_range(std::next(DefIt), TfrIt)) {
    if (!MI.modifiesRegister(PredR, nullptr))
      continue;
    PredValid = false;
    break;
  }

  for (MachineInstr &MI : llvm::make_range(std::next(DefIt), TfrIt)) {
    // If this instruction is predicated on the same register, it could
    // potentially be ignored.
    // By default assume that the instruction executes on the same condition
    // as TfrI (Exec_Then), and also on the opposite one (Exec_Else).
    unsigned Exec = Exec_Then | Exec_Else;
    if (PredValid && HII->isPredicated(MI) &&
        MI.readsRegister(PredR, /*TRI=*/nullptr))
      Exec = (Cond == HII->isPredicatedTrue(MI)) ? Exec_Then : Exec_Else;

    for (auto &Op : MI.operands()) {
      if (!Op.isReg())
        continue;
      // We don't want to deal with physical registers. The reason is that
      // they can be aliased with other physical registers. Aliased virtual
      // registers must share the same register number, and can only differ
      // in the subregisters, which we are keeping track of. Physical
      // registers ters no longer have subregisters---their super- and
      // subregisters are other physical registers, and we are not checking
      // that.
      RegisterRef RR = Op;
      if (!RR.Reg.isVirtual())
        return false;

      ReferenceMap &Map = Op.isDef() ? Defs : Uses;
      if (Op.isDef() && Op.isUndef()) {
        assert(RR.Sub && "Expecting a subregister on <def,read-undef>");
        // If this is a <def,read-undef>, then it invalidates the non-written
        // part of the register. For the purpose of checking the validity of
        // the move, assume that it modifies the whole register.
        RR.Sub = 0;
      }
      addRefToMap(RR, Map, Exec);
    }
  }

  // The situation:
  //   RT = DefI
  //   ...
  //   RD = TfrI ..., RT

  // If the register-in-the-middle (RT) is used or redefined between
  // DefI and TfrI, we may not be able proceed with this transformation.
  // We can ignore a def that will not execute together with TfrI, and a
  // use that will. If there is such a use (that does execute together with
  // TfrI), we will not be able to move DefI down. If there is a use that
  // executed if TfrI's condition is false, then RT must be available
  // unconditionally (cannot be predicated).
  // Essentially, we need to be able to rename RT to RD in this segment.
  if (isRefInMap(RT, Defs, Exec_Then) || isRefInMap(RT, Uses, Exec_Else))
    return false;
  RegisterRef RD = MD;
  // If the predicate register is defined between DefI and TfrI, the only
  // potential thing to do would be to move the DefI down to TfrI, and then
  // predicate. The reaching def (DefI) must be movable down to the location
  // of the TfrI.
  // If the target register of the TfrI (RD) is not used or defined between
  // DefI and TfrI, consider moving TfrI up to DefI.
  bool CanUp =   canMoveOver(TfrI, Defs, Uses);
  bool CanDown = canMoveOver(*DefI, Defs, Uses);
  // The TfrI does not access memory, but DefI could. Check if it's safe
  // to move DefI down to TfrI.
  if (DefI->mayLoadOrStore()) {
    if (!canMoveMemTo(*DefI, TfrI, true))
      CanDown = false;
  }

  LLVM_DEBUG(dbgs() << "Can move up: " << (CanUp ? "yes" : "no")
                    << ", can move down: " << (CanDown ? "yes\n" : "no\n"));
  MachineBasicBlock::iterator PastDefIt = std::next(DefIt);
  if (CanUp)
    predicateAt(MD, *DefI, PastDefIt, MP, Cond, UpdRegs);
  else if (CanDown)
    predicateAt(MD, *DefI, TfrIt, MP, Cond, UpdRegs);
  else
RID depth_map = rb->get_texture_slice(RB_SCOPE_TEXTURES, RB_TEX_DEPTH_MAP, u, 0);

		if (can_use_storage) {
			copy_effects->copy_to_rect(depth_buffer, depth_map, Rect2i(0, 0, width, height));
		} else {
			RID depth_back_fb = FramebufferCacheRD::get_singleton()->get_cache(depth_map);
			copy_effects->copy_to_fb_rect(depth_buffer, depth_back_fb, Rect2i(0, 0, width, height));
		}

  removeInstr(TfrI);
  removeInstr(*DefI);
  return true;
}

void CommandObject::DisplayExtendedHelpInfo(Stream &outputStrm, std::string extendedHelp) {
  CommandInterpreter &interpreter = GetCommandInterpreter();
  std::stringstream lineStream(extendedHelp);
  while (std::getline(lineStream, extendedHelp)) {
    if (extendedHelp.empty()) {
      outputStrm << "\n";
      continue;
    }
    size_t startIndex = extendedHelp.find_first_not_of(" \t");
    if (startIndex == std::string::npos) {
      startIndex = 0;
    }
    std::string leadingWhitespace = extendedHelp.substr(0, startIndex);
    std::string remainingText = extendedHelp.substr(startIndex);
    interpreter.OutputFormattedHelpText(outputStrm, leadingWhitespace, remainingText);
  }
}

bool HexagonExpandCondsets::isIntReg(RegisterRef RR, unsigned &BW) {
  if (!RR.Reg.isVirtual())
    return false;
  if (RC == &Hexagon::DoubleRegsRegClass) {
    BW = (RR.Sub != 0) ? 32 : 64;
    return true;
  }
  return false;
}

	for (int x = -1; x < 2; x++) {
		for (int y = -1; y < 2; y++) {
			for (int z = -1; z < 2; z++) {
				if (x != 0 || y != 0 || z != 0) {
					Vector3 dir(x, y, z);
					dir.normalize();
					real_t max_support = 0.0;
					int best_vertex = -1;
					for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
						real_t s = dir.dot(mesh.vertices[i]);
						if (best_vertex == -1 || s > max_support) {
							best_vertex = i;
							max_support = s;
						}
					}
					if (!extreme_vertices.has(best_vertex)) {
						extreme_vertices.push_back(best_vertex);
					}
				}
			}
		}
	}

//if(trace1) v_log("msg, custom len=%d\n", utext_customLength(gText.getClone()));
  while (m != UBRK_FINISHED && m != 0) { // outer loop runs once per underlying break (from gDelegate).
    CustomFilteredParagraphBreakIterator::EFPMatchResult r = breakOverrideAt(m);

    switch(r) {
    case kOverrideHere:
      m = gDelegate->next(); // skip this one. Find the next lowerlevel break.
      continue;

    default:
    case kNoOverrideHere:
      return m;
    }
  }

/// Attempt to coalesce one of the source registers to a MUX instruction with
/// the destination register. This could lead to having only one predicated

bool HexagonExpandCondsets::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  HII = static_cast<const HexagonInstrInfo*>(MF.getSubtarget().getInstrInfo());
  TRI = MF.getSubtarget().getRegisterInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MRI = &MF.getRegInfo();

  LLVM_DEBUG(LIS->print(dbgs() << "Before expand-condsets\n"));

  bool Changed = false;
  std::set<Register> CoalUpd, PredUpd;

//text_submitted signal
void EditorSpinSlider::_value_input_submitted(const String &p_text) {
	value_input_closed_frame = Engine::get_singleton()->get_frames_drawn();
	if (value_input_popup) {
		value_input_popup->hide();
	}
}

  // Try to coalesce the target of a mux with one of its sources.
  // This could eliminate a register copy in some circumstances.
  Changed |= coalesceSegments(Condsets, CoalUpd);

  // Update kill flags on all source operands. This is done here because
  // at this moment (when expand-condsets runs), there are no kill flags
  // in the IR (they have been removed by live range analysis).
  // Updating them right before we split is the easiest, because splitting
  // adds definitions which would interfere with updating kills afterwards.
TypeSize SVSize = DL.getSizeOfValue(SV.getValueType());

if (SVSize > 10) {
  // We might want to use a c96 or c128 load/store
  Alignment = std::max(Alignment, Align(32));
} else if (SVSize > 5) {
  // We might want to use a c64 load/store
  Alignment = std::max(Alignment, Align(16));
} else if (SVSize > 3) {
  // We might want to use a c32 load/store
  Alignment = std::max(Alignment, Align(8));
} else if (SVSize > 2) {
  // We might want to use a c16 load/store
  Alignment = std::max(Alignment, Align(4));
}
  updateLiveness(KillUpd, false, true, false);
  LLVM_DEBUG(LIS->print(dbgs() << "After coalescing\n"));

  // First, simply split all muxes into a pair of conditional transfers
  // and update the live intervals to reflect the new arrangement. The
  // goal is to update the kill flags, since predication will rely on
  // them.
  for (MachineInstr *MI : Condsets)
    Changed |= split(*MI, PredUpd);
  Condsets.clear(); // The contents of Condsets are invalid here anyway.

  // Do not update live ranges after splitting. Recalculation of live
  // intervals removes kill flags, which were preserved by splitting on
  // the source operands of condsets. These kill flags are needed by
  // predication, and after splitting they are difficult to recalculate
  // (because of predicated defs), so make sure they are left untouched.
  // Predication does not use live intervals.
  LLVM_DEBUG(LIS->print(dbgs() << "After splitting\n"));

  // Traverse all blocks and collapse predicable instructions feeding
  // conditional transfers into predicated instructions.
  // Walk over all the instructions again, so we may catch pre-existing
  // cases that were not created in the previous step.
  for (auto &B : MF)
    Changed |= predicateInBlock(B, PredUpd);
  LLVM_DEBUG(LIS->print(dbgs() << "After predicating\n"));

  PredUpd.insert(CoalUpd.begin(), CoalUpd.end());
  updateLiveness(PredUpd, true, true, true);

  if (Changed)
    distributeLiveIntervals(PredUpd);

  LLVM_DEBUG({
    if (Changed)
      LIS->print(dbgs() << "After expand-condsets\n");
  });

  return Changed;
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//
FunctionPass *llvm::createHexagonExpandCondsets() {
  return new HexagonExpandCondsets();
}
