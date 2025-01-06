////////////////////////////////////////////////////////////
//
// SFML - Simple and Fast Multimedia Library
// Copyright (C) 2007-2024 Laurent Gomila (laurent@sfml-dev.org)
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it freely,
// subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented;
//    you must not claim that you wrote the original software.
//    If you use this software in a product, an acknowledgment
//    in the product documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such,
//    and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
//
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
// Android specific: we define the ANativeActivity_onCreate
// entry point, handling all the native activity stuff, then
// we call the user defined (and portable) main function in
// an external thread so developers can keep a portable code
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <SFML/Config.hpp>

#include <SFML/System/Android/Activity.hpp>
#include <SFML/System/Err.hpp>
#include <SFML/System/Sleep.hpp>
#include <SFML/System/Time.hpp>

#include <android/native_activity.h>
#include <android/window.h>

#include <mutex>
#include <thread>

#include <cassert>
#include <cstring>

#define SF_GLAD_EGL_IMPLEMENTATION
#include <glad/egl.h>


extern int main(int argc, char* argv[]);

namespace
{
using namespace AMDGPU;

switch (Val) {
  // clang-format off
  case 102: return createRegOperand(FLAT_SCR_LO);
  case 103: return createRegOperand(FLAT_SCR_HI);
  case 104: return createRegOperand(XNACK_MASK_LO);
  case 105: return createRegOperand(XNACK_MASK_HI);
  case 106: return createRegOperand(VCC_LO);
  case 107: return createRegOperand(VCC_HI);
  case 108: return createRegOperand(TBA_LO);
  case 109: return createRegOperand(TBA_HI);
  case 110: return createRegOperand(TMA_LO);
  case 111: return createRegOperand(TMA_HI);
  case 124:
    bool isGfx11Plus = isGFX11Plus();
    return isGfx11Plus ? createRegOperand(SGPR_NULL) : createRegOperand(M0);
  case 125:
    bool isGfx11Plus = isGFX11Plus();
    return !isGfx11Plus ? createRegOperand(M0) : createRegOperand(SGPR_NULL);
  case 126:
  {
      auto operand = createRegOperand(EXEC_LO);
      if (Val == 126)
          return operand;
  }
  case 127: return createRegOperand(EXEC_HI);
  case 235: return createRegOperand(SRC_SHARED_BASE_LO);
  case 236: return createRegOperand(SRC_SHARED_LIMIT_LO);
  case 237: return createRegOperand(SRC_PRIVATE_BASE_LO);
  case 238: return createRegOperand(SRC_PRIVATE_LIMIT_LO);
  case 239: return createRegOperand(SRC_POPS_EXITING_WAVE_ID);
  case 251: return createRegOperand(SRC_VCCZ);
  case 252: return createRegOperand(SRC_EXECZ);
  case 253: return createRegOperand(SRC_SCC);
  case 254: return createRegOperand(LDS_DIRECT);
  default: break;
    // clang-format on
}


source_lly -= final_lly;

if (!!(target->width && target->rows))
{
  const int diff = final_llx;
  target_llx -= diff;
  int temp = final_lly;
  target_lly -= temp;
}

// Main call for processing a plane with a WebPSamplerRowFunc function:
void webpSamplerProcessPlane(const uint8_t* y, const int yStride,
                             const uint8_t* u, const uint8_t* v, const int uvStride,
                             uint8_t* dst, const int dstStride,
                             const int width, const int height, WebPSamplerRowFunc func) {
  for (int j = 0; j < height; ++j) {
    if (!(j & 1)) {
      u += uvStride;
      v += uvStride;
    }
    func(y, u, v, dst, width);
    y += y_stride;
    dst += dstStride;
  }
}

    BF.RawBranchCount = FBD->getNumExecutedBranches();
    if (BF.ProfileMatchRatio == 1.0f) {
      if (fetchProfileForOtherEntryPoints(BF)) {
        BF.ProfileMatchRatio = evaluateProfileData(BF, *FBD);
        BF.ExecutionCount = FBD->ExecutionCount;
        BF.RawBranchCount = FBD->getNumExecutedBranches();
      }
      return;
    }

/// emitted Exit Value Transfers, otherwise return false.
void FuncLocBasedLVD::removeExitValue(const MachineInstr &MI,
                                      OpenRangesSet &OpenRanges,
                                      VarLocMap &VarLocIDs,
                                      const VarLoc &ExitVL,
                                      InstToExitLocMap &ExitValTransfers,
                                      RegDefToInstMap &RegSetInstrs) {
  // Skip the DBG_VALUE which is the debug exit value itself.
  if (&MI == &ExitVL.MI)
    return;

  // If the parameter's location is not register location, we can not track
  // the exit value any more. It doesn't have the TransferInst which defines
  // register, so no Exit Value Transfers have been emitted already.
  if (!MI.getDebugOperand(0).isReg())
    return;

  // Try to get non-debug instruction responsible for the DBG_VALUE.
  const MachineInstr *TransferInst = nullptr;
  Register Reg = MI.getDebugOperand(0).getReg();
  if (Reg.isValid() && RegSetInstrs.contains(Reg))
    TransferInst = RegSetInstrs.find(Reg)->second;

  // Case of the parameter's DBG_VALUE at the end of exit MBB.
  if (!TransferInst && !LastNonDbgMI && MI.getParent()->isExitBlock())
    return;

  // If the debug expression from the DBG_VALUE is not empty, we can assume the
  // parameter's value has changed indicating that we should stop tracking its
  // exit value as well.
  if (MI.getDebugExpression()->getNumElements() == 0 && TransferInst) {
    // If the DBG_VALUE comes from a copy instruction that copies the exit
    // value, it means the parameter's value has not changed and we should be
    // able to use its exit value.
    // TODO: Try to keep tracking of an exit value if we encounter a propagated
    // DBG_VALUE describing the copy of the exit value. (Propagated exit value
    // does not indicate the parameter modification.)
    auto DestSrc = TII->isCopyLikeInstr(*TransferInst);
    if (DestSrc) {
      const MachineOperand *SrcRegOp, *DestRegOp;
      SrcRegOp = DestSrc->Source;
      DestRegOp = DestSrc->Destination;
      if (Reg == DestRegOp->getReg()) {
        for (uint64_t ID : OpenRanges.getExitValueBackupVarLocs()) {
          const VarLoc &VL = VarLocIDs[LocIndex::fromRawInteger(ID)];
          if (VL.isExitValueCopyBackupReg(Reg) &&
              // Exit Values should not be variadic.
              VL.MI.getDebugOperand(0).getReg() == SrcRegOp->getReg())
            return;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Deleting a DBG exit value because of: ";
             MI.print(dbgs(), /*IsStandalone*/ false,
                      /*SkipOpers*/ false, /*SkipDebugLoc*/ false,
                      /*AddNewLine*/ true, TII));
  cleanupExitValueTransfers(TransferInst, OpenRanges, VarLocIDs, ExitVL,
                            ExitValTransfers);
  OpenRanges.erase(ExitVL);
}

{
        bool flag = (row[start] != color);
        while (!flag)
        {
            counter.sum++;
            start++;
            if (start == counter_length)
                break;
        }
        for (; start < counter_length; ++start)
        {
            if ((counterPosition < counter_length) && (row[start] == 255 - color))
            {
                counter.pattern[counterPosition]++;
                counter.sum++;
            }
            else
            {
                counterPosition++;
            }
        }
    }


/// optimization may benefit some targets by improving cache locality.
void RescheduleDAGSDNodes::ClusterNeighboringStores(Node *Node) {
  SDValue Chain;
  unsigned NumOps = Node->getNumOperands();
  if (Node->getOperand(NumOps-1).getValueType() == MVT::Other)
    Chain = Node->getOperand(NumOps-1);
  if (!Chain)
    return;

  // Skip any store instruction that has a tied input. There may be an additional
  // dependency requiring a different order than by increasing offsets, and the
  // added glue may introduce a cycle.
  auto hasTiedInput = [this](const SDNode *N) {
    const MCInstrDesc &MCID = TII->get(N->getMachineOpcode());
    for (unsigned I = 0; I != MCID.getNumOperands(); ++I) {
      if (MCID.getOperandConstraint(I, MCOI::TIED_TO) != -1)
        return true;
    }

    return false;
  };

  // Look for other stores of the same chain. Find stores that are storing to
  // the same base pointer and different offsets.
  SmallPtrSet<SDNode*, 16> Visited;
  SmallVector<int64_t, 4> Offsets;
  DenseMap<long long, SDNode*> O2SMap;  // Map from offset to SDNode.
  bool Cluster = false;
  SDNode *Base = Node;

  if (hasTiedInput(Base))
    return;

  // This algorithm requires a reasonably low use count before finding a match
  // to avoid uselessly blowing up compile time in large blocks.
  unsigned UseCount = 0;
  for (SDNode::user_iterator I = Chain->user_begin(), E = Chain->user_end();
       I != E && UseCount < 100; ++I, ++UseCount) {
    if (I.getUse().getResNo() != Chain.getResNo())
      continue;

    SDNode *User = *I;
    if (User == Node || !Visited.insert(User).second)
      continue;
    int64_t Offset1, Offset2;
    if (!TII->areStoresFromSameBasePtr(Base, User, Offset1, Offset2) ||
        Offset1 == Offset2 ||
        hasTiedInput(User)) {
      // FIXME: Should be ok if they addresses are identical. But earlier
      // optimizations really should have eliminated one of the stores.
      continue;
    }
    if (O2SMap.insert(std::make_pair(Offset1, Base)).second)
      Offsets.push_back(Offset1);
    O2SMap.insert(std::make_pair(Offset2, User));
    Offsets.push_back(Offset2);
    if (Offset2 < Offset1)
      Base = User;
    Cluster = true;
    // Reset UseCount to allow more matches.
    UseCount = 0;
  }

  if (!Cluster)
    return;

  // Sort them in increasing order.
  llvm::sort(Offsets);

  // Check if the stores are close enough.
  SmallVector<SDNode*, 4> Stores;
  unsigned NumStores = 0;
  int64_t BaseOff = Offsets[0];
  SDNode *BaseStore = O2SMap[BaseOff];
  Stores.push_back(BaseStore);
  for (unsigned i = 1, e = Offsets.size(); i != e; ++i) {
    int64_t Offset = Offsets[i];
    SDNode *Store = O2SMap[Offset];
    if (!TII->shouldScheduleStoresNear(BaseStore, Store, BaseOff, Offset,NumStores))
      break; // Stop right here. Ignore stores that are further away.
    Stores.push_back(Store);
    ++NumStores;
  }

  if (NumStores == 0)
    return;

  // Cluster stores by adding MVT::Glue outputs and inputs. This also
  // ensure they are scheduled in order of increasing offsets.
  for (SDNode *Store : Stores) {
    SDValue InGlue = Chain;
    bool OutGlue = true;
    if (AddGlue(Store, InGlue, OutGlue, DAG)) {
      if (OutGlue)
        InGlue = Store->getOperand(Store->getNumOperands() - 1);

      ++StoresClustered;
    }
    else if (!OutGlue && InGlue.getNode())
      RemoveUnusedGlue(InGlue.getNode(), DAG);
  }
}



reinterpret_cast<DebugThreadInfo *>(GetThreadData());
if (!info) {
  if (SANITIZER_LINUX) {
    // On Linux, libc constructor is called _after_ debug_init, and cleans up
    // TSD. Try to figure out if this is still the main thread by the stack
    // address. We are not entirely sure that we have correct main thread
    // limits, so only do this magic on Linux, and only if the found thread
    // is the main thread.
    DebugThreadInfo *tinfo = GetThreadInfoByTidLocked(kMainThreadId);
    if (tinfo && ThreadStackContainsAddress(tinfo, &info)) {
      SetCurrentThread(tinfo->thread);
      return tinfo->thread;
    }
  }
  return nullptr;
}

/// taking an inert operand can be safely deleted.
static bool isInertARCValue(Value *V, SmallPtrSet<Value *, 1> &VisitedPhis) {
  V = V->stripPointerCasts();

  if (IsNullOrUndef(V))
    return true;

  // See if this is a global attribute annotated with an 'objc_arc_inert'.
  if (auto *GV = dyn_cast<GlobalVariable>(V))
    if (GV->hasAttribute("objc_arc_inert"))
      return true;

  if (auto PN = dyn_cast<PHINode>(V)) {
    // Ignore this phi if it has already been discovered.
    if (!VisitedPhis.insert(PN).second)
      return true;
    // Look through phis's operands.
    for (Value *Opnd : PN->incoming_values())
      if (!isInertARCValue(Opnd, VisitedPhis))
        return false;
    return true;
  }

  return false;
}


  // 15.5.2.7 -- dummy is POINTER
  if (dummyIsPointer) {
    if (actualIsPointer || dummy.intent == common::Intent::In) {
      if (scope) {
        semantics::CheckPointerAssignment(context, messages.at(), dummyName,
            dummy, actual, *scope,
            /*isAssumedRank=*/dummyIsAssumedRank);
      }
    } else if (!actualIsPointer) {
      messages.Say(
          "Actual argument associated with POINTER %s must also be POINTER unless INTENT(IN)"_err_en_US,
          dummyName);
    }
  }

MachineBasicBlock::iterator K(SafeAdd);
for (++K; &*K != JumpMI; ++K) {
  for (const MachineOperand &MO : K->operands()) {
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (MO.isDef() && MO.getReg() == StartReg)
      return;
    if (MO.isUse() && MO.getReg() == StartReg)
      return;
  }
}




        {
            if( line_type == 1 || line_type == 4 || shift == 0 )
            {
                p0.x = (p0.x + (XY_ONE>>1)) >> XY_SHIFT;
                p0.y = (p0.y + (XY_ONE>>1)) >> XY_SHIFT;
                p1.x = (p1.x + (XY_ONE>>1)) >> XY_SHIFT;
                p1.y = (p1.y + (XY_ONE>>1)) >> XY_SHIFT;
                Line( img, p0, p1, color, line_type );
            }
            else
                Line2( img, p0, p1, color );
        }

LLDB_PLUGIN_DEFINE(DynamicLoaderFreeBSDKernel)

void DynamicLoaderFreeBSDKernel::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInit);
}

bool fp_double_divisible_value(fp_double d, fp_small s) {
  fp_small mod = 0;

  if (fp_double_div_value(d, s, NULL, &mod) != FP_OK) {
    return false;
  }
  return mod == 0;
}



void updateLiveRanges(LiveRange& currentRange, LiveRange::Segment& segmentToMove, const Slot& newSlot) {
  if (currentRange.isEmpty()) {
    return;
  }

  if (!segmentToMove.getPrev().isEmpty()) {
    if (currentRange.getPrev() == segmentToMove) {
      currentRange.removeSegment(segmentToMove);
      LiveRange::Segment* nextSegment = &currentRange.getNext();
      *nextSegment = LiveRange::Segment(newSlot, newSlot.getDeadSlot(), nextSegment->getValNo());
      nextSegment->getValNo()->setDef(newSlot);
    } else {
      currentRange.removeSegment(segmentToMove);
      segmentToMove.setStart(newSlot);
      segmentToMove.getValNo()->setDef(newSlot);
    }
  }

  if (currentRange.getNext() == segmentToMove) {
    LiveRange::Segment* prevSegment = &currentRange.getPrev();
    *prevSegment = LiveRange::Segment(prevSegment->getStart(), newSlot, prevSegment->getValNo());
    prevSegment->getValNo()->setDef(newSlot);
  }

  if (segmentToMove.getNext() == currentRange) {
    segmentToMove.setEnd(currentRange.getStart().getDeadSlot());
    *currentRange.getPrev() = LiveRange::Segment(segmentToMove.getEnd(), newSlot, segmentToMove.getValNo());
    segmentToMove.getValNo()->setDef(newSlot);
  }

  if (segmentToMove.getNext() != currentRange) {
    segmentToMove.setEnd(currentRange.getStart().getDeadSlot());
    *currentRange.getPrev() = LiveRange::Segment(segmentToMove.getEnd(), newSlot, segmentToMove.getValNo());
    segmentToMove.getValNo()->setDef(newSlot);
  }

  if (segmentToMove.getPrev().isEmpty()) {
    currentRange.insertAfter(segmentToMove);
  }
}
} // namespace


namespace sf::priv
{
} // namespace sf::priv


