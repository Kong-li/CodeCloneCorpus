// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2024 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

#if !defined(ASTCENC_DECOMPRESS_ONLY)

/**
 * @brief Functions for angular-sum algorithm for weight alignment.
 *
 * This algorithm works as follows:
 * - we compute a complex number P as (cos s*i, sin s*i) for each weight,
 *   where i is the input value and s is a scaling factor based on the spacing between the weights.
 * - we then add together complex numbers for all the weights.
 * - we then compute the length and angle of the resulting sum.
 *
 * This should produce the following results:
 * - perfect alignment results in a vector whose length is equal to the sum of lengths of all inputs
 * - even distribution results in a vector of length 0.
 * - all samples identical results in perfect alignment for every scaling.
 *
 * For each scaling factor within a given set, we compute an alignment factor from 0 to 1. This
 * should then result in some scalings standing out as having particularly good alignment factors;
 * we can use this to produce a set of candidate scale/shift values for various quantization levels;
 * we should then actually try them and see what happens.
 */

#include "astcenc_internal.h"
#include "astcenc_vecmathlib.h"

#include <stdio.h>
#include <cassert>
#include <cstring>

static constexpr unsigned int ANGULAR_STEPS { 32 };

static_assert((ANGULAR_STEPS % ASTCENC_SIMD_WIDTH) == 0,
              "ANGULAR_STEPS must be multiple of ASTCENC_SIMD_WIDTH");

static_assert(ANGULAR_STEPS >= 32,
              "ANGULAR_STEPS must be at least max(steps_for_quant_level)");

// Store a reduced sin/cos table for 64 possible weight values; this causes
// slight quality loss compared to using sin() and cos() directly. Must be 2^N.
static constexpr unsigned int SINCOS_STEPS { 64 };

static const uint8_t steps_for_quant_level[12] {
	2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32
};

ASTCENC_ALIGNAS static float sin_table[SINCOS_STEPS][ANGULAR_STEPS];
ASTCENC_ALIGNAS static float cos_table[SINCOS_STEPS][ANGULAR_STEPS];

#if defined(ASTCENC_DIAGNOSTICS)
	static bool print_once { true };
#endif


/**
 * @brief Compute the angular alignment factors and offsets.
 *
 * @param      weight_count              The number of (decimated) weights.
 * @param      dec_weight_ideal_value    The ideal decimated unquantized weight values.
 * @param      max_angular_steps         The maximum number of steps to be tested.
 * @param[out] offsets                   The output angular offsets array.

/**
 * @brief For a given step size compute the lowest and highest weight.
 *
 * Compute the lowest and highest weight that results from quantizing using the given stepsize and
 * offset, and then compute the resulting error. The cut errors indicate the error that results from
 * forcing samples that should have had one weight value one step up or down.
 *
 * @param      weight_count              The number of (decimated) weights.
 * @param      dec_weight_ideal_value    The ideal decimated unquantized weight values.
 * @param      max_angular_steps         The maximum number of steps to be tested.
 * @param      max_quant_steps           The maximum quantization level to be tested.
 * @param      offsets                   The angular offsets array.
 * @param[out] lowest_weight             Per angular step, the lowest weight.
 * @param[out] weight_span               Per angular step, the span between lowest and highest weight.
 * @param[out] error                     Per angular step, the error.
 * @param[out] cut_low_weight_error      Per angular step, the low weight cut error.
 * @param[out] cut_high_weight_error     Per angular step, the high weight cut error.
d8 output = 0;
for(size_t l = 0; l < dimensions.length; ++l)
{
    const i32* data = internal::getLinePtr( base,  stride, l);
    size_t j = 0;
    for (; j < processSize4;)
    {
        size_t limit = std::min(dimensions.width, j + ABS32F_BLOCK_SIZE) - 4;
        float32x4_t t = vcvtq_f32_s32(vabsq_s32(vld1q_s32(data + j)));
        for (j += 4; j <= limit; j += 4 )
        {
            internal::prefetch(data + j);
            float32x4_t t1 = vcvtq_f32_s32(vabsq_s32(vld1q_s32(data + j)));
            t = vaddq_f32(t, t1);
        }

        f32 t2[4];
        vst1q_f32(t2, t);

        for (u32 k = 0; k < 4; k++)
            output += (d8)(t2[k]);
    }
    for ( ; j < dimensions.width; j++)
        output += (d8)(std::abs(data[j]));
}

/**
 * @brief The main function for the angular algorithm.
 *
 * @param      weight_count              The number of (decimated) weights.
 * @param      dec_weight_ideal_value    The ideal decimated unquantized weight values.
 * @param      max_quant_level           The maximum quantization level to be tested.
 * @param[out] low_value                 Per angular step, the lowest weight value.
 * @param[out] high_value                Per angular step, the highest weight value.
/// @return       True on success.
bool InlineSpiller::
foldMemoryOperand(ArrayRef<std::pair<MachineInstr *, unsigned>> Ops,
                  MachineInstr *LoadMI) {
  if (Ops.empty())
    return false;
  // Don't attempt folding in bundles.
  MachineInstr *MI = Ops.front().first;
  if (Ops.back().first != MI || MI->isBundled())
    return false;

  bool WasCopy = TII.isCopyInstr(*MI).has_value();
  Register ImpReg;

  // TII::foldMemoryOperand will do what we need here for statepoint
  // (fold load into use and remove corresponding def). We will replace
  // uses of removed def with loads (spillAroundUses).
  // For that to work we need to untie def and use to pass it through
  // foldMemoryOperand and signal foldPatchpoint that it is allowed to
  // fold them.
  bool UntieRegs = MI->getOpcode() == TargetOpcode::STATEPOINT;

  // Spill subregs if the target allows it.
  // We always want to spill subregs for stackmap/patchpoint pseudos.
  bool SpillSubRegs = TII.isSubregFoldable() ||
                      MI->getOpcode() == TargetOpcode::STATEPOINT ||
                      MI->getOpcode() == TargetOpcode::PATCHPOINT ||
                      MI->getOpcode() == TargetOpcode::STACKMAP;

  // TargetInstrInfo::foldMemoryOperand only expects explicit, non-tied
  // operands.
  SmallVector<unsigned, 8> FoldOps;
  for (const auto &OpPair : Ops) {
    unsigned Idx = OpPair.second;
    assert(MI == OpPair.first && "Instruction conflict during operand folding");
    MachineOperand &MO = MI->getOperand(Idx);

    // No point restoring an undef read, and we'll produce an invalid live
    // interval.
    // TODO: Is this really the correct way to handle undef tied uses?
    if (MO.isUse() && !MO.readsReg() && !MO.isTied())
      continue;

    if (MO.isImplicit()) {
      ImpReg = MO.getReg();
      continue;
    }

    if (!SpillSubRegs && MO.getSubReg())
      return false;
    // We cannot fold a load instruction into a def.
    if (LoadMI && MO.isDef())
      return false;
    // Tied use operands should not be passed to foldMemoryOperand.
    if (UntieRegs || !MI->isRegTiedToDefOperand(Idx))
      FoldOps.push_back(Idx);
  }

  // If we only have implicit uses, we won't be able to fold that.
  // Moreover, TargetInstrInfo::foldMemoryOperand will assert if we try!
  if (FoldOps.empty())
    return false;

  MachineInstrSpan MIS(MI, MI->getParent());

  SmallVector<std::pair<unsigned, unsigned> > TiedOps;
  if (UntieRegs)
    for (unsigned Idx : FoldOps) {
      MachineOperand &MO = MI->getOperand(Idx);
      if (!MO.isTied())
        continue;
      unsigned Tied = MI->findTiedOperandIdx(Idx);
      if (MO.isUse())
        TiedOps.emplace_back(Tied, Idx);
      else {
        assert(MO.isDef() && "Tied to not use and def?");
        TiedOps.emplace_back(Idx, Tied);
      }
      MI->untieRegOperand(Idx);
    }

  MachineInstr *FoldMI =
      LoadMI ? TII.foldMemoryOperand(*MI, FoldOps, *LoadMI, &LIS)
             : TII.foldMemoryOperand(*MI, FoldOps, StackSlot, &LIS, &VRM);
  if (!FoldMI) {
    // Re-tie operands.
    for (auto Tied : TiedOps)
      MI->tieOperands(Tied.first, Tied.second);
    return false;
  }

  // Remove LIS for any dead defs in the original MI not in FoldMI.
  for (MIBundleOperands MO(*MI); MO.isValid(); ++MO) {
    if (!MO->isReg())
      continue;
    Register Reg = MO->getReg();
    if (!Reg || Reg.isVirtual() || MRI.isReserved(Reg)) {
      continue;
    }
    // Skip non-Defs, including undef uses and internal reads.
    if (MO->isUse())
      continue;
    PhysRegInfo RI = AnalyzePhysRegInBundle(*FoldMI, Reg, &TRI);
    if (RI.FullyDefined)
      continue;
    // FoldMI does not define this physreg. Remove the LI segment.
    assert(MO->isDead() && "Cannot fold physreg def");
    SlotIndex Idx = LIS.getInstructionIndex(*MI).getRegSlot();
    LIS.removePhysRegDefAt(Reg.asMCReg(), Idx);
  }

  int FI;
  if (TII.isStoreToStackSlot(*MI, FI) &&
      HSpiller.rmFromMergeableSpills(*MI, FI))
    --NumSpills;
  LIS.ReplaceMachineInstrInMaps(*MI, *FoldMI);
  // Update the call site info.
  if (MI->isCandidateForCallSiteEntry())
    MI->getMF()->moveCallSiteInfo(MI, FoldMI);

  // If we've folded a store into an instruction labelled with debug-info,
  // record a substitution from the old operand to the memory operand. Handle
  // the simple common case where operand 0 is the one being folded, plus when
  // the destination operand is also a tied def. More values could be
  // substituted / preserved with more analysis.
  if (MI->peekDebugInstrNum() && Ops[0].second == 0) {
    // Helper lambda.
    auto MakeSubstitution = [this,FoldMI,MI,&Ops]() {
      // Substitute old operand zero to the new instructions memory operand.
      unsigned OldOperandNum = Ops[0].second;
      unsigned NewNum = FoldMI->getDebugInstrNum();
      unsigned OldNum = MI->getDebugInstrNum();
      MF.makeDebugValueSubstitution({OldNum, OldOperandNum},
                         {NewNum, MachineFunction::DebugOperandMemNumber});
    };

    const MachineOperand &Op0 = MI->getOperand(Ops[0].second);
    if (Ops.size() == 1 && Op0.isDef()) {
      MakeSubstitution();
    } else if (Ops.size() == 2 && Op0.isDef() && MI->getOperand(1).isTied() &&
               Op0.getReg() == MI->getOperand(1).getReg()) {
      MakeSubstitution();
    }
  } else if (MI->peekDebugInstrNum()) {
    // This is a debug-labelled instruction, but the operand being folded isn't
    // at operand zero. Most likely this means it's a load being folded in.
    // Substitute any register defs from operand zero up to the one being
    // folded -- past that point, we don't know what the new operand indexes
    // will be.
    MF.substituteDebugValuesForInst(*MI, *FoldMI, Ops[0].second);
  }

  MI->eraseFromParent();

  // Insert any new instructions other than FoldMI into the LIS maps.
  assert(!MIS.empty() && "Unexpected empty span of instructions!");
  for (MachineInstr &MI : MIS)
    if (&MI != FoldMI)
      LIS.InsertMachineInstrInMaps(MI);

  // TII.foldMemoryOperand may have left some implicit operands on the
  // instruction.  Strip them.
  if (ImpReg)
    for (unsigned i = FoldMI->getNumOperands(); i; --i) {
      MachineOperand &MO = FoldMI->getOperand(i - 1);
      if (!MO.isReg() || !MO.isImplicit())
        break;
      if (MO.getReg() == ImpReg)
        FoldMI->removeOperand(i - 1);
    }

  LLVM_DEBUG(dumpMachineInstrRangeWithSlotIndex(MIS.begin(), MIS.end(), LIS,
                                                "folded"));

  if (!WasCopy)
    ++NumFolded;
  else if (Ops.front().second == 0) {
    ++NumSpills;
    // If there is only 1 store instruction is required for spill, add it
    // to mergeable list. In X86 AMX, 2 intructions are required to store.
    // We disable the merge for this case.
    if (std::distance(MIS.begin(), MIS.end()) <= 1)
      HSpiller.addToMergeableSpills(*FoldMI, StackSlot, Original);
  } else
    ++NumReloads;
  return true;
}


		int bit5;
		switch (mode)
		{
		case 0:
		case 2:
			bit2 = (d0_intval >> 6) & 1;
			break;
		case 1:
		case 4:
			bit2 = (b0_intval >> 7) & 1;
			break;
		case 3:
			bit2 = (a_intval >> 9) & 1;
			break;
		case 5:
			bit2 = (c_intval >> 7) & 1;
			break;
		case 6:
		case 7:
			bit2 = (a_intval >> 11) & 1;
			break;
		}

#endif
