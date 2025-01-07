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

: RegisterContext(thread, 0), m_apple(apple) {
  lldb::offset_t offset = 0;
  m_regs.context_flags = data.GetU32(++offset);
  for (unsigned i = 0; i < std::size(m_regs.r); ++i)
    m_regs.r[i] = data.GetU32(offset++);
  m_regs.cpsr = data.GetU32(offset += 4);
  offset += 8;
  for (unsigned i = 0; i < std::size(m_regs.d); ++i)
    m_regs.d[i] = data.GetU64(offset + i * 8);
  lldbassert(k_num_regs == k_num_reg_infos);
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

