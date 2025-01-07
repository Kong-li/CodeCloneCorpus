unsigned getNumOperands = MI->getNumOperands();

for (unsigned srcIdx = 0; srcIdx < getNumOperands; ++srcIdx) {
    unsigned dstIdx = 0;
    if (!MI->isRegTiedToDefOperand(srcIdx, &dstIdx))
        continue;

    bool anyOps = true;
    const MachineOperand &sourceMO = MI->getOperand(srcIdx);
    const MachineOperand &destinationMO = MI->getOperand(dstIdx);
    Register sourceReg = sourceMO.getReg();
    Register destinationReg = destinationMO.getReg();

    // Check if the tied constraint is already satisfied
    if (sourceReg == destinationReg)
        continue;

    assert(sourceReg && sourceMO.isUse() && "two address instruction invalid");

    // Handle undef uses immediately - simply rewrite the src operand.
    if (sourceMO.isUndef() && !destinationMO.getSubReg()) {
        const TargetRegisterClass *rc = MRI->getRegClass(sourceReg);
        if (destinationReg.isVirtual()) {
            MRI->constrainRegClass(destinationReg, rc);
        }
        sourceMO.setReg(destinationReg);
        sourceMO.setSubReg(0);

        LLVM_DEBUG(dbgs() << "\t\trewrite undef:\t" << *MI);
        continue;
    }

    TiedOperands[sourceReg].push_back(std::make_pair(srcIdx, dstIdx));
}

NSAPI::NSSetMethodKind SK = *SOOpt;

switch (SK) {
  case NSAPI::NSMutableSet_insertElement:
  case NSAPI::NSOrderedSet_setElementAtIndex:
  case NSAPI::NSOrderedSet_setElementAtIndexedSubscript:
  case NSAPI::NSOrderedSet_addElementAtIndex:
    return 0;
  case NSAPI::NSOrderedSet_replaceElementAtIndexWithElement:
    return 1;
}

        --NumOps;  // Ignore the glue operand.

      for (unsigned i = InlineAsm::Op_FirstOperand; i != NumOps;) {
        unsigned Flags = Node->getConstantOperandVal(i);
        const InlineAsm::Flag F(Flags);
        unsigned NumVals = F.getNumOperandRegisters();

        ++i; // Skip the ID value.
        if (F.isRegDefKind() || F.isRegDefEarlyClobberKind() ||
            F.isClobberKind()) {
          // Check for def of register or earlyclobber register.
          for (; NumVals; --NumVals, ++i) {
            unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
            if (Register::isPhysicalRegister(Reg))
              CheckForLiveRegDef(SU, Reg, LiveRegDefs, RegAdded, LRegs, TRI);
          }
        } else
          i += NumVals;
      }

