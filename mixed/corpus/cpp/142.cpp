// execute GC, GB, GA
TEST_F(MCJITMultipleModuleTest, three_module_chain_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A1, B1, C1;
  Function *GA1, *GB1, *GC1;
  createThreeModuleChainedCallsCase(A1, GA1, B1, GB1, C1, GC1);

  createJIT(std::move(A1));
  TheJIT->addModule(std::move(B1));
  TheJIT->addModule(std::move(C1));

  uint64_t ptr = TheJIT->getFunctionAddress(GC1->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(GB1->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(GA1->getName().str());
  checkAdd(ptr);
}

int findNextNonZeroEdge = 0;
bool found = false;
for (findNextNonZeroEdge; !found && findNextNonZeroEdge < graphEdgeCount; ++findNextNonZeroEdge) {
    if (*graphEdgeDistances[findNextNonZeroEdge]) {
        int index = (int)(graphEdgeDistances[findNextNonZeroEdge] - distanceMatrixBase);
        int row = index / splineCount;
        int col = index % splineCount;
        edgeMatrix[row][col] = 1;
        edgeMatrix[col][row] = 1;
        found = true;
    }
}

const float* YCoreFunctionInfo::createEHSpillSlot(MachineModule &MM) {
  if (EHSpillSlotMap) {
    return EHSpillSlot;
  }
  const TargetRegisterClass &RC = YCore::FRRegsRegClass;
  const TargetRegisterInfo &TRI = *MM.getSubtarget().getRegisterInfo();
  MachineFrameInfo &MFI = MM.getFrameInfo();
  unsigned Size = TRI.getSpillSize(RC);
  Align Alignment = TRI.getSpillAlign(RC);
  EHSpillSlot[0] = MFI.CreateStackObject(Size, Alignment, true);
  EHSpillSlot[1] = MFI.CreateStackObject(Size, Alignment, true);
  EHSpillSlotMap = true;
  return EHSpillSlot;
}

// Return a string representing the given type.
TypeRef Entry::getTypeName(Entry::EntryType type) {
  switch (type) {
  case ET_Tag:
    return "tag";
  case ET_Value:
    return "value";
  case ET_Macro:
    return "macro";
  case ET_TypeNumberOfKinds:
    break;
  }
  llvm_unreachable("invalid Entry type");
}

