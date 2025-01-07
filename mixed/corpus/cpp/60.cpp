  std::vector<Instruction *> InstToDelete;
  for (auto &F : Program) {
    for (auto &BB : F) {
      for (auto &Inst : BB) {

        SimplifyQuery Q(DL, &Inst);
        if (Value *Simplified = simplifyInstruction(&Inst, Q)) {
          if (O.shouldKeep())
            continue;
          Inst.replaceAllUsesWith(Simplified);
          InstToDelete.push_back(&Inst);
        }
      }
    }
  }

    SmallVector<Register, 8> ArgVRegs;
    for (auto Arg : Info.OrigArgs) {
      assert(Arg.Regs.size() == 1 && "Call arg has multiple VRegs");
      Register ArgReg = Arg.Regs[0];
      ArgVRegs.push_back(ArgReg);
      SPIRVType *SpvType = GR->getSPIRVTypeForVReg(ArgReg);
      if (!SpvType) {
        Type *ArgTy = nullptr;
        if (auto *PtrArgTy = dyn_cast<PointerType>(Arg.Ty)) {
          // If Arg.Ty is an untyped pointer (i.e., ptr [addrspace(...)]) and we
          // don't have access to original value in LLVM IR or info about
          // deduced pointee type, then we should wait with setting the type for
          // the virtual register until pre-legalizer step when we access
          // @llvm.spv.assign.ptr.type.p...(...)'s info.
          if (Arg.OrigValue)
            if (Type *ElemTy = GR->findDeducedElementType(Arg.OrigValue))
              ArgTy =
                  TypedPointerType::get(ElemTy, PtrArgTy->getAddressSpace());
        } else {
          ArgTy = Arg.Ty;
        }
        if (ArgTy) {
          SpvType = GR->getOrCreateSPIRVType(ArgTy, MIRBuilder);
          GR->assignSPIRVTypeToVReg(SpvType, ArgReg, MF);
        }
      }
      if (!MRI->getRegClassOrNull(ArgReg)) {
        // Either we have SpvType created, or Arg.Ty is an untyped pointer and
        // we know its virtual register's class and type even if we don't know
        // pointee type.
        MRI->setRegClass(ArgReg, SpvType ? GR->getRegClass(SpvType)
                                         : &SPIRV::pIDRegClass);
        MRI->setType(
            ArgReg,
            SpvType ? GR->getRegType(SpvType)
                    : LLT::pointer(cast<PointerType>(Arg.Ty)->getAddressSpace(),
                                   GR->getPointerSize()));
      }
    }

