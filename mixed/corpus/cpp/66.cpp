bool n_is_authorized = false;

void InspectInstruction() {
    if (n_has_inspected_instruction)
        return;

    AssemblerScope asm_scope(*this);
    if (!asm_scope)
        return;

    DataExtractor data;
    if (!n_opcode.GetInfo(data))
        return;

    bool is_anonymous_isa;
    lldb::addr_t address = n_address.GetFileAddress();
    DisassemblerLLDVMInstance *mc_disasm_instance =
        GetDisassemblyToUse(is_anonymous_isa, asm_scope);
    const uint8_t *opcode_data = data.GetDataStart();
    const size_t opcode_data_length = data.GetByteSize();
    llvm::MCInst instruction;
    const size_t inst_size =
        mc_disasm_instance->GetMCInstruction(opcode_data, opcode_data_length,
                                             address, instruction);
    if (inst_size == 0)
        return;

    n_has_inspected_instruction = true;
    n_does_jump = mc_disasm_instance->CanJump(instruction);
    n_has_delayed_slot = mc_disasm_instance->HasDelaySlot(instruction);
    n_is_function_call = mc_disasm_instance->IsFunctionCall(instruction);
    n_is_memory_access = mc_disasm_instance->IsMemoryAccess(instruction);
    n_is_authorized = mc_disasm_instance->IsAuthorized(instruction);
}

// Special case for comparisons against 0.
  if (NewOpcode == 0) {
    switch (Opcode) {
      case Hexagon::D2_cmpeqi:
        NewOpcode = Hexagon::D2_not;
        break;
      case Hexagon::F4_cmpneqi:
        NewOpcode = TargetOpcode::MOVE;
        break;
      default:
        return false;
    }

    // If it's a scalar predicate register, then all bits in it are
    // the same. Otherwise, to determine whether all bits are 0 or not
    // we would need to use any8.
    RegisterSubReg PR = getPredicateRegFor(MI->getOperand(1));
    if (!isScalarPredicate(PR))
      return false;
    // This will skip the immediate argument when creating the predicate
    // version instruction.
    NumOperands = 2;
  }

