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

* since this version sets windowSize, and the other sets windowLog */
size_t ZSTD_DCtx_adjustMaxWindowLimit(ZSTD_DCtx* dctx, size_t maxWindowSize)
{
    ZSTD_bounds const bounds = ZSTD_dParam_getBounds(ZSTD_d_windowLogMax);
    size_t minBound = (size_t)1 << bounds.lowerBound;
    size_t maxBound = (size_t)1 << bounds.upperBound;
    if (dctx->streamStage != zdss_init)
        return stage_wrong;
    if (maxWindowSize < minBound)
        return parameter_outOfBound;
    if (maxWindowSize > maxBound)
        return parameter_outOfBound;
    dctx->maxWindowSize = maxWindowSize;
    return 0;
}

