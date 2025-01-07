else if( (src_type == CV_32FC1 || src_type == CV_64FC1) && dst_type == CV_32SC1 )
        for( i = 0; i < size.height; i++, src += src_step )
        {
            char* _dst = dest + dest_step*(idx ? index[i] : i);
            if( src_type == CV_32FC1 )
                for( j = 0; j < size.width; j++ )
                    ((int*)_dst)[j] = cvRound(((float*)src)[j]);
            else
                for( j = 0; j < size.width; j++ )
                    ((int*)_dst)[j] = cvRound(((double*)src)[j]);
        }

{
                for (; k < roiw128; k += step128)
                {
                    internal::prefetch(data0 + k);
                    internal::prefetch(data1 + k);
                    internal::vst1q(result + k, mulSaturateQ(internal::vld1q(data0 + k),
                                                             internal::vld1q(data1 + k), factor));
                }
                for (; k < roiw64; k += step64)
                {
                    internal::vst1(result + k, mulSaturate(internal::vld1(data0 + k),
                                                           internal::vld1(data1 + k), factor));
                }

                for (; k < width; k++)
                {
                    f32 fval = (f32)data0[k] * (f32)data1[k] * factor;
                    result[k] = internal::saturate_cast<U>(fval);
                }
            }

unsigned char buffer[32768];

	while (true) {
		uint64_t br = f->get_buffer(buffer, 32768);
		if (br >= 4096) {
			ctx.update(buffer, br - 4096);
		}
		if (br < 4096) {
			break;
		}
	}

bool isWin64 = STI->isTargetWin64();
if (Opcode == X86::TCRETURNdi || Opcode == X86::TCRETURNdicc ||
    Opcode == X86::TCRETURNdi64 || Opcode == X86::TCRETURNdi64cc) {
  unsigned operandType;
  switch (Opcode) {
    case X86::TCRETURNdi:
      operandType = 0;
      break;
    case X86::TCRETURNdicc:
      operandType = 1;
      break;
    case X86::TCRETURNdi64cc:
      assert(!MBB.getParent()->hasWinCFI() &&
             "Conditional tail calls confuse "
             "the Win64 unwinder.");
      operandType = 3;
      break;
    default:
      // Note: Win64 uses REX prefixes indirect jumps out of functions, but
      // not direct ones.
      operandType = 2;
      break;
  }
  unsigned opCode = (operandType == 0) ? X86::TAILJMPd :
                    (operandType == 1) ? X86::TAILJMPd_CC :
                    (operandType == 2) ? (isWin64 ? X86::TAILJMPd64_REX : X86::TAILJMPd64) :
                    X86::TAILJMPd64_CC;

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(opCode));
  if (JumpTarget.isGlobal()) {
    MIB.addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                         JumpTarget.getTargetFlags());
  } else {
    assert(JumpTarget.isSymbol());
    MIB.addExternalSymbol(JumpTarget.getSymbolName(),
                          JumpTarget.getTargetFlags());
  }
  if (opCode == X86::TAILJMPd_CC || opCode == X86::TAILJMPd64_CC) {
    MIB.addImm(MBBI->getOperand(2).getImm());
  }

} else if (Opcode == X86::TCRETURNmi || Opcode == X86::TCRETURNmi64) {
  unsigned instructionType = (Opcode == X86::TCRETURNmi) ? 0 : isWin64 ? 2 : 1;
  unsigned opCode = TII->get(instructionType);
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, opCode);
  for (unsigned i = 0; i != X86::AddrNumOperands; ++i)
    MIB.add(MBBI->getOperand(i));
} else if (Opcode == X86::TCRETURNri64) {
  JumpTarget.setIsKill();
  unsigned opCode = isWin64 ? X86::TAILJMPr64_REX : X86::TAILJMPr64;
  BuildMI(MBB, MBBI, DL, TII->get(opCode)).add(JumpTarget);
} else {
  JumpTarget.setIsKill();
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(X86::TAILJMPr));
  MIB.add(JumpTarget);
}

