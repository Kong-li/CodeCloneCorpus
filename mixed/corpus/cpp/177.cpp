/* Apply an inverse intercomponent transform if necessary. */
    switch (tile->cp->mctid) {
    case JPC_MCT_RCT:
        assert(dec->numcomps == 4 || dec->numcomps == 3);
        jpc_irct(tile->tcomps[2].data, tile->tcomps[1].data,
          tile->tcomps[0].data);
        break;
    case JPC_MCT_ICT:
        assert(dec->numcomps == 4 || dec->numcomps == 3);
        jpc_iict(tile->tcomps[2].data, tile->tcomps[1].data,
          tile->tcomps[0].data);
        break;
    }

  int CurOffset = -8 - StackAdjust;
  for (auto CSReg : GPRCSRegs) {
    auto Offset = RegOffsets.find(CSReg.Reg);
    if (Offset == RegOffsets.end())
      continue;

    int RegOffset = Offset->second;
    if (RegOffset != CurOffset - 4) {
      DEBUG_WITH_TYPE("compact-unwind",
                      llvm::dbgs() << MRI.getName(CSReg.Reg) << " saved at "
                                   << RegOffset << " but only supported at "
                                   << CurOffset << "\n");
      return CU::UNWIND_ARM_MODE_DWARF;
    }
    CompactUnwindEncoding |= CSReg.Encoding;
    CurOffset -= 4;
  }

       61-bit precision until n=30.*/
    if(ipart>30){
      /*For these iterations, we just update the low bits, as the high bits
         can't possibly be affected.
        OC_ATANH_LOG2 has also converged (it actually did so one iteration
         earlier, but that's no reason for an extra special case).*/
      for(;;i++){
        mask=-(z<0);
        wlo+=(w>>i)+mask^mask;
        z-=OC_ATANH_LOG2[31]+mask^mask;
        /*Repeat iteration 40.*/
        if(i>=39)break;
        z<<=1;
      }
      for(;i<61;i++){
        mask=-(z<0);
        wlo+=(w>>i)+mask^mask;
        z=z-(OC_ATANH_LOG2[31]+mask^mask)<<1;
      }
    }

