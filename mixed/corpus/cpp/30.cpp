{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }
    if (data == NULL) {
        SDL_InvalidParamError("data");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
    }

    COMMAND_BUFFER_DEVICE->PushComputeUniformData(
        command_buffer,
        slot_index,
        data,
        length);
}

#ifdef FT_CONFIG_OPTION_USE_PNG

  static FT_Error
  sbitDecoderLoadPng( TT_SBitDecoder   decoder,
                      FT_Byte*         buffer,
                      FT_UInt          end,
                      int              horizontalPosition,
                      int              verticalPosition,
                      unsigned short   recursionDepth )
  {
    FT_Error       error = FT_Err_Ok;
    FT_ULong       pngLength;

    FT_UNUSED( recursionDepth );


    if ( static_cast<FT_Int>(end - buffer) < 4 )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_png: broken bitmap\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    pngLength = FT_NEXT_ULONG( buffer );
    if ( (FT_ULong)( end - buffer ) < pngLength )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_png: broken bitmap\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    bool loadSuccess = Load_SBit_Png( &decoder->face->root.glyph,
                                      horizontalPosition,
                                      verticalPosition,
                                      decoder->bit_depth,
                                      decoder->metrics,
                                      decoder->stream->memory,
                                      buffer,
                                      pngLength,
                                      false,
                                      false );

    if (loadSuccess)
    {
      FT_TRACE3(( "tt_sbit_decoder_load_png: loaded\n" ));
    }

  Exit:
    return error;
  }

MachineInstr *createBranchInstruction(MachineBasicBlock *MBB, Instruction *I, DebugLoc DL, BranchType BR_C, int offset, const CondInfo &Cond) {
  MachineInstr *MI = nullptr;

  if (BR_C == Xtensa::BEQ || BR_C == Xtensa::BNE || BR_C == Xtensa::BLT || BR_C == Xtensa::BLTU || BR_C == Xtensa::BGE || BR_C == Xtensa::BGEU) {
    MI = BuildMI(MBB, I, DL, get(BR_C))
             .addImm(offset)
             .addReg(Cond[1].getReg())
             .addReg(Cond[2].getReg());
  } else if (BR_C == Xtensa::BEQI || BR_C == Xtensa::BNEI || BR_C == Xtensa::BLTI || BR_C == Xtensa::BLTUI || BR_C == Xtensa::BGEI || BR_C == Xtensa::BGEUI) {
    MI = BuildMI(MBB, I, DL, get(BR_C))
             .addImm(offset)
             .addReg(Cond[1].getReg())
             .addImm(Cond[2].getImm());
  } else if (BR_C == Xtensa::BEQZ || BR_C == Xtensa::BNEZ || BR_C == Xtensa::BLTZ || BR_C == Xtensa::BGEZ) {
    MI = BuildMI(MBB, I, DL, get(BR_C)).addImm(offset).addReg(Cond[1].getReg());
  } else {
    llvm_unreachable("Invalid branch type!");
  }

  return MI;
}

{
    if (computePass == nullptr) {
        SDL_InvalidParamError("computePass");
        return;
    }

    const bool isDebugMode = COMPUTEPASS_DEVICE->debug_mode;

    if (!isDebugMode) {
        COMPUTEPASS_DEVICE->DispatchComputeIndirect(
            COMPUTEPASS_COMMAND_BUFFER,
            buffer,
            offset);
        return;
    }

    CHECK_COMPUTEPASS();
    CHECK_COMPUTE_PIPELINE_BOUND();

    COMPUTEPASS_DEVICE->DispatchComputeIndirect(
        COMPUTEPASS_COMMAND_BUFFER,
        buffer,
        offset);
}

