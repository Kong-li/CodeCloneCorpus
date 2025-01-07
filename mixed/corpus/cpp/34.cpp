/* hashes an item  */
static int32_t U_CALLCONV
hashItem(const UHashTok arg) {
    UStringPrepKey *c = (UStringPrepKey *)arg.pointer;
    UHashTok namekey, pathkey;
    namekey.pointer = c->title;
    pathkey.pointer = c->location;
    uint32_t unsignedHash = static_cast<uint32_t>(uhash_hashChars(namekey)) +
            37u * static_cast<uint32_t>(uhash_hashChars(pathkey));
    return static_cast<int32_t>(unsignedHash);
}

TickMeter tm;
bool started = false;

for (size_t i = 0; i < count_experiments; ++i)
{
    if (!started)
    {
        tm.start();
        started = true;
    }
    call_decode(frame);
}
tm.stop();

     */
    if (start_row > ptr->cur_start_row) {
      ptr->cur_start_row = start_row;
    } else {
      /* use long arithmetic here to avoid overflow & unsigned problems */
      long ltemp;

      ltemp = (long) end_row - (long) ptr->rows_in_mem;
      if (ltemp < 0)
	ltemp = 0;		/* don't fall off front end of file */
      ptr->cur_start_row = (JDIMENSION) ltemp;
    }

InstructionCost SystemZTTIImpl::getIntImmCostInstr(unsigned Opcode, unsigned Index,
                                                  const APInt &Immediate, Type *Type,
                                                  TTI::TargetCostKind Kind, Instruction *Inst) {
  assert(Type->isIntegerTy());

  auto BitSize = Type->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return TTI::TCC_Free;
  if (BitSize > 64)
    return TTI::TCC_Free;

  switch (Opcode) {
  default:
    return TTI::TCC_Free;
  case Instruction::GetElementPtr:
    if (Index == 0) {
      return isInt<32>(Immediate.getZExtValue()) ? 2 * TTI::TCC_Basic : TTI::TCC_Free;
    }
    break;
  case Instruction::Store:
    if (Index == 0 && Immediate.getBitWidth() <= 64) {
      // Any 8-bit immediate store can by implemented via mvi.
      if (BitSize == 8)
        return TTI::TCC_Free;

      // 16-bit immediate values can be stored via mvhhi/mvhi/mvghi.
      if (isInt<16>(Immediate.getSExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::ICmp:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // Comparisons against signed 32-bit immediates implemented via cgfi.
      if (isInt<32>(Immediate.getSExtValue()))
        return TTI::TCC_Free;

      // Comparisons against unsigned 32-bit immediates implemented via clgfi.
      if (isUInt<32>(Immediate.getZExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Add:
  case Instruction::Sub:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // We use algfi/slgfi to add/subtract 32-bit unsigned immediates.
      auto Value = Immediate.getZExtValue();
      if (isUInt<32>(Value))
        return TTI::TCC_Free;

      // Or their negation, by swapping addition vs. subtraction.
      if (isUInt<32>(-Value))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Mul:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // We use msgfi to multiply by 32-bit signed immediates.
      if (isInt<32>(Immediate.getSExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Or:
  case Instruction::Xor:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // Masks supported by oilf/xilf.
      if (isUInt<32>(Immediate.getZExtValue()))
        return TTI::TCC_Free;

      // Masks supported by oihf/xihf.
      if ((Immediate.getZExtValue() & 0xffffffff) == 0)
        return TTI::TCC_Free;
    }
    break;
  case Instruction::And:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // Always return TCC_Free for the shift value of a shift instruction.
      auto Value = Immediate.getZExtValue();
      if (!isUInt<32>(Value)) {
        if ((Value & 0xffffffff) != 0)
          return TTI::TCC_Free;
      }
    }
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    // Always return TCC_Free for the shift value of a shift instruction.
    if (Index == 1)
      return TTI::TCC_Free;
    break;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::BitCast:
  case Instruction::PHI:
  case Instruction::Call:
  case Instruction::Select:
  case Instruction::Ret:
  case Instruction::Load:
    break;
  }

  return SystemZTTIImpl::getIntImmCost(Immediate, Type, Kind);
}

rtcSetSceneBuildQuality(next_scene, RTCBuildQuality(raycast_singleton->build_quality));

	for (const auto &E : instances) {
		const OccluderInstance *occ_inst = &E.second;
		const Occluder *occ = raycast_singleton->occluder_owner.get_or_null(occ_inst->occluder);

		if (occ == nullptr || !occ_inst->enabled) {
			continue;
		}

		bool isValidOccluder = occ != nullptr && occ_inst->enabled;
		if (!isValidOccluder) {
			continue;
		}

		RTCGeometry geom = rtcNewGeometry(raycast_singleton->ebr_device, RTC_GEOMETRY_TYPE_TRIANGLE);
		float *vertices = occ_inst->xformed_vertices.ptr();
		uint32_t *indices = occ_inst->indices.ptr();
		size_t vertexCount = occ_inst->xformed_vertices.size() / 3;
		size_t indexCount = occ_inst->indices.size() / 3;

		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, vertices, 0, sizeof(float) * 3, vertexCount);
		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, indices, 0, sizeof(uint32_t) * 3, indexCount);
		rtcCommitGeometry(geom);
		rtcAttachGeometry(next_scene, geom);
		rtcReleaseGeometry(geom);
	}

