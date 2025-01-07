using RDComputer = std::function<uint64_t(RS1, RS2, PC)>;

static void ProcessInstruction(RISCVEmulatorTester *tester, DecodeResult inst,
                               bool rs2Present, RDComputer expectedValue) {
  tester->WritePC(0x114514);
  uint32_t rd = DecodeRD(inst.inst);
  uint32_t rs1 = DecodeRS1(inst.inst);
  uint32_t rs2 = 0;

  uint64_t rs1Val = 0x19;
  uint64_t rs2Val = 0x81;

  if (rs1 != 0)
    tester->gpr.gpr[rs1] = rs1Val;

  bool hasRs2 = static_cast<bool>(rs2Present);
  if (hasRs2) {
    rs2 = DecodeRS2(inst.inst);
    if (rs2 != 0) {
      if (rs1 == rs2)
        rs2Val = rs1Val;
      tester->gpr.gpr[rs2] = rs2Val;
    }
  }

  ASSERT_TRUE(tester->Execute(inst, false));
  CheckRD(tester, rd, expectedValue(rs1Val, hasRs2 ? rs2Val : 0UL, static_cast<PC>(0x114514)));
}

if (probe->getInitialOffset() != Vector3(0.0, 0.0, 0.0)) {
		for (int j = 0; j < 3; ++j) {
			Vector3 offset = probe->getInitialOffset();
			lines.push_back(offset);
			offset[j] -= 0.25;
			lines.push_back(offset);

			offset[j] += 0.5;
			lines.push_back(offset);
		}
	}

int j, k;
for (k = 0; k < max_width / 16 * 2; ++k) {
    int i = k * 3 * 16;
    __m128i rgb_plane[6];
    __m128i zero = _mm_setzero_si128();

    RGB24PackedToPlanar_SSE2(rgb + i, rgb_plane);

    const __m128i r0 = _mm_unpacklo_epi8(rgb_plane[0], zero);
    const __m128i g0 = _mm_unpacklo_epi8(rgb_plane[2], zero);
    const __m128i b0 = _mm_unpacklo_epi8(rgb_plane[4], zero);
    ConvertRGBToY_SSE2(&r0, &g0, &b0, &rgb[i]);

    const __m128i r1 = _mm_unpackhi_epi8(rgb_plane[0], zero);
    const __m128i g1 = _mm_unpackhi_epi8(rgb_plane[2], zero);
    const __m128i b1 = _mm_unpackhi_epi8(rgb_plane[4], zero);
    ConvertRGBToY_SSE2(&r1, &g1, &b1, &rgb[i + 16]);

    for (j = 0; j < 2; ++j) {
        STORE_16(_mm_packus_epi16(rgb[i + 16 * j], rgb[i + 32 + 16 * j]), y + i);
    }
}

FT_Pos   tempSwap;
int      innerIndex, outerIndex;

for (outerIndex = 1; outerIndex < count; outerIndex++)
{
    for (innerIndex = outerIndex; innerIndex > 0; innerIndex--)
    {
        if (!(table[innerIndex] >= table[innerIndex - 1]))
            break;

        tempSwap         = table[innerIndex];
        table[innerIndex]     = table[innerIndex - 1];
        table[innerIndex - 1] = tempSwap;
    }
}

