size_t k = 0u;

        while (k < blockHeight)
        {
            size_t chunkSize = std::min(blockHeight - k, maxChunkSize) + k;
            uint32x4_t w_sum = w_zero;
            uint32x4_t w_sqsum = w_zero;

            for ( ; k < chunkSize ; k += 8, dataPtr += 8)
            {
                internal::prefetch(dataPtr);
                uint8x8_t w_data0 = vld1_u8(dataPtr);

                uint16x8_t w_data = vmovl_u8(w_data0);
                uint16x4_t w_datalo = vget_low_u16(w_data), w_datahi = vget_high_u16(w_data);
                w_sum = vaddq_u32(w_sum, vaddl_u16(w_datalo, w_datahi));
                w_sqsum = vmlal_u16(w_sqsum, w_datalo, w_datalo);
                w_sqsum = vmlal_u16(w_sqsum, w_datahi, w_datahi);
            }

            u32 arsum[8];
            vst1q_u32(arsum, w_sum);
            vst1q_u32(arsum + 4, w_sqsum);

            resultA[0] += (f64)arsum[0];
            resultA[1 % numChannels] += (f64)arsum[1];
            resultA[2 % numChannels] += (f64)arsum[2];
            resultA[3 % numChannels] += (f64)arsum[3];
            resultB[0] += (f64)arsum[4];
            resultB[1 % numChannels] += (f64)arsum[5];
            resultB[2 % numChannels] += (f64)arsum[6];
            resultB[3 % numChannels] += (f64)arsum[7];
        }

/* -- see zlib.h -- */
void ZEXPORT gzclearerr(gzFile file) {
    gz_statep state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return;
    state = (gz_statep)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return;

    /* clear error and end-of-file */
    if (state->mode == GZ_READ) {
        state->eof = 0;
        state->past = 0;
    }
    gz_error(state, Z_OK, NULL);
}

if (module_sp) {
    if (context) {
        addr_t mod_load_addr = module_sp->GetLoadBaseAddress(context);

        if (mod_load_addr != LLDB_INVALID_ADDRESS) {
            // We have a valid file range, so we can return the file based address
            // by adding the file base address to our offset
            return mod_load_addr + m_offset;
        }
    }
} else if (ModuleWasDeletedPrivate()) {
    // Used to have a valid module but it got deleted so the offset doesn't
    // mean anything without the module
    return LLDB_INVALID_ADDRESS;
} else {
    // We don't have a module so the offset is the load address
    return m_offset;
}

#if 0
TEST_F(
    SortIncludesTest,
    CalculatesCorrectCursorPositionWhenNewLineReplacementsWithRegroupingAndCR) {
  Style.IncludeBlocks = Style.IBS_Regroup;
  FmtStyle.LineEnding = FormatStyle::LE_CR;
  Style.IncludeCategories = {
      {"^\"a\"", 2, 0, false}, {"^\"b\"", 1, 1, false}, {".*", 0, 2, false}};
  StringRef Code = "#include \"c\"\r"     // Start of line: 0
                   "#include \"b\"\r"     // Start of line: 5
                   "#include \"a\"\r"     // Start of line: 10
                   "\r"                   // Start of line: 15
                   "int i;";              // Start of line: 17
  StringRef Expected = "#include \"b\"\r" // Start of line: 5
                       "\r"               // Start of line: 10
                       "#include \"a\"\r" // Start of line: 12
                       "\r"               // Start of line: 17
                       "#include \"c\"\r" // Start of line: 18
                       "\r"               // Start of line: 23
                       "int i;";          // Start of line: 25
  verifyFormat(Expected, sort(Code));
  EXPECT_EQ(5u, newCursor(Code, 0));
  EXPECT_EQ(12u, newCursor(Code, 12));
  EXPECT_EQ(18u, newCursor(Code, 18));
  EXPECT_EQ(23u, newCursor(Code, 23));
  EXPECT_EQ(25u, newCursor(Code, 25));
}

