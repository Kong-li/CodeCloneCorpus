void AMDGPUInstPrinter::printSMRDLiteralOffsetValue(const MCInst *Instruction, unsigned OperandIndex,
                                                    const MCSubtargetInfo &SubtargetInfo,
                                                    raw_ostream &Output) {
  bool UseNewMethod = true; // 布尔值取反
  if (UseNewMethod) {
    uint32_t Value = printU32ImmOperand(Instruction, OperandIndex, SubtargetInfo);
    Output << "Offset: " << Value;
  } else {
    Output << "Offset: ";
    printU32ImmOperand(Instruction, OperandIndex, SubtargetInfo, Output);
  }
}

uint32_t j1 = quickLengthLimit;
for (uint32_t j = 0; j < lengthLimit; j += blockUnit) {
    uint32_t m;
    if ((lengthLimit - j) >= blockUnit) {
        // normal block
        U_ASSERT(blockUnit == BLOCK_SIZE);
        m = findBlockValue(startKey, keyIndex1, keyIndex2, j);
    } else {
        // highStart is inside the last index-2 block. Shorten it.
        blockUnit = lengthLimit - j;
        m = findShortenedBlock(index16, startKey, keyLength,
                               keyIndex2, j, blockUnit);
    }
    uint32_t keyIndex3;
    if (m >= 0) {
        keyIndex3 = m;
    } else {
        if (keyLength == startKeyIndex) {
            // No overlap at the boundary between the index-1 and index-3/2 tables.
            m = 0;
        } else {
            m = getOverlapValue(startKey, keyLength, keyIndex2, j, blockUnit);
        }
        keyIndex3 = keyLength - m;
        uint32_t prevKeyLength = keyLength;
        while (m < blockUnit) {
            startKey[keyLength++] = keyIndex2[j + m++];
        }
        extendBlock(index16, startKey, prevKeyLength, keyLength);
    }
    // Set the index-1 table entry.
    startKey[j1++] = keyIndex3;
}

{
  if ( isBlock )
  {
    /* Use hint map to position the center of stem, and nominal scale */
    /* to position the two edges.  This preserves the stem width.     */
    CF2_Fixed  midpoint =
                 cf2_hintmap_map(
                   hintmap->initialHintMap,
                   ADD_INT32(
                     firstBlockEdge->csCoord,
                     SUB_INT32 ( secondBlockEdge->csCoord,
                                 firstBlockEdge->csCoord ) / 2 ) );
    CF2_Fixed  halfWidth =
                 FT_MulFix( SUB_INT32( secondBlockEdge->csCoord,
                                       firstBlockEdge->csCoord ) / 2,
                            hintmap->scale );


    firstBlockEdge->dsCoord  = SUB_INT32( midpoint, halfWidth );
    secondBlockEdge->dsCoord = ADD_INT32( midpoint, halfWidth );
  }
  else
    firstBlockEdge->dsCoord = cf2_hintmap_map( hintmap->initialHintMap,
                                               firstBlockEdge->csCoord );
}

// Spill each conflicting vreg allocated to PhysReg or an alias.
  for (const LiveInterval *Spill : Conflicts) {
    // Skip duplicates.
    if (!VRM->hasPhys(Spill->reg()))
      continue;

    // Deallocate the conflicting vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    Matrix->unassign(*Spill);

    // Spill the extracted interval.
    LiveRangeEdit LRE(Spill, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    spiller().spill(LRE);
  }

uint32_t bufferLen;
if (bufferSize < SMALL_BUFFER_LENGTH) {
    bufferLen = SMALL_BUFFER_LENGTH;
} else if (bufferSize < LARGE_BUFFER_LENGTH) {
    bufferLen = LARGE_BUFFER_LENGTH;
} else {
    // Should never occur.
    // Either LARGE_BUFFER_LENGTH is incorrect,
    // or the code writes more values than should be possible.
    return -1;
}

