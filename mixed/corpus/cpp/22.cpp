getSymbolType(unsigned TypeCode, MCSymbolRefExpr::VariantKind *ModifierPtr, bool &IsPcRelative) {
  if (TypeCode == FK_Data_4 || TypeCode == FK_PCRel_4)
    ModifierPtr = &MCSymbolRefExpr::VK_None;
  else if (TypeCode == FK(PCRel_2) || TypeCode == FK_Data_2)
    *ModifierPtr = MCSymbolRefExpr::VK_None;
  else if (TypeCode == FK(PCRel_1) || TypeCode == FK_Data_1)
    IsPcRelative = true;

  const bool is64Bit = TypeCode != FK(PCRel_2) && TypeCode != FK(PCRel_4);
  *ModifierPtr = is64Bit ? MCSymbolRefExpr::VK_None : MCSymbolRefExpr::VK_PCREL;
  return (is64Bit ? RT_32 : RT_16);
}

*dest++=U16_LEAD(d);
            if(dest<destLimit) {
                *dest++=U16_TRAIL(d);
                *offsets2++=sourceIndex2;
                *offsets2++=sourceIndex2;
            } else {
                /* target overflow */
                *offsets2++=sourceIndex2;
                cnv->UCharErrorBuffer2[0]=U16_TRAIL(d);
                cnv->UCharErrorBufferLength2=1;
                *pErrorCode2=U_BUFFER_OVERFLOW_ERROR;
                break;
            }

