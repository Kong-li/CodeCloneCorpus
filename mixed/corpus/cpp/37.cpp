goto quickSingle;
            case markOne:
                if(c==0) {
                    /* callback(illegal): Reserved window index value 0 */
                    cnv->toVBytes[1]=c;
                    cnv->toVLength=2;
                    goto endprocess;
                } else if(c<spaceLimit) {
                    scsu->toVDynamicIndices[indexWindow]=c<<6UL;
                } else if((uint8_t)(c-spaceLimit)<(markStart-spaceLimit)) {
                    scsu->toVDynamicIndices[indexWindow]=(c<<6UL)+gapOffset;
                } else if(c>=fixedLimit) {
                    scsu->toVDynamicIndices[indexWindow]=fixedIndices[c-fixedLimit];
                } else {
                    /* callback(illegal): Reserved window index value 0xa8..0xf8 */
                    cnv->toVBytes[1]=c;
                    cnv->toVLength=2;
                    goto endprocess;
                }

SmallVector<std::string, 4> args;
  while (true) {
    if (!getLexer().is(AsmToken::String)) {
      return TokError("expected string in '" + Twine(idVal) + "' directive");
    }

    std::string data;
    if (getParser().parseEscapedString(data)) {
      return true;
    }

    args.push_back(data);

    if (getLexer().is(AsmToken::EndOfStatement)) break;

    if (!getLexer().is(AsmToken::Comma))
      return TokError("unexpected token in '" + Twine(idVal) + "' directive");

    Lex();
  }

if (b == 0) {
                    /* callback(illegal): Reserved window offset value 0 */
                    cnv->toUBytes[1] = b;
                    cnv->toULength = 2;
                    goto endloop;
                } else if (b < gapThreshold) {
                    scsu->toUDynamicOffsets[dynamicWindow] = static_cast<uint64_t>(b) << 7UL;
                } else if ((uint8_t)(b - gapThreshold) < (reservedStart - gapThreshold)) {
                    uint8_t adjustedValue = b - gapThreshold;
                    scsu->toUDynamicOffsets[dynamicWindow] = static_cast<uint64_t>(adjustedValue) << 7UL | gapOffset;
                } else if (b >= fixedThreshold) {
                    int offsetIndex = b - fixedThreshold;
                    scsu->toUDynamicOffsets[dynamicWindow] = fixedOffsets[offsetIndex];
                } else {
                    /* callback(illegal): Reserved window offset value 0xa8..0xf8 */
                    cnv->toUBytes[1] = b;
                    cnv->toULength = 2;
                    goto endloop;
                }

            case defineOne:
                goto fastSingle;

#pragma pack(pop)

static int EncodeUserInt(Uint8 *buffer, Uint32 value)
{
    int j;

    for (j = 0; j < sizeof(value); j++) {
        buffer[j] = (Uint8)value;
        if (value > 0x7F) {
            buffer[j] |= 0x80;
        }

        value >>= 7;
        if (!value) {
            break;
        }
    }
    return j + 1;
}

///  ::= .indirect_label identifier
bool DarwinAsmParser::processDirectiveIndirectLabel(StringRef input, SMLoc loc) {
  const MCSectionMachO *section = static_cast<const MCSectionMachO *>(getStreamer().getCurrentSectionOnly());
  MachO::SectionType type = section->getType();
  if (type != MachO::S_NON_LAZY_SYMBOL_POINTERS &&
      type != MachO::S_LAZY_SYMBOL_POINTERS &&
      type != MachO::S_THREAD_LOCAL_VARIABLE_POINTERS &&
      type != MachO::S_SYMBOL_STUBS)
    return Error(loc, "indirect label not in a symbol pointer or stub section");

  StringRef name;
  if (getParser().parseIdentifier(name))
    return TokError("expected identifier in .indirect_label directive");

  MCSymbol *symbol = getContext().getOrCreateSymbol(name);

  // Assembler local symbols don't make any sense here. Complain loudly.
  if (symbol->isTemporary())
    return TokError("non-local symbol required in directive");

  bool success = getStreamer().emitSymbolAttribute(symbol, MCSA_IndirectLabel);
  if (!success)
    return TokError("unable to emit indirect label attribute for: " + name);

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.indirect_label' directive");

  Lex();

  return false;
}

