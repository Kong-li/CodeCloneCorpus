{
    for (int i = 0; i < MAX_CAMERAS; ++i)
    {
        std::string devicePath = "/dev/video" + std::to_string(i);
        int fileHandle = ::open(devicePath.c_str(), O_RDONLY);
        if (fileHandle != -1)
        {
            ::close(fileHandle);
            _index = i;
            break;
        }
    }
    if (_index < 0)
    {
        CV_LOG_WARNING(NULL, "VIDEOIO(V4L2): can't find camera device");
        name.clear();
        return false;
    }
}

  if (!isAIXBigArchive(Kind)) {
    if (ShouldWriteSymtab) {
      if (!HeadersSize)
        HeadersSize = computeHeadersSize(
            Kind, Data.size(), StringTableSize, NumSyms, SymNamesBuf.size(),
            isCOFFArchive(Kind) ? &SymMap : nullptr);
      writeSymbolTable(Out, Kind, Deterministic, Data, SymNamesBuf,
                       *HeadersSize, NumSyms);

      if (isCOFFArchive(Kind))
        writeSymbolMap(Out, Kind, Deterministic, Data, SymMap, *HeadersSize);
    }

    if (StringTableSize)
      Out << StringTableMember.Header << StringTableMember.Data
          << StringTableMember.Padding;

    if (ShouldWriteSymtab && SymMap.ECMap.size())
      writeECSymbols(Out, Kind, Deterministic, Data, SymMap);

    for (const MemberData &M : Data)
      Out << M.Header << M.Data << M.Padding;
  } else {
    HeadersSize = sizeof(object::BigArchive::FixLenHdr);
    LastMemberEndOffset += *HeadersSize;
    LastMemberHeaderOffset += *HeadersSize;

    // For the big archive (AIX) format, compute a table of member names and
    // offsets, used in the member table.
    uint64_t MemberTableNameStrTblSize = 0;
    std::vector<size_t> MemberOffsets;
    std::vector<StringRef> MemberNames;
    // Loop across object to find offset and names.
    uint64_t MemberEndOffset = sizeof(object::BigArchive::FixLenHdr);
    for (size_t I = 0, Size = NewMembers.size(); I != Size; ++I) {
      const NewArchiveMember &Member = NewMembers[I];
      MemberTableNameStrTblSize += Member.MemberName.size() + 1;
      MemberEndOffset += Data[I].PreHeadPadSize;
      MemberOffsets.push_back(MemberEndOffset);
      MemberNames.push_back(Member.MemberName);
      // File member name ended with "`\n". The length is included in
      // BigArMemHdrType.
      MemberEndOffset += sizeof(object::BigArMemHdrType) +
                         alignTo(Data[I].Data.size(), 2) +
                         alignTo(Member.MemberName.size(), 2);
    }

    // AIX member table size.
    uint64_t MemberTableSize = 20 + // Number of members field
                               20 * MemberOffsets.size() +
                               MemberTableNameStrTblSize;

    SmallString<0> SymNamesBuf32;
    SmallString<0> SymNamesBuf64;
    raw_svector_ostream SymNames32(SymNamesBuf32);
    raw_svector_ostream SymNames64(SymNamesBuf64);

    if (ShouldWriteSymtab && NumSyms)
      // Generate the symbol names for the members.
      for (const auto &M : Data) {
        Expected<std::vector<unsigned>> SymbolsOrErr = getSymbols(
            M.SymFile.get(), 0,
            is64BitSymbolicFile(M.SymFile.get()) ? SymNames64 : SymNames32,
            nullptr);
        if (!SymbolsOrErr)
          return SymbolsOrErr.takeError();
      }

    uint64_t MemberTableEndOffset =
        LastMemberEndOffset +
        alignTo(sizeof(object::BigArMemHdrType) + MemberTableSize, 2);

    // In AIX OS, The 'GlobSymOffset' field in the fixed-length header contains
    // the offset to the 32-bit global symbol table, and the 'GlobSym64Offset'
    // contains the offset to the 64-bit global symbol table.
    uint64_t GlobalSymbolOffset =
        (ShouldWriteSymtab &&
         (WriteSymtab != SymtabWritingMode::BigArchive64) && NumSyms32 > 0)
            ? MemberTableEndOffset
            : 0;

    uint64_t GlobalSymbolOffset64 = 0;
    uint64_t NumSyms64 = NumSyms - NumSyms32;
    if (ShouldWriteSymtab && (WriteSymtab != SymtabWritingMode::BigArchive32) &&
        NumSyms64 > 0) {
      if (GlobalSymbolOffset == 0)
        GlobalSymbolOffset64 = MemberTableEndOffset;
      else
        // If there is a global symbol table for 32-bit members,
        // the 64-bit global symbol table is after the 32-bit one.
        GlobalSymbolOffset64 =
            GlobalSymbolOffset + sizeof(object::BigArMemHdrType) +
            (NumSyms32 + 1) * 8 + alignTo(SymNamesBuf32.size(), 2);
    }

    // Fixed Sized Header.
    printWithSpacePadding(Out, NewMembers.size() ? LastMemberEndOffset : 0,
                          20); // Offset to member table
    // If there are no file members in the archive, there will be no global
    // symbol table.
    printWithSpacePadding(Out, GlobalSymbolOffset, 20);
    printWithSpacePadding(Out, GlobalSymbolOffset64, 20);
    printWithSpacePadding(Out,
                          NewMembers.size()
                              ? sizeof(object::BigArchive::FixLenHdr) +
                                    Data[0].PreHeadPadSize
                              : 0,
                          20); // Offset to first archive member
    printWithSpacePadding(Out, NewMembers.size() ? LastMemberHeaderOffset : 0,
                          20); // Offset to last archive member
    printWithSpacePadding(
        Out, 0,
        20); // Offset to first member of free list - Not supported yet

    for (const MemberData &M : Data) {
      Out << std::string(M.PreHeadPadSize, '\0');
      Out << M.Header << M.Data;
      if (M.Data.size() % 2)
        Out << '\0';
    }

    if (NewMembers.size()) {
      // Member table.
      printBigArchiveMemberHeader(Out, "", sys::toTimePoint(0), 0, 0, 0,
                                  MemberTableSize, LastMemberHeaderOffset,
                                  GlobalSymbolOffset ? GlobalSymbolOffset
                                                     : GlobalSymbolOffset64);
      printWithSpacePadding(Out, MemberOffsets.size(), 20); // Number of members
      for (uint64_t MemberOffset : MemberOffsets)
        printWithSpacePadding(Out, MemberOffset,
                              20); // Offset to member file header.
      for (StringRef MemberName : MemberNames)
        Out << MemberName << '\0'; // Member file name, null byte padding.

      if (MemberTableNameStrTblSize % 2)
        Out << '\0'; // Name table must be tail padded to an even number of
                     // bytes.

      if (ShouldWriteSymtab) {
        // Write global symbol table for 32-bit file members.
        if (GlobalSymbolOffset) {
          writeSymbolTable(Out, Kind, Deterministic, Data, SymNamesBuf32,
                           *HeadersSize, NumSyms32, LastMemberEndOffset,
                           GlobalSymbolOffset64);
          // Add padding between the symbol tables, if needed.
          if (GlobalSymbolOffset64 && (SymNamesBuf32.size() % 2))
            Out << '\0';
        }

        // Write global symbol table for 64-bit file members.
        if (GlobalSymbolOffset64)
          writeSymbolTable(Out, Kind, Deterministic, Data, SymNamesBuf64,
                           *HeadersSize, NumSyms64,
                           GlobalSymbolOffset ? GlobalSymbolOffset
                                              : LastMemberEndOffset,
                           0, true);
      }
    }
  }

