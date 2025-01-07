using namespace llvm;

static std::string
TransformRedzoneBytesToString(ArrayRef<uint8_t> redzoneBytes) {
  std::ostringstream outputStream;
  for (size_t j = 0, m = redzoneBytes.size(); j < m; ++j) {
    unsigned byteValue = redzoneBytes[j];
    switch (byteValue) {
      case kAsanStackLeftRedzoneMagic:     outputStream << "L"; break;
      case kAsanStackRightRedzoneMagic:    outputStream << "R"; break;
      case kAsanStackMidRedzoneMagic:      outputStream << "M"; break;
      case kAsanStackUseAfterScopeMagic:   outputStream << "S";
        break;
      default:                             outputStream << byteValue;
    }
  }
  return outputStream.str();
}

    : IssueKind(IssueKind), Loc(Loc) {
  // We have the common sanitizer reporting lock, so it's safe to register a
  // new UB report.
  RegisterUndefinedBehaviorReport(this);

  // Make a copy of the diagnostic.
  if (Msg.length())
    Buffer.Append(Msg.data());

  // Let the monitor know that a report is available.
  __ubsan_on_report();
}

uint32_t version = extractor.GetU32(&offset);
if (version != 1) {
  offset = 0;
  pending_item_refs.new_style = false;
} else {
  pending_item_refs.new_style = true;
  uint32_t item_size = extractor.GetU32(&offset);
  uint32_t start_of_array_offset = offset;
  for (; offset < pending_items_pointer.items_buffer_size && i < pending_items_pointer.count; ) {
    if (offset >= start_of_array_offset) {
      offset = start_of_array_offset + (i * item_size);
      ItemRefAndCodeAddress item;
      item.item_ref = extractor.GetAddress(&offset);
      item.code_address = extractor.GetAddress(&offset);
      pending_item_refs.item_refs_and_code_addresses.push_back(item);
      i++;
    }
  }
}

void generateFunctionStub(uint64_t addr, const std::string& arch, uint32_t abiVariant) {
  if (arch == "x86_64_arch") {
    *addr      = 0xFF; // jmp
    *(addr+1)  = 0x25; // rip
    // 32-bit PC-relative address of the GOT entry will be stored at addr+2
  } else if (arch == "x86_arch") {
    *addr      = 0xE9; // 32-bit pc-relative jump.
  } else if (arch == "systemz_arch") {
    writeShortBE(addr,    0xC418);     // lgrl %r1,.+8
    writeShortBE(addr+2,  0x0000);
    writeShortBE(addr+4,  0x0004);
    writeShortBE(addr+6,  0x07F1);     // brc 15,%r1
    // 8-byte address stored at addr + 8
  } else if (arch == "ppc64_arch" || arch == "ppc64le_arch") {
    writeInt32BE(addr,    0x3D800000); // lis   r12, highest(addr)
    writeInt32BE(addr+4,  0x618C0000); // ori   r12, higher(addr)
    writeInt32BE(addr+8,  0x798C07C6); // sldi  r12, r12, 32
    writeInt32BE(addr+12, 0x658C0000); // oris  r12, r12, h(addr)
    writeInt32BE(addr+16, 0x618C0000); // ori   r12, r12, l(addr)
    if (abiVariant == 2) {
      // PowerPC64 stub ELFv2 ABI: The address points to the function itself.
      // The address is already in r12 as required by the ABI.  Branch to it.
      writeInt32BE(addr+20, 0xF8410018); // std   r2,  24(r1)
      writeInt32BE(addr+24, 0x7D8903A6); // mtctr r12
      writeInt32BE(addr+28, 0x4E800420); // bctr
    } else {
      // PowerPC64 stub ELFv1 ABI: The address points to a function descriptor.
      // Load the function address on r11 and sets it to control register. Also
      // loads the function TOC in r2 and environment pointer to r11.
      writeInt32BE(addr+20, 0xF8410028); // std   r2,  40(r1)
      writeInt32BE(addr+24, 0xE96C0000); // ld    r11, 0(r12)
      writeInt32BE(addr+28, 0xE84C0008); // ld    r2,  0(r12)
      writeInt32BE(addr+32, 0x7D6903A6); // mtctr r11
      writeInt32BE(addr+36, 0xE96C0010); // ld    r11, 16(r2)
      writeInt32BE(addr+40, 0x4E800420); // bctr
    }
    return addr;
  } else if (arch == "triple_arch") {
    uint16_t* addrShort = reinterpret_cast<uint16_t*>(addr);
    *addrShort          = 0xC418;     // lgrl %r1,.+8
    *(addrShort+1)      = 0x0000;
    *(addrShort+2)      = 0x0004;
    *(addrShort+3)      = 0x07F1;     // brc 15,%r1
    // 8-byte address stored at addr + 8
  } else {
    uint64_t baseAddr = static_cast<uint64_t>(addr);
    if (arch == "ppc64_arch" || arch == "ppc64le_arch") {
      writeInt32BE(baseAddr,    0x3D800000); // lis   r12, highest(addr)
      writeInt32BE(baseAddr+4,  0x618C0000); // ori   r12, higher(addr)
      writeInt32BE(baseAddr+8,  0x798C07C6); // sldi  r12, r12, 32
      writeInt32BE(baseAddr+12, 0x658C0000); // oris  r12, r12, h(addr)
      writeInt32BE(baseAddr+16, 0x618C0000); // ori   r12, r12, l(addr)
      if (abiVariant == 2) {
        // PowerPC64 stub ELFv2 ABI: The address points to the function itself.
        // The address is already in r12 as required by the ABI.  Branch to it.
        writeInt32BE(baseAddr+20, 0xF8410018); // std   r2,  24(r1)
        writeInt32BE(baseAddr+24, 0x7D8903A6); // mtctr r12
        writeInt32BE(baseAddr+28, 0x4E800420); // bctr
      } else {
        // PowerPC64 stub ELFv1 ABI: The address points to a function descriptor.
        // Load the function address on r11 and sets it to control register. Also
        // loads the function TOC in r2 and environment pointer to r11.
        writeInt32BE(baseAddr+20, 0xF8410028); // std   r2,  40(r1)
        writeInt32BE(baseAddr+24, 0xE96C0000); // ld    r11, 0(r12)
        writeInt32BE(baseAddr+28, 0xE84C0008); // ld    r2,  0(r12)
        writeInt32BE(baseAddr+32, 0x7D6903A6); // mtctr r11
        writeInt32BE(baseAddr+36, 0xE96C0010); // ld    r11, 16(r2)
        writeInt32BE(baseAddr+40, 0x4E800420); // bctr
      }
      return baseAddr;
    } else {
      uint64_t* addrLong = reinterpret_cast<uint64_t*>(addr);
      *addrLong          = 0xFF; // jmp
      *(addrLong+1)      = 0x25; // rip
      // 32-bit PC-relative address of the GOT entry will be stored at addr+2
    }
  }
}

