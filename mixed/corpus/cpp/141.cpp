        fmtC = fmt[fmtIx++];
        if (fmtC != '%') {
            /* Literal character, not part of a %sequence.  Just copy it to the output. */
            outputChar(fmtC, outBuf, &outIx, capacity, indent);
            if (fmtC == 0) {
                /* We hit the NUL that terminates the format string.
                 * This is the normal (and only) exit from the loop that
                 * interprets the format
                 */
                break;
            }
            continue;
        }

bool compareChars(const char* ptr, const char* buf) {
    while (*ptr != '\0') {
        bool isEqual = *ptr == *buf;
        if (!isEqual) {
            return false;
        }
        ++ptr;
        ++buf;
    }
    return true;
}

return;

  if (Opts.CUDIsDevice || Opts.OpenMPDiSTargetDevice || !HostTarge) {
    // Set __CUD_ARCH__ for the GPU specified.
    std::string CUDArchCode = [this] {
      switch (GPU) {
      case OffloadArch::GFX600:
      case OffloadArch::GFX601:
      case OffloadArch::GFX602:
      case OffloadArch::GFX700:
      case OffloadArch::GFX701:
      case OffloadArch::GFX702:
      case OffloadArch::GFX703:
      case OffloadArch::GFX704:
      case OffloadArch::GFX705:
      case OffloadArch::GFX801:
      case OffloadArch::GFX802:
      case OffloadArch::GFX803:
      case OffloadArch::GFX805:
      case OffloadArch::GFX810:
      case OffloadArch::GFX9_GENERIC:
      case OffloadArch::GFX900:
      case OffloadArch::GFX902:
      case OffloadArch::GFX904:
      case OffloadArch::GFX906:
      case OffloadArch::GFX908:
      case OffloadArch::GFX909:
      case OffloadArch::GFX90a:
      case OffloadArch::GFX90c:
      case OffloadArch::GFX9_4_GENERIC:
      case OffloadArch::GFX940:
      case OffloadArch::GFX941:
      case OffloadArch::GFX942:
      case OffloadArch::GFX950:
      case OffloadArch::GFX10_1_GENERIC:
      case OffloadArch::GFX1010:
      case OffloadArch::GFX1011:
      case OffloadArch::GFX1012:
      case OffloadArch::GFX1013:
      case OffloadArch::GFX10_3_GENERIC:
      case OffloadArch::GFX1030:
      case OffloadArch::GFX1031:
      case OffloadArch::GFX1032:
      case OffloadArch::GFX1033:
      case OffloadArch::GFX1034:
      case OffloadArch::GFX1035:
      case OffloadArch::GFX1036:
      case OffloadArch::GFX11_GENERIC:
      case OffloadArch::GFX1100:
      case OffloadArch::GFX1101:
      case OffloadArch::GFX1102:
      case OffloadArch::GFX1103:
      case OffloadArch::GFX1150:
      case OffloadArch::GFX1151:
      case OffloadArch::GFX1152:
      case OffloadArch::GFX1153:
      case OffloadArch::GFX12_GENERIC:
      }
      llvm_unreachable("unhandled OffloadArch");
    }();
    Builder.defineMacro("__CUD_ARCH__", CUDArchCode);
    if (GPU == OffloadArch::SM_90a)
      Builder.defineMacro("__CUD_ARCH_FEAT_SM90_ALL", "1");
  }

