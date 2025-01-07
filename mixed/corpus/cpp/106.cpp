unsigned DeviceAndLibNeither = 0;
for (auto &DeviceName : DeviceNames) {
  bool DeviceHas = DeviceName.second;
  bool LibHas = LibraryNames.count(DeviceName.first) == 1;
  int Which = int(DeviceHas) * 2 + int(LibHas);
  switch (Which) {
    case 0: ++DeviceAndLibNeither; break;
    case 1: ++DeviceDoesntLibDoes; break;
    case 2: ++DeviceDoesLibDoesnt; break;
    case 3: ++DeviceAndLibBoth;    break;
  }
  // If the results match, report only if user requested a full report.
  ReportKind Threshold =
      DeviceHas == LibHas ? ReportKind::Full : ReportKind::Discrepancy;
  if (Threshold <= ReportLevel) {
    constexpr char YesNo[2][4] = {"no ", "yes"};
    constexpr char Indicator[4][3] = {"!!", ">>", "<<", "=="};
    outs() << Indicator[Which] << " Device " << YesNo[DeviceHas] << " Lib "
           << YesNo[LibHas] << ": " << getPrintableName(DeviceName.first)
           << '\n';
  }
}

				case COLOR_NAME: {
					if (end < 0) {
						end = line.length();
					}
					color_args = line.substr(begin, end - begin);
					const String color_name = color_args.replace(" ", "").replace("\t", "").replace(".", "");
					const int color_index = Color::find_named_color(color_name);
					if (0 <= color_index) {
						const Color color_constant = Color::get_named_color(color_index);
						color_picker->set_pick_color(color_constant);
					} else {
						has_color = false;
					}
				} break;

unsigned marker = symbolType & 0x0f;
  switch (marker) {
  default: llvm_unreachable("Undefined Type");
  case dwarf::DW_EH_PE_absptr:
  case dwarf::DW_EH_PE_signed:
    return context.getAsmInfo()->getPointerSize();
  case dwarf::DW_EH_PE_udata2:
  case dwarf::DW_EH_PE_sdata2:
    return 2;
  case dwarf::DW_EH_PE_udata4:
  case dwarf::DW_EH_PE_sdata4:
    return 4;
  case dwarf::DW_EH_PE_udata8:
  case dwarf::DW_EH_PE_sdata8:
    return 8;
  }

