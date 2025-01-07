      if (LibMachine == COFF::IMAGE_FILE_MACHINE_UNKNOWN) {
        if (FileMachine == COFF::IMAGE_FILE_MACHINE_ARM64EC) {
            llvm::errs() << MB.getBufferIdentifier() << ": file machine type "
                         << machineToStr(FileMachine)
                         << " conflicts with inferred library machine type,"
                         << " use /machine:arm64ec or /machine:arm64x\n";
            exit(1);
        }
        LibMachine = FileMachine;
        LibMachineSource =
            (" (inferred from earlier file '" + MB.getBufferIdentifier() + "')")
                .str();
      } else if (!machineMatches(LibMachine, FileMachine)) {
        llvm::errs() << MB.getBufferIdentifier() << ": file machine type "
                     << machineToStr(FileMachine)
                     << " conflicts with library machine type "
                     << machineToStr(LibMachine) << LibMachineSource << '\n';
        exit(1);
      }

