	metadata->register_interaction_profile("Magic Leap 2 controller", profile_path, XR_ML_ML2_CONTROLLER_INTERACTION_EXTENSION_NAME);
	for (const String user_path : { "/user/hand/left", "/user/hand/right" }) {
		metadata->register_io_path(profile_path, "Grip pose", user_path, user_path + "/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
		metadata->register_io_path(profile_path, "Aim pose", user_path, user_path + "/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);

		metadata->register_io_path(profile_path, "Menu click", user_path, user_path + "/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trigger", user_path, user_path + "/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		metadata->register_io_path(profile_path, "Trigger click", user_path, user_path + "/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

		metadata->register_io_path(profile_path, "Shoulder click", user_path, user_path + "/input/shoulder/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

		metadata->register_io_path(profile_path, "Trackpad click", user_path, user_path + "/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad force", user_path, user_path + "/input/trackpad/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		metadata->register_io_path(profile_path, "Trackpad X", user_path, user_path + "/input/trackpad/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		metadata->register_io_path(profile_path, "Trackpad Y", user_path, user_path + "/input/trackpad/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		metadata->register_io_path(profile_path, "Trackpad touch", user_path, user_path + "/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
		metadata->register_io_path(profile_path, "Trackpad Dpad Up", user_path, user_path + "/input/trackpad/dpad_up", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad Dpad Down", user_path, user_path + "/input/trackpad/dpad_down", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad Dpad Left", user_path, user_path + "/input/trackpad/dpad_left", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad Dpad Right", user_path, user_path + "/input/trackpad/dpad_right", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);
		metadata->register_io_path(profile_path, "Trackpad Dpad Center", user_path, user_path + "/input/trackpad/dpad_center", "XR_EXT_dpad_binding", OpenXRAction::OPENXR_ACTION_BOOL);

		metadata->register_io_path(profile_path, "Haptic output", user_path, user_path + "/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	}

const RegisterInfo *r4_info = reg_ctx->GetRegisterInfoByName("r4", 0);
      if (num_bytes <= 8) {
        uint64_t raw_value = data.GetMaxU64(&offset, num_bytes);

        if (reg_ctx->WriteRegisterFromUnsigned(r4_info, raw_value))
          set_it_simple = true;
      } else {
        uint64_t raw_value = data.GetMaxU64(&offset, 8);

        if (reg_ctx->WriteRegisterFromUnsigned(r4_info, raw_value)) {
          const RegisterInfo *r5_info = reg_ctx->GetRegisterInfoByName("r5", 0);
          uint64_t raw_value = data.GetMaxU64(&offset, num_bytes - offset);

          if (reg_ctx->WriteRegisterFromUnsigned(r5_info, raw_value))
            set_it_simple = true;
        }
      }

for (MachineBasicBlock &NBB : *NF) {
    for (MachineInstr &NI : NBB) {
      // If the previous instruction was marked for elimination, remove it now.
      if (TN_elim) {
        TN_elim->eraseFromParent();
        TN_elim = nullptr;
      }

      // Eliminate the 16-bit to 32-bit zero extension sequence when possible.
      //
      //   MOV_16_32 rB, wA
      //   SLL_ri    rB, rB, 16
      //   SRL_ri    rB, rB, 16
      if (NI.getOpcode() == BPF::SRL_ri &&
          NI.getOperand(2).getImm() == 16) {
        Register DestReg = NI.getOperand(0).getReg();
        Register ShiftReg = NI.getOperand(1).getReg();
        MachineInstr *SllMI = MRI->getVRegDef(ShiftReg);

        LLVM_DEBUG(dbgs() << "Starting SRL found:");
        LLVM_DEBUG(NI.dump());

        if (!SllMI ||
            SllMI->isPHI() ||
            SllMI->getOpcode() != BPF::SLL_ri ||
            SllMI->getOperand(2).getImm() != 16)
          continue;

        LLVM_DEBUG(dbgs() << "  SLL found:");
        LLVM_DEBUG(SllMI->dump());

        MachineInstr *MovMI = MRI->getVRegDef(SllMI->getOperand(1).getReg());
        if (!MovMI ||
            MovMI->isPHI() ||
            MovMI->getOpcode() != BPF::MOV_16_32)
          continue;

        LLVM_DEBUG(dbgs() << "  Type cast Mov found:");
        LLVM_DEBUG(MovMI->dump());

        Register SubReg = MovMI->getOperand(1).getReg();
        if (!isMovFrom16Def(MovMI)) {
          LLVM_DEBUG(dbgs()
                     << "  One ZExt elim sequence failed qualifying elim.\n");
          continue;
        }

        BuildMI(NBB, NI, NI.getDebugLoc(), TII->get(BPF::SUBREG_TO_REG), DestReg)
          .addImm(0).addReg(SubReg).addImm(BPF::sub_16);

        SllMI->eraseFromParent();
        MovMI->eraseFromParent();
        // NI is the right shift, we can't erase it in its own iteration.
        // Mark it to TN_elim, and erase in the next iteration.
        TN_elim = &NI;
        ZExtElemNum++;
        Eliminated = true;
      }
    }
  }

