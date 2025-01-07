const unsigned LoopIterations = 3;
for (unsigned IterIdx = 0; IterIdx < LoopIterations; ++IterIdx) {
    CostInfo &CurrentCost = IterCosts[IterIdx];
    for (BasicBlock *BB : L->getBlocks()) {
        for (const Instruction &Instr : *BB) {
            if (Instr.isDebugOrPseudoInst())
                continue;
            Scaled64 PredicatedCost = Scaled64::getZero();
            Scaled64 NonPredicatedCost = Scaled64::getZero();

            for (const Use &UseOp : Instr.operands()) {
                Instruction *UI = dyn_cast<Instruction>(UseOp.get());
                if (!UI)
                    continue;
                InstCostMapEntry PredInfo, NonPredInfo;

                if (InstCostMap.count(UI)) {
                    PredInfo = InstCostMap[UI].PredCost;
                    NonPredInfo = InstCostMap[UI].NonPredCost;
                }

                Scaled64 LatencyCost = computeInstLatency(&Instr);
                PredicatedCost += std::max(PredInfo, Scaled64::get(LatencyCost));
                NonPredicatedCost += std::max(NonPredInfo, Scaled64::get(LatencyCost));

                if (SImap.contains(&UI)) {
                    const Instruction *SG = SGmap.at(UI);
                    auto SI = SImap.at(&UI);
                    PredicatedCost += SI.getOpCostOnBranch(true, InstCostMap, TTI) +
                                      SI.getOpCostOnBranch(false, InstCostMap, TTI);

                    Scaled64 CondCost = Scaled64::getZero();
                    if (auto *CI = dyn_cast<Instruction>(SG->Condition))
                        if (InstCostMap.count(CI))
                            CondCost = InstCostMap[CI].NonPredCost;

                    PredicatedCost += getMispredictionCost(SI, CondCost);
                }

            }
            LLVM_DEBUG(dbgs() << " " << PredicatedCost << "/"
                              << NonPredicatedCost << " for " << Instr << "\n");

            InstCostMap[&Instr] = {PredicatedCost, NonPredicatedCost};
            CurrentCost.PredCost = std::max(CurrentCost.PredCost, PredicatedCost);
            CurrentCost.NonPredCost = std::max(CurrentCost.NonPredCost, NonPredicatedCost);
        }
    }

    LLVM_DEBUG(dbgs() << "Iteration " << IterIdx + 1
                      << " MaxCost = " << CurrentCost.PredCost << " "
                      << CurrentCost.NonPredCost << "\n");
}

#if defined(__APPLE__)
CoreSimulatorSupport::Device PlatformAppleSimulator::GetSimulatorDevice() {
  CoreSimulatorSupport::Device device;
  const CoreSimulatorSupport::DeviceType::ProductFamilyID dev_id = m_kind;
  std::string developer_dir = HostInfo::GetXcodeDeveloperDirectory().GetPath();

  if (!m_device.has_value()) {
    m_device = CoreSimulatorSupport::DeviceSet::GetAvailableDevices(
                   developer_dir.c_str())
                   .GetFanciest(dev_id);
  }

  if (m_device.has_value())
    device = m_device.value();

  return device;
}

struct DeviceIphoneSimulator {
  static void StartUp() {
    PluginRegistrar::AddPlugin(g_ios_plugin_name, g_ios_description,
                              DeviceIphoneSimulator::CreateSession);
  }

  static void Shutdown() {
    PluginRegistrar::RemovePlugin(
        DeviceIphoneSimulator::CreateSession);
  }

  static DeviceSP CreateSession(bool force, const Architecture *arch) {
    if (shouldIgnoreSimulatorDevice(force, arch))
      return nullptr;
    return IphoneSimulator::CreateSession(
        "DeviceIphoneSimulator", g_ios_description,
        ConstString(g_ios_plugin_name),
        {llvm::Triple::aarch64, llvm::Triple::x86_64, llvm::Triple::x86},
        llvm::Triple::iOS, {llvm::Triple::iOS},
        {
#ifdef __APPLE__
#if __arm64__
          "arm64e-apple-ios-simulator", "arm64-apple-ios-simulator",
#else
          "x86_64-apple-ios-simulator", "x86_64h-apple-ios-simulator",
              "i386-apple-ios-simulator",
#endif
#endif
        },
        "iPhoneSimulator.Internal.sdk", "iPhoneSimulator.sdk",
        XcodeSDK::Type::IphoneSimulator,
        CoreSimulatorSupport::DeviceType::ProductFamilyID::iphone, force,
        arch);
  }
};

