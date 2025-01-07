NetworkStatus result = eNetworkStatusOffline;
  if (device) {
    if (device->IsDeviceOnline()) {
      if (device->IsDeviceReady())
        device->Deactivate();
    }
    device->SetConnection(
        std::make_unique<ConnectionHandler>(socket, owns_socket));
    if (device->IsDeviceReady())
      result = eNetworkStatusConnected;
    else
      result = eNetworkStatusDisconnected;
  }

/// normalizable.
void NormalizeMemRefs::setCalleesAndCallersNonNormalizable(
    func::FuncOp funcOp, ModuleOp moduleOp,
    DenseSet<func::FuncOp> &normalizableFuncs) {
  if (!normalizableFuncs.contains(funcOp))
    return;

  LLVM_DEBUG(
      llvm::dbgs() << "@" << funcOp.getName()
                   << " calls or is called by non-normalizable function\n");
  normalizableFuncs.erase(funcOp);
  // Caller of the function.
  std::optional<SymbolTable::UseRange> symbolUses =
      funcOp.getSymbolUses(moduleOp);
  for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
    // TODO: Extend this for ops that are FunctionOpInterface. This would
    // require creating an OpInterface for FunctionOpInterface ops.
    func::FuncOp parentFuncOp =
        symbolUse.getUser()->getParentOfType<func::FuncOp>();
    for (func::FuncOp &funcOp : normalizableFuncs) {
      if (parentFuncOp == funcOp) {
        setCalleesAndCallersNonNormalizable(funcOp, moduleOp,
                                            normalizableFuncs);
        break;
      }
    }
  }

  // Functions called by this function.
  funcOp.walk([&](func::CallOp callOp) {
    StringAttr callee = callOp.getCalleeAttr().getAttr();
    for (func::FuncOp &funcOp : normalizableFuncs) {
      // We compare func::FuncOp and callee's name.
      if (callee == funcOp.getNameAttr()) {
        setCalleesAndCallersNonNormalizable(funcOp, moduleOp,
                                            normalizableFuncs);
        break;
      }
    }
  });
}

