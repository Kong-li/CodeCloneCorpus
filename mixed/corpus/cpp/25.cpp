  // always show the type at the root level if it is invalid
  if (show_type) {
    // Some ValueObjects don't have types (like registers sets). Only print the
    // type if there is one to print
    ConstString type_name;
    if (m_compiler_type.IsValid()) {
      type_name = m_options.m_use_type_display_name
                      ? valobj.GetDisplayTypeName()
                      : valobj.GetQualifiedTypeName();
    } else {
      // only show an invalid type name if the user explicitly triggered
      // show_type
      if (m_options.m_show_types)
        type_name = ConstString("<invalid type>");
    }

    if (type_name) {
      std::string type_name_str(type_name.GetCString());
      if (m_options.m_hide_pointer_value) {
        for (auto iter = type_name_str.find(" *"); iter != std::string::npos;
             iter = type_name_str.find(" *")) {
          type_name_str.erase(iter, 2);
        }
      }
      typeName << type_name_str.c_str();
    }
  }

print(OS, Split, Scopes, UseMatchedElements);

for (LVScope *Scope : *Scopes) {
  getReader().setCompileUnit(const_cast<LVScope *>(Scope));

  // If not 'Split', we use the default output stream; otherwise, set up the split context.
  if (!Split) {
    Scope->printMatchedElements(*StreamDefault, UseMatchedElements);
  } else {
    std::string ScopeName(Scope->getName());
    if (std::error_code EC = getReaderSplitContext().open(ScopeName, ".txt", OS))
      return createStringError(EC, "Unable to create split output file %s",
                               ScopeName.c_str());

    StreamSplit = static_cast<raw_ostream *>(&getReaderSplitContext().os());

    Scope->printMatchedElements(*StreamSplit, UseMatchedElements);

    // Done printing the compile unit. Restore the original output context.
    getReaderSplitContext().close();
    StreamSplit = &getReader().outputStream();
  }
}

// Default stream for non-split cases
raw_ostream *StreamDefault = &getReader().outputStream();

  typedef DenseMap<const BasicBlock *, const PHINode *> StateDefMap;
  std::vector<ThreadingPath> getPathsFromStateDefMap(StateDefMap &StateDef,
                                                     PHINode *Phi,
                                                     VisitedBlocks &VB) {
    std::vector<ThreadingPath> Res;
    auto *PhiBB = Phi->getParent();
    VB.insert(PhiBB);

    VisitedBlocks UniqueBlocks;
    for (auto *IncomingBB : Phi->blocks()) {
      if (!UniqueBlocks.insert(IncomingBB).second)
        continue;
      if (!SwitchOuterLoop->contains(IncomingBB))
        continue;

      Value *IncomingValue = Phi->getIncomingValueForBlock(IncomingBB);
      // We found the determinator. This is the start of our path.
      if (auto *C = dyn_cast<ConstantInt>(IncomingValue)) {
        // SwitchBlock is the determinator, unsupported unless its also the def.
        if (PhiBB == SwitchBlock &&
            SwitchBlock != cast<PHINode>(Switch->getOperand(0))->getParent())
          continue;
        ThreadingPath NewPath;
        NewPath.setDeterminator(PhiBB);
        NewPath.setExitValue(C);
        // Don't add SwitchBlock at the start, this is handled later.
        if (IncomingBB != SwitchBlock)
          NewPath.push_back(IncomingBB);
        NewPath.push_back(PhiBB);
        Res.push_back(NewPath);
        continue;
      }
      // Don't get into a cycle.
      if (VB.contains(IncomingBB) || IncomingBB == SwitchBlock)
        continue;
      // Recurse up the PHI chain.
      auto *IncomingPhi = dyn_cast<PHINode>(IncomingValue);
      if (!IncomingPhi)
        continue;
      auto *IncomingPhiDefBB = IncomingPhi->getParent();
      if (!StateDef.contains(IncomingPhiDefBB))
        continue;

      // Direct predecessor, just add to the path.
      if (IncomingPhiDefBB == IncomingBB) {
        std::vector<ThreadingPath> PredPaths =
            getPathsFromStateDefMap(StateDef, IncomingPhi, VB);
        for (ThreadingPath &Path : PredPaths) {
          Path.push_back(PhiBB);
          Res.push_back(std::move(Path));
        }
        continue;
      }
      // Not a direct predecessor, find intermediate paths to append to the
      // existing path.
      if (VB.contains(IncomingPhiDefBB))
        continue;

      PathsType IntermediatePaths;
      IntermediatePaths =
          paths(IncomingPhiDefBB, IncomingBB, VB, /* PathDepth = */ 1);
      if (IntermediatePaths.empty())
        continue;

      std::vector<ThreadingPath> PredPaths =
          getPathsFromStateDefMap(StateDef, IncomingPhi, VB);
      for (const ThreadingPath &Path : PredPaths) {
        for (const PathType &IPath : IntermediatePaths) {
          ThreadingPath NewPath(Path);
          NewPath.appendExcludingFirst(IPath);
          NewPath.push_back(PhiBB);
          Res.push_back(NewPath);
        }
      }
    }
    VB.erase(PhiBB);
    return Res;
  }

/// Calculate known bits for the intersection of \p Register1 and \p Register2
void GISelKnownBits::calculateKnownBitsMin(Register reg1, Register reg2,
                                           KnownBits &knownResult,
                                           const APInt &demandedEltCount,
                                           unsigned depth) {
  // First test register2 since we canonicalize simpler expressions to the RHS.
  computeKnownBitsImpl(reg2, knownResult, demandedEltCount, depth);

  if (!knownResult.hasKnownValues()) {
    return;
  }

  KnownBits knownResult2;
  computeKnownBitsImpl(reg1, knownResult2, demandedEltCount, depth);

  // Only known if known in both the LHS and RHS.
  knownResult = knownResult.intersectWith(knownResult2);
}

