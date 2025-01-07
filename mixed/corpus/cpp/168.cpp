  /// \returns true if such an ancestor was found, false otherwise.
  bool findContainingAncestor(DynTypedNode Start, SourceLocation MacroLoc,
                              DynTypedNode &Result) {
    // Below we're only following the first parent back up the AST. This should
    // be fine since for the statements we care about there should only be one
    // parent, except for the case specified below.

    assert(MacroLoc.isFileID());

    while (true) {
      const auto &Parents = Context.getParents(Start);
      if (Parents.empty())
        return false;
      if (Parents.size() > 1) {
        // If there are more than one parents, don't do the replacement unless
        // they are InitListsExpr (semantic and syntactic form). In this case we
        // can choose any one here, and the ASTVisitor will take care of
        // traversing the right one.
        for (const auto &Parent : Parents) {
          if (!Parent.get<InitListExpr>())
            return false;
        }
      }

      const DynTypedNode &Parent = Parents[0];

      SourceLocation Loc;
      if (const auto *D = Parent.get<Decl>())
        Loc = D->getBeginLoc();
      else if (const auto *S = Parent.get<Stmt>())
        Loc = S->getBeginLoc();

      // TypeLoc and NestedNameSpecifierLoc are members of the parent map. Skip
      // them and keep going up.
      if (Loc.isValid()) {
        if (!expandsFrom(Loc, MacroLoc)) {
          Result = Parent;
          return true;
        }
      }
      Start = Parent;
    }

    llvm_unreachable("findContainingAncestor");
  }

Components.clear();

  if (HSOpts->ImplicitHeaderMaps) {
    // Load header maps for each of the directory search directories.
    for (DirectoryLookup &DL : directory_search_range()) {
      bool IsSystem = DL.isSystemDirectory();
      if (DL.isFramework()) {
        std::error_code EC;
        SmallString<128> DirNative;
        llvm::sys::path::native(DL.getFrameworkDirRef()->getFileName(), DirNative);

        // Search each of the ".framework" directories to load them as components.
        llvm::vfs::FileSystem &FS = FileMgr.getVirtualFileSystem();
        for (llvm::vfs::directory_iterator Dir = FS.dir_begin(DirNative, EC),
                                           DirEnd;
             Dir != DirEnd && !EC; Dir.increment(EC)) {
          if (llvm::sys::path::extension(Dir->path()) != ".framework")
            continue;

          auto FrameworkDir = FileMgr.getOptionalDirectoryRef(Dir->path());
          if (!FrameworkDir)
            continue;

          // Load this framework component.
          loadFrameworkComponent(llvm::sys::path::stem(Dir->path()), *FrameworkDir,
                                 IsSystem);
        }
        continue;
      }

      // FIXME: Deal with header maps.
      if (DL.isHeaderMap())
        continue;

      // Try to load a header map file for the search directory.
      loadHeaderMapFile(*DL.getDirectoryRef(), IsSystem, /*IsFramework*/ false);

      // Try to load header map files for immediate subdirectories of this
      // search directory.
      loadSubdirectoryHeaderMaps(DL);
    }
  }

using namespace lldb_private;

ABISP
ABI::FindPlugin(lldb::ProcessSP process_sp, const ArchSpec &arch) {
  ABISP abi_sp;
  ABICreateInstance create_callback;

  for (uint32_t idx = 0;
       (create_callback = PluginManager::GetABICreateCallbackAtIndex(idx)) !=
       nullptr;
       ++idx) {
    abi_sp = create_callback(process_sp, arch);

    if (abi_sp)
      return abi_sp;
  }
  abi_sp.reset();
  return abi_sp;
}

