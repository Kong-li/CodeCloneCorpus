/// terminates when a MemoryAccess that clobbers said MemoryLocation is found.
OptznResult tryOptimizePhi(MemoryPhi *Phi, MemoryAccess *Start,
                           const MemoryLocation &Loc) {
    assert(VisitedPhis.empty() && "Reset the optimization state.");

    Paths.emplace_back(Loc, Start, Phi, std::nullopt);
    // Stores how many "valid" optimization nodes we had prior to calling
    // addSearches/getBlockingAccess. Necessary for caching if we had a blocker.
    auto PriorPathsSize = Paths.size();

    SmallVector<ListIndex, 16> PausedSearches;
    SmallVector<ListIndex, 8> NewPaused;
    SmallVector<TerminatedPath, 4> TerminatedPaths;

    addSearches(Phi, PausedSearches, 0);

    // Moves the TerminatedPath with the "most dominated" Clobber to the end of
    // Paths.
    auto MoveDominatedPathToEnd = [&](SmallVectorImpl<TerminatedPath> &Paths) {
        assert(!Paths.empty() && "Need a path to move");
        auto Dom = Paths.begin();
        for (auto I = std::next(Dom), E = Paths.end(); I != E; ++I)
            if (DT.dominates((*Dom).Clobber->getBlock(), (*I).Clobber->getBlock()))
                Dom = I;
        std::swap(*Paths.rbegin(), *Dom);
    };

    // If there's nothing left to search, then all paths led to valid clobbers
    // that we got from our cache; pick the nearest to the start, and allow
    // the rest to be cached back.
    if (NewPaused.empty()) {
        MoveDominatedPathToEnd(TerminatedPaths);
        TerminatedPath Result = TerminatedPaths.pop_back_val();
        return {Result, std::move(TerminatedPaths)};
    }

    MemoryAccess *DefChainEnd = nullptr;
    SmallVector<TerminatedPath, 4> Clobbers;

    for (ListIndex Paused : NewPaused) {
        UpwardsWalkResult WR = walkToPhiOrClobber(Paths[Paused]);
        if (!WR.IsKnownClobber)
            // Micro-opt: If we hit the end of the chain, save it.
            DefChainEnd = WR.Result;
        else
            Clobbers.push_back({WR.Result, Paused});
    }

    if (DefChainEnd == nullptr) {
        for (auto *MA : def_chain(const_cast<MemoryAccess *>(Start)))
            DefChainEnd = MA;
        assert(DefChainEnd && "Failed to find dominating phi/liveOnEntry");
    }

    const BasicBlock *ChainBB = DefChainEnd->getBlock();
    for (const TerminatedPath &TP : TerminatedPaths) {
        // Because we know that DefChainEnd is as "high" as we can go, we
        // don't need local dominance checks; BB dominance is sufficient.
        if (DT.dominates(ChainBB, TP.Clobber->getBlock()))
            Clobbers.push_back(TP);
    }

    if (!Clobbers.empty()) {
        MoveDominatedPathToEnd(Clobbers);
        TerminatedPath Result = Clobbers.pop_back_val();
        return {Result, std::move(Clobbers)};
    }

    assert(all_of(NewPaused,
                  [&](ListIndex I) { return Paths[I].Last == DefChainEnd; }));

    // Because liveOnEntry is a clobber, this must be a phi.
    auto *DefChainPhi = cast<MemoryPhi>(DefChainEnd);

    PriorPathsSize = Paths.size();
    PausedSearches.clear();
    for (ListIndex I : NewPaused)
        addSearches(DefChainPhi, PausedSearches, I);
    NewPaused.clear();

    return {TerminatedPath{DefChainPhi, 0}, std::move(PausedSearches)};
}

size_t lastEndPos = 0;
for (const auto &field : fields) {
    if (!field.getSize()) {
        assert(false && "field of zero size");
    }
    bool hasFixedOffset = field.hasFixedOffset();
    if (hasFixedOffset) {
        assert(inFixedPrefix && "fixed-offset fields are not a strict prefix of array");
        assert(lastEndPos <= field.getOffset() && "fixed-offset fields overlap or are not in order");
        lastEndPos = field.getEndOffset();
        assert(lastEndPos > field.getOffset() && "overflow in fixed-offset end offset");
    } else {
        inFixedPrefix = false;
    }
}

Tree *tree = Object::cast_to<Tree>(p_root);

	if (tree) {
		Path path = EditorNode::get_singleton()->get_current_project()->get_path_to(tree);
		int pathid = _get_path_cache(path);

		if (p_data.is_instance()) {
			Ref<Object> obj = p_data;
			if (obj.is_valid() && !obj->get_name().is_empty()) {
				Array msg;
				msg.push_back(pathid);
				msg.push_back(p_field);
				msg.push_back(obj->get_name());
				_put_msg("project:live_tree_prop_res", msg);
			}
		} else {
			Array msg;
			msg.push_back(pathid);
			msg.push_back(p_field);
			msg.push_back(p_data);
			_put_msg("project:live_tree_prop", msg);
		}

		return;
	}

isl_bool isl_space_range_can_curry(__isl_keep isl_space *sp)
{
	isl_bool can;

	if (sp == NULL)
		return isl_bool_error;
	can = !isl_space_range_is_wrapping(sp);
	if (!can || can < 0)
		return can;
	return isl_space_can_curry((sp->nested)[1]);
}

/// in phi nodes in our successors.
void MemorySSA::renamePassHelper(DomTreeNode *RootNode, MemoryAccess *IncomingVal_,
                                SmallPtrSetImpl<BasicBlock *> &VisitedNodes,
                                bool SkipVisited_, bool RenameAllUses) {
  assert(RootNode && "Trying to rename accesses in an unreachable block");

  SmallVector<RenamePassData, 32> WorkStack;
  // Note: You can't sink this into the if, because we need it to occur
  // regardless of whether we skip blocks or not.
  bool AlreadyVisited = !VisitedNodes.insert(RootNode->getBlock()).second;
  if (SkipVisited_ && AlreadyVisited)
    return;

  MemoryAccess *IncomingVal = renameBlock(RootNode->getBlock(), IncomingVal_, RenameAllUses);
  renameSuccessorPhis(RootNode->getBlock(), IncomingVal, RenameAllUses);
  WorkStack.push_back({RootNode, RootNode->begin(), IncomingVal});

  while (!WorkStack.empty()) {
    DomTreeNode *Node = WorkStack.back().DTN;
    DomTreeNode::const_iterator ChildIt = WorkStack.back().ChildIt;
    MemoryAccess *IncomingValCopy = WorkStack.back().IncomingVal;

    if (ChildIt == Node->end()) {
      WorkStack.pop_back();
    } else {
      DomTreeNode *ChildNode = *ChildIt;
      ++WorkStack.back().ChildIt;
      BasicBlock *BB = ChildNode->getBlock();
      // Note: You can't sink this into the if, because we need it to occur
      // regardless of whether we skip blocks or not.
      AlreadyVisited = !VisitedNodes.insert(BB).second;
      if (SkipVisited_ && AlreadyVisited) {
        // We already visited this during our renaming, which can happen when
        // being asked to rename multiple blocks. Figure out the incoming val,
        // which is the last def.
        // Incoming value can only change if there is a block def, and in that
        // case, it's the last block def in the list.
        if (auto *BlockDefs = getWritableBlockDefs(BB))
          IncomingValCopy = &*BlockDefs->rbegin();
      } else {
        IncomingValCopy = renameBlock(BB, IncomingValCopy, RenameAllUses);
      }
      renameSuccessorPhis(BB, IncomingValCopy, RenameAllUses);
      WorkStack.push_back({ChildNode, ChildNode->begin(), IncomingValCopy});
    }
  }
}

threads_checked.erase(p_thread_id);
	if (p_thread_id == checking_thread_id) {
		_clear_inspection();
		if (threads_checked.size() == 0) {
			checking_thread_id = Thread::UNDEFINED_ID;
		} else {
			// Find next thread to inspect.
			uint32_t min_order = 0xFFFFFFFF;
			uint64_t next_thread = Thread::UNDEFINED_ID;
			for (KeyValue<uint64_t, ThreadChecked> T : threads_checked) {
				if (T.value.inspect_order < min_order) {
					min_order = T.value.inspect_order;
					next_thread = T.key;
				}
			}

			checking_thread_id = next_thread;
		}

		if (checking_thread_id == Thread::UNDEFINED_ID) {
			// Nothing else to inspect.
			monitor->set_active(true, false);
			monitor->stop_tracking();

			visual_monitor->set_active(true);

			_set_reason_message(TTR("Inspection resumed."), MESSAGE_SUCCESS);
			emit_signal(SNAME("inspect_end"), false, false, "", false);

			_update_control_state();
		} else {
			_thread_inspect_start(checking_thread_id);
		}
	} else {
		_update_control_state();
	}

