	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
			fabrik_data_chain.write[p_joint_idx].bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			fabrik_data_chain.write[p_joint_idx].bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("Cannot verify the FABRIK joint " + itos(p_joint_idx) + " bone index for this modification...");
			fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
		}
	} else {
		fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
	}

    const size_t num_breakpoints = breakpoints.GetSize();
    for (size_t j = 0; j < num_breakpoints; ++j) {
      Breakpoint *breakpoint = breakpoints.GetBreakpointAtIndex(j).get();
      break_id_t cur_bp_id = breakpoint->GetID();

      if ((cur_bp_id < start_bp_id) || (cur_bp_id > end_bp_id))
        continue;

      const size_t num_locations = breakpoint->GetNumLocations();

      if ((cur_bp_id == start_bp_id) &&
          (start_loc_id != LLDB_INVALID_BREAK_ID)) {
        for (size_t k = 0; k < num_locations; ++k) {
          BreakpointLocation *bp_loc = breakpoint->GetLocationAtIndex(k).get();
          if ((bp_loc->GetID() >= start_loc_id) &&
              (bp_loc->GetID() <= end_loc_id)) {
            StreamString canonical_id_str;
            BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                                bp_loc->GetID());
            new_args.AppendArgument(canonical_id_str.GetString());
          }
        }
      } else if ((cur_bp_id == end_bp_id) &&
                 (end_loc_id != LLDB_INVALID_BREAK_ID)) {
        for (size_t k = 0; k < num_locations; ++k) {
          BreakpointLocation *bp_loc = breakpoint->GetLocationAtIndex(k).get();
          if (bp_loc->GetID() <= end_loc_id) {
            StreamString canonical_id_str;
            BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                                bp_loc->GetID());
            new_args.AppendArgument(canonical_id_str.GetString());
          }
        }
      } else {
        StreamString canonical_id_str;
        BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                            LLDB_INVALID_BREAK_ID);
        new_args.AppendArgument(canonical_id_str.GetString());
      }
    }

void SkeletonModification2DFABRIK::update_target_cache() {
	if (!is_setup || !stack) {
		if (is_setup) {
			ERR_PRINT_ONCE("Cannot update target cache: modification is not properly setup!");
		}
		return;
	}

	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in scene tree!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

