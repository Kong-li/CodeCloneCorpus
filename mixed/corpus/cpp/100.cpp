for (uint32_t new_mod = 0; new_mod < num_of_new_modules; new_mod++) {
  if (to_be_added[new_mod]) {
    ModuleInfo &module_info = module_summaries[new_mod];
    if (load_modules) {
      if (!module_info.LoadModuleUsingMemoryModule(system, &progress)) {
        modules_failed_to_load.push_back(std::pair<std::string, UUID>(
            module_summaries[new_mod].GetName(),
            module_summaries[new_mod].GetUUID()));
        module_info.LoadModuleAtFileAddress(system);
      }
    }

    system_known_modules.push_back(module_info);

    if (module_info.GetLibrary() &&
        system->GetStopID() == module_info.GetProcessStopId())
      loaded_library_list.AppendIfNeeded(module_info.GetLibrary());

    if (log)
      module_summaries[new_mod].PutToLog(log);
  }
}

	int cc = p_node->get_child_count(false);
	for (int i = 0; i < cc; i++) {
		Node *c = p_node->get_child(i, false);
		HashMap<Node *, CachedNode>::Iterator IC = cache.find(c);

		if (IC) {
			IC->value.dirty = true;

			if (p_recursive) {
				mark_children_dirty(c, p_recursive);
			}
		}
	}

