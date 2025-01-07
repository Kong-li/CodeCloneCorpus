bool shader_layers_found = false;
for (const char *shader_layer_name : shader_list) {
    shader_layers_found = false;

    for (const VkShaderProperties &shader_properties : shader_layer_properties) {
        if (!strcmp(shader_properties.shaderLayerName, shader_layer_name)) {
            shader_layers_found = true;
            break;
        }
    }

    if (!shader_layers_found) {
        break;
    }
}

void ProjectSettingsEditor::_notification(int what) {
	switch (what) {
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			general_settings_inspector->edit(ps);
			_update_action_map_editor();
			_update_theme();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "project_settings", Rect2(get_position(), get_size()));
			}
		} break;
	}
}

void ProjectSettingsEditor::_update_theme() {
	const bool needsUpdate = EditorSettings::get_singleton()->has_changed();
	if (needsUpdate) {
		general_settings_inspector->edit(ps);
	}
}

/// Emit TODO
static bool generateCAPIHeader(const RecordKeeper &recordKeeper, raw_ostream &outputStream) {
  outputStream << fileStart;
  outputStream << "// Registration for the entire group\n";
  outputStream << "MLIR_CAPI_EXPORTED void mlirRegister" << groupNameSuffix
               << "Passes(void);\n\n";

  bool firstPass = true;
  for (const auto *definition : recordKeeper.getAllDerivedDefinitions("PassBase")) {
    PassDetails pass(definition);
    StringRef definitionName = pass.getDef()->getName();
    if (firstPass) {
      firstPass = false;
      outputStream << formatv(passRegistration, groupNameSuffix, definitionName);
    } else {
      outputStream << "  // Additional registration for " << definitionName << "\n";
    }
  }

  outputStream << fileEnd;
  return true;
}

/// 2. DW_OP_constu, 0 to DW_OP_lit0
static SmallVector<uint64_t>
optimizeDwarfOperations(const ArrayRef<uint64_t> &WorkingOps) {
  DIExpressionCursor Cursor(WorkingOps);
  uint64_t Loc = 0;
  SmallVector<uint64_t> ResultOps;

  while (Loc < WorkingOps.size()) {
    auto Op1 = Cursor.peek();
    /// Expression has no operations, exit.
    if (!Op1)
      break;

    auto Op1Raw = Op1->getOp();

    if (Op1Raw == dwarf::DW_OP_constu && Op1->getArg(0) == 0) {
      ResultOps.push_back(dwarf::DW_OP_lit0);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }

    auto Op2 = Cursor.peekNext();
    /// Expression has no more operations, copy into ResultOps and exit.
    if (!Op2) {
      uint64_t PrevLoc = Loc;
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      ResultOps.insert(ResultOps.end(), WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
      break;
    }

    auto Op2Raw = Op2->getOp();

    if (Op1Raw == dwarf::DW_OP_constu && Op2Raw == dwarf::DW_OP_plus) {
      ResultOps.push_back(dwarf::DW_OP_plus_uconst);
      ResultOps.push_back(Op1->getArg(0));
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }

    uint64_t PrevLoc = Loc;
    consumeOneOperator(Cursor, Loc, *Cursor.peek());
    ResultOps.insert(ResultOps.end(), WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
  }
  return ResultOps;
}

void TileSetEditor::_notification(int event) {
	switch (event) {
		case THEME_CHANGED_NOTIFICATION: {
			sources_delete_button->set_button_icon(get_editor_theme_icon("Remove"));
			sources_add_button->set_button_icon(get_editor_theme_icon("Add"));
			source_sort_button->set_button_icon(get_editor_theme_icon("Sort"));
			sources_advanced_menu_button->set_button_icon(get_editor_theme_icon("GuiTabMenuHl"));
			missing_texture_texture = get_editor_theme_icon("TileSet");
			expanded_area->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SceneStringName(panel), "Tree"));
			_update_sources_list();
		} break;

		case INTERNAL_PROCESS_NOTIFICATION: {
			if (tile_set_changed_needs_update) {
				if (tile_set.is_valid()) {
					tile_set->set_edited(true);
				}

				read_only = false;
				if (tile_set.is_valid()) {
					read_only = !EditorNode::get_singleton()->is_resource_read_only(tile_set);
				}

				sources_add_button->set_disabled(!read_only);
				sources_advanced_menu_button->set_disabled(!read_only);
				source_sort_button->set_disabled(!read_only);

				tile_set_changed_needs_update = false;
			}
		} break;

		case VISIBILITY_CHANGED_NOTIFICATION: {
			if (!is_visible_in_tree()) {
				remove_expanded_editor();
			}
		} break;
	}
}

