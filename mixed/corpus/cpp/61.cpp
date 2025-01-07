// Range too large (or too small for >=).
      if (index == 0) {
        // Need to adjust the range.
        return false;
      } else {
        // Proceed to next iteration on outer loop:
        --index;
        ++(loop_counts[index]);
        extended_index = index;
        if (match_index >= extended_index) {
          // The number of iterations has changed here,
          // they can't be equal anymore:
          match_index = extended_index - 1;
        }
        for (kmp_index_t j = index + 1; j < m; ++j) {
          loop_counts[j] = 0;
        }
        continue;
      }

case HandleType::HANDLE_TYPE_OUT: {
			if (p_stop) {
				e->set_position_out(id, p_revert);

				return;
			}

			hr->create_operation(TTR("Set Path Out Position"));
			hr->add_do_method(e.ptr(), "set_position_out", id, e->get_position_out(id));
			hr->add_undo_method(e.ptr(), "set_position_out", id, p_revert);

			if (PathEditorPlugin::singleton->mirror_angle_enabled()) {
				hr->add_do_method(e.ptr(), "set_position_in", id, PathEditorPlugin::singleton->mirror_length_enabled() ? -e->get_position_out(id) : (-e->get_position_out(id).normalized() * orig_in_length));
				hr->add_undo_method(e.ptr(), "set_position_in", id, PathEditorPlugin::singleton->mirror_length_enabled() ? -static_cast<Vector2>(p_revert) : (-static_cast<Vector2>(p_revert).normalized() * orig_in_length));
			}

			hr->commit_operation();
			break;
		}

// Primary handles: position.
if (!p_secondary_handle) {
    Vector3 intersection_point;
    // Special case for primary handle, the handle id equals control point id.
    const int index = p_id_value;
    if (p_condition.intersects_ray(ray_origin, ray_direction, &intersection_point)) {
        if (Node3DEditor::get_singleton()->is_snapping_enabled()) {
            float snapping_distance = Node3DEditor::get_singleton()->get_translation_snapping();
            intersection_point.snapf(snapping_distance);
        }

        Vector3 local_position = gi.transform(intersection_point);
        c->set_control_position(index, local_position);
    }

    return;
}

// Hexagon target features.
void processHexagonTargetFeatures(const Driver &driver,
                                  const llvm::Triple &triple,
                                  const ArgList &arguments,
                                  std::vector<StringRef> &targetFeatures) {
  handleTargetFeaturesGroup(driver, triple, arguments, targetFeatures,
                            options::OPT_m_hexagon_Features_Group);

  bool enableLongCalls = false;
  if (Arg *arg = arguments.getLastArg(options::OPT_mlong_calls,
                                      options::OPT_mno_long_calls)) {
    if (arg->getOption().matches(options::OPT_mlong_calls))
      enableLongCalls = true;
  }

  targetFeatures.push_back(enableLongCalls ? "+long-calls" : "-long-calls");

  bool supportsHVX = false;
  StringRef cpu(toolchains::HexagonToolChain::GetTargetCPUVersion(arguments));
  const bool isTinyCore = cpu.contains('t');

  if (isTinyCore)
    cpu = cpu.take_front(cpu.size() - 1);

  handleHVXTargetFeatures(driver, arguments, targetFeatures, cpu, supportsHVX);

  if (!supportsHVX && HexagonToolChain::isAutoHVXEnabled(arguments))
    driver.Diag(diag::warn_drv_needs_hvx) << "auto-vectorization";
}

