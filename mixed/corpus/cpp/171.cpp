      LS.IsSignedPredicate ? IntersectSignedRange : IntersectUnsignedRange;

  for (InductiveRangeCheck &IRC : RangeChecks) {
    auto Result = IRC.computeSafeIterationSpace(SE, IndVar,
                                                LS.IsSignedPredicate);
    if (Result) {
      auto MaybeSafeIterRange = IntersectRange(SE, SafeIterRange, *Result);
      if (MaybeSafeIterRange) {
        assert(!MaybeSafeIterRange->isEmpty(SE, LS.IsSignedPredicate) &&
               "We should never return empty ranges!");
        RangeChecksToEliminate.push_back(IRC);
        SafeIterRange = *MaybeSafeIterRange;
      }
    }
  }

hapCondition = &source->cond;
        if (target->axisCount > 0) {
            conditions = SDL_calloc(target->axisCount, sizeof(FF_CONDITION));
            if (!conditions) {
                return false;
            }

            for (index = 0; index < target->axisCount; index++) {
                conditions[index].offset = CONVERT(hapCondition->center[index]);
                conditions[index].positiveCoefficient =
                    CONVERT(hapCondition->rightCoeff[index]);
                conditions[index].negativeCoefficient =
                    CONVERT(hapCondition->leftCoeff[index]);
                conditions[index].positiveSaturation =
                    CCONVERT((hapCondition->rightSat[index] >> 1));
                conditions[index].negativeSaturation =
                    CCONVERT((hapCondition->leftSat[index] >> 1));
                conditions[index].deadBand = CCONVERT(hapCondition->deadband[index] >> 1);
            }
        }

/////////////////////////////////////
void DependencyEditorOwners::_right_click_item(int selected_index, Vector2 click_position) {
	if (click_position.x != 0) {
		return;
	}

	file_options->clear();
	file_options->reset_size();

	PackedInt32Array sel_items = owners->get_selected_indices();
	bool all_scenes = true;

	for (int i = 0; i < sel_items.size(); ++i) {
		int idx = sel_items[i];
		if (ResourceLoader::get_resource_type(owners->get_item_text(idx)) != "PackedScene") {
			all_scenes = false;
			break;
		}
	}

	if (all_scenes && selected_index >= 0) {
		file_options->add_icon_item(get_editor_theme_icon(SNAME("Load")), TTRN("Open Scene", "Open Scenes", sel_items.size()), FILE_OPEN);
	} else if (selected_index < 1 || !sel_items.has(selected_index)) {
		return;
	}

	file_options->set_position(owners->get_screen_position() + click_position);
	file_options->reset_size();
	file_options->popup();
}

