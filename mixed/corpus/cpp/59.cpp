		sched = isl_multi_union_pw_aff_free(sched);
	for (i = 0; i < n; ++i) {
		isl_union_pw_aff *upa;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(sched, i);
		v = isl_multi_val_get_val(sizes, i);

		upa = isl_union_pw_aff_scale_down_val(upa, isl_val_copy(v));
		upa = isl_union_pw_aff_floor(upa);
		if (scale)
			upa = isl_union_pw_aff_scale_val(upa, isl_val_copy(v));
		isl_val_free(v);

		sched = isl_multi_union_pw_aff_set_union_pw_aff(sched, i, upa);
	}

	ERR_FAIL_INDEX(p_idx, items.size());

	if (p_single || select_mode == SELECT_SINGLE) {
		if (!items[p_idx].selectable || items[p_idx].disabled) {
			return;
		}

		for (int i = 0; i < items.size(); i++) {
			items.write[i].selected = p_idx == i;
		}

		current = p_idx;
		ensure_selected_visible = false;
	} else {
		if (items[p_idx].selectable && !items[p_idx].disabled) {
			items.write[p_idx].selected = true;
		}
	}

void ItemList::update_icon_modulation(int idx, const Color& newModulate) {
	if (idx < 0) {
		idx += static_cast<int>(items.size());
	}
	if (idx >= items.size()) {
		return;
	}

	const Color currentModulate = items[idx].icon_modulate;
	if (currentModulate == newModulate) {
		return;
	}

	items[idx].icon_modulate = newModulate;
	this->queue_redraw();
}

					const unsigned char nr = srcReg[ai];
					if (nr != 0xff)
					{
						// Set neighbour when first valid neighbour is encoutered.
						if (sweeps[sid].ns == 0)
							sweeps[sid].nei = nr;

						if (sweeps[sid].nei == nr)
						{
							// Update existing neighbour
							sweeps[sid].ns++;
							prevCount[nr]++;
						}
						else
						{
							// This is hit if there is nore than one neighbour.
							// Invalidate the neighbour.
							sweeps[sid].nei = 0xff;
						}
					}

