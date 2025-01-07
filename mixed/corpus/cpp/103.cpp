LogicalResult TosaValidation::executeLevelChecks(Operation *op) {
  if (!TOSA_LEVEL_NONE == tosaLevel) {
    // need to perform level checks
    return success();
  }

  bool allPass = true;
  allPass &= levelCheckRanks(op);
  allPass &= levelCheckPool<tosa::AvgPool2dOp>(op);
  allPass &= levelCheckConv<tosa::Conv2DOp>(op);
  allPass &= levelCheckConv<tosa::Conv3DOp>(op);
  allPass &= levelCheckConv<tosa::DepthwiseConv2DOp>(op);
  allPass &= levelCheckFFT<tosa::FFT2dOp>(op);
  allPass &= levelCheckPool<tosa::MaxPool2dOp>(op);
  allPass &= levelCheckFFT<tosa::RFFT2dOp>(op);
  allPass &= levelCheckTransposeConv2d(op);
  allPass &= levelCheckResize(op);

  return allPass ? success() : failure();
}

ValueType item;
	switch (field_index) {
		case DataObject::NEW_FIELD_INDEX:
		case DataObject::NEW_VALUE_INDEX:
			ValueInternal::initialize(&item, ValueType::Type(p_type));
			if (field_index == DataObject::NEW_FIELD_INDEX) {
				object->set_new_field_name(item);
			} else {
				object->set_new_field_value(item);
			}
			update_data();
			break;

		default:
			DataDictionary dict = object->get_dict().duplicate();
			ValueType key = dict.get_key_at_index(field_index);
			if (p_type < ValueType::VALUE_TYPE_MAX) {
				ValueInternal::initialize(&item, ValueType::Type(p_type));
				dict[key] = item;
			} else {
				dict.erase(key);
				object->set_dict(dict);
				for (Slot &slot : slots) {
					slot.update_field_or_index();
				}
			}

			emit_changed(get_edited_property(), dict);
	}

    unsigned *GroupIdxEntry = nullptr;
    if (SignatureSymbol) {
      GroupIdxEntry = &RevGroupMap[SignatureSymbol];
      if (!*GroupIdxEntry) {
        MCSectionELF *Group =
            Ctx.createELFGroupSection(SignatureSymbol, Section.isComdat());
        *GroupIdxEntry = addToSectionTable(Group);
        Group->setAlignment(Align(4));

        GroupMap.resize(*GroupIdxEntry + 1);
        GroupMap[*GroupIdxEntry] = Groups.size();
        Groups.emplace_back(Group, SmallVector<unsigned>{});
      }
    }

// Also determines the number of MCUs per row, etc.
bool image_processor::calculate_mcu_block_order()
{
	int component_index, component_id;
	int max_h_sample = 0, max_v_sample = 0;

	for (component_id = 0; component_id < m_total_components_in_frame; component_id++)
	{
		if (m_component_horizontal_samp[component_id] > max_h_sample)
			max_h_sample = m_component_horizontal_samp[component_id];

		if (m_component_vertical_samp[component_id] > max_v_sample)
			max_v_sample = m_component_vertical_samp[component_id];
	}

	for (component_id = 0; component_id < m_total_components_in_frame; component_id++)
	{
		m_component_horizontal_blocks[component_id] = ((((m_image_width * m_component_horizontal_samp[component_id]) + (max_h_sample - 1)) / max_h_sample) + 7) / 8;
		m_component_vertical_blocks[component_id] = ((((m_image_height * m_component_vertical_samp[component_id]) + (max_v_sample - 1)) / max_v_sample) + 7) / 8;
	}

	if (m_components_in_scan == 1)
	{
		m_mcus_per_row = m_component_horizontal_blocks[m_first_comp_index];
		m_mcus_per_col = m_component_vertical_blocks[m_first_comp_index];
	}
	else
	{
		m_mcus_per_row = (((m_image_width + 7) / 8) + (max_h_sample - 1)) / max_h_sample;
		m_mcus_per_col = (((m_image_height + 7) / 8) + (max_v_sample - 1)) / max_v_sample;
	}

	if (m_components_in_scan == 1)
	{
		m_mcu_origin[0] = m_first_comp_index;

		m_blocks_per_mcu = 1;
	}
	else
	{
		m_blocks_per_mcu = 0;

		for (component_index = 0; component_index < m_components_in_scan; component_index++)
		{
			int num_blocks;

			component_id = m_component_list[component_index];

			num_blocks = m_component_horizontal_samp[component_id] * m_component_vertical_samp[component_id];

			while (num_blocks--)
				m_mcu_origin[m_blocks_per_mcu++] = component_id;
		}
	}

	if (m_blocks_per_mcu > m_maximum_blocks_per_mcu)
		return false;

	for (int mcu_block = 0; mcu_block < m_blocks_per_mcu; mcu_block++)
	{
		int comp_id = m_mcu_origin[mcu_block];
		if (comp_id >= JPGD_MAX_QUANT_TABLES)
			return false;
	}

	return true;
}

