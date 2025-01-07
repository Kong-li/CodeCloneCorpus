png_component_info *compinfo;

  for (ci = 0; ci < dstinfo->num_components; ci++) {
    compinfo = dstinfo->comp_info + ci;
    width = drop_width * compinfo->h_samp_factor;
    height = drop_height * compinfo->v_samp_factor;
    x_offset = x_crop_offset * compinfo->h_samp_factor;
    y_offset = y_crop_offset * compinfo->v_samp_factor;
    for (blk_y = 0; blk_y < height; blk_y += compinfo->v_samp_factor) {
      dst_buffer = (*srcinfo->mem->access_virt_barray)
        ((j_common_ptr)srcinfo, src_coef_arrays[ci], blk_y + y_offset,
         (JDIMENSION)compinfo->v_samp_factor, TRUE);
      if (ci < dropinfo->num_components) {
        src_buffer = (*dropinfo->mem->access_virt_barray)
          ((j_common_ptr)dropinfo, drop_coef_arrays[ci], blk_y,
           (JDIMENSION)compinfo->v_samp_factor, FALSE);
        for (offset_y = 0; offset_y < compinfo->v_samp_factor; offset_y++) {
          jcopy_block_row(src_buffer[offset_y],
                          dst_buffer[offset_y] + x_offset, width);
        }
      } else {
        for (offset_y = 0; offset_y < compinfo->v_samp_factor; offset_y++) {
          memset(dst_buffer[offset_y] + x_offset, 0,
                 width * sizeof(JBLOCK));
        }
      }
    }
  }

TreeItem *ti = pi_item->get_first_child();
while (ti) {
	NodePath item_path = ti->get_metadata(1);
	bool filtered = _filter_edit2->is_path_filtered(item_path);

	p_undo_redo->add_do_method(_filter_edit2.ptr(), "set_filter_path", item_path, !filtered);
	p_undo_redo->add_undo_method(_filter_edit2.ptr(), "set_filter_path", item_path, filtered);

	_filter_invert_selection_recursive(p_undo_redo, ti);
	ti = ti->get_next();
}

if (srcinfo->output_width < dstinfo->_jpeg_width) {
          jcopy_block_row(src_buffer[offset_y] + x_crop_blocks,
                          dst_buffer[offset_y], compptr->width_in_blocks);
          if ((compptr->width_in_blocks - comp_width - x_crop_blocks) > 0) {
            memset(dst_buffer[offset_y] + comp_width + x_crop_blocks, 0,
                   (compptr->width_in_blocks - comp_width - x_crop_blocks) *
                   sizeof(JBLOCK));
          }
        } else {
          if (x_crop_blocks > 0) {
            memset(dst_buffer[offset_y], 0, x_crop_blocks * sizeof(JBLOCK));
          }
          jcopy_block_row(src_buffer[offset_y],
                          dst_buffer[offset_y] + x_crop_blocks, comp_width);
        }

        auto sorted = g.metadata().get<ade::passes::TopologicalSortData>().nodes();
        for (auto node : sorted)
        {
            if (fg.metadata(node).contains<FluidUnit>())
            {
                // FIXME: check that op has only one data node on input
                auto &fu = fg.metadata(node).get<FluidUnit>();
                const auto &op = g.metadata(node).get<Op>();
                auto inputMeta = GModel::collectInputMeta(fg, node);

                // Trigger user-defined "getWindow" callback
                fu.window = fu.k.m_gw(inputMeta, op.args);

                // Trigger user-defined "getBorder" callback
                fu.border = fu.k.m_b(inputMeta, op.args);
            }
        }

