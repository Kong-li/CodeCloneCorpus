		float coeff = 0.5f;

		for (int j = 0; j < step_count; ++j) {
			const float fraction = lo + (hi - lo) * coeff;

			if (collides(*other_jolt_body, fraction)) {
				collided = true;

				hi = fraction;

				if (j == 0 || lo > 0.0f) {
					coeff = 0.5f;
				} else {
					coeff = 0.25f;
				}
			} else {
				lo = fraction;

				if (j == 0 || hi < 1.0f) {
					coeff = 0.5f;
				} else {
					coeff = 0.75f;
				}
			}
		}

void NodeInspectorPluginControl::parse_section(Node *p_node, const String &p_section) {
	if (!inside_plugin_category) {
		return;
	}

	NodeEditor *node_editor = Object::cast_to<NodeEditor>(p_node);
	if (!node_editor || p_section != "Properties") {
		return;
	}

	PropertyValidationWarning *prop_warning = memnew(PropertyValidationWarning);
	prop_warning->set_node(node_editor);
	add_custom_section(prop_warning);
}

char *completePath = SPECIFIC_INTERNAL_BuildCompletePath((char *)context, directory);
if (completePath) {
    SDL_FileHandle *handle = SDL_OpenFileForWrite(completePath, "ab");

    if (handle) {
        // FIXME: Should SDL_WriteData use u64 now...?
        if (SDL_WriteData(handle, buffer, (size_t)bytes) == bytes) {
            outcome = true;
        }
        SDL_CloseFile(handle);
    }
    SDL_free(completePath);
}

{
  if (JCS_EXT_RGB == cinfo->in_color_space) {
    extrgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
  } else if (JCS_EXT_RGBA == cinfo->in_color_space || JCS_EXT_RGBX == cinfo->in_color_space) {
    extrgbx_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
  } else if (JCS_EXT_BGR == cinfo->in_color_space) {
    extbgr_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
  } else if (JCS_EXT_BGRA == cinfo->in_color_space || JCS_EXT_BGRX == cinfo->in_color_space) {
    extbgrx_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
  } else if (JCS_EXT_ABGR == cinfo->in_color_space || JCS_EXT_XBGR == cinfo->in_color_space) {
    extxbgr_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
  } else if (JCS_EXT_ARGB == cinfo->in_color_space || JCS_EXT_XRGB == cinfo->in_color_space) {
    extxrgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
  } else {
    rgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                              num_rows);
  }
}

    char *fullpath = GENERIC_INTERNAL_CreateFullPath((char *)userdata, path);
    if (fullpath) {
        SDL_IOStream *stream = SDL_IOFromFile(fullpath, "wb");

        if (stream) {
            // FIXME: Should SDL_WriteIO use u64 now...?
            if (SDL_WriteIO(stream, source, (size_t)length) == length) {
                result = true;
            }
            SDL_CloseIO(stream);
        }
        SDL_free(fullpath);
    }

