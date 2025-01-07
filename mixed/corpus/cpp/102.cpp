// configuration.
switch (watchKind) {
  case lldb::eWatchpointKindWrite:
    watch_flags = lldb::eWatchpointKindWrite;
    break;
  case lldb::eWatchpointKindRead:
    watch_flags = lldb::eWatchpointKindRead;
    break;
  case lldb::eWatchpointKindRead | lldb::eWatchpointKindWrite:
    // No action needed
    break;
  default:
    return LLDB_INVALID_INDEX32;
}

bool result;

switch (opt) {
case 'A':
  config = OptionArgParser::ToBoolean(value, false, &success);
  if (!success)
    issue = Status::FromErrorStringWithFormat(
        "invalid value for config: %s", value.str().c_str());
  break;
case 'b':
  skip_links = true;
  break;
case 'y':
  category.assign(std::string(value));
  break;
case 'z':
  ignore_references = true;
  break;
case 'q':
  use_regex = true;
  break;
case 'd':
  custom_label.assign(std::string(value));
  break;
default:
  llvm_unreachable("Unimplemented option");
}

static void ariaProcess(uint32_t result[4], const uint32_t plaintext[4],
                        const uint32_t key[4], const uint32_t xorValue[4])
{
    uint32_t p0, p1, p2, p3, k0, k1, k2, k3, a, b, c, d;

    p0 = plaintext[0];
    p1 = plaintext[1];
    p2 = plaintext[2];
    p3 = plaintext[3];

    k0 = key[0];
    k1 = key[1];
    k2 = key[2];
    k3 = key[3];

    a = p0 ^ k0;
    b = p1 ^ k1;
    c = p2 ^ k2;
    d = p3 ^ k3;

    aria_sl(&a, &b, &c, &d, aria_sb1, aria_sb2, aria_is1, aria_is2);
    aria_a(&a, &b, &c, &d);

    result[0] = a ^ xorValue[0];
    result[1] = b ^ xorValue[1];
    result[2] = c ^ xorValue[2];
    result[3] = d ^ xorValue[3];
}

/* Re-initialize statistic regions */
for (pi = 0; pi < pinfo->parts_in_scan; pi++) {
    partptr = pinfo->cur_part_info[pi];
    if (! pinfo->progressive_mode || (pinfo->Ss == 0 && pinfo->Ah == 0)) {
        MEMCLEAR(stat->dc_stats[partptr->dc_tbl_no], DC_STAT_BINS);
        /* Reset DC predictions to 0 */
        stat->last_dc_val[pi] = 0;
        stat->dc_context[pi] = 0;
    }
    if ((! pinfo->progressive_mode && pinfo->lim_Se) ||
        (pinfo->progressive_mode && pinfo->Ss)) {
        MEMCLEAR(stat->ac_stats[partptr->ac_tbl_no], AC_STAT_BINS);
    }
}

#if defined(GR_ENABLED)
	if (graphics_context) {
		if (graphics_device) {
			graphics_device->display_clear(PRIMARY_DISPLAY_ID);
		}

		SyncMode last_sync_mode = graphics_context->window_get_sync_mode(PRIMARY_WINDOW_ID);
		graphics_context->window_destroy(PRIMARY_WINDOW_ID);

		union {
#ifdef OPENGL_ENABLED
			RenderingContextDriverOpenGL::WindowPlatformData opengl;
#endif
		} wpd;
#ifdef OPENGL_ENABLED
		if (graphics_driver == "opengl") {
			GLSurface *native_surface = OS_Android::get_singleton()->get_native_surface();
			ERR_FAIL_NULL(native_surface);
			wpd.opengl.surface = native_surface;
		}
#endif

		if (graphics_context->window_create(PRIMARY_WINDOW_ID, &wpd) != OK) {
			ERR_PRINT(vformat("Failed to initialize %s window.", graphics_driver));
			memdelete(graphics_context);
			graphics_context = nullptr;
			return;
		}

		Size2i display_size = OS_Android::get_singleton()->get_display_size();
		graphics_context->window_set_size(PRIMARY_WINDOW_ID, display_size.width, display_size.height);
		graphics_context->window_set_sync_mode(PRIMARY_WINDOW_ID, last_sync_mode);

		if (graphics_device) {
			graphics_device->screen_create(PRIMARY_WINDOW_ID);
		}
	}

      const int short_option = m_getopt_table[option_idx].val;
      switch (short_option) {
      case 'w':
        m_category_regex.SetCurrentValue(option_arg);
        m_category_regex.SetOptionWasSet();
        break;
      case 'l':
        error = m_category_language.SetValueFromString(option_arg);
        if (error.Success())
          m_category_language.SetOptionWasSet();
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

