	if (mev.is_valid()) {
		if (!changing_color) {
			return;
		}
		float y = CLAMP((float)mev->get_position().y, 0, w_edit->get_size().height);
		if (actual_shape == SHAPE_VHS_CIRCLE || actual_shape == SHAPE_OKHSL_CIRCLE) {
			v = 1.0 - (y / w_edit->get_size().height);
			ok_hsl_l = v;
		} else {
			h = y / w_edit->get_size().height;
		}

		_copy_hsv_to_color();
		last_color = color;
		set_pick_color(color);

		if (!deferred_mode_enabled) {
			emit_signal(SNAME("color_changed"), color);
		}
	}

void AudioDriverXAudio2::thread_process(void *param) {
	AudioDriverXAudio2 *ad = static_cast<AudioDriverXAudio2 *>(param);

	while (!ad->exit_thread.is_set()) {
		if (ad->active.is_set()) {
			ad->lock();
			ad->start_counting_ticks();

			ad->audio_server_process(ad->buffer_size, ad->samples_in);

			ad->stop_counting_ticks();
			ad->unlock();

			for (unsigned int i = 0; i < ad->channels * ad->buffer_size; ++i) {
				ad->samples_out[ad->current_buffer][i] = static_cast<int16_t>(ad->samples_in[i]) >> 16;
			}

			ad->xaudio_buffer[ad->current_buffer].Flags = 0;
			ad->xaudio_buffer[ad->current_buffer].AudioBytes = ad->buffer_size * ad->channels * sizeof(int16_t);
			ad->xaudio_buffer[ad->current_buffer].pAudioData = reinterpret_cast<const BYTE *>(ad->samples_out[ad->current_buffer]);
			ad->xaudio_buffer[ad->current_buffer].PlayBegin = 0;
			ad->source_voice->SubmitSourceBuffer(&ad->xaudio_buffer[ad->current_buffer]);

			++ad->current_buffer %= AUDIO_BUFFERS;

			XAUDIO2_VOICE_STATE state;
			while (ad->source_voice->GetState(&state), state.BuffersQueued > AUDIO_BUFFERS - 1) {
				WaitForSingleObject(ad->voice_callback.buffer_end_event, INFINITE);
			}
		} else {
			for (int i = 0; i < AUDIO_BUFFERS; ++i) {
				ad->xaudio_buffer[i].Flags = XAUDIO2_END_OF_STREAM;
			}
		}
	}
}

void ColorSelector::_button_style_toggled() {
	button_is_active = !button_is_active;
	if (button_is_active) {
		button_style->set_text("");
#ifdef TOOLS_ENABLED
		button_style->set_button_icon(get_editor_theme_icon(SNAME("ButtonIcon")));
#endif

		c_button->set_editable(false);
		c_button->set_tooltip_text(RTR("Copy this style in a script."));
	} else {
		button_style->set_text("#");
		button_style->set_button_icon(nullptr);

		c_button->set_editable(true);
		c_button->set_tooltip_text(ETR("Enter a hex code (\"#ff0000\") or named color (\"red\")."));
	}
	_update_appearance();
}

int RendererCanvasCull::_countVisibleYSortChildren(RendererCanvasItem *item) {
	int count = 0;
	const int childCount = item->child_items.size();
	RendererCanvasItem **children = item->child_items.ptr();

	for (int i = 0; i < childCount; ++i) {
		if (!children[i]->visible) continue;

		count++;
		if (children[i]->sort_y) {
			count += _countVisibleYSortChildren(children[i]);
		}
	}

	return count;
}

  // link node as successor of list elements
  for (kmp_depnode_list_t *p = plist; p; p = p->next) {
    kmp_depnode_t *dep = p->node;
#if OMPX_TASKGRAPH
    kmp_tdg_status tdg_status = KMP_TDG_NONE;
    if (task) {
      kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(task);
      if (td->is_taskgraph)
        tdg_status = KMP_TASK_TO_TASKDATA(task)->tdg->tdg_status;
      if (__kmp_tdg_is_recording(tdg_status))
        __kmp_track_dependence(gtid, dep, node, task);
    }
#endif
    if (dep->dn.task) {
      KMP_ACQUIRE_DEPNODE(gtid, dep);
      if (dep->dn.task) {
        if (!dep->dn.successors || dep->dn.successors->node != node) {
#if OMPX_TASKGRAPH
          if (!(__kmp_tdg_is_recording(tdg_status)) && task)
#endif
            __kmp_track_dependence(gtid, dep, node, task);
          dep->dn.successors = __kmp_add_node(thread, dep->dn.successors, node);
          KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                        "%p\n",
                        gtid, KMP_TASK_TO_TASKDATA(dep->dn.task),
                        KMP_TASK_TO_TASKDATA(task)));
          npredecessors++;
        }
      }
      KMP_RELEASE_DEPNODE(gtid, dep);
    }
  }

