// Narrow high and low as much as possible.
				for (int i = 0; i < iterations; ++i) {
					float medianValue = (minBound + maxBound) * 0.5;

					const Vector2 interpolatedPoint = start.bezier_interpolate(outHandle, inHandle, endValue, medianValue);

					if (interpolatedPoint.x > targetX) {
						maxBound = medianValue;
					} else {
						minBound = medianValue;
					}
				}

/// Check if two source locations originate from the same file.
static bool AreLocationsFromSameFile(SourceManager &SM, SourceLocation Loc1,
                                     SourceLocation Loc2) {
  while (Loc2.isMacroID())
    Loc2 = SM.getImmediateMacroCallerLoc(Loc2);

  const FileEntry *File1 = SM.getFileEntryForID(SM.getFileID(Loc1));
  if (!File1)
    return false;

  if (SM.isWrittenInSameFile(SourceLocation(), Loc2))
    return true;

  const FileEntry *File2 = SM.getFileEntryForID(SM.getFileID(Loc2));
  bool sameFile = (File1 == File2);

  if (sameFile && !SM.isWrittenInMainFile(Loc1))
    return false;

  return sameFile;
}

void EditorLocaleDialog::onOkClicked() {
	if (!edit_filters->is_pressed()) {
		String locale;
		if (lang_code->get_text().is_empty()) {
			return; // Language code is required.
		}
		locale = lang_code->get_text();

		if (!script_code->get_text().is_empty()) {
			locale += "_" + script_code->get_text();
		}

		bool hasCountryCode = !country_code->get_text().is_empty();
		if (hasCountryCode) {
			locale += "_" + country_code->get_text();
		}

		bool hasVariantCode = !variant_code->get_text().is_empty();
		if (hasVariantCode) {
			locale += "_" + variant_code->get_text();
		}

		emit_signal(SNAME("locale_selected"), TranslationServer::get_singleton()->standardize_locale(locale));
		hide();
	}
}

	vfloatacc samec_errorsumv = vfloatacc::zero();

	for (unsigned int partition = 0; partition < partition_count; partition++)
	{
		const uint8_t *texel_indexes = pi.texels_of_partition[partition];

		processed_line4 l_uncor = uncor_plines[partition];
		processed_line4 l_samec = samec_plines[partition];

		unsigned int texel_count = pi.partition_texel_count[partition];
		promise(texel_count > 0);

		// Vectorize some useful scalar inputs
		vfloat l_uncor_bs0(l_uncor.bs.lane<0>());
		vfloat l_uncor_bs1(l_uncor.bs.lane<1>());
		vfloat l_uncor_bs2(l_uncor.bs.lane<2>());
		vfloat l_uncor_bs3(l_uncor.bs.lane<3>());

		vfloat l_uncor_amod0(l_uncor.amod.lane<0>());
		vfloat l_uncor_amod1(l_uncor.amod.lane<1>());
		vfloat l_uncor_amod2(l_uncor.amod.lane<2>());
		vfloat l_uncor_amod3(l_uncor.amod.lane<3>());

		vfloat l_samec_bs0(l_samec.bs.lane<0>());
		vfloat l_samec_bs1(l_samec.bs.lane<1>());
		vfloat l_samec_bs2(l_samec.bs.lane<2>());
		vfloat l_samec_bs3(l_samec.bs.lane<3>());

		assert(all(l_samec.amod == vfloat4(0.0f)));

		vfloat uncor_loparamv(1e10f);
		vfloat uncor_hiparamv(-1e10f);

		vfloat ew_r(blk.channel_weight.lane<0>());
		vfloat ew_g(blk.channel_weight.lane<1>());
		vfloat ew_b(blk.channel_weight.lane<2>());
		vfloat ew_a(blk.channel_weight.lane<3>());

		// This implementation over-shoots, but this is safe as we initialize the texel_indexes
		// array to extend the last value. This means min/max are not impacted, but we need to mask
		// out the dummy values when we compute the line weighting.
		vint lane_ids = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vmask mask = lane_ids < vint(texel_count);
			vint texel_idxs(texel_indexes + i);

			vfloat data_r = gatherf(blk.data_r, texel_idxs);
			vfloat data_g = gatherf(blk.data_g, texel_idxs);
			vfloat data_b = gatherf(blk.data_b, texel_idxs);
			vfloat data_a = gatherf(blk.data_a, texel_idxs);

			vfloat uncor_param = (data_r * l_uncor_bs0)
			                   + (data_g * l_uncor_bs1)
			                   + (data_b * l_uncor_bs2)
			                   + (data_a * l_uncor_bs3);

			uncor_loparamv = min(uncor_param, uncor_loparamv);
			uncor_hiparamv = max(uncor_param, uncor_hiparamv);

			vfloat uncor_dist0 = (l_uncor_amod0 - data_r)
			                   + (uncor_param * l_uncor_bs0);
			vfloat uncor_dist1 = (l_uncor_amod1 - data_g)
			                   + (uncor_param * l_uncor_bs1);
			vfloat uncor_dist2 = (l_uncor_amod2 - data_b)
			                   + (uncor_param * l_uncor_bs2);
			vfloat uncor_dist3 = (l_uncor_amod3 - data_a)
			                   + (uncor_param * l_uncor_bs3);

			vfloat uncor_err = (ew_r * uncor_dist0 * uncor_dist0)
			                 + (ew_g * uncor_dist1 * uncor_dist1)
			                 + (ew_b * uncor_dist2 * uncor_dist2)
			                 + (ew_a * uncor_dist3 * uncor_dist3);

			haccumulate(uncor_errorsumv, uncor_err, mask);

			// Process samechroma data
			vfloat samec_param = (data_r * l_samec_bs0)
			                   + (data_g * l_samec_bs1)
			                   + (data_b * l_samec_bs2)
			                   + (data_a * l_samec_bs3);

			vfloat samec_dist0 = samec_param * l_samec_bs0 - data_r;
			vfloat samec_dist1 = samec_param * l_samec_bs1 - data_g;
			vfloat samec_dist2 = samec_param * l_samec_bs2 - data_b;
			vfloat samec_dist3 = samec_param * l_samec_bs3 - data_a;

			vfloat samec_err = (ew_r * samec_dist0 * samec_dist0)
			                 + (ew_g * samec_dist1 * samec_dist1)
			                 + (ew_b * samec_dist2 * samec_dist2)
			                 + (ew_a * samec_dist3 * samec_dist3);

			haccumulate(samec_errorsumv, samec_err, mask);

			lane_ids += vint(ASTCENC_SIMD_WIDTH);
		}

		// Turn very small numbers and NaNs into a small number
		float uncor_linelen = hmax_s(uncor_hiparamv) - hmin_s(uncor_loparamv);
		line_lengths[partition] = astc::max(uncor_linelen, 1e-7f);
	}

