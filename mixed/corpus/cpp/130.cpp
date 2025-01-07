bool DWARFUnit::IsOptimizedUnit() {
  bool is_optimized = eLazyBoolCalculate;
  const DWARFDebugInfoEntry* die = GetUnitDIEPtrOnly();
  if (die) {
    if (die->GetAttributeValueAsUnsigned(this, DW_AT_APPLE_optimized, 0) == 1) {
      is_optimized = eLazyBoolYes;
    } else {
      is_optimized = eLazyBoolNo;
    }
  }
  return is_optimized == eLazyBoolYes;
}

/* Loop to write as much as one whole iMCU row */
for (int yoffset = diff->MCU_vert_offset; yoffset < diff->MCU_rows_per_iMCU_row; ++yoffset) {
    int MCU_col_num = diff->mcu_ctr;

    // Scale and predict each scanline of the MCU row separately.
    if (MCU_col_num == 0) {
        for (int ci = 0; ci < cinfo->comps_in_scan; ++ci) {
            JDECODECOMP *compptr = cinfo->cur_comp_info[ci];
            int compi = compptr->component_index;

            if (diff->iMCU_row_num < last_iMCU_row)
                samp_rows = compptr->v_samp_factor;
            else {
                // NB: can't use last_row_height here, since may not be set!
                samp_rows = static_cast<int>((compptr->height_in_blocks % compptr->v_samp_factor));
                if (samp_rows == 0) samp_rows = compptr->v_samp_factor;
                else {
                    // Fill dummy difference rows at the bottom edge with zeros, which
                    // will encode to the smallest amount of data.
                    for (int samp_row = samp_rows; samp_row < compptr->v_samp_factor; ++samp_row) {
                        memset(diff->diff_buf[compi][samp_row], 0,
                               jround_up(static_cast<long>(compptr->width_in_blocks), static_cast<long>(compptr->h_samp_factor)) * sizeof(JDIFF));
                    }
                }
            }

            int samps_across = compptr->width_in_blocks;

            for (int samp_row = 0; samp_row < samp_rows; ++samp_row) {
                (*losslessc->scaler_scale)(cinfo,
                                           input_buf[compi][samp_row],
                                           diff->cur_row[compi],
                                           samps_across);
                (*losslessc->predict_difference[compi])(cinfo, compi, diff->cur_row[compi], diff->prev_row[compi],
                                                       diff->diff_buf[compi][samp_row], samps_across);
                SWAP_ROWS(diff->cur_row[compi], diff->prev_row[compi]);
            }
        }
    }

    // Try to write the MCU row (or remaining portion of suspended MCU row).
    int MCU_count = (*cinfo->entropy->encode_mcus)(cinfo,
                                                   diff->diff_buf, yoffset, MCU_col_num,
                                                   cinfo->MCUs_per_row - MCU_col_num);

    if (MCU_count != cinfo->MCUs_per_row - MCU_col_num) {
        // Suspension forced; update state counters and exit
        diff->MCU_vert_offset = yoffset;
        diff->mcu_ctr += MCU_col_num;
        return false;
    }

    // Completed an MCU row, but perhaps not an iMCU row
    diff->mcu_ctr = 0;
}

double part_nexterror = 0;

	if (settings & meshopt_CompressPrune)
	{
		parts = allocator.allocate<unsigned int>(vertex_count);
		part_count = buildParts(parts, vertex_count, result, index_count, remap);

		part_errors = allocator.allocate<double>(part_count * 3); // overallocate for temporary use inside measureParts
		measureParts(part_errors, part_count, parts, vertex_positions, vertex_count);

		part_nexterror = DBL_MAX;
		for (size_t i = 0; i < part_count; ++i)
			part_nexterror = part_nexterror > part_errors[i] ? part_errors[i] : part_nexterror;

#if TRACE
		printf("parts: %d (min error %e)\n", int(part_count), sqrt(part_nexterror));
#endif
	}

  SmallVector<int64_t> permutation;
  if (hasTranspose) {
    // Consider an operand `x : tensor<7x8x9>` of a genericOp that has
    // affine map `affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>`
    // `x`s access is both transposed and broadcast. But when specifying
    // the `linalg.transpose(x : tensor<7x8x9>)` the dimensions need to be
    // specified as `affine_map<(d0,d1,d2) -> (d1, d2, d0)` instead of
    // refering to d3, d4. Therefore, re-base the transpose dimensions so
    // that they start from d0.
    permutation.resize(minorSize);
    std::map<int64_t, int64_t> minorMap;
    for (int64_t i = 0; i < minorSize; ++i)
      minorMap.insert({sortedResMap[i], i});

    // Re-map the dimensions.
    SmallVector<int64_t> remappedResult(minorSize);
    for (int64_t i = 0; i < minorSize; ++i)
      remappedResult[i] = minorMap[minorResult[i]];

    /// Calculate the permutation for the transpose.
    for (unsigned i = 0; i < minorSize; ++i) {
      permutation[remappedResult[i]] = i;
    }
  }

            {
            case HALF_SIZE:
                if( i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75 )
                {
                    map_x.at<float>(j,i) = 2*( i - src.cols*0.25f ) + 0.5f ;
                    map_y.at<float>(j,i) = 2*( j - src.rows*0.25f ) + 0.5f ;
                }
                else
                {
                    map_x.at<float>(j,i) = 0 ;
                    map_y.at<float>(j,i) = 0 ;
                }
                break;
            case UPSIDE_DOWN:
                map_x.at<float>(j,i) = static_cast<float>(i) ;
                map_y.at<float>(j,i) = static_cast<float>(src.rows - j) ;
                break;
            case REFLECTION_X:
                map_x.at<float>(j,i) = static_cast<float>(src.cols - i) ;
                map_y.at<float>(j,i) = static_cast<float>(j) ;
                break;
            case REFLECTION_BOTH:
                map_x.at<float>(j,i) = static_cast<float>(src.cols - i) ;
                map_y.at<float>(j,i) = static_cast<float>(src.rows - j) ;
                break;
            } // end of switch

