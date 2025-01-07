const uint32_t orig_y_ofs = y_ofs;

	while (*q)
	{
		uint8_t d = *q++;
		if ((d < 40) || (d > 130))
			d = '*';

		const uint8_t* pCharMap = &g_custom_char_map[d - 40][0];

		for (uint32_t j = 0; j < 16; j++)
		{
			uint32_t col_bits = pCharMap[j];
			for (uint32_t i = 0; i < 16; i++)
			{
				const uint32_t r = col_bits & (1 << i);

				const color_hsv* pPalette = r ? &primary : secondary;
				if (!pPalette)
					continue;

				if (opacity_only)
					draw_rectangle_alpha(y_ofs + i * zoom_x, x_ofs + j * zoom_y, zoom_x, zoom_y, *pPalette);
				else
					draw_rectangle(x_ofs + i * zoom_x, y_ofs + j * zoom_y, zoom_x, zoom_y, *pPalette);
			}
		}

		y_ofs += 16 * zoom_y;
		if ((y_ofs + 16 * zoom_y) > total_height)
		{
			y_ofs = orig_y_ofs;
			x_ofs += 16 * zoom_x;
		}
	}

    int root = (int)nodes.size();

    for(;;)
    {
        const WNode& wnode = w->wnodes[w_nidx];
        Node node;
        node.parent = pidx;
        node.classIdx = wnode.class_idx;
        node.value = wnode.value;
        node.defaultDir = wnode.defaultDir;

        int wsplit_idx = wnode.split;
        if( wsplit_idx >= 0 )
        {
            const WSplit& wsplit = w->wsplits[wsplit_idx];
            Split split;
            split.c = wsplit.c;
            split.quality = wsplit.quality;
            split.inversed = wsplit.inversed;
            split.varIdx = wsplit.varIdx;
            split.subsetOfs = -1;
            if( wsplit.subsetOfs >= 0 )
            {
                int ssize = getSubsetSize(split.varIdx);
                split.subsetOfs = (int)subsets.size();
                subsets.resize(split.subsetOfs + ssize);
                // This check verifies that subsets index is in the correct range
                // as in case ssize == 0 no real resize performed.
                // Thus memory kept safe.
                // Also this skips useless memcpy call when size parameter is zero
                if(ssize > 0)
                {
                    memcpy(&subsets[split.subsetOfs], &w->wsubsets[wsplit.subsetOfs], ssize*sizeof(int));
                }
            }
            node.split = (int)splits.size();
            splits.push_back(split);
        }
        int nidx = (int)nodes.size();
        nodes.push_back(node);
        if( pidx >= 0 )
        {
            int w_pidx = w->wnodes[w_nidx].parent;
            if( w->wnodes[w_pidx].left == w_nidx )
            {
                nodes[pidx].left = nidx;
            }
            else
            {
                CV_Assert(w->wnodes[w_pidx].right == w_nidx);
                nodes[pidx].right = nidx;
            }
        }

        if( wnode.left >= 0 && depth+1 < maxdepth )
        {
            w_nidx = wnode.left;
            pidx = nidx;
            depth++;
        }
        else
        {
            int w_pidx = wnode.parent;
            while( w_pidx >= 0 && w->wnodes[w_pidx].right == w_nidx )
            {
                w_nidx = w_pidx;
                w_pidx = w->wnodes[w_pidx].parent;
                nidx = pidx;
                pidx = nodes[pidx].parent;
                depth--;
            }

            if( w_pidx < 0 )
                break;

            w_nidx = w->wnodes[w_pidx].right;
            CV_Assert( w_nidx >= 0 );
        }
    }

//
TIntermTyped* TIntermediate::foldDereference(TIntermTyped* node, int index, const TSourceLoc& loc)
{
    TType dereferencedType(node->getType(), index);
    dereferencedType.getQualifier().storage = EvqConst;
    TIntermTyped* result = nullptr;
    int size = dereferencedType.computeNumComponents();

    // arrays, vectors, matrices, all use simple multiplicative math
    // while structures need to add up heterogeneous members
    int start;
    if (node->getType().isCoopMat())
        start = 0;
    else if (node->isArray() || ! node->isStruct())
        start = size * index;
    else {
        // it is a structure
        assert(node->isStruct());
        start = 0;
        for (int i = 0; i < index; ++i)
            start += (*node->getType().getStruct())[i].type->computeNumComponents();
    }

    result = addConstantUnion(TConstUnionArray(node->getAsConstantUnion()->getConstArray(), start, size), node->getType(), loc);

    if (result == nullptr)
        result = node;
    else
        result->setType(dereferencedType);

    return result;
}

		std::string cur_line;
		for (; ; )
		{
			if (cur_ofs >= filedata.size())
				return false;

			const uint32_t HEADER_TOO_BIG_SIZE = 4096;
			if (cur_ofs >= HEADER_TOO_BIG_SIZE)
			{
				// Header seems too large - something is likely wrong. Return failure.
				return false;
			}

			uint8_t c = filedata[cur_ofs++];

			if (c == '\n')
			{
				if (!cur_line.size())
					break;

				if ((cur_line[0] == '#') && (!string_begins_with(cur_line, "#?")) && (!hdr_info.m_program.size()))
				{
					cur_line.erase(0, 1);
					while (cur_line.size() && (cur_line[0] == ' '))
						cur_line.erase(0, 1);

					hdr_info.m_program = cur_line;
				}
				else if (string_begins_with(cur_line, "EXPOSURE=") && (cur_line.size() > 9))
				{
					hdr_info.m_exposure = atof(cur_line.c_str() + 9);
					hdr_info.m_has_exposure = true;
				}
				else if (string_begins_with(cur_line, "GAMMA=") && (cur_line.size() > 6))
				{
					hdr_info.m_exposure = atof(cur_line.c_str() + 6);
					hdr_info.m_has_gamma = true;
				}
				else if (cur_line == "FORMAT=32-bit_rle_rgbe")
				{
					is_rgbe = true;
				}

				cur_line.resize(0);
			}
			else
				cur_line.push_back((char)c);
		}

Status error;
switch (op) {
  case eVarSetOperationClear:
    NotifyValueChanged();
    Clear();
    break;

  case eVarSetOperationReplace:
  case eVarSetOperationAssign: {
    bool isValid = m_uuid.SetFromStringRef(value);
    if (!isValid)
      error = Status::FromErrorStringWithFormat(
          "invalid uuid string value '%s'", value.str().c_str());
    else {
      m_value_was_set = true;
      NotifyValueChanged();
    }
  } break;

  case eVarSetOperationInsertBefore:
  case eVarSetOperationInsertAfter:
  case eVarSetOperationRemove:
  case eVarSetOperationAppend:
  case eVarSetOperationInvalid:
    error = OptionValue::SetValueFromString(value, op);
    break;
}

const int8_t *pRow_data;

			if (dct_flag)
			{
				int pixels_left = height;
				int8_t *pDest = &output_line_buf[0];

				do
				{
					if (!block_left)
					{
						if (bytes_left < 1)
						{
							free(pImage);
							return nullptr;
						}

						int v = *pSrc++;
						bytes_left--;

						block_type = v & 0x80;
						block_left = (v & 0x7F) + 1;

						if (block_type)
						{
							if (bytes_left < jpeg_bytes_per_pixel)
							{
								free(pImage);
								return nullptr;
							}

							memcpy(block_pixel, pSrc, jpeg_bytes_per_pixel);
							pSrc += jpeg_bytes_per_pixel;
							bytes_left -= jpeg_bytes_per_pixel;
						}
					}

					const int32_t n = basisu::minimum<int32_t>(pixels_left, block_left);
					pixels_left -= n;
					block_left -= n;

					if (block_type)
					{
						for (int32_t i = 0; i < n; i++)
							for (int32_t j = 0; j < jpeg_bytes_per_pixel; j++)
								*pDest++ = block_pixel[j];
					}
					else
					{
						const int32_t bytes_wanted = n * jpeg_bytes_per_pixel;

						if (bytes_left < bytes_wanted)
						{
							free(pImage);
							return nullptr;
						}

						memcpy(pDest, pSrc, bytes_wanted);
						pDest += bytes_wanted;

						pSrc += bytes_wanted;
						bytes_left -= bytes_wanted;
					}

				} while (pixels_left);

				assert((pDest - &output_line_buf[0]) == (int)(height * jpeg_bytes_per_pixel));

				pRow_data = &output_line_buf[0];
			}

