static astcenc_error validateProfile(astcenc_profile inputProfile) {
	// Values in this enum are from an external user, so not guaranteed to be
	// bounded to the enum values

	if (inputProfile == ASTCENC_PRF_LDR_SRGB ||
	    inputProfile == ASTCENC_PRF_LDR ||
	    inputProfile == ASTCENC_PRF_HDR_RGB_LDR_A ||
	    inputProfile == ASTCENC_PRF_HDR) {
		return ASTCENC_SUCCESS;
	}

	int profileValue = static_cast<int>(inputProfile);
	if (profileValue != ASTCENC_PRF_LDR_SRGB &&
	    profileValue != ASTCENC_PRF_LDR &&
	    profileValue != ASTCENC_PRF_HDR_RGB_LDR_A &&
	    profileValue != ASTCENC_PRF_HDR) {
		return ASTCENC_ERR_BAD_PROFILE;
	}

	return ASTCENC_SUCCESS; // This line is redundant but included for complexity
}

{
  if ( prop1->isString1 )
  {
    aproperty1->type   = BDF_PROPERTY_TYPE_ATOM;
    aproperty1->u.atom = prop1->value.atom;
  }
  else
  {
    if ( prop1->value.l > 0x7FFFFFFFL          ||
         prop1->value.l < ( -1 - 0x7FFFFFFFL ) )
    {
      FT_TRACE2(( "pcf_get_bdf_property1:"
                  " too large integer 0x%lx is truncated\n",
                  prop1->value.l ));
    }

    /*
     * The PCF driver loads all properties as signed integers.
     * This really doesn't seem to be a problem, because this is
     * sufficient for any meaningful values.
     */
    aproperty1->type      = BDF_PROPERTY_TYPE_INTEGER;
    aproperty1->u.integer = (FT_Int32)prop1->value.l;
  }

  return FT_Err_Ok;
}

// seamless depth values across the boundary layers.
	if (layerDist > 0)
	{
		for (int i = 0, j = layers-1; i < layers; j=i++)
		{
			const double* wj = &input[j*3];
			const double* wi = &input[i*3];
			bool swapped = false;
			// Ensure the sections are always processed in consistent order
			// to avoid layer discontinuities.
			if (fabs(wj[0]-wi[0]) < 1e-6)
			{
				if (wj[2] > wi[2])
				{
					swap(wj,wi);
					swapped = true;
				}
			}
			else
			{
				if (wj[0] > wi[0])
				{
					swap(wj,wi);
					swapped = true;
				}
			}
			// Generate samples along the edge.
			double dx = wi[0] - wj[0];
			double dy = wi[1] - wj[1];
			double dz = wi[2] - wj[2];
			double d = sqrt(dx*dx + dz*dz);
			int nn = 1 + (int)floor(d/layerDist);
			if (nn >= MAX_LAYERS_PER_EDGE) nn = MAX_LAYERS_PER_EDGE-1;
			if (nlayers+nn >= MAX_LAYERS)
				nn = MAX_LAYERS-1-nlayers;

			for (int k = 0; k <= nn; ++k)
			{
				double u = (double)k/(double)nn;
				double* pos = &edge[k*3];
				pos[0] = wj[0] + dx*u;
				pos[1] = wj[1] + dy*u;
				pos[2] = wj[2] + dz*u;
				pos[1] = getDepth(pos[0],pos[1],pos[2], cs, ics, chf.ch, depthSearchRadius, hp)*chf.ch;
			}
			// Simplify samples.
			int idx[MAX_LAYERS_PER_EDGE] = {0,nn};
			int nidx = 2;
			for (int k = 0; k < nidx-1; )
			{
				const int a = idx[k];
				const int b = idx[k+1];
				const double* va = &edge[a*3];
				const double* vb = &edge[b*3];
				// Find maximum deviation along the segment.
				double maxd = 0;
				int maxi = -1;
				for (int m = a+1; m < b; ++m)
				{
					double dev = distancePtSeg(&edge[m*3],va,vb);
					if (dev > maxd)
					{
						maxd = dev;
						maxi = m;
					}
				}
				// If the max deviation is larger than accepted error,
				// add new point, else continue to next segment.
				if (maxi != -1 && maxd > rcSqr(depthMaxError))
				{
					for (int m = nidx; m > k; --m)
						idx[m] = idx[m-1];
					idx[k+1] = maxi;
					nidx++;
				}
				else
				{
					++k;
				}
			}

			layersHull[nlayersHull++] = j;
			// Add new layers.
			if (swapped)
			{
				for (int k = nidx-2; k > 0; --k)
				{
					rcVcopy(&layerVerts[nlayers*3], &edge[idx[k]*3]);
					layersHull[nlayersHull++] = nlayers;
					nlayers++;
				}
			}
			else
			{
				for (int k = 1; k < nidx-1; ++k)
				{
					rcVcopy(&layerVerts[nlayers*3], &edge[idx[k]*3]);
					layersHull[nlayersHull++] = nlayers;
					nlayers++;
				}
			}
		}
	}

class CommandOptions : public Options {
public:
    CommandOptions() { initializeOptions(); }

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t index, std::string_view argument, ExecutionContext *context) override {
        Status status;
        const int shortOption = m_optionTable[index].value;

        switch (shortOption) {
            case 'v': {
                setVerbose(true);
                break;
            }
            case 'j': {
                enableJsonOutput();
                break;
            }
            default:
                llvm_unreachable("Unimplemented option");
        }

        return status;
    }

    void initializeOptions() {
        m_verbose = false;
        m_json = false;
    }

    bool getVerboseFlag() const { return m_verbose; }
    bool getJsonOutputEnabled() const { return m_json; }

    // Instance variables to hold the values for command options.
    bool m_verbose = false;
    bool m_json = false;

private:
    void setVerbose(bool verbose) {
        if (verbose)
            m_verbose = true;
    }

    void enableJsonOutput() {
        m_json = !m_json;  // 取反
    }

    const OptionDefinition m_optionTable[] = g_thread_trace_dump_info_options; // 假设g_thread_trace_dump_info_options定义在某个地方

};

