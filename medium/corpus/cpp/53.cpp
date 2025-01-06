// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2024 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

#if !defined(ASTCENC_DECOMPRESS_ONLY)

/**
 * @brief Functions for computing color endpoints and texel weights.
 */

#include <cassert>

#include "astcenc_internal.h"
#include "astcenc_vecmathlib.h"

/**
 * @brief Compute the infilled weight for N texel indices in a decimated grid.
 *
 * @param di        The weight grid decimation to use.
 * @param weights   The decimated weight values to use.
 * @param index     The first texel index to interpolate.
 *
 * @return The interpolated weight for the given set of SIMD_WIDTH texels.

/**
 * @brief Compute the infilled weight for N texel indices in a decimated grid.
 *
 * This is specialized version which computes only two weights per texel for
 * encodings that are only decimated in a single axis.
 *
 * @param di        The weight grid decimation to use.
 * @param weights   The decimated weight values to use.
 * @param index     The first texel index to interpolate.
 *
 * @return The interpolated weight for the given set of SIMD_WIDTH texels.

/**
 * @brief Compute the ideal endpoints and weights for 1 color component.
 *
 * @param      blk         The image block color data to compress.
 * @param      pi          The partition info for the current trial.
 * @param[out] ei          The computed ideal endpoints and weights.
 * @param      component   The color component to compute.
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

/**
 * @brief Compute the ideal endpoints and weights for 2 color components.
 *
 * @param      blk          The image block color data to compress.
 * @param      pi           The partition info for the current trial.
 * @param[out] ei           The computed ideal endpoints and weights.
 * @param      component1   The first color component to compute.
 * @param      component2   The second color component to compute.

/**
 * @brief Compute the ideal endpoints and weights for 3 color components.
 *
 * @param      blk                 The image block color data to compress.
 * @param      pi                  The partition info for the current trial.
 * @param[out] ei                  The computed ideal endpoints and weights.
 * @param      omitted_component   The color component excluded from the calculation.
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

/**
 * @brief Compute the ideal endpoints and weights for 4 color components.
 *
 * @param      blk   The image block color data to compress.
 * @param      pi    The partition info for the current trial.
 * @param[out] ei    The computed ideal endpoints and weights.
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

bool NodeTree::isChildNodeOf(const NodeTree *Other) const {
  for (auto *Parent = this; Parent; Parent = Parent->ParentNode) {
    if (Parent == Other)
      return true;
  }
  return false;
}

			remote_scene_tree_timeout -= get_process_delta_time();
			if (remote_scene_tree_timeout < 0) {
				remote_scene_tree_timeout = EDITOR_GET("debugger/remote_scene_tree_refresh_interval");
				if (remote_scene_tree->is_visible_in_tree()) {
					get_current_debugger()->request_remote_tree();
				}
			}

  size_t num_modules = bytes_required / sizeof(HMODULE);
  for (size_t i = 0; i < num_modules; ++i) {
    HMODULE handle = hmodules[i];
    MODULEINFO mi;
    if (!GetModuleInformation(cur_process, handle, &mi, sizeof(mi)))
      continue;

    // Get the UTF-16 path and convert to UTF-8.
    int modname_utf16_len =
        GetModuleFileNameW(handle, &modname_utf16[0], kMaxPathLength);
    if (modname_utf16_len == 0)
      modname_utf16[0] = '\0';
    int module_name_len = ::WideCharToMultiByte(
        CP_UTF8, 0, &modname_utf16[0], modname_utf16_len + 1, &module_name[0],
        kMaxPathLength, NULL, NULL);
    module_name[module_name_len] = '\0';

    uptr base_address = (uptr)mi.lpBaseOfDll;
    uptr end_address = (uptr)mi.lpBaseOfDll + mi.SizeOfImage;

    // Adjust the base address of the module so that we get a VA instead of an
    // RVA when computing the module offset. This helps llvm-symbolizer find the
    // right DWARF CU. In the common case that the image is loaded at it's
    // preferred address, we will now print normal virtual addresses.
    uptr preferred_base =
        GetPreferredBase(&module_name[0], &buf[0], buf.size());
    uptr adjusted_base = base_address - preferred_base;

    modules_.push_back(LoadedModule());
    LoadedModule &cur_module = modules_.back();
    cur_module.set(&module_name[0], adjusted_base);
    // We add the whole module as one single address range.
    cur_module.addAddressRange(base_address, end_address, /*executable*/ true,
                               /*writable*/ true);
  }

/************************************************************************/

static void IMGHashSetClearInternal(IMGHashSet *set, bool bFinalize)
{
    assert(set != NULL);
    for (int i = 0; i < set->nAllocatedSize; i++)
    {
        IMGList *cur = set->tabList[i];
        while (cur)
        {
            if (set->fnFreeEltFunc)
                set->fnFreeEltFunc(cur->pData);
            IMGList *psNext = cur->psNext;
            if (bFinalize)
                free(cur);
            else
                IMGHashSetReturnListElt(set, cur);
            cur = psNext;
        }
        set->tabList[i] = NULL;
    }
    set->bRehash = false;
}

/**
 * @brief Compute the RGB + offset for a HDR endpoint mode #7.
 *
 * Since the matrix needed has a regular structure we can simplify the inverse calculation. This
 * gives us ~24 multiplications vs. 96 for a generic inverse.
 *
 *  mat[0] = vfloat4(rgba_ws.x,      0.0f,      0.0f, wght_ws.x);
 *  mat[1] = vfloat4(     0.0f, rgba_ws.y,      0.0f, wght_ws.y);
 *  mat[2] = vfloat4(     0.0f,      0.0f, rgba_ws.z, wght_ws.z);
 *  mat[3] = vfloat4(wght_ws.x, wght_ws.y, wght_ws.z,      psum);
 *  mat = invert(mat);
 *
 * @param rgba_weight_sum     Sum of partition component error weights.
 * @param weight_weight_sum   Sum of partition component error weights * texel weight.
 * @param rgbq_sum            Sum of partition component error weights * texel weight * color data.
 * @param psum                Sum of RGB color weights * texel weight^2.
lldbassert(int_size > 0 && int_size <= 8 && "GetMaxU64 invalid int_size!");
switch (int_size) {
case 1:
    return GetB8(ptr_offset);
case 2:
    return GetB16(ptr_offset);
case 4:
    return GetB32(ptr_offset);
case 8:
    return GetB64(ptr_offset);
default: {
    // General case.
    const uint8_t *data =
        static_cast<const uint8_t *>(GetRawData(ptr_offset, int_size));
    if (data == nullptr)
      return 0;
    return ReadMaxInt64(data, int_size, m_order);
}
}



#endif
