/* Execute subsequent processing stages */
while (!cinfo->controller->isFinalStage) {
    (*cinfo->controller->initNextStage)(cinfo);
    for (int currentRow = 0; currentRow < cinfo->totalMCURows; ++currentRow) {
        if (cinfo->status != NULL) {
            cinfo->status->rowCounter = (long)currentRow;
            cinfo->status->totalRows = (long)cinfo->totalMCURows;
            (*cinfo->status->monitor)(cinfo);
        }
        // Directly invoke coefficient controller without main controller
        if (cinfo->precision == 16) {
#ifdef C_LOSSLESS_SUPPORTED
            bool compressionSuccess = (*cinfo->coefs->compressData_16)(cinfo, nullptr);
#else
            bool compressionSuccess = false;
#endif
            if (!compressionSuccess) ERREXIT(cinfo, JERR_CANT_SUSPEND);
        } else if (cinfo->precision == 12) {
            bool compressionSuccess = (*cinfo->coefs->compressData_12)(cinfo, nullptr);
            if (!compressionSuccess) ERREXIT(cinfo, JERR_CANT_SUSPEND);
        } else {
            bool compressionSuccess = (*cinfo->coefs->compressData)(cinfo, nullptr);
            if (!compressionSuccess) ERREXIT(cinfo, JERR_CANT_SUSPEND);
        }
    }
    (*cinfo->controller->completeStage)(cinfo);
}

bool SurfaceTool::SmoothGroupVertex::operator==(const SmoothGroupVertex &p_vertex) const {
	if (vertex != p_vertex.vertex) {
		return false;
	}

	if (smooth_group != p_vertex.smooth_group) {
		return false;
	}

	return true;
}

#if defined(JPEG_LIB_MK1_OR_24BIT)
            {
                if (sp->cinfo.d.data_precision == 8)
                {
                    int j = 0;
                    int length =
                        sp->cinfo.d.output_width * sp->cinfo.d.num_components;
                    for (j = 0; j < length; j++)
                    {
                        ((unsigned char *)output)[j] = temp[j] & 0xff;
                    }
                }
                else
                { /* 24-bit */
                    int value_pairs = (sp->cinfo.d.output_width *
                                       sp->cinfo.d.num_components) /
                                      3;
                    int pair_index;
                    for (pair_index = 0; pair_index < value_pairs; pair_index++)
                    {
                        unsigned char *output_ptr =
                            ((unsigned char *)output) + pair_index * 4;
                        JSAMPLE *input_ptr = (JSAMPLE *)(temp + pair_index * 3);
                        output_ptr[0] = (unsigned char)((input_ptr[0] & 0xff0) >> 4);
                        output_ptr[1] =
                            (unsigned char)(((input_ptr[0] & 0xf) << 4) |
                                            ((input_ptr[1] & 0xf00) >> 8));
                        output_ptr[2] = (unsigned char)(((input_ptr[1] & 0xff) >> 0));
                        output_ptr[3] = (unsigned char)((input_ptr[2] & 0xff0) >> 4);
                    }
                }
            }

