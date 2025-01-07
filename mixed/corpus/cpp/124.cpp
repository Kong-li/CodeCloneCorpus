if(config&TREE_CONFIG_DATA_SIZE_IS_64_BIT) {
    if(size<8*node->dataCount) {
        *pErrorStatus=U_INVALID_DATA_ERROR;
        return -1;
    }
    node->data64=(const uint64_t *)pData;
    node->startValue=node->data64[0];
    size=(int64_t)sizeof(NodeHeader)+4*node->indexCount+8*node->dataCount;
} else {
    if(size<4*node->dataCount) {
        *pErrorStatus=U_INVALID_DATA_ERROR;
        return -1;
    }

    /* the "data32" data is used via the index pointer */
    node->data64=nullptr;
    node->startValue=node->index[node->indexCount];
    size=(int64_t)sizeof(NodeHeader)+4*node->indexCount+4*node->dataCount;
}

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_) {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        const auto &input = inputs[0], &scale = inputs[1], &bias = inputs[2];
        auto &output = outputs[0];

        const auto input_shape = shape(input);
        size_t N = input_shape[0], C = input_shape[1];
        size_t num_groups = this->num_groups;
        size_t channels_per_group = C / num_groups;
        size_t loops = N * num_groups, norm_size = static_cast<size_t>(total(input_shape, 2)) * channels_per_group;
        float inv_norm_size = 1.f / norm_size;

        // no fp16 support
        if (input.depth() == CV_16F) {
            return false;
        }

        String base_opts = format(" -DT=float -DT4=float4 -Dconvert_T=convert_float4");

        // Calculate mean
        UMat one = UMat::ones(norm_size, 1, CV_32F);
        UMat mean = UMat(loops, 1, CV_32F);
        UMat mean_square = UMat(loops, 1, CV_32F);
        UMat tmp = UMat(loops, norm_size, CV_32F);
        bool ret = ocl4dnn::ocl4dnnGEMV<float>(ocl4dnn::CblasNoTrans, loops, norm_size, inv_norm_size,
                                               input, 0, one, 0, 0.f, mean, 0);
        if (!ret) {
            return false;
        }
        // Calculate mean_square
        int num_vector = (norm_size % 8 == 0) ? 8 : ((norm_size % 4 == 0) ? 4 : 1);
        size_t global[] = {loops, static_cast<size_t>(norm_size / num_vector)};
        String build_opt = format(" -DNUM=%d", num_vector) + base_opts;
        String mean_square_kernel_name = format("calc_mean%d", num_vector);
        ocl::Kernel mean_square_kernel(mean_square_kernel_name.c_str(), ocl::dnn::mvn_oclsrc, build_opt + " -DKERNEL_MEAN");
        if (mean_square_kernel.empty()) {
            return false;
        }
        mean_square_kernel.set(0, ocl::KernelArg::PtrReadOnly(input));
        mean_square_kernel.set(1, (int)loops);
        mean_square_kernel.set(2, (int)norm_size);
        mean_square_kernel.set(3, ocl::KernelArg::PtrReadOnly(mean));
        mean_square_kernel.set(4, ocl::KernelArg::PtrWriteOnly(tmp));
        ret = mean_square_kernel.run(2, global, NULL, false);
        if (!ret) {
            return false;
        }
        ret = ocl4dnn::ocl4dnnGEMV<float>(ocl4dnn::CblasNoTrans, loops, norm_size, inv_norm_size,
                                          tmp, 0, one, 0, 0.f, mean_square, 0);
        if (!ret) {
            return false;
        }
        // Calculate group norm: output = scale * (x - mean) / sqrt(var + eps) + bias
        String mvn_group_kernel_name = format("mvn_group%d", num_vector);
        build_opt += " -DNORM_VARIANCE -DKERNEL_MVN_GROUP";
        ocl::Kernel mvn_group_kernel(mvn_group_kernel_name.c_str(), ocl::dnn::mvn_oclsrc, build_opt);
        if (mvn_group_kernel.empty()) {
            return false;
        }
        mvn_group_kernel.set(0, ocl::KernelArg::PtrReadOnly(input));
        mvn_group_kernel.set(1, (int)loops);
        mvn_group_kernel.set(2, (int)norm_size);
        mvn_group_kernel.set(3, (float)epsilon);
        mvn_group_kernel.set(4, ocl::KernelArg::PtrReadOnly(mean));
        mvn_group_kernel.set(5, ocl::KernelArg::PtrReadOnly(mean_square));
        mvn_group_kernel.set(6, ocl::KernelArg::PtrReadOnly(scale));
        mvn_group_kernel.set(7, ocl::KernelArg::PtrReadOnly(bias));
        mvn_group_kernel.set(8, (int)C);
        mvn_group_kernel.set(9, (int)num_groups);
        mvn_group_kernel.set(10, (float)0.f);
        mvn_group_kernel.set(11, ocl::KernelArg::PtrWriteOnly(output));
        ret = mvn_group_kernel.run(2, global, NULL, false);
        if (!ret) {
            return false;
        }

        return true;
        }

for (i = 0; i < aff->n_eq; ++i) {
		if (isl_basic_set_eq_is_stride(aff, i)) {
			k = isl_basic_map_alloc_equality(bmap);
			if (k < 0)
				continue;
			isl_seq_clr(bmap->eq[k], 1 + nparam);
			isl_seq_cpy(bmap->eq[k] + 1 + nparam + d,
					aff->eq[i] + 1 + nparam, d);
			isl_int_set_si(bmap->eq[k][1 + total + aff->n_div], 0);
			isl_seq_neg(bmap->eq[k] + 1 + nparam,
					aff->eq[i] + 1 + nparam, d);
			isl_seq_cpy(bmap->eq[k] + 1 + nparam + 2 * d,
					aff->eq[i] + 1 + nparam + d, aff->n_div);
		}
	}

static llvm::ArrayRef<const char *> GetCompatibleArchs(ArchSpec::Core core) {
  switch (core) {
  default:
    [[fallthrough]];
  case ArchSpec::eCore_arm_arm64e: {
    static const char *g_arm64e_compatible_archs[] = {
        "arm64e",    "arm64",    "armv7",    "armv7f",   "armv7k",   "armv7s",
        "armv7m",    "armv7em",  "armv6m",   "armv6",    "armv5",    "armv4",
        "arm",       "thumbv7",  "thumbv7f", "thumbv7k", "thumbv7s", "thumbv7m",
        "thumbv7em", "thumbv6m", "thumbv6",  "thumbv5",  "thumbv4t", "thumb",
    };
    return {g_arm64e_compatible_archs};
  }
  case ArchSpec::eCore_arm_arm64: {
    static const char *g_arm64_compatible_archs[] = {
        "arm64",    "armv7",    "armv7f",   "armv7k",   "armv7s",   "armv7m",
        "armv7em",  "armv6m",   "armv6",    "armv5",    "armv4",    "arm",
        "thumbv7",  "thumbv7f", "thumbv7k", "thumbv7s", "thumbv7m", "thumbv7em",
        "thumbv6m", "thumbv6",  "thumbv5",  "thumbv4t", "thumb",
    };
    return {g_arm64_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7: {
    static const char *g_armv7_compatible_archs[] = {
        "armv7",   "armv6m",   "armv6",   "armv5",   "armv4",    "arm",
        "thumbv7", "thumbv6m", "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7f: {
    static const char *g_armv7f_compatible_archs[] = {
        "armv7f",  "armv7",   "armv6m",   "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7f", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7f_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7k: {
    static const char *g_armv7k_compatible_archs[] = {
        "armv7k",  "armv7",   "armv6m",   "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7k", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7k_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7s: {
    static const char *g_armv7s_compatible_archs[] = {
        "armv7s",  "armv7",   "armv6m",   "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7s", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7s_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7m: {
    static const char *g_armv7m_compatible_archs[] = {
        "armv7m",  "armv7",   "armv6m",   "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7m", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7m_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7em: {
    static const char *g_armv7em_compatible_archs[] = {
        "armv7em", "armv7",   "armv6m",    "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7em", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t",  "thumb",
    };
    return {g_armv7em_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv6m: {
    static const char *g_armv6m_compatible_archs[] = {
        "armv6m",   "armv6",   "armv5",   "armv4",    "arm",
        "thumbv6m", "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv6m_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv6: {
    static const char *g_armv6_compatible_archs[] = {
        "armv6",   "armv5",   "armv4",    "arm",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv6_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv5: {
    static const char *g_armv5_compatible_archs[] = {
        "armv5", "armv4", "arm", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv5_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv4: {
    static const char *g_armv4_compatible_archs[] = {
        "armv4",
        "arm",
        "thumbv4t",
        "thumb",
    };
    return {g_armv4_compatible_archs};
  }
  }
  return {};
}

