    {
        switch (property_id) {
            case CAP_PROP_FRAME_WIDTH:
                desiredWidth = value;
                settingWidth = true;
                if (settingWidth && settingHeight) {
                    setWidthHeight();
                    settingWidth = false;
                    settingHeight = false;
                }
                return true;
            case CAP_PROP_FRAME_HEIGHT:
                desiredHeight = value;
                settingHeight = true;
                if (settingWidth && settingHeight) {
                    setWidthHeight();
                    settingWidth = false;
                    settingHeight = false;
                }
                return true;
            case CAP_PROP_FOURCC:
                {
                    uint32_t newFourCC = cvRound(value);
                    if (fourCC == newFourCC) {
                        return true;
                    } else {
                        switch (newFourCC) {
                            case FOURCC_BGR:
                            case FOURCC_RGB:
                            case FOURCC_BGRA:
                            case FOURCC_RGBA:
                            case FOURCC_GRAY:
                                fourCC = newFourCC;
                                return true;
                            case FOURCC_YV12:
                                if (colorFormat == COLOR_FormatYUV420Planar) {
                                    fourCC = newFourCC;
                                    return true;
                                } else {
                                    LOGE("Unsupported FOURCC conversion COLOR_FormatYUV420SemiPlanar -> COLOR_FormatYUV420Planar");
                                    return false;
                                }
                            case FOURCC_NV21:
                                if (colorFormat == COLOR_FormatYUV420SemiPlanar) {
                                    fourCC = newFourCC;
                                    return true;
                                } else {
                                    LOGE("Unsupported FOURCC conversion COLOR_FormatYUV420Planar -> COLOR_FormatYUV420SemiPlanar");
                                    return false;
                                }
                            default:
                                LOGE("Unsupported FOURCC value: %d\n", fourCC);
                                return false;
                        }
                    }
                }
            case CAP_PROP_AUTO_EXPOSURE:
                aeMode = (value != 0) ? ACAMERA_CONTROL_AE_MODE_ON : ACAMERA_CONTROL_AE_MODE_OFF;
                if (isOpened()) {
                    return submitRequest(ACaptureRequest_setEntry_u8, ACAMERA_CONTROL_AE_MODE, aeMode);
                }
                return true;
            case CAP_PROP_EXPOSURE:
                if (isOpened() && exposureRange.Supported()) {
                    exposureTime = exposureRange.clamp(static_cast<int64_t>(value));
                    LOGI("Setting CAP_PROP_EXPOSURE will have no effect unless CAP_PROP_AUTO_EXPOSURE is off");
                    return submitRequest(ACaptureRequest_setEntry_i64, ACAMERA_SENSOR_EXPOSURE_TIME, exposureTime);
                }
                return false;
            case CAP_PROP_ISO_SPEED:
                if (isOpened() && sensitivityRange.Supported()) {
                    sensitivity = sensitivityRange.clamp(static_cast<int32_t>(value));
                    LOGI("Setting CAP_PROP_ISO_SPEED will have no effect unless CAP_PROP_AUTO_EXPOSURE is off");
                    return submitRequest(ACaptureRequest_setEntry_i32, ACAMERA_SENSOR_SENSITIVITY, sensitivity);
                }
                return false;
            case CAP_PROP_ANDROID_DEVICE_TORCH:
                flashMode = (value != 0) ? ACAMERA_FLASH_MODE_TORCH : ACAMERA_FLASH_MODE_OFF;
                if (isOpened()) {
                    return submitRequest(ACaptureRequest_setEntry_u8, ACAMERA_FLASH_MODE, flashMode);
                }
                return true;
            default:
                break;
        }
        return false;
    }

                    _JtErr.rowRange(eofs, eofs + 6) += Je.t()*err;

                    if( recomputeIntrinsics )
                    {
                        _JtJ(Rect(iofs, iofs, NINTRINSIC, NINTRINSIC)) += Ji.t()*Ji;
                        _JtJ(Rect(iofs, eofs, NINTRINSIC, 6)) += Je.t()*Ji;
                        if( k == 1 )
                        {
                            _JtJ(Rect(iofs, 0, NINTRINSIC, 6)) += J_LR.t()*Ji;
                        }
                        _JtErr.rowRange(iofs, iofs + NINTRINSIC) += Ji.t()*err;
                    }

{
  std::string getProgramPath() {
    const int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
    char buffer[4096] = {0};
    size_t length = sizeof(buffer) - 1;
    if (sysctl(mib, 4, buffer, &length, nullptr, 0) < 0)
      return std::string{};
    return std::string(buffer);
  }

  size_t getMemoryUsage() {
    return 0;
  }
}

// Parse any Product ID values that we can get
  for (uint32_t j = 0; j < entries_count; j++) {
    if (!product_infos[j].IDValid()) {
      DataExtractor data; // Load command data
      if (!ExtractMachInfo(product_infos[j].address, &product_infos[j].header,
                           &data))
        continue;

      ProcessLoadData(data, product_infos[j], nullptr);

      if (product_infos[j].header.filetype == llvm::MachO::MH_BUNDLE)
        bundle_idx = j;
    }
  }

// addresses.
void DynamicLoaderMacOSXDYLD::DoInitialImageFetch() {
  if (m_dyld_all_image_infos_addr == LLDB_INVALID_ADDRESS) {
    // Check the image info addr as it might point to the mach header for dyld,
    // or it might point to the dyld_all_image_infos struct
    const addr_t shlib_addr = m_process->GetImageInfoAddress();
    if (shlib_addr != LLDB_INVALID_ADDRESS) {
      ByteOrder byte_order =
          m_process->GetTarget().GetArchitecture().GetByteOrder();
      uint8_t buf[4];
      DataExtractor data(buf, sizeof(buf), byte_order, 4);
      Status error;
      if (m_process->ReadMemory(shlib_addr, buf, 4, error) == 4) {
        lldb::offset_t offset = 0;
        uint32_t magic = data.GetU32(&offset);
        switch (magic) {
        case llvm::MachO::MH_MAGIC:
        case llvm::MachO::MH_MAGIC_64:
        case llvm::MachO::MH_CIGAM:
        case llvm::MachO::MH_CIGAM_64:
          m_process_image_addr_is_all_images_infos = false;
          ReadDYLDInfoFromMemoryAndSetNotificationCallback(shlib_addr);
          return;

        default:
          break;
        }
      }
      // Maybe it points to the all image infos?
      m_dyld_all_image_infos_addr = shlib_addr;
      m_process_image_addr_is_all_images_infos = true;
    }
  }

  if (m_dyld_all_image_infos_addr != LLDB_INVALID_ADDRESS) {
    if (ReadAllImageInfosStructure()) {
      if (m_dyld_all_image_infos.dyldImageLoadAddress != LLDB_INVALID_ADDRESS)
        ReadDYLDInfoFromMemoryAndSetNotificationCallback(
            m_dyld_all_image_infos.dyldImageLoadAddress);
      else
        ReadDYLDInfoFromMemoryAndSetNotificationCallback(
            m_dyld_all_image_infos_addr & 0xfffffffffff00000ull);
      return;
    }
  }

  // Check some default values
  Module *executable = m_process->GetTarget().GetExecutableModulePointer();

  if (executable) {
    const ArchSpec &exe_arch = executable->GetArchitecture();
    if (exe_arch.GetAddressByteSize() == 8) {
      ReadDYLDInfoFromMemoryAndSetNotificationCallback(0x7fff5fc00000ull);
    } else if (exe_arch.GetMachine() == llvm::Triple::arm ||
               exe_arch.GetMachine() == llvm::Triple::thumb ||
               exe_arch.GetMachine() == llvm::Triple::aarch64 ||
               exe_arch.GetMachine() == llvm::Triple::aarch64_32) {
      ReadDYLDInfoFromMemoryAndSetNotificationCallback(0x2fe00000);
    } else {
      ReadDYLDInfoFromMemoryAndSetNotificationCallback(0x8fe00000);
    }
  }
}

        size_t j = 0;

        for (; j < roiw_base; j += step_base)
        {
            prefetch(src + j);
            vec128 v_src0 = vld1q(src + j), v_src1 = vld1q(src + j + 16 / sizeof(T));
            v_min_base = vminq(v_min_base, v_src0);
            v_max_base = vmaxq(v_max_base, v_src0);
            v_min_base = vminq(v_min_base, v_src1);
            v_max_base = vmaxq(v_max_base, v_src1);
        }

