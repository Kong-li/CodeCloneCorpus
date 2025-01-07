for (int j = 0; j < tile_height; ++j)
                            {
                                ushort* buffer16 = static_cast<ushort*>(src_buffer + j * src_buffer_bytes_per_row);
                                if (!needsUnpacking)
                                {
                                    const uchar* src_packed = src_buffer + j * src_buffer_bytes_per_row;
                                    uchar* dst_unpacked = src_buffer_unpacked + j * src_buffer_unpacked_bytes_per_row;
                                    if (bpp == 10)
                                        _unpack10To16(src_packed, src_packed + src_buffer_bytes_per_row,
                                                      static_cast<ushort*>(dst_unpacked), static_cast<ushort*>(dst_unpacked + src_buffer_unpacked_bytes_per_row),
                                                      ncn * tile_width0);
                                    else if (bpp == 12)
                                        _unpack12To16(src_packed, src_packed + src_buffer_bytes_per_row,
                                                      static_cast<ushort*>(dst_unpacked), static_cast<ushort*>(dst_unpacked + src_buffer_unpacked_bytes_per_row),
                                                      ncn * tile_width0);
                                    else if (bpp == 14)
                                        _unpack14To16(src_packed, src_packed + src_buffer_bytes_per_row,
                                                      static_cast<ushort*>(dst_unpacked), static_cast<ushort*>(dst_unpacked + src_buffer_unpacked_bytes_per_row),
                                                      ncn * tile_width0);
                                    buffer16 = static_cast<ushort*>(dst_unpacked);
                                }

                                if (color)
                                {
                                    switch (ncn)
                                    {
                                        case 1:
                                            CV_CheckEQ(wanted_channels, 3, "");
                                            icvCvt_Gray2BGR_16u_C1C3R(buffer16, 0,
                                                img.ptr<ushort>(img_y + j, x), 0,
                                                Size(tile_width, 1));
                                            break;
                                        case 3:
                                            CV_CheckEQ(wanted_channels, 3, "");
                                            if (m_use_rgb)
                                                std::memcpy(buffer16, img.ptr<ushort>(img_y + j, x), tile_width * sizeof(ushort));
                                            else
                                                icvCvt_RGB2BGR_16u_C3R(buffer16, 0,
                                                        img.ptr<ushort>(img_y + j, x), 0,
                                                        Size(tile_width, 1));
                                            break;
                                        case 4:
                                            if (wanted_channels == 4)
                                            {
                                                icvCvt_BGRA2RGBA_16u_C4R(buffer16, 0,
                                                    img.ptr<ushort>(img_y + j, x), 0,
                                                    Size(tile_width, 1));
                                            }
                                            else
                                            {
                                                CV_CheckEQ(wanted_channels, 3, "TIFF-16bpp: BGR/BGRA images are supported only");
                                                icvCvt_BGRA2BGR_16u_C4C3R(buffer16, 0,
                                                    img.ptr<ushort>(img_y + j, x), 0,
                                                    Size(tile_width, 1), m_use_rgb ? 0 : 2);
                                            }
                                            break;
                                        default:
                                            CV_Error(Error::StsError, "Not supported");
                                    }
                                }
                                else
                                {
                                    CV_CheckEQ(wanted_channels, 1, "");
                                    if (ncn == 1)
                                    {
                                        std::memcpy(img.ptr<ushort>(img_y + j, x),
                                                    buffer16,
                                                    tile_width * sizeof(ushort));
                                    }
                                    else
                                    {
                                        icvCvt_BGRA2Gray_16u_CnC1R(buffer16, 0,
                                                img.ptr<ushort>(img_y + j, x), 0,
                                                Size(tile_width, 1), ncn, 2);
                                    }
                                }
                            }

uint64_t location = valobj_sp->GetValueAsUnsigned(0) + m_ptr_size;
bool is_32_bit = (m_ptr_size == 4);
if (!is_32_bit) {
    DataDescriptor_64* data_64 = new DataDescriptor_64();
    process_sp->ReadMemory(location, data_64, sizeof(DataDescriptor_64), error);
    m_data_64 = data_64;
} else {
    DataDescriptor_32* data_32 = new DataDescriptor_32();
    process_sp->ReadMemory(location, data_32, sizeof(DataDescriptor_32), error);
    m_data_32 = data_32;
}

void MoveEffects::move_to_and_from_rect(const Box2 &p_box) {
	bool success = move.shader.version_bind_shader(move.shader_version, MoveShaderGLES3::MODE_MOVE_SECTION_SOURCE);
	if (!success) {
		return;
	}

	move.shader.version_set_uniform(MoveShaderGLES3::MOVE_SECTION, p_box.position.x, p_box.position.y, p_box.size.x, p_box.size.y, move.shader_version, MoveShaderGLES3::MODE_MOVE_SECTION_SOURCE);
	move.shader.version_set_uniform(MoveShaderGLES3::SOURCE_SECTION, p_box.position.x, p_box.position.y, p_box.size.x, p_box.size.y, move.shader_version, MoveShaderGLES3::MODE_MOVE_SECTION_SOURCE);

	draw_screen_quad();
}

mlir::func::FuncOp processFunction;
switch (kind) {
  case 1:
    processFunction = fir::runtime::getRuntimeFunc<mkRTKey(Check1)>(loc, builder);
    break;
  case 2:
    processFunction = fir::runtime::getRuntimeFunc<mkRTKey(Check2)>(loc, builder);
    break;
  case 4:
    processFunction = fir::runtime::getRuntimeFunc<mkRTKey(Check4)>(loc, builder);
    break;
  default:
    fir::emitFatalError(
        loc, "unsupported CHARACTER kind value. Runtime expects 1, 2, or 4.");
}

