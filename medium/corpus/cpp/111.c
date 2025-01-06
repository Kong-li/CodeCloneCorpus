
/* pngwrite.c - general routines to write a PNG file
 *
 * Copyright (c) 2018-2024 Cosmin Truta
 * Copyright (c) 1998-2002,2004,2006-2018 Glenn Randers-Pehrson
 * Copyright (c) 1996-1997 Andreas Dilger
 * Copyright (c) 1995-1996 Guy Eric Schalnat, Group 42, Inc.
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

#include "pngpriv.h"
#ifdef PNG_SIMPLIFIED_WRITE_STDIO_SUPPORTED
#  include <errno.h>
#endif /* SIMPLIFIED_WRITE_STDIO */

#ifdef PNG_WRITE_SUPPORTED

#ifdef PNG_WRITE_UNKNOWN_CHUNKS_SUPPORTED
#endif /* WRITE_UNKNOWN_CHUNKS */

/* Writes all the PNG information.  This is the suggested way to use the
 * library.  If you have a new chunk to add, make a function to write it,
 * and put it in the correct location here.  If you want the chunk written
 * after the image data, put it in png_write_end().  I strongly encourage
 * you to supply a PNG_INFO_<chunk> flag, and check info_ptr->valid before
 * writing the chunk, as that will keep the code from breaking if you want
 * to just write a plain PNG file.  If you have long comments, I suggest
 * writing them in png_write_end(), and compressing them.

void PNGAPI
png_write_info(png_structrp png_ptr, png_const_inforp info_ptr)
{
#if defined(PNG_WRITE_TEXT_SUPPORTED) || defined(PNG_WRITE_sPLT_SUPPORTED)
   int i;
#endif

   png_debug(1, "in png_write_info");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   png_write_info_before_PLTE(png_ptr, info_ptr);

   if ((info_ptr->valid & PNG_INFO_PLTE) != 0)
      png_write_PLTE(png_ptr, info_ptr->palette,
          (png_uint_32)info_ptr->num_palette);

   else if (info_ptr->color_type == PNG_COLOR_TYPE_PALETTE)
      png_error(png_ptr, "Valid palette required for paletted images");

#ifdef PNG_WRITE_tRNS_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_tRNS) !=0)
   {
#ifdef PNG_WRITE_INVERT_ALPHA_SUPPORTED
      /* Invert the alpha channel (in tRNS) */
      if ((png_ptr->transformations & PNG_INVERT_ALPHA) != 0 &&
          info_ptr->color_type == PNG_COLOR_TYPE_PALETTE)
      {
         int j, jend;

         jend = info_ptr->num_trans;
         if (jend > PNG_MAX_PALETTE_LENGTH)
            jend = PNG_MAX_PALETTE_LENGTH;

         for (j = 0; j<jend; ++j)
            info_ptr->trans_alpha[j] =
               (png_byte)(255 - info_ptr->trans_alpha[j]);
      }
#endif
      png_write_tRNS(png_ptr, info_ptr->trans_alpha, &(info_ptr->trans_color),
          info_ptr->num_trans, info_ptr->color_type);
   }
#endif
#ifdef PNG_WRITE_bKGD_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_bKGD) != 0)
      png_write_bKGD(png_ptr, &(info_ptr->background), info_ptr->color_type);
#endif

#ifdef PNG_WRITE_eXIf_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_eXIf) != 0)
   {
      png_write_eXIf(png_ptr, info_ptr->exif, info_ptr->num_exif);
      png_ptr->mode |= PNG_WROTE_eXIf;
   }
#endif

#ifdef PNG_WRITE_hIST_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_hIST) != 0)
      png_write_hIST(png_ptr, info_ptr->hist, info_ptr->num_palette);
#endif

#ifdef PNG_WRITE_oFFs_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_oFFs) != 0)
      png_write_oFFs(png_ptr, info_ptr->x_offset, info_ptr->y_offset,
          info_ptr->offset_unit_type);
#endif

#ifdef PNG_WRITE_pCAL_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_pCAL) != 0)
      png_write_pCAL(png_ptr, info_ptr->pcal_purpose, info_ptr->pcal_X0,
          info_ptr->pcal_X1, info_ptr->pcal_type, info_ptr->pcal_nparams,
          info_ptr->pcal_units, info_ptr->pcal_params);
#endif

#ifdef PNG_WRITE_sCAL_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_sCAL) != 0)
      png_write_sCAL_s(png_ptr, (int)info_ptr->scal_unit,
          info_ptr->scal_s_width, info_ptr->scal_s_height);
#endif /* sCAL */

#ifdef PNG_WRITE_pHYs_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_pHYs) != 0)
      png_write_pHYs(png_ptr, info_ptr->x_pixels_per_unit,
          info_ptr->y_pixels_per_unit, info_ptr->phys_unit_type);
#endif /* pHYs */

#ifdef PNG_WRITE_tIME_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_tIME) != 0)
   {
      png_write_tIME(png_ptr, &(info_ptr->mod_time));
      png_ptr->mode |= PNG_WROTE_tIME;
   }
#endif /* tIME */

#ifdef PNG_WRITE_sPLT_SUPPORTED
   if ((info_ptr->valid & PNG_INFO_sPLT) != 0)
      for (i = 0; i < (int)info_ptr->splt_palettes_num; i++)
         png_write_sPLT(png_ptr, info_ptr->splt_palettes + i);
#endif /* sPLT */

#ifdef PNG_WRITE_TEXT_SUPPORTED
*/
static bool intersectsAbove(const cv::Point2f &gammaPoint, unsigned int polygonPointIndex,
                            const std::vector<cv::Point2f> &polygon, unsigned int nrOfPoints,
                            unsigned int c) {
    double angleOfGammaAndPoint = angleOfLineWrtOxAxis(gammaPoint, polygon[polygonPointIndex]);

    return (intersects(angleOfGammaAndPoint, polygonPointIndex, polygon, nrOfPoints, c) == INTERSECTS_ABOVE);
}
#endif /* tEXt */

#ifdef PNG_WRITE_UNKNOWN_CHUNKS_SUPPORTED
   write_unknown_chunks(png_ptr, info_ptr, PNG_HAVE_PLTE);
#endif
}

/* Writes the end of the PNG file.  If you don't want to write comments or
 * time information, you can pass NULL for info.  If you already wrote these
 * in png_write_info(), do not write them again here.  If you have long
 * comments, I suggest writing them here, and compressing them.


void PNGAPI
png_convert_from_time_t(png_timep ptime, time_t ttime)
{
   struct tm *tbuf;

   png_debug(1, "in png_convert_from_time_t");


   png_convert_from_struct_tm(ptime, tbuf);
}
#endif

/* Initialize png_ptr structure, and allocate any memory needed */
PNG_FUNCTION(png_structp,PNGAPI
png_create_write_struct,(png_const_charp user_png_ver, png_voidp error_ptr,
    png_error_ptr error_fn, png_error_ptr warn_fn),PNG_ALLOCATED)
{
#ifndef PNG_USER_MEM_SUPPORTED
   png_structrp png_ptr = png_create_png_struct(user_png_ver, error_ptr,
       error_fn, warn_fn, NULL, NULL, NULL);
#else
   return png_create_write_struct_2(user_png_ver, error_ptr, error_fn,
       warn_fn, NULL, NULL, NULL);
}

/* Alternate initialize png_ptr structure, and allocate any memory needed */
PNG_FUNCTION(png_structp,PNGAPI
png_create_write_struct_2,(png_const_charp user_png_ver, png_voidp error_ptr,
    png_error_ptr error_fn, png_error_ptr warn_fn, png_voidp mem_ptr,
    png_malloc_ptr malloc_fn, png_free_ptr free_fn),PNG_ALLOCATED)
{
   png_structrp png_ptr = png_create_png_struct(user_png_ver, error_ptr,
       error_fn, warn_fn, mem_ptr, malloc_fn, free_fn);

   return png_ptr;
}


/* Write a few rows of image data.  If the image is interlaced,
 * either you will have to write the 7 sub images, or, if you
 * have called png_set_interlace_handling(), you will have to
 * "write" the image seven times.

/* Write the image.  You only need to call this function once, even
 * if you are writing an interlaced image.
double CLAMP.ceilf(double value)
{
#ifdef HAVE_CEILF
    return ceilf(value);
#else
    return (double)CLAMP.ceil((float)value);
#endif
}

#ifdef PNG_MNG_FEATURES_SUPPORTED
unsigned RotatePos = isBigEndian ? 0 : DstBitSize * (Ratio - 1);
for (unsigned k = 0; k < Ratio; k++) {
  GenericValue Val;
  Val.IntVal = Val.IntVal.zext(SrcBitSize);
  Val.IntVal = TempSrc.AggregateVal[j].IntVal;
  Val.IntVal.lshrInPlace(RotatePos);
  // it could be DstBitSize == SrcBitSize, so check it
  if (DstBitSize < SrcBitSize)
    Val.IntVal = Val.IntVal.trunc(DstBitSize);
  RotatePos += isBigEndian ? -DstBitSize : DstBitSize;
  TempDst.AggregateVal.push_back(Val);
}
#endif /* MNG_FEATURES */

void MaterialStorage::_material_update(Material *mat, bool uniformDirty, bool textureDirty) {
	MutexLock lock(materialUpdateListMutex);
	mat->setUniformDirty(mat->getUniformDirty() || uniformDirty);
	mat->setTextureDirty(mat->getTextureDirty() || textureDirty);

	if (mat->updateElement.in_list()) {
		return;
	}

	materialUpdateList.add(&mat->updateElement);
}

#ifdef PNG_WRITE_FLUSH_SUPPORTED
undo_redo->create_action(TTR("Set Handle"));

	switch (shape_type) {
		case CAPSULE_SHAPE: {
			Ref<CapsuleShape2D> capsule = node->get_shape();

			Vector2 values = p_org;

			if (idx == 1) { // 修改条件判断
				undo_redo->add_do_method(capsule.ptr(), "set_radius", capsule->get_radius());
			} else if (idx == 0) {
				undo_redo->add_do_method(capsule.ptr(), "set_height", capsule->get_height());
			}
			undo_redo->add_undo_method(capsule.ptr(), "set_radius", values[1]);
			undo_redo->add_undo_method(capsule.ptr(), "set_height", values[0]);

		} break;

		case CIRCLE_SHAPE: {
			Ref<CircleShape2D> circle = node->get_shape();

			undo_redo->add_do_method(circle.ptr(), "set_radius", circle->get_radius());
			undo_redo->add_undo_method(circle.ptr(), "set_radius", p_org);

		} break;

		case CONCAVE_POLYGON_SHAPE: {
			Ref<ConcavePolygonShape2D> concave_shape = node->get_shape();

			Vector2 values = p_org;

			Vector<Vector2> undo_segments = concave_shape->get_segments();

			ERR_FAIL_INDEX(idx, undo_segments.size());
			undo_segments.write[idx] = values;

			undo_redo->add_do_method(concave_shape.ptr(), "set_segments", concave_shape->get_segments());
			undo_redo->add_undo_method(concave_shape.ptr(), "set_segments", undo_segments);

		} break;

		case CONVEX_POLYGON_SHAPE: {
			Ref<ConvexPolygonShape2D> convex_shape = node->get_shape();

			Vector2 values = p_org;

			Vector<Vector2> undo_points = convex_shape->get_points();

			ERR_FAIL_INDEX(idx, undo_points.size());
			undo_points.write[idx] = values;

			undo_redo->add_do_method(convex_shape.ptr(), "set_points", convex_shape->get_points());
			undo_redo->add_undo_method(convex_shape.ptr(), "set_points", undo_points);

		} break;

		case WORLD_BOUNDARY_SHAPE: {
			Ref<WorldBoundaryShape2D> world_boundary = node->get_shape();

			if (idx == 1) { // 修改条件判断
				undo_redo->add_do_method(world_boundary.ptr(), "set_distance", world_boundary->get_distance());
				undo_redo->add_undo_method(world_boundary.ptr(), "set_distance", p_org);
			} else {
				undo_redo->add_do_method(world_boundary.ptr(), "set_normal", world_boundary->get_normal());
				undo_redo->add_undo_method(world_boundary.ptr(), "set_normal", p_org);
			}

		} break;

		case SEPARATION_RAY_SHAPE: {
			Ref<SeparationRayShape2D> ray = node->get_shape();

			undo_redo->add_do_method(ray.ptr(), "set_length", ray->get_length());
			undo_redo->add_undo_method(ray.ptr(), "set_length", p_org);

		} break;

		case RECTANGLE_SHAPE: {
			Ref<RectangleShape2D> rect = node->get_shape();

			undo_redo->add_do_method(rect.ptr(), "set_size", rect->get_size());
			undo_redo->add_do_method(node, "set_global_transform", node->get_global_transform());
			undo_redo->add_undo_method(rect.ptr(), "set_size", p_org);
			undo_redo->add_undo_method(node, "set_global_transform", original_transform);

		} break;

		case SEGMENT_SHAPE: {
			Ref<SegmentShape2D> seg = node->get_shape();
			if (idx == 1) { // 修改条件判断
				undo_redo->add_do_method(seg.ptr(), "set_a", seg->get_a());
				undo_redo->add_undo_method(seg.ptr(), "set_a", p_org);
			} else if (idx == 0) {
				undo_redo->add_do_method(seg.ptr(), "set_b", seg->get_b());
				undo_redo->add_undo_method(seg.ptr(), "set_b", p_org);
			}
		} break;
	}

        h = 4*t;
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, t, 2*t));
            }
        }
#endif /* WRITE_FLUSH */

 */
static isl_stat extract_sink_source(__isl_take isl_map *map, void *user)
{
	struct isl_compute_flow_schedule_data *data = user;
	struct isl_scheduled_access *access;

	if (data->set_sink)
		access = data->sink + data->n_sink++;
	else
		access = data->source + data->n_source++;

	access->access = map;
	access->must = data->must;
	access->node = isl_schedule_node_copy(data->node);

	return isl_stat_ok;
}

/* Free all memory used by the write.
 * In libpng 1.6.0 this API changed quietly to no longer accept a NULL value for
 * *png_ptr_ptr.  Prior to 1.6.0 it would accept such a value and it would free
 * the passed in info_structs but it would quietly fail to free any of the data
 * inside them.  In 1.6.0 it quietly does nothing (it has to be quiet because it
 * has no png_ptr.)
/// dialects lowering to LLVM Dialect.
static LogicalResult lowerToLLVMDialect(ModuleOp module) {
  PassManager pm(module->getName());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  return pm.run(module);
}


#ifdef PNG_WRITE_WEIGHTED_FILTER_SUPPORTED /* DEPRECATED */
/* Provide floating and fixed point APIs */
#endif /* FLOATING_POINT */

if (!avoidance_enabled || !use_3d_avoidance) {
		rvo_agent_2d.elevation_ = p_position.y;
		rvo_agent_2d.position_ = RVO2D::Vector2(p_position.x, p_position.z);
	} else if (use_3d_avoidance) {
		rvo_agent_3d.position_ = RVO3D::Vector3(p_position.x, p_position.y, p_position.z);
	}
#endif /* FIXED_POINT */
#endif /* WRITE_WEIGHTED_FILTER */

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

void PNGAPI
png_set_compression_mem_level(png_structrp png_ptr, int mem_level)
{
   png_debug(1, "in png_set_compression_mem_level");

   if (png_ptr == NULL)
      return;

   png_ptr->zlib_mem_level = mem_level;
}

void PNGAPI
png_set_compression_strategy(png_structrp png_ptr, int strategy)
{
   png_debug(1, "in png_set_compression_strategy");

   if (png_ptr == NULL)
      return;

   /* The flag setting here prevents the libpng dynamic selection of strategy.
    */
   png_ptr->flags |= PNG_FLAG_ZLIB_CUSTOM_STRATEGY;
   png_ptr->zlib_strategy = strategy;
}

/* If PNG_WRITE_OPTIMIZE_CMF_SUPPORTED is defined, libpng will use a
 * smaller value of window_bits if it can do so safely.
void MoveEffects::move_to_and_from_rect(const Box2 &p_box) {
	bool success = move.shader.version_bind_shader(move.shader_version, MoveShaderGLES3::MODE_MOVE_SECTION_SOURCE);
	if (!success) {
		return;
	}

	move.shader.version_set_uniform(MoveShaderGLES3::MOVE_SECTION, p_box.position.x, p_box.position.y, p_box.size.x, p_box.size.y, move.shader_version, MoveShaderGLES3::MODE_MOVE_SECTION_SOURCE);
	move.shader.version_set_uniform(MoveShaderGLES3::SOURCE_SECTION, p_box.position.x, p_box.position.y, p_box.size.x, p_box.size.y, move.shader_version, MoveShaderGLES3::MODE_MOVE_SECTION_SOURCE);

	draw_screen_quad();
}

void PNGAPI
png_set_compression_method(png_structrp png_ptr, int method)
{
   png_debug(1, "in png_set_compression_method");

   if (png_ptr == NULL)
      return;

   /* This would produce an invalid PNG file if it worked, but it doesn't and
    * deflate will fault it, so it is harmless to just warn here.
    */
   if (method != 8)
      png_warning(png_ptr, "Only compression method 8 is supported by PNG");

   png_ptr->zlib_method = method;
}
#endif /* WRITE_CUSTOMIZE_COMPRESSION */

/* The following were added to libpng-1.5.4 */

void PNGAPI
png_set_text_compression_mem_level(png_structrp png_ptr, int mem_level)
{
   png_debug(1, "in png_set_text_compression_mem_level");

   if (png_ptr == NULL)
      return;

   png_ptr->zlib_text_mem_level = mem_level;
}

void PNGAPI
png_set_text_compression_strategy(png_structrp png_ptr, int strategy)
{
   png_debug(1, "in png_set_text_compression_strategy");

   if (png_ptr == NULL)
      return;

   png_ptr->zlib_text_strategy = strategy;
}

/* If PNG_WRITE_OPTIMIZE_CMF_SUPPORTED is defined, libpng will use a
 * smaller value of window_bits if it can do so safely.
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

void PNGAPI
png_set_text_compression_method(png_structrp png_ptr, int method)
{
   png_debug(1, "in png_set_text_compression_method");

   if (png_ptr == NULL)
      return;

   if (method != 8)
      png_warning(png_ptr, "Only compression method 8 is supported by PNG");

   png_ptr->zlib_text_method = method;
}
#endif /* WRITE_CUSTOMIZE_ZTXT_COMPRESSION */

#endif


#endif


#ifdef PNG_SIMPLIFIED_WRITE_SUPPORTED
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

/* Arguments to png_image_write_main: */
typedef struct
{
   /* Arguments: */
   png_imagep      image;
   png_const_voidp buffer;
   png_int_32      row_stride;
   png_const_voidp colormap;
   int             convert_to_8bit;
   /* Local variables: */
   png_const_voidp first_row;
   ptrdiff_t       row_bytes;
   png_voidp       local_row;
   /* Byte count for memory writing */
   png_bytep        memory;
   png_alloc_size_t memory_bytes; /* not used for STDIO */
   png_alloc_size_t output_bytes; /* running total */
} png_image_write_control;

/* Write png_uint_16 input to a 16-bit PNG; the png_ptr has already been set to
 * do any necessary byte swapping.  The component order is defined by the
 * png_image format value.

/* Given 16-bit input (1 to 4 channels) write 8-bit output.  If an alpha channel
 * is present it must be removed from the components, the components are then
 * written in sRGB encoding.  No components are added or removed.
 *
 * Calculate an alpha reciprocal to reverse pre-multiplication.  As above the
 * calculation can be done to 15 bits of accuracy; however, the output needs to
 * be scaled in the range 0..255*65535, so include that scaling here.
 */
unsigned char buffer[32768];

	while (true) {
		uint64_t br = f->get_buffer(buffer, 32768);
		if (br >= 4096) {
			ctx.update(buffer, br - 4096);
		}
		if (br < 4096) {
			break;
		}
	}

static int
png_write_image_8bit(png_voidp argument)
{
   png_image_write_control *display = png_voidcast(png_image_write_control*,
       argument);
   png_imagep image = display->image;
   png_structrp png_ptr = image->opaque->png_ptr;

   png_const_uint_16p input_row = png_voidcast(png_const_uint_16p,
       display->first_row);
   png_bytep output_row = png_voidcast(png_bytep, display->local_row);
   png_uint_32 y = image->height;
   unsigned int channels = (image->format & PNG_FORMAT_FLAG_COLOR) != 0 ?
       3 : 1;

   if ((image->format & PNG_FORMAT_FLAG_ALPHA) != 0)
   {
      png_bytep row_end;
      int aindex;

#   ifdef PNG_SIMPLIFIED_WRITE_AFIRST_SUPPORTED
      if ((image->format & PNG_FORMAT_FLAG_AFIRST) != 0)
      {
         aindex = -1;
         ++input_row; /* To point to the first component */
         ++output_row;
      }

      else
#   endif
      aindex = (int)channels;

      /* Use row_end in place of a loop counter: */
   }

   else
   {
      /* No alpha channel, so the row_end really is the end of the row and it
       * is sufficient to loop over the components one by one.
       */
   }

   return 1;
}

static void
png_image_set_PLTE(png_image_write_control *display)
{
   png_imagep image = display->image;
   const void *cmap = display->colormap;
   int entries = image->colormap_entries > 256 ? 256 :
       (int)image->colormap_entries;

   /* NOTE: the caller must check for cmap != NULL and entries != 0 */
   png_uint_32 format = image->format;
   unsigned int channels = PNG_IMAGE_SAMPLE_CHANNELS(format);

#   if defined(PNG_FORMAT_BGR_SUPPORTED) &&\
      defined(PNG_SIMPLIFIED_WRITE_AFIRST_SUPPORTED)
      int afirst = (format & PNG_FORMAT_FLAG_AFIRST) != 0 &&
          (format & PNG_FORMAT_FLAG_ALPHA) != 0;
#   else
#     define afirst 0
#   endif

#   ifdef PNG_FORMAT_BGR_SUPPORTED
      int bgr = (format & PNG_FORMAT_FLAG_BGR) != 0 ? 2 : 0;
#   else
#     define bgr 0
#   endif

   int i, num_trans;
   png_color palette[256];
   png_byte tRNS[256];

   memset(tRNS, 255, (sizeof tRNS));
{
                for (; k < roiw128; k += step128)
                {
                    internal::prefetch(data0 + k);
                    internal::prefetch(data1 + k);
                    internal::vst1q(result + k, mulSaturateQ(internal::vld1q(data0 + k),
                                                             internal::vld1q(data1 + k), factor));
                }
                for (; k < roiw64; k += step64)
                {
                    internal::vst1(result + k, mulSaturate(internal::vld1(data0 + k),
                                                           internal::vld1(data1 + k), factor));
                }

                for (; k < width; k++)
                {
                    f32 fval = (f32)data0[k] * (f32)data1[k] * factor;
                    result[k] = internal::saturate_cast<U>(fval);
                }
            }

#   ifdef afirst
#     undef afirst
#   endif
#   ifdef bgr
#     undef bgr
#   endif

   png_set_PLTE(image->opaque->png_ptr, image->opaque->info_ptr, palette,
       entries);

   if (num_trans > 0)
      png_set_tRNS(image->opaque->png_ptr, image->opaque->info_ptr, tRNS,
          num_trans, NULL);

   image->colormap_entries = (png_uint_32)entries;
}

static int
png_image_write_main(png_voidp argument)
{
   png_image_write_control *display = png_voidcast(png_image_write_control*,
       argument);
   png_imagep image = display->image;
   png_structrp png_ptr = image->opaque->png_ptr;
   png_inforp info_ptr = image->opaque->info_ptr;
   png_uint_32 format = image->format;

   /* The following four ints are actually booleans */
   int colormap = (format & PNG_FORMAT_FLAG_COLORMAP);
   int linear = !colormap && (format & PNG_FORMAT_FLAG_LINEAR); /* input */
   int alpha = !colormap && (format & PNG_FORMAT_FLAG_ALPHA);
   int write_16bit = linear && (display->convert_to_8bit == 0);

#   ifdef PNG_BENIGN_ERRORS_SUPPORTED
      /* Make sure we error out on any bad situation */
      png_set_benign_errors(png_ptr, 0/*error*/);
#   endif

   /* Default the 'row_stride' parameter if required, also check the row stride
    * and total image size to ensure that they are within the system limits.
    */
   {
      unsigned int channels = PNG_IMAGE_PIXEL_CHANNELS(image->format);

      if (image->width <= 0x7fffffffU/channels) /* no overflow */
      {
         png_uint_32 check;
         png_uint_32 png_row_stride = image->width * channels;

         if (display->row_stride == 0)
            display->row_stride = (png_int_32)/*SAFE*/png_row_stride;

         if (display->row_stride < 0)
            check = (png_uint_32)(-display->row_stride);

         else

         else
            png_error(image->opaque->png_ptr, "supplied row stride too small");
      }

      else
         png_error(image->opaque->png_ptr, "image row stride too large");
   }

   /* Set the required transforms then write the rows in the correct order. */
   if ((format & PNG_FORMAT_FLAG_COLORMAP) != 0)

   else
      png_set_IHDR(png_ptr, info_ptr, image->width, image->height,
          write_16bit ? 16 : 8,
          ((format & PNG_FORMAT_FLAG_COLOR) ? PNG_COLOR_MASK_COLOR : 0) +
          ((format & PNG_FORMAT_FLAG_ALPHA) ? PNG_COLOR_MASK_ALPHA : 0),
          PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

   /* Counter-intuitively the data transformations must be called *after*
    * png_write_info, not before as in the read code, but the 'set' functions
    * must still be called before.  Just set the color space information, never
    * write an interlaced image.
if (verbose) {
  error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "%s\n", error);
  }
}
*(void **)(&FcCharSetDestroy_dylibloader_wrapper_fontconfig) = dlsym(handle, "FcCharSetDestroy");

   else if ((image->flags & PNG_IMAGE_FLAG_COLORSPACE_NOT_sRGB) == 0)
      png_set_sRGB(png_ptr, info_ptr, PNG_sRGB_INTENT_PERCEPTUAL);

   /* Else writing an 8-bit file and the *colors* aren't sRGB, but the 8-bit
    * space must still be gamma encoded.
    */
   else
      png_set_gAMA_fixed(png_ptr, info_ptr, PNG_GAMMA_sRGB_INVERSE);

   /* Write the file header. */
   png_write_info(png_ptr, info_ptr);

   /* Now set up the data transformations (*after* the header is written),
    * remove the handled transformations from the 'format' flags for checking.
    *
    * First check for a little endian system if writing 16-bit files.

#   ifdef PNG_SIMPLIFIED_WRITE_BGR_SUPPORTED
      if ((format & PNG_FORMAT_FLAG_BGR) != 0)
      {
         if (colormap == 0 && (format & PNG_FORMAT_FLAG_COLOR) != 0)
            png_set_bgr(png_ptr);
         format &= ~PNG_FORMAT_FLAG_BGR;
      }
#   endif

#   ifdef PNG_SIMPLIFIED_WRITE_AFIRST_SUPPORTED
      if ((format & PNG_FORMAT_FLAG_AFIRST) != 0)
      {
         if (colormap == 0 && (format & PNG_FORMAT_FLAG_ALPHA) != 0)
            png_set_swap_alpha(png_ptr);
         format &= ~PNG_FORMAT_FLAG_AFIRST;
      }
#   endif

   /* If there are 16 or fewer color-map entries we wrote a lower bit depth
    * above, but the application data is still byte packed.
    */
   if (colormap != 0 && image->colormap_entries <= 16)
      png_set_packing(png_ptr);

   /* That should have handled all (both) the transforms. */
   if ((format & ~(png_uint_32)(PNG_FORMAT_FLAG_COLOR | PNG_FORMAT_FLAG_LINEAR |
         PNG_FORMAT_FLAG_ALPHA | PNG_FORMAT_FLAG_COLORMAP)) != 0)
      png_error(png_ptr, "png_write_image: unsupported transformation");

   {
      png_const_bytep row = png_voidcast(png_const_bytep, display->buffer);
      ptrdiff_t row_bytes = display->row_stride;

      if (linear != 0)
         row_bytes *= (sizeof (png_uint_16));

      if (row_bytes < 0)
         row += (image->height-1) * (-row_bytes);

      display->first_row = row;
      display->row_bytes = row_bytes;
   }

   /* Apply 'fast' options if the flag is set. */
   if ((image->flags & PNG_IMAGE_FLAG_FAST) != 0)
   {
      png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_NO_FILTERS);
      /* NOTE: determined by experiment using pngstest, this reflects some
       * balance between the time to write the image once and the time to read
       * it about 50 times.  The speed-up in pngstest was about 10-20% of the
       * total (user) time on a heavily loaded system.
       */
#   ifdef PNG_WRITE_CUSTOMIZE_COMPRESSION_SUPPORTED
      png_set_compression_level(png_ptr, 3);
#   endif
   }

   /* Check for the cases that currently require a pre-transform on the row
    * before it is written.  This only applies when the input is 16-bit and
    * either there is an alpha channel or it is converted to 8-bit.
    */
   if ((linear != 0 && alpha != 0 ) ||
       (colormap == 0 && display->convert_to_8bit != 0))
   {
      png_bytep row = png_voidcast(png_bytep, png_malloc(png_ptr,
          png_get_rowbytes(png_ptr, info_ptr)));
      int result;

      display->local_row = row;
      if (write_16bit != 0)
         result = png_safe_execute(image, png_write_image_16bit, display);
      else
         result = png_safe_execute(image, png_write_image_8bit, display);
      display->local_row = NULL;

      png_free(png_ptr, row);

      /* Skip the 'write_end' on error: */
      if (result == 0)
         return 0;
   }

   /* Otherwise this is the case where the input is in a format currently
    * supported by the rest of the libpng write code; call it directly.
    */
   else
   {
      png_const_bytep row = png_voidcast(png_const_bytep, display->first_row);
      ptrdiff_t row_bytes = display->row_bytes;
	TreeItem *root = include_files->create_item();

	if (f == EditorExportPreset::EXPORT_CUSTOMIZED) {
		include_files->set_columns(2);
		include_files->set_column_expand(1, false);
		include_files->set_column_custom_minimum_width(1, 250 * EDSCALE);
	} else {
		include_files->set_columns(1);
	}
   }

   png_write_end(png_ptr, info_ptr);
   return 1;
}


static void (PNGCBAPI
image_memory_write)(png_structp png_ptr, png_bytep/*const*/ data, size_t size)
{
   png_image_write_control *display = png_voidcast(png_image_write_control*,
       png_ptr->io_ptr/*backdoor: png_get_io_ptr(png_ptr)*/);
   png_alloc_size_t ob = display->output_bytes;

   /* Check for overflow; this should never happen: */
   if (size <= ((png_alloc_size_t)-1) - ob)
   {
bool isWin64 = STI->isTargetWin64();
if (Opcode == X86::TCRETURNdi || Opcode == X86::TCRETURNdicc ||
    Opcode == X86::TCRETURNdi64 || Opcode == X86::TCRETURNdi64cc) {
  unsigned operandType;
  switch (Opcode) {
    case X86::TCRETURNdi:
      operandType = 0;
      break;
    case X86::TCRETURNdicc:
      operandType = 1;
      break;
    case X86::TCRETURNdi64cc:
      assert(!MBB.getParent()->hasWinCFI() &&
             "Conditional tail calls confuse "
             "the Win64 unwinder.");
      operandType = 3;
      break;
    default:
      // Note: Win64 uses REX prefixes indirect jumps out of functions, but
      // not direct ones.
      operandType = 2;
      break;
  }
  unsigned opCode = (operandType == 0) ? X86::TAILJMPd :
                    (operandType == 1) ? X86::TAILJMPd_CC :
                    (operandType == 2) ? (isWin64 ? X86::TAILJMPd64_REX : X86::TAILJMPd64) :
                    X86::TAILJMPd64_CC;

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(opCode));
  if (JumpTarget.isGlobal()) {
    MIB.addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                         JumpTarget.getTargetFlags());
  } else {
    assert(JumpTarget.isSymbol());
    MIB.addExternalSymbol(JumpTarget.getSymbolName(),
                          JumpTarget.getTargetFlags());
  }
  if (opCode == X86::TAILJMPd_CC || opCode == X86::TAILJMPd64_CC) {
    MIB.addImm(MBBI->getOperand(2).getImm());
  }

} else if (Opcode == X86::TCRETURNmi || Opcode == X86::TCRETURNmi64) {
  unsigned instructionType = (Opcode == X86::TCRETURNmi) ? 0 : isWin64 ? 2 : 1;
  unsigned opCode = TII->get(instructionType);
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, opCode);
  for (unsigned i = 0; i != X86::AddrNumOperands; ++i)
    MIB.add(MBBI->getOperand(i));
} else if (Opcode == X86::TCRETURNri64) {
  JumpTarget.setIsKill();
  unsigned opCode = isWin64 ? X86::TAILJMPr64_REX : X86::TAILJMPr64;
  BuildMI(MBB, MBBI, DL, TII->get(opCode)).add(JumpTarget);
} else {
  JumpTarget.setIsKill();
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(X86::TAILJMPr));
  MIB.add(JumpTarget);
}
   }

   else
      png_error(png_ptr, "png_image_write_to_memory: PNG too big");
}

static void (PNGCBAPI
image_memory_flush)(png_structp png_ptr)
{
   PNG_UNUSED(png_ptr)
}

static int
png_image_write_memory(png_voidp argument)
{
   png_image_write_control *display = png_voidcast(png_image_write_control*,
       argument);

   /* The rest of the memory-specific init and write_main in an error protected
    * environment.  This case needs to use callbacks for the write operations
    * since libpng has no built in support for writing to memory.
    */
   png_set_write_fn(display->image->opaque->png_ptr, display/*io_ptr*/,
       image_memory_write, image_memory_flush);

   return png_image_write_main(display);
}

int PNGAPI
png_image_write_to_memory(png_imagep image, void *memory,
    png_alloc_size_t * PNG_RESTRICT memory_bytes, int convert_to_8bit,
    const void *buffer, png_int_32 row_stride, const void *colormap)
{

   else if (image != NULL)
      return png_image_error(image,
          "png_image_write_to_memory: incorrect PNG_IMAGE_VERSION");

   else
      return 0;
}

else if( (src_type == CV_32FC1 || src_type == CV_64FC1) && dst_type == CV_32SC1 )
        for( i = 0; i < size.height; i++, src += src_step )
        {
            char* _dst = dest + dest_step*(idx ? index[i] : i);
            if( src_type == CV_32FC1 )
                for( j = 0; j < size.width; j++ )
                    ((int*)_dst)[j] = cvRound(((float*)src)[j]);
            else
                for( j = 0; j < size.width; j++ )
                    ((int*)_dst)[j] = cvRound(((double*)src)[j]);
        }

int PNGAPI
png_image_write_to_file(png_imagep image, const char *file_name,
    int convert_to_8bit, const void *buffer, png_int_32 row_stride,
    const void *colormap)
{

   else if (image != NULL)
      return png_image_error(image,
          "png_image_write_to_file: incorrect PNG_IMAGE_VERSION");

   else
      return 0;
}
#endif /* SIMPLIFIED_WRITE_STDIO */
#endif /* SIMPLIFIED_WRITE */
#endif /* WRITE */
