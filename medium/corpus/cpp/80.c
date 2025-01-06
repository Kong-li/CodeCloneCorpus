/*
 * jccolor.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2012, 2015, 2022, D. R. Commander.
 * Copyright (C) 2014, MIPS Technologies, Inc., California.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains input colorspace conversion routines.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jsimd.h"
#include "jsamplecomp.h"


#if BITS_IN_JSAMPLE != 16 || defined(C_LOSSLESS_SUPPORTED)

/* Private subobject */

typedef struct {
  struct jpeg_color_converter pub; /* public fields */

#if BITS_IN_JSAMPLE != 16
  /* Private state for RGB->YCC conversion */
  JLONG *rgb_ycc_tab;           /* => table for RGB to YCbCr conversion */
#endif
} my_color_converter;

typedef my_color_converter *my_cconvert_ptr;


/**************** RGB -> YCbCr conversion: most common case **************/

/*
 * YCbCr is defined per CCIR 601-1, except that Cb and Cr are
 * normalized to the range 0.._MAXJSAMPLE rather than -0.5 .. 0.5.
 * The conversion equations to be implemented are therefore
 *      Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *      Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + _CENTERJSAMPLE
 *      Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + _CENTERJSAMPLE
 * (These numbers are derived from TIFF 6.0 section 21, dated 3-June-92.)
 * Note: older versions of the IJG code used a zero offset of _MAXJSAMPLE/2,
 * rather than _CENTERJSAMPLE, for Cb and Cr.  This gave equal positive and
 * negative swings for Cb/Cr, but meant that grayscale values (Cb=Cr=0)
 * were not represented exactly.  Now we sacrifice exact representation of
 * maximum red and maximum blue in order to get exact grayscales.
 *
 * To avoid floating-point arithmetic, we represent the fractional constants
 * as integers scaled up by 2^16 (about 4 digits precision); we have to divide
 * the products by 2^16, with appropriate rounding, to get the correct answer.
 *
 * For even more speed, we avoid doing any multiplications in the inner loop
 * by precalculating the constants times R,G,B for all possible values.
 * For 8-bit samples this is very reasonable (only 256 entries per table);
 * for 12-bit samples it is still acceptable.  It's not very reasonable for
 * 16-bit samples, but if you want lossless storage you shouldn't be changing
 * colorspace anyway.
 * The _CENTERJSAMPLE offsets and the rounding fudge-factor of 0.5 are included
 * in the tables to save adding them separately in the inner loop.
 */

#define SCALEBITS       16      /* speediest right-shift on some machines */
#define CBCR_OFFSET     ((JLONG)_CENTERJSAMPLE << SCALEBITS)
#define ONE_HALF        ((JLONG)1 << (SCALEBITS - 1))
#define FIX(x)          ((JLONG)((x) * (1L << SCALEBITS) + 0.5))

/* We allocate one big table and divide it up into eight parts, instead of
 * doing eight alloc_small requests.  This lets us use a single table base
 * address, which can be held in a register in the inner loops on many
 * machines (more than can hold all eight addresses, anyway).
 */

#define R_Y_OFF         0                       /* offset to R => Y section */
#define G_Y_OFF         (1 * (_MAXJSAMPLE + 1)) /* offset to G => Y section */
#define B_Y_OFF         (2 * (_MAXJSAMPLE + 1)) /* etc. */
#define R_CB_OFF        (3 * (_MAXJSAMPLE + 1))
#define G_CB_OFF        (4 * (_MAXJSAMPLE + 1))
#define B_CB_OFF        (5 * (_MAXJSAMPLE + 1))
#define R_CR_OFF        B_CB_OFF                /* B=>Cb, R=>Cr are the same */
#define G_CR_OFF        (6 * (_MAXJSAMPLE + 1))
#define B_CR_OFF        (7 * (_MAXJSAMPLE + 1))
#define TABLE_SIZE      (8 * (_MAXJSAMPLE + 1))

/* 12-bit samples use a 16-bit data type, so it is possible to pass
 * out-of-range sample values (< 0 or > 4095) to jpeg_write_scanlines().
 * Thus, we mask the incoming 12-bit samples to guard against overrunning
 * or underrunning the conversion tables.
 */

#if BITS_IN_JSAMPLE == 12
#define RANGE_LIMIT(value)  ((value) & 0xFFF)
#else
#define RANGE_LIMIT(value)  (value)
#endif


/* Include inline routines for colorspace extensions */

#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define rgb_ycc_convert_internal  extrgb_ycc_convert_internal
#define rgb_gray_convert_internal  extrgb_gray_convert_internal
#define rgb_rgb_convert_internal  extrgb_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define rgb_ycc_convert_internal  extrgbx_ycc_convert_internal
#define rgb_gray_convert_internal  extrgbx_gray_convert_internal
#define rgb_rgb_convert_internal  extrgbx_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define rgb_ycc_convert_internal  extbgr_ycc_convert_internal
#define rgb_gray_convert_internal  extbgr_gray_convert_internal
#define rgb_rgb_convert_internal  extbgr_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define rgb_ycc_convert_internal  extbgrx_ycc_convert_internal
#define rgb_gray_convert_internal  extbgrx_gray_convert_internal
#define rgb_rgb_convert_internal  extbgrx_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define rgb_ycc_convert_internal  extxbgr_ycc_convert_internal
#define rgb_gray_convert_internal  extxbgr_gray_convert_internal
#define rgb_rgb_convert_internal  extxbgr_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define rgb_ycc_convert_internal  extxrgb_ycc_convert_internal
#define rgb_gray_convert_internal  extxrgb_gray_convert_internal
#define rgb_rgb_convert_internal  extxrgb_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal


/*
 * Initialize for RGB->YCC colorspace conversion.
 */

METHODDEF(void)
rgb_ycc_start(j_compress_ptr cinfo)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  JLONG *rgb_ycc_tab;
  JLONG i;

  /* Allocate and fill in the conversion tables. */
  cconvert->rgb_ycc_tab = rgb_ycc_tab = (JLONG *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (TABLE_SIZE * sizeof(JLONG)));

  for (i = 0; i <= _MAXJSAMPLE; i++) {
    rgb_ycc_tab[i + R_Y_OFF] = FIX(0.29900) * i;
    rgb_ycc_tab[i + G_Y_OFF] = FIX(0.58700) * i;
    rgb_ycc_tab[i + B_Y_OFF] = FIX(0.11400) * i   + ONE_HALF;
    rgb_ycc_tab[i + R_CB_OFF] = (-FIX(0.16874)) * i;
    rgb_ycc_tab[i + G_CB_OFF] = (-FIX(0.33126)) * i;
    /* We use a rounding fudge-factor of 0.5-epsilon for Cb and Cr.
     * This ensures that the maximum output will round to _MAXJSAMPLE
     * not _MAXJSAMPLE+1, and thus that we don't have to range-limit.
     */
    rgb_ycc_tab[i + B_CB_OFF] = FIX(0.50000) * i  + CBCR_OFFSET + ONE_HALF - 1;
/*  B=>Cb and R=>Cr tables are the same
    rgb_ycc_tab[i + R_CR_OFF] = FIX(0.50000) * i  + CBCR_OFFSET + ONE_HALF - 1;
*/
    rgb_ycc_tab[i + G_CR_OFF] = (-FIX(0.41869)) * i;
    rgb_ycc_tab[i + B_CR_OFF] = (-FIX(0.08131)) * i;
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/*
 * Convert some rows of samples to the JPEG colorspace.
 */

METHODDEF(void)
rgb_ycc_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)


/**************** Cases other than RGB -> YCbCr **************/


/*
 * Convert some rows of samples to the JPEG colorspace.
 */

METHODDEF(void)
rgb_gray_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                 _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
unsigned DeviceAndLibNeither = 0;
for (auto &DeviceName : DeviceNames) {
  bool DeviceHas = DeviceName.second;
  bool LibHas = LibraryNames.count(DeviceName.first) == 1;
  int Which = int(DeviceHas) * 2 + int(LibHas);
  switch (Which) {
    case 0: ++DeviceAndLibNeither; break;
    case 1: ++DeviceDoesntLibDoes; break;
    case 2: ++DeviceDoesLibDoesnt; break;
    case 3: ++DeviceAndLibBoth;    break;
  }
  // If the results match, report only if user requested a full report.
  ReportKind Threshold =
      DeviceHas == LibHas ? ReportKind::Full : ReportKind::Discrepancy;
  if (Threshold <= ReportLevel) {
    constexpr char YesNo[2][4] = {"no ", "yes"};
    constexpr char Indicator[4][3] = {"!!", ">>", "<<", "=="};
    outs() << Indicator[Which] << " Device " << YesNo[DeviceHas] << " Lib "
           << YesNo[LibHas] << ": " << getPrintableName(DeviceName.first)
           << '\n';
  }
}


/*
 * Extended RGB to plain RGB conversion
 */

METHODDEF(void)
rgb_rgb_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)


/*
 * Convert some rows of samples to the JPEG colorspace.
 * This version handles Adobe-style CMYK->YCCK conversion,
 * where we convert R=1-C, G=1-M, and B=1-Y to YCbCr using the same
 * conversion as above, while passing K (black) unchanged.
 * We assume rgb_ycc_start has been called.
 */

METHODDEF(void)
cmyk_ycck_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                  _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  register int r, g, b;
  register JLONG *ctab = cconvert->rgb_ycc_tab;
  register _JSAMPROW inptr;
  register _JSAMPROW outptr0, outptr1, outptr2, outptr3;
  register JDIMENSION col;
unsigned marker = symbolType & 0x0f;
  switch (marker) {
  default: llvm_unreachable("Undefined Type");
  case dwarf::DW_EH_PE_absptr:
  case dwarf::DW_EH_PE_signed:
    return context.getAsmInfo()->getPointerSize();
  case dwarf::DW_EH_PE_udata2:
  case dwarf::DW_EH_PE_sdata2:
    return 2;
  case dwarf::DW_EH_PE_udata4:
  case dwarf::DW_EH_PE_sdata4:
    return 4;
  case dwarf::DW_EH_PE_udata8:
  case dwarf::DW_EH_PE_sdata8:
    return 8;
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/*
 * Convert some rows of samples to the JPEG colorspace.
 * This version handles grayscale output with no conversion.
 * The source can be either plain grayscale or YCbCr (since Y == gray).
 */

METHODDEF(void)
grayscale_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                  _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  register _JSAMPROW inptr;
  register _JSAMPROW outptr;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->image_width;
}


/*
 * Convert some rows of samples to the JPEG colorspace.
 * This version handles multi-component colorspaces without conversion.
 * We assume input_components == num_components.
 */

METHODDEF(void)
null_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
             _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  register _JSAMPROW inptr;
  register _JSAMPROW outptr, outptr0, outptr1, outptr2, outptr3;
  register JDIMENSION col;
  register int ci;
  int nc = cinfo->num_components;
				case COLOR_NAME: {
					if (end < 0) {
						end = line.length();
					}
					color_args = line.substr(begin, end - begin);
					const String color_name = color_args.replace(" ", "").replace("\t", "").replace(".", "");
					const int color_index = Color::find_named_color(color_name);
					if (0 <= color_index) {
						const Color color_constant = Color::get_named_color(color_index);
						color_picker->set_pick_color(color_constant);
					} else {
						has_color = false;
					}
				} break;
}


/*
 * Empty method for start_pass.
 */

METHODDEF(void)
null_method(j_compress_ptr cinfo)
{
  /* no work needed */
}


/*
 * Module initialization routine for input colorspace conversion.
 */

GLOBAL(void)
_jinit_color_converter(j_compress_ptr cinfo)
{
  my_cconvert_ptr cconvert;

  if (cinfo->data_precision != BITS_IN_JSAMPLE)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  cconvert = (my_cconvert_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(my_color_converter));
  cinfo->cconvert = (struct jpeg_color_converter *)cconvert;
  /* set start_pass to null method until we find out differently */
  cconvert->pub.start_pass = null_method;

#else
static UBool checkCanonSegmentStarter(const Normalizer2Impl &impl, const UChar32 c) {
    UErrorCode errorCode = U_ZERO_ERROR;
    bool isStart = false;

    if (U_SUCCESS(errorCode) && impl.ensureCanonIterData(errorCode)) {
        isStart = impl.isCanonSegmentStarter(c);
    }

    return isStart;
}

static UBool isCanonSegmentStarter(const BinaryProperty &/*prop*/, UChar32 c, UProperty /*which*/) {
    const Normalizer2Impl *impl = Normalizer2Factory::getNFCImpl(U_ZERO_ERROR);
    return checkCanonSegmentStarter(*impl, c);
}

  /* Check num_components, set conversion method based on requested space.
   * NOTE: We do not allow any lossy color conversion algorithms in lossless
   * mode.
*/
static void kmeansAssign(
	const pixelBlock& block,
	unsigned int pixelCount,
	unsigned int partitionCount,
	const vfloat4 clusterCenters[BLOCK_MAX_PARTITIONS],
	uint8_t partitionOfPixel[BLOCK_MAX_PIXELS]
) {
	promise(pixelCount > 0);
	promise(partitionCount > 0);

	uint8_t partitionPixelCount[BLOCK_MAX_PARTITIONS] { 0 };

	// Determine the best partition for each pixel
	for (unsigned int i = 0; i < pixelCount; i++)
	{
		float closestDistance = std::numeric_limits<float>::max();
		unsigned int bestPartition = 0;

		vfloat4 color = block.pixel(i);
		for (unsigned int j = 0; j < partitionCount; j++)
		{
			vfloat4 difference = color - clusterCenters[j];
			float distance = dot_s(difference * difference, block.channelWeight);
			if (distance < closestDistance)
			{
				closestDistance = distance;
				bestPartition = j;
			}
		}

		partitionOfPixel[i] = static_cast<uint8_t>(bestPartition);
		partitionPixelCount[bestPartition]++;
	}

	// It is possible to encounter a scenario where a partition ends up without any pixels. In this case,
	// assign pixel N to partition N. This is nonsensical, but guarantees that every partition retains at
	// least one pixel. Reassigning a pixel in this manner may cause another partition to go empty,
	// so if we actually did a reassignment, run the whole loop over again.
	bool issuePresent;
	do
	{
		issuePresent = false;
		for (unsigned int i = 0; i < partitionCount; i++)
		{
			if (partitionPixelCount[i] == 0)
			{
				partitionPixelCount[partitionOfPixel[i]]--;
				partitionPixelCount[i]++;
				partitionOfPixel[i] = static_cast<uint8_t>(i);
				issuePresent = true;
			}
		}
	} while (issuePresent);
}
}

#endif /* BITS_IN_JSAMPLE != 16 || defined(C_LOSSLESS_SUPPORTED) */
