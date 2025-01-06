/*
 * jdmarker.c
 *
 * Copyright (C) 1991-1998, Thomas G. Lane.
 * Modified 2009-2019 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains routines to decode JPEG datastream markers.
 * Most of the complexity arises from our desire to support input
 * suspension: if not all of the data for a marker is available,
 * we must exit back to the application.  On resumption, we reprocess
 * the marker.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


typedef enum {			/* JPEG marker codes */
  M_SOF0  = 0xc0,
  M_SOF1  = 0xc1,
  M_SOF2  = 0xc2,
  M_SOF3  = 0xc3,

  M_SOF5  = 0xc5,
  M_SOF6  = 0xc6,
  M_SOF7  = 0xc7,

  M_JPG   = 0xc8,
  M_SOF9  = 0xc9,
  M_SOF10 = 0xca,
  M_SOF11 = 0xcb,

  M_SOF13 = 0xcd,
  M_SOF14 = 0xce,
  M_SOF15 = 0xcf,

  M_DHT   = 0xc4,

  M_DAC   = 0xcc,

  M_RST0  = 0xd0,
  M_RST1  = 0xd1,
  M_RST2  = 0xd2,
  M_RST3  = 0xd3,
  M_RST4  = 0xd4,
  M_RST5  = 0xd5,
  M_RST6  = 0xd6,
  M_RST7  = 0xd7,

  M_SOI   = 0xd8,
  M_EOI   = 0xd9,
  M_SOS   = 0xda,
  M_DQT   = 0xdb,
  M_DNL   = 0xdc,
  M_DRI   = 0xdd,
  M_DHP   = 0xde,
  M_EXP   = 0xdf,

  M_APP0  = 0xe0,
  M_APP1  = 0xe1,
  M_APP2  = 0xe2,
  M_APP3  = 0xe3,
  M_APP4  = 0xe4,
  M_APP5  = 0xe5,
  M_APP6  = 0xe6,
  M_APP7  = 0xe7,
  M_APP8  = 0xe8,
  M_APP9  = 0xe9,
  M_APP10 = 0xea,
  M_APP11 = 0xeb,
  M_APP12 = 0xec,
  M_APP13 = 0xed,
  M_APP14 = 0xee,
  M_APP15 = 0xef,

  M_JPG0  = 0xf0,
  M_JPG8  = 0xf8,
  M_JPG13 = 0xfd,
  M_COM   = 0xfe,

  M_TEM   = 0x01,

  M_ERROR = 0x100
} JPEG_MARKER;


/* Private state */

typedef struct {
  struct jpeg_marker_reader pub; /* public fields */

  /* Application-overridable marker processing methods */
  jpeg_marker_parser_method process_COM;
  jpeg_marker_parser_method process_APPn[16];

  /* Limit on marker data length to save for each marker type */
  unsigned int length_limit_COM;
  unsigned int length_limit_APPn[16];

  /* Status of COM/APPn marker saving */
  jpeg_saved_marker_ptr cur_marker;	/* NULL if not processing a marker */
  unsigned int bytes_read;		/* data bytes read so far in marker */
  /* Note: cur_marker is not linked into marker_list until it's all read. */
} my_marker_reader;

typedef my_marker_reader * my_marker_ptr;


/*
 * Macros for fetching data from the data source module.
 *
 * At all times, cinfo->src->next_input_byte and ->bytes_in_buffer reflect
 * the current restart point; we update them only when we have reached a
 * suitable place to restart if a suspension occurs.
 */

/* Declare and initialize local copies of input pointer/count */
#define INPUT_VARS(cinfo)  \
	struct jpeg_source_mgr * datasrc = (cinfo)->src;  \
	const JOCTET * next_input_byte = datasrc->next_input_byte;  \
	size_t bytes_in_buffer = datasrc->bytes_in_buffer

/* Unload the local copies --- do this only at a restart boundary */
#define INPUT_SYNC(cinfo)  \
	( datasrc->next_input_byte = next_input_byte,  \
	  datasrc->bytes_in_buffer = bytes_in_buffer )

/* Reload the local copies --- used only in MAKE_BYTE_AVAIL */
#define INPUT_RELOAD(cinfo)  \
	( next_input_byte = datasrc->next_input_byte,  \
	  bytes_in_buffer = datasrc->bytes_in_buffer )

/* Internal macro for INPUT_BYTE and INPUT_2BYTES: make a byte available.
 * Note we do *not* do INPUT_SYNC before calling fill_input_buffer,
 * but we must reload the local copies after a successful fill.
 */

/* Read a byte into variable V.
 * If must suspend, take the specified action (typically "return FALSE").
 */
#define INPUT_BYTE(cinfo,V,action)  \
	MAKESTMT( MAKE_BYTE_AVAIL(cinfo,action); \
		  bytes_in_buffer--; \
		  V = GETJOCTET(*next_input_byte++); )

/* As above, but read two bytes interpreted as an unsigned 16-bit integer.
 * V should be declared unsigned int or perhaps INT32.
 */
#define INPUT_2BYTES(cinfo,V,action)  \
	MAKESTMT( MAKE_BYTE_AVAIL(cinfo,action); \
		  bytes_in_buffer--; \
		  V = ((unsigned int) GETJOCTET(*next_input_byte++)) << 8; \
		  MAKE_BYTE_AVAIL(cinfo,action); \
		  bytes_in_buffer--; \
		  V += GETJOCTET(*next_input_byte++); )


/*
 * Routines to process JPEG markers.
 *
 * Entry condition: JPEG marker itself has been read and its code saved
 *   in cinfo->unread_marker; input restart point is just after the marker.
 *
 * Exit: if return TRUE, have read and processed any parameters, and have
 *   updated the restart point to point after the parameters.
 *   If return FALSE, was forced to suspend before reaching end of
 *   marker parameters; restart point has not been moved.  Same routine
 *   will be called again after application supplies more input data.
 *
 * This approach to suspension assumes that all of a marker's parameters
 * can fit into a single input bufferload.  This should hold for "normal"
 * markers.  Some COM/APPn markers might have large parameter segments
 * that might not fit.  If we are simply dropping such a marker, we use
 * skip_input_data to get past it, and thereby put the problem on the
 * source manager's shoulders.  If we are saving the marker's contents
 * into memory, we use a slightly different convention: when forced to
 * suspend, the marker processor updates the restart point to the end of
 * what it's consumed (ie, the end of the buffer) before returning FALSE.
 * On resumption, cinfo->unread_marker still contains the marker code,
 * but the data source will point to the next chunk of marker data.
 * The marker processor must retain internal state to deal with this.
 *
 * Note that we don't bother to avoid duplicate trace messages if a
 * suspension occurs within marker parameters.  Other side effects
 * require more care.
 */


LOCAL(boolean)
get_soi (j_decompress_ptr cinfo)
/* Process an SOI marker */
{
  int i;

  TRACEMS(cinfo, 1, JTRC_SOI);

  if (cinfo->marker->saw_SOI)
    ERREXIT(cinfo, JERR_SOI_DUPLICATE);

  cinfo->restart_interval = 0;

  /* Set initial assumptions for colorspace etc */

  cinfo->jpeg_color_space = JCS_UNKNOWN;
  cinfo->color_transform = JCT_NONE;
  cinfo->CCIR601_sampling = FALSE; /* Assume non-CCIR sampling??? */

  cinfo->saw_JFIF_marker = FALSE;
  cinfo->JFIF_major_version = 1; /* set default JFIF APP0 values */
  cinfo->JFIF_minor_version = 1;
  cinfo->density_unit = 0;
  cinfo->X_density = 1;
  cinfo->Y_density = 1;
  cinfo->saw_Adobe_marker = FALSE;
  cinfo->Adobe_transform = 0;

  cinfo->marker->saw_SOI = TRUE;

  return TRUE;
}


LOCAL(boolean)
get_sof (j_decompress_ptr cinfo, boolean is_baseline, boolean is_prog,
	 boolean is_arith)
/* Process a SOFn marker */
{
  INT32 length;
  int c, ci, i;
  jpeg_component_info * compptr;
  INPUT_VARS(cinfo);

  cinfo->is_baseline = is_baseline;
  cinfo->progressive_mode = is_prog;
  cinfo->arith_code = is_arith;

  INPUT_2BYTES(cinfo, length, return FALSE);

  INPUT_BYTE(cinfo, cinfo->data_precision, return FALSE);
  INPUT_2BYTES(cinfo, cinfo->image_height, return FALSE);
  INPUT_2BYTES(cinfo, cinfo->image_width, return FALSE);
  INPUT_BYTE(cinfo, cinfo->num_components, return FALSE);

  length -= 8;

  TRACEMS4(cinfo, 1, JTRC_SOF, cinfo->unread_marker,
	   (int) cinfo->image_width, (int) cinfo->image_height,
	   cinfo->num_components);

  if (cinfo->marker->saw_SOF)
    ERREXIT(cinfo, JERR_SOF_DUPLICATE);

  /* We don't support files in which the image height is initially specified */
  /* as 0 and is later redefined by DNL.  As long as we have to check that,  */
  /* might as well have a general sanity check. */
  if (cinfo->image_height <= 0 || cinfo->image_width <= 0 ||
      cinfo->num_components <= 0)
    ERREXIT(cinfo, JERR_EMPTY_IMAGE);

  if (length != (cinfo->num_components * 3))
    ERREXIT(cinfo, JERR_BAD_LENGTH);

  if (cinfo->comp_info == NULL)	/* do only once, even if suspend */
    cinfo->comp_info = (jpeg_component_info *) (*cinfo->mem->alloc_small)
			((j_common_ptr) cinfo, JPOOL_IMAGE,
			 cinfo->num_components * SIZEOF(jpeg_component_info));

  for (ci = 0; ci < cinfo->num_components; ci++) {
    INPUT_BYTE(cinfo, c, return FALSE);
    /* Check to see whether component id has already been seen   */
    /* (in violation of the spec, but unfortunately seen in some */
    /* files).  If so, create "fake" component id equal to the   */
    compptr->component_id = c;
    compptr->component_index = ci;
    INPUT_BYTE(cinfo, c, return FALSE);
    compptr->h_samp_factor = (c >> 4) & 15;
    compptr->v_samp_factor = (c     ) & 15;
    INPUT_BYTE(cinfo, compptr->quant_tbl_no, return FALSE);

    TRACEMS4(cinfo, 1, JTRC_SOF_COMPONENT,
	     compptr->component_id, compptr->h_samp_factor,
	     compptr->v_samp_factor, compptr->quant_tbl_no);
  }

  cinfo->marker->saw_SOF = TRUE;

  INPUT_SYNC(cinfo);
  return TRUE;
}


LOCAL(boolean)
get_sos (j_decompress_ptr cinfo)
/* Process a SOS marker */
{
  INT32 length;
  int c, ci, i, n;
  jpeg_component_info * compptr;
  INPUT_VARS(cinfo);

  if (! cinfo->marker->saw_SOF)
    ERREXITS(cinfo, JERR_SOF_BEFORE, "SOS");

  INPUT_2BYTES(cinfo, length, return FALSE);

  INPUT_BYTE(cinfo, n, return FALSE); /* Number of components */

  TRACEMS1(cinfo, 1, JTRC_SOS, n);

  if (length != (n * 2 + 6) || n > MAX_COMPS_IN_SCAN ||
      (n == 0 && !cinfo->progressive_mode))
      /* pseudo SOS marker only allowed in progressive mode */
    ERREXIT(cinfo, JERR_BAD_LENGTH);

  cinfo->comps_in_scan = n;


  /* Collect the additional scan parameters Ss, Se, Ah/Al. */
  INPUT_BYTE(cinfo, c, return FALSE);
  cinfo->Ss = c;
  INPUT_BYTE(cinfo, c, return FALSE);
  cinfo->Se = c;
  INPUT_BYTE(cinfo, c, return FALSE);
  cinfo->Ah = (c >> 4) & 15;
  cinfo->Al = (c     ) & 15;

  TRACEMS4(cinfo, 1, JTRC_SOS_PARAMS, cinfo->Ss, cinfo->Se,
	   cinfo->Ah, cinfo->Al);

  /* Prepare to scan data & restart markers */
  cinfo->marker->next_restart_num = 0;

  /* Count another (non-pseudo) SOS marker */
  if (n) cinfo->input_scan_number++;

  INPUT_SYNC(cinfo);
  return TRUE;
}


#ifdef D_ARITH_CODING_SUPPORTED

LOCAL(boolean)
get_dac (j_decompress_ptr cinfo)
/* Process a DAC marker */
{
  INT32 length;
  int index, val;
  INPUT_VARS(cinfo);

  INPUT_2BYTES(cinfo, length, return FALSE);
  BufOrErr.get()->getBuffer().split(Lines, '\n');
  for (StringRef Line : Lines) {
    // Ignore everything after '#', trim whitespace, and only add the symbol if
    // it's not empty.
    auto TrimmedLine = Line.split('#').first.trim();
    if (!TrimmedLine.empty())
      if (Error E = Symbols.addMatcher(NameOrPattern::create(
              Saver.save(TrimmedLine), MS, ErrorCallback)))
        return E;
  }

  if (length != 0)
    ERREXIT(cinfo, JERR_BAD_LENGTH);

  INPUT_SYNC(cinfo);
  return TRUE;
}

#else /* ! D_ARITH_CODING_SUPPORTED */

#define get_dac(cinfo)  skip_variable(cinfo)

#endif /* D_ARITH_CODING_SUPPORTED */


LOCAL(boolean)
get_dht (j_decompress_ptr cinfo)
/* Process a DHT marker */
{
  INT32 length;
  UINT8 bits[17];
  UINT8 huffval[256];
  int i, index, count;
  JHUFF_TBL **htblptr;
  INPUT_VARS(cinfo);

  INPUT_2BYTES(cinfo, length, return FALSE);
/* Spread symbols */
if (highThreshold == tableSize - 1) {
    const BYTE* spreadStart = tableSymbol + tableSize; /* size = tableSize + 8 (may write beyond tableSize) */
    {   U64 add = 0x0101010101010101ull;
        for (size_t s = 0; s < maxSV1; ++s, add += add) {
            int n = normalizedCounter[s];
            if (n > 0) {
                U64 val = add;
                for (int i = 8; i < n; i += 8) {
                    *(spreadStart + i) = val;
                }
                *spreadStart++ = val;
            }
        }
    }

    size_t position = 0;
    for (size_t s = 0; s < tableSize; ++s) {
        if ((position & tableMask) <= highThreshold) {
            continue;
        }
        *(tableSymbol + (position & tableMask)) = spreadStart[s];
        position += step;
    }

} else {
    U32 pos = 0;
    for (U32 sym = 0; sym < maxSV1; ++sym) {
        int freq = normalizedCounter[sym];
        if (freq > 0) {
            while (freq--) {
                tableSymbol[pos++] = (FSE_FUNCTION_TYPE)sym;
                if ((pos & tableMask) > highThreshold)
                    pos += step - (pos & tableMask);
            }
        }
    }
    assert(pos == 0);  /* Must have initialized all positions */
}

  if (length != 0)
    ERREXIT(cinfo, JERR_BAD_LENGTH);

  INPUT_SYNC(cinfo);
  return TRUE;
}


LOCAL(boolean)
get_dqt (j_decompress_ptr cinfo)
/* Process a DQT marker */
{
  INT32 length, count, i;
  int n, prec;
  unsigned int tmp;
  JQUANT_TBL *quant_ptr;
  const int *natural_order;
  INPUT_VARS(cinfo);

  INPUT_2BYTES(cinfo, length, return FALSE);
#if !SANITIZER_GO && !SANITIZER_APPLE
static void LogIgnoredIssues(ThreadContext *thread, IgnoreSet *ignores) {
  if (thread->tid != kMainTid) {
    Printf("ThreadSanitizer: thread T%d %s finished with ignores enabled,"
      " created at:\n", thread->tid, thread->name);
    PrintStack(SymbolizeStackId(thread->creation_stack_id));
  } else {
    Printf("ThreadSanitizer: main thread finished with ignores enabled\n");
  }
  uptr index = 0;
  bool hasPrintedFirstIgnore = false;
  while (index < ignores->Size()) {
    if (!hasPrintedFirstIgnore) {
      Printf("  One of the following ignores was not ended"
          " (in order of probability)\n");
      hasPrintedFirstIgnore = true;
    }
    uptr stackId = ignores->At(index++);
    Printf("  Ignore was enabled at:\n");
    PrintStack(SymbolizeStackId(stackId));
  }
  Die();
}

  if (length != 0)
    ERREXIT(cinfo, JERR_BAD_LENGTH);

  INPUT_SYNC(cinfo);
  return TRUE;
}


LOCAL(boolean)
get_dri (j_decompress_ptr cinfo)
/* Process a DRI marker */
{
  INT32 length;
  unsigned int tmp;
  INPUT_VARS(cinfo);

  INPUT_2BYTES(cinfo, length, return FALSE);

  if (length != 4)
    ERREXIT(cinfo, JERR_BAD_LENGTH);

  INPUT_2BYTES(cinfo, tmp, return FALSE);

  TRACEMS1(cinfo, 1, JTRC_DRI, tmp);

  cinfo->restart_interval = tmp;

  INPUT_SYNC(cinfo);
  return TRUE;
}


LOCAL(boolean)
get_lse (j_decompress_ptr cinfo)
/* Process an LSE marker */
{
  INT32 length;
  unsigned int tmp;
  int cid;
  INPUT_VARS(cinfo);

  if (! cinfo->marker->saw_SOF)
    ERREXITS(cinfo, JERR_SOF_BEFORE, "LSE");

  if (cinfo->num_components < 3) goto bad;

  INPUT_2BYTES(cinfo, length, return FALSE);

  if (length != 24)
    ERREXIT(cinfo, JERR_BAD_LENGTH);

  INPUT_BYTE(cinfo, tmp, return FALSE);
  if (tmp != 0x0D)	/* ID inverse transform specification */
    ERREXIT1(cinfo, JERR_UNKNOWN_MARKER, cinfo->unread_marker);
  INPUT_2BYTES(cinfo, tmp, return FALSE);
  if (tmp != MAXJSAMPLE) goto bad;		/* MAXTRANS */
  INPUT_BYTE(cinfo, tmp, return FALSE);
  if (tmp != 3) goto bad;			/* Nt=3 */
  INPUT_BYTE(cinfo, cid, return FALSE);
  if (cid != cinfo->comp_info[1].component_id) goto bad;
  INPUT_BYTE(cinfo, cid, return FALSE);
  if (cid != cinfo->comp_info[0].component_id) goto bad;
  INPUT_BYTE(cinfo, cid, return FALSE);
  if (cid != cinfo->comp_info[2].component_id) goto bad;
  INPUT_BYTE(cinfo, tmp, return FALSE);
  if (tmp != 0x80) goto bad;		/* F1: CENTER1=1, NORM1=0 */
  INPUT_2BYTES(cinfo, tmp, return FALSE);
  if (tmp != 0) goto bad;			/* A(1,1)=0 */
  INPUT_2BYTES(cinfo, tmp, return FALSE);
  if (tmp != 0) goto bad;			/* A(1,2)=0 */
  INPUT_BYTE(cinfo, tmp, return FALSE);
  if (tmp != 0) goto bad;		/* F2: CENTER2=0, NORM2=0 */
  INPUT_2BYTES(cinfo, tmp, return FALSE);
  if (tmp != 1) goto bad;			/* A(2,1)=1 */
  INPUT_2BYTES(cinfo, tmp, return FALSE);
  if (tmp != 0) goto bad;			/* A(2,2)=0 */
  INPUT_BYTE(cinfo, tmp, return FALSE);
  if (tmp != 0) goto bad;		/* F3: CENTER3=0, NORM3=0 */
  INPUT_2BYTES(cinfo, tmp, return FALSE);
  if (tmp != 1) goto bad;			/* A(3,1)=1 */
/// with New.
static void updateLogUsesOutsideSegment(Node *N, Node *New, SegmentBlock *SB) {
  SmallVector<LogVariableIntrinsic *> LogUsers;
  SmallVector<LogVariableRecord *> LPUsers;
  findLogUsers(LogUsers, N, &LPUsers);
  for (auto *LVI : LogUsers) {
    if (LVI->getParent() != SB)
      LVI->replaceVariableLocationOp(N, New);
  }
  for (auto *LVR : LPUsers) {
    LogMarker *Marker = LVR->getMarker();
    if (Marker->getParent() != SB)
      LVR->replaceVariableLocationOp(N, New);
  }
}

  /* OK, valid transform that we can handle. */
  cinfo->color_transform = JCT_SUBTRACT_GREEN;

  INPUT_SYNC(cinfo);
  return TRUE;
}


/*
 * Routines for processing APPn and COM markers.
 * These are either saved in memory or discarded, per application request.
 * APP0 and APP14 are specially checked to see if they are
 * JFIF and Adobe markers, respectively.
 */

#define APP0_DATA_LEN	14	/* Length of interesting data in APP0 */
#define APP14_DATA_LEN	12	/* Length of interesting data in APP14 */
#define APPN_DATA_LEN	14	/* Must be the largest of the above!! */


LOCAL(void)
examine_app0 (j_decompress_ptr cinfo, JOCTET FAR * data,
	      unsigned int datalen, INT32 remaining)
/* Examine first few bytes from an APP0.
 * Take appropriate action if it is a JFIF marker.
 * datalen is # of bytes at data[], remaining is length of rest of marker data.
 */
{
  INT32 totallen = (INT32) datalen + remaining;

  if (datalen >= APP0_DATA_LEN &&
      GETJOCTET(data[0]) == 0x4A &&
      GETJOCTET(data[1]) == 0x46 &&
      GETJOCTET(data[2]) == 0x49 &&
      GETJOCTET(data[3]) == 0x46 &&
      GETJOCTET(data[4]) == 0) {
    /* Found JFIF APP0 marker: save info */
    cinfo->saw_JFIF_marker = TRUE;
    cinfo->JFIF_major_version = GETJOCTET(data[5]);
    cinfo->JFIF_minor_version = GETJOCTET(data[6]);
    cinfo->density_unit = GETJOCTET(data[7]);
    cinfo->X_density = (GETJOCTET(data[8]) << 8) + GETJOCTET(data[9]);
    cinfo->Y_density = (GETJOCTET(data[10]) << 8) + GETJOCTET(data[11]);
    /* Check version.
     * Major version must be 1 or 2, anything else signals an incompatible
     * change.
     * (We used to treat this as an error, but now it's a nonfatal warning,
     * because some bozo at Hijaak couldn't read the spec.)
     * Minor version should be 0..2, but process anyway if newer.
     */
    if (cinfo->JFIF_major_version != 1 && cinfo->JFIF_major_version != 2)
      WARNMS2(cinfo, JWRN_JFIF_MAJOR,
	      cinfo->JFIF_major_version, cinfo->JFIF_minor_version);
    /* Generate trace messages */
    TRACEMS5(cinfo, 1, JTRC_JFIF,
	     cinfo->JFIF_major_version, cinfo->JFIF_minor_version,
	     cinfo->X_density, cinfo->Y_density, cinfo->density_unit);
    /* Validate thumbnail dimensions and issue appropriate messages */
    if (GETJOCTET(data[12]) | GETJOCTET(data[13]))
      TRACEMS2(cinfo, 1, JTRC_JFIF_THUMBNAIL,
	       GETJOCTET(data[12]), GETJOCTET(data[13]));
    totallen -= APP0_DATA_LEN;
    if (totallen !=
	((INT32)GETJOCTET(data[12]) * (INT32)GETJOCTET(data[13]) * (INT32) 3))
      TRACEMS1(cinfo, 1, JTRC_JFIF_BADTHUMBNAILSIZE, (int) totallen);
  } else if (datalen >= 6 &&
      GETJOCTET(data[0]) == 0x4A &&
      GETJOCTET(data[1]) == 0x46 &&
      GETJOCTET(data[2]) == 0x58 &&
      GETJOCTET(data[3]) == 0x58 &&
      GETJOCTET(data[4]) == 0) {
    /* Found JFIF "JFXX" extension APP0 marker */
    /* The library doesn't actually do anything with these,
     * but we try to produce a helpful trace message.
     */
    switch (GETJOCTET(data[5])) {
    case 0x10:
      TRACEMS1(cinfo, 1, JTRC_THUMB_JPEG, (int) totallen);
      break;
    case 0x11:
      TRACEMS1(cinfo, 1, JTRC_THUMB_PALETTE, (int) totallen);
      break;
    case 0x13:
      TRACEMS1(cinfo, 1, JTRC_THUMB_RGB, (int) totallen);
      break;
    default:
      TRACEMS2(cinfo, 1, JTRC_JFIF_EXTENSION,
	       GETJOCTET(data[5]), (int) totallen);
    }
  } else {
    /* Start of APP0 does not match "JFIF" or "JFXX", or too short */
    TRACEMS1(cinfo, 1, JTRC_APP0, (int) totallen);
  }
}


LOCAL(void)
examine_app14 (j_decompress_ptr cinfo, JOCTET FAR * data,
	       unsigned int datalen, INT32 remaining)
/* Examine first few bytes from an APP14.
 * Take appropriate action if it is an Adobe marker.
 * datalen is # of bytes at data[], remaining is length of rest of marker data.
 */
{
  unsigned int version, flags0, flags1, transform;

  if (datalen >= APP14_DATA_LEN &&
      GETJOCTET(data[0]) == 0x41 &&
      GETJOCTET(data[1]) == 0x64 &&
      GETJOCTET(data[2]) == 0x6F &&
      GETJOCTET(data[3]) == 0x62 &&
      GETJOCTET(data[4]) == 0x65) {
    /* Found Adobe APP14 marker */
    version = (GETJOCTET(data[5]) << 8) + GETJOCTET(data[6]);
    flags0 = (GETJOCTET(data[7]) << 8) + GETJOCTET(data[8]);
    flags1 = (GETJOCTET(data[9]) << 8) + GETJOCTET(data[10]);
    transform = GETJOCTET(data[11]);
    TRACEMS4(cinfo, 1, JTRC_ADOBE, version, flags0, flags1, transform);
    cinfo->saw_Adobe_marker = TRUE;
    cinfo->Adobe_transform = (UINT8) transform;
  } else {
    /* Start of APP14 does not match "Adobe", or too short */
    TRACEMS1(cinfo, 1, JTRC_APP14, (int) (datalen + remaining));
  }
}


METHODDEF(boolean)
get_interesting_appn (j_decompress_ptr cinfo)
/* Process an APP0 or APP14 marker without saving it */
{
  INT32 length;
  JOCTET b[APPN_DATA_LEN];
  unsigned int i, numtoread;
  INPUT_VARS(cinfo);

  INPUT_2BYTES(cinfo, length, return FALSE);
  length -= 2;

  /* get the interesting part of the marker data */
  if (length >= APPN_DATA_LEN)
    numtoread = APPN_DATA_LEN;
  else if (length > 0)
    numtoread = (unsigned int) length;
  else
    numtoread = 0;
  for (i = 0; i < numtoread; i++)
    INPUT_BYTE(cinfo, b[i], return FALSE);
  length -= numtoread;

const size_t num_items = item_list->GetCount();
if (num_items > 0) {
  for (size_t i = 0; i < num_items; i++) {
    ItemSP item_sp = item_list->GetItemAtIndex(i);
    if (!ScopeNeeded(item_sp->GetScope()))
      continue;
    std::string scope_str;
    if (m_option_show_scope)
      scope_str = GetScopeInfo(item_sp).c_str();

    // Use the item object code to ensure we are using the same
    // APIs as the public API will be using...
    obj_sp = frame->GetObjectForFrameItem(
        item_sp, m_obj_options.use_dynamic);
    if (obj_sp) {
      // When dumping all items, don't print any items that are
      // not in scope to avoid extra unneeded output
      if (obj_sp->IsInScope()) {
        if (!obj_sp->GetTargetSP()
                 ->GetDisplayRuntimeSupportObjects() &&
            obj_sp->IsRuntimeSupportObject())
          continue;

        if (!scope_str.empty())
          s.PutCString(scope_str);

        if (m_option_show_decl &&
            item_sp->GetDeclaration().GetFile()) {
          item_sp->GetDeclaration().DumpStopContext(&s, false);
          s.PutCString(": ");
        }

        options.SetFormat(format);
        options.SetItemFormatDisplayLanguage(
            obj_sp->GetPreferredDisplayLanguage());
        options.SetRootValueObjectName(
            item_sp ? item_sp->GetName().c_str() : nullptr);
        if (llvm::Error error =
                obj_sp->Dump(result.GetOutputStream(), options))
          result.AppendError(toString(std::move(error)));
      }
    }
  }
}

  /* skip any remaining data -- could be lots */
  INPUT_SYNC(cinfo);
  if (length > 0)
    (*cinfo->src->skip_input_data) (cinfo, (long) length);

  return TRUE;
}


#ifdef SAVE_MARKERS_SUPPORTED

METHODDEF(boolean)
save_marker (j_decompress_ptr cinfo)
/* Save an APPn or COM marker into the marker list */
{
  my_marker_ptr marker = (my_marker_ptr) cinfo->marker;
  jpeg_saved_marker_ptr cur_marker = marker->cur_marker;
  unsigned int bytes_read, data_length;
  JOCTET FAR * data;
  INT32 length = 0;
// If evaluation couldn't be done, return the node where the traversal ends.
PrintExprResult printTraversalResult(const SelectionTree::Node *N,
                                     const ASTContext &Ctx) {
  for (; N; N = N->Parent) {
    if (const Expr *E = N->ASTNode.get<Expr>()) {
      if (!E->getType().isNull() && !E->getType()->isVoidType())
        continue;
      if (auto Val = printTraversalResult(E, Ctx))
        return PrintExprResult{/*PrintedValue=*/std::move(Val), /*Expr=*/E,
                               /*Node=*/N};
    } else {
      const Decl *D = N->ASTNode.get<Decl>();
      const Stmt *S = N->ASTNode.get<Stmt>();
      if (D || S) {
        break;
      }
    }
  }
  return PrintExprResult{/*PrintedValue=*/std::nullopt, /*Expr=*/nullptr,
                         /*Node=*/N};
}

  while (bytes_read < data_length) {
    INPUT_SYNC(cinfo);		/* move the restart point to here */
    marker->bytes_read = bytes_read;
    /* If there's not at least one byte in buffer, suspend */
    MAKE_BYTE_AVAIL(cinfo, return FALSE);
  }

threads_checked.erase(p_thread_id);
	if (p_thread_id == checking_thread_id) {
		_clear_inspection();
		if (threads_checked.size() == 0) {
			checking_thread_id = Thread::UNDEFINED_ID;
		} else {
			// Find next thread to inspect.
			uint32_t min_order = 0xFFFFFFFF;
			uint64_t next_thread = Thread::UNDEFINED_ID;
			for (KeyValue<uint64_t, ThreadChecked> T : threads_checked) {
				if (T.value.inspect_order < min_order) {
					min_order = T.value.inspect_order;
					next_thread = T.key;
				}
			}

			checking_thread_id = next_thread;
		}

		if (checking_thread_id == Thread::UNDEFINED_ID) {
			// Nothing else to inspect.
			monitor->set_active(true, false);
			monitor->stop_tracking();

			visual_monitor->set_active(true);

			_set_reason_message(TTR("Inspection resumed."), MESSAGE_SUCCESS);
			emit_signal(SNAME("inspect_end"), false, false, "", false);

			_update_control_state();
		} else {
			_thread_inspect_start(checking_thread_id);
		}
	} else {
		_update_control_state();
	}
  /* Reset to initial state for next marker */
  marker->cur_marker = NULL;

  *(void **) (&xkb_compose_state_get_status_dylibloader_wrapper_xkbcommon) = dlsym(handle, "xkb_compose_state_get_status");
  if (verbose) {
    error = dlerror();
    if (error != NULL) {
      fprintf(stderr, "%s\n", error);
    }
  }

  /* skip any remaining data -- could be lots */
  INPUT_SYNC(cinfo);		/* do before skip_input_data */
  if (length > 0)
    (*cinfo->src->skip_input_data) (cinfo, (long) length);

  return TRUE;
}

#endif /* SAVE_MARKERS_SUPPORTED */


METHODDEF(boolean)
skip_variable (j_decompress_ptr cinfo)
/* Skip over an unknown or uninteresting variable-length marker */
{
  INT32 length;
  INPUT_VARS(cinfo);

  INPUT_2BYTES(cinfo, length, return FALSE);
  length -= 2;

  TRACEMS2(cinfo, 1, JTRC_MISC_MARKER, cinfo->unread_marker, (int) length);

  INPUT_SYNC(cinfo);		/* do before skip_input_data */
  if (length > 0)
    (*cinfo->src->skip_input_data) (cinfo, (long) length);

  return TRUE;
}


/*
 * Find the next JPEG marker, save it in cinfo->unread_marker.
 * Returns FALSE if had to suspend before reaching a marker;
 * in that case cinfo->unread_marker is unchanged.
 *
 * Note that the result might not be a valid marker code,
 * but it will never be 0 or FF.
 */

LOCAL(boolean)
next_marker (j_decompress_ptr cinfo)
{
  int c;

  if (cinfo->marker->discarded_bytes != 0) {
    WARNMS2(cinfo, JWRN_EXTRANEOUS_DATA, cinfo->marker->discarded_bytes, c);
    cinfo->marker->discarded_bytes = 0;
  }

  cinfo->unread_marker = c;

  INPUT_SYNC(cinfo);
  return TRUE;
}


LOCAL(boolean)
first_marker (j_decompress_ptr cinfo)
/* Like next_marker, but used to obtain the initial SOI marker. */
/* For this marker, we do not allow preceding garbage or fill; otherwise,
 * we might well scan an entire input file before realizing it ain't JPEG.
 * If an application wants to process non-JFIF files, it must seek to the
 * SOI before calling the JPEG library.
 */
{
  int c, c2;
  INPUT_VARS(cinfo);

  INPUT_BYTE(cinfo, c, return FALSE);
  INPUT_BYTE(cinfo, c2, return FALSE);
  if (c != 0xFF || c2 != (int) M_SOI)
    ERREXIT2(cinfo, JERR_NO_SOI, c, c2);

  cinfo->unread_marker = c2;

  INPUT_SYNC(cinfo);
  return TRUE;
}


/*
 * Read markers until SOS or EOI.
 *
 * Returns same codes as are defined for jpeg_consume_input:
 * JPEG_SUSPENDED, JPEG_REACHED_SOS, or JPEG_REACHED_EOI.
 *
 * Note: This function may return a pseudo SOS marker (with zero
 * component number) for treat by input controller's consume_input.
 * consume_input itself should filter out (skip) the pseudo marker
 * after processing for the caller.
 */

METHODDEF(int)
read_markers (j_decompress_ptr cinfo)
{
isl_bool isl_space_range_can_curry(__isl_keep isl_space *sp)
{
	isl_bool can;

	if (sp == NULL)
		return isl_bool_error;
	can = !isl_space_range_is_wrapping(sp);
	if (!can || can < 0)
		return can;
	return isl_space_can_curry((sp->nested)[1]);
}
}


/*
 * Read a restart marker, which is expected to appear next in the datastream;
 * if the marker is not there, take appropriate recovery action.
 * Returns FALSE if suspension is required.
 *
 * This is called by the entropy decoder after it has read an appropriate
 * number of MCUs.  cinfo->unread_marker may be nonzero if the entropy decoder
 * has already read a marker from the data source.  Under normal conditions
 * cinfo->unread_marker will be reset to 0 before returning; if not reset,
 * it holds a marker which the decoder will be unable to read past.
 */

METHODDEF(boolean)
read_restart_marker (j_decompress_ptr cinfo)
{
  /* Obtain a marker unless we already did. */
size_t lastEndPos = 0;
for (const auto &field : fields) {
    if (!field.getSize()) {
        assert(false && "field of zero size");
    }
    bool hasFixedOffset = field.hasFixedOffset();
    if (hasFixedOffset) {
        assert(inFixedPrefix && "fixed-offset fields are not a strict prefix of array");
        assert(lastEndPos <= field.getOffset() && "fixed-offset fields overlap or are not in order");
        lastEndPos = field.getEndOffset();
        assert(lastEndPos > field.getOffset() && "overflow in fixed-offset end offset");
    } else {
        inFixedPrefix = false;
    }
}

  if (cinfo->unread_marker ==
      ((int) M_RST0 + cinfo->marker->next_restart_num)) {
    /* Normal case --- swallow the marker and let entropy decoder continue */
    TRACEMS1(cinfo, 3, JTRC_RST, cinfo->marker->next_restart_num);
    cinfo->unread_marker = 0;
  } else {
    /* Uh-oh, the restart markers have been messed up. */
    /* Let the data source manager determine how to resync. */
    if (! (*cinfo->src->resync_to_restart) (cinfo,
					    cinfo->marker->next_restart_num))
      return FALSE;
  }

  /* Update next-restart state */
  cinfo->marker->next_restart_num = (cinfo->marker->next_restart_num + 1) & 7;

  return TRUE;
}


/*
 * This is the default resync_to_restart method for data source managers
 * to use if they don't have any better approach.  Some data source managers
 * may be able to back up, or may have additional knowledge about the data
 * which permits a more intelligent recovery strategy; such managers would
 * presumably supply their own resync method.
 *
 * read_restart_marker calls resync_to_restart if it finds a marker other than
 * the restart marker it was expecting.  (This code is *not* used unless
 * a nonzero restart interval has been declared.)  cinfo->unread_marker is
 * the marker code actually found (might be anything, except 0 or FF).
 * The desired restart marker number (0..7) is passed as a parameter.
 * This routine is supposed to apply whatever error recovery strategy seems
 * appropriate in order to position the input stream to the next data segment.
 * Note that cinfo->unread_marker is treated as a marker appearing before
 * the current data-source input point; usually it should be reset to zero
 * before returning.
 * Returns FALSE if suspension is required.
 *
 * This implementation is substantially constrained by wanting to treat the
 * input as a data stream; this means we can't back up.  Therefore, we have
 * only the following actions to work with:
 *   1. Simply discard the marker and let the entropy decoder resume at next
 *      byte of file.
 *   2. Read forward until we find another marker, discarding intervening
 *      data.  (In theory we could look ahead within the current bufferload,
 *      without having to discard data if we don't find the desired marker.
 *      This idea is not implemented here, in part because it makes behavior
 *      dependent on buffer size and chance buffer-boundary positions.)
 *   3. Leave the marker unread (by failing to zero cinfo->unread_marker).
 *      This will cause the entropy decoder to process an empty data segment,
 *      inserting dummy zeroes, and then we will reprocess the marker.
 *
 * #2 is appropriate if we think the desired marker lies ahead, while #3 is
 * appropriate if the found marker is a future restart marker (indicating
 * that we have missed the desired restart marker, probably because it got
 * corrupted).
 * We apply #2 or #3 if the found marker is a restart marker no more than
 * two counts behind or ahead of the expected one.  We also apply #2 if the
 * found marker is not a legal JPEG marker code (it's certainly bogus data).
 * If the found marker is a restart marker more than 2 counts away, we do #1
 * (too much risk that the marker is erroneous; with luck we will be able to
 * resync at some future point).
 * For any valid non-restart JPEG marker, we apply #3.  This keeps us from
 * overrunning the end of a scan.  An implementation limited to single-scan
 * files might find it better to apply #2 for markers other than EOI, since
 * any other marker would have to be bogus data in that case.
 */

GLOBAL(boolean)
jpeg_resync_to_restart (j_decompress_ptr cinfo, int desired)
{
  int marker = cinfo->unread_marker;
  int action = 1;

  /* Always put up a warning. */
  WARNMS2(cinfo, JWRN_MUST_RESYNC, marker, desired);

}


/*
 * Reset marker processing state to begin a fresh datastream.
 */

METHODDEF(void)
reset_marker_reader (j_decompress_ptr cinfo)
{
  my_marker_ptr marker = (my_marker_ptr) cinfo->marker;

  cinfo->comp_info = NULL;		/* until allocated by get_sof */
  cinfo->input_scan_number = 0;		/* no SOS seen yet */
  cinfo->unread_marker = 0;		/* no pending marker */
  marker->pub.saw_SOI = FALSE;		/* set internal state too */
  marker->pub.saw_SOF = FALSE;
  marker->pub.discarded_bytes = 0;
  marker->cur_marker = NULL;
}


/*
 * Initialize the marker reader module.
 * This is called only once, when the decompression object is created.
 */

GLOBAL(void)
jinit_marker_reader (j_decompress_ptr cinfo)
{
  my_marker_ptr marker;
  int i;

  /* Create subobject in permanent pool */
  marker = (my_marker_ptr) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_PERMANENT, SIZEOF(my_marker_reader));
  cinfo->marker = &marker->pub;
  /* Initialize public method pointers */
  marker->pub.reset_marker_reader = reset_marker_reader;
  marker->pub.read_markers = read_markers;
  marker->pub.read_restart_marker = read_restart_marker;
  /* Initialize COM/APPn processing.
   * By default, we examine and then discard APP0 and APP14,
   * but simply discard COM and all other APPn.
   */
  marker->process_COM = skip_variable;
/// in phi nodes in our successors.
void MemorySSA::renamePassHelper(DomTreeNode *RootNode, MemoryAccess *IncomingVal_,
                                SmallPtrSetImpl<BasicBlock *> &VisitedNodes,
                                bool SkipVisited_, bool RenameAllUses) {
  assert(RootNode && "Trying to rename accesses in an unreachable block");

  SmallVector<RenamePassData, 32> WorkStack;
  // Note: You can't sink this into the if, because we need it to occur
  // regardless of whether we skip blocks or not.
  bool AlreadyVisited = !VisitedNodes.insert(RootNode->getBlock()).second;
  if (SkipVisited_ && AlreadyVisited)
    return;

  MemoryAccess *IncomingVal = renameBlock(RootNode->getBlock(), IncomingVal_, RenameAllUses);
  renameSuccessorPhis(RootNode->getBlock(), IncomingVal, RenameAllUses);
  WorkStack.push_back({RootNode, RootNode->begin(), IncomingVal});

  while (!WorkStack.empty()) {
    DomTreeNode *Node = WorkStack.back().DTN;
    DomTreeNode::const_iterator ChildIt = WorkStack.back().ChildIt;
    MemoryAccess *IncomingValCopy = WorkStack.back().IncomingVal;

    if (ChildIt == Node->end()) {
      WorkStack.pop_back();
    } else {
      DomTreeNode *ChildNode = *ChildIt;
      ++WorkStack.back().ChildIt;
      BasicBlock *BB = ChildNode->getBlock();
      // Note: You can't sink this into the if, because we need it to occur
      // regardless of whether we skip blocks or not.
      AlreadyVisited = !VisitedNodes.insert(BB).second;
      if (SkipVisited_ && AlreadyVisited) {
        // We already visited this during our renaming, which can happen when
        // being asked to rename multiple blocks. Figure out the incoming val,
        // which is the last def.
        // Incoming value can only change if there is a block def, and in that
        // case, it's the last block def in the list.
        if (auto *BlockDefs = getWritableBlockDefs(BB))
          IncomingValCopy = &*BlockDefs->rbegin();
      } else {
        IncomingValCopy = renameBlock(BB, IncomingValCopy, RenameAllUses);
      }
      renameSuccessorPhis(BB, IncomingValCopy, RenameAllUses);
      WorkStack.push_back({ChildNode, ChildNode->begin(), IncomingValCopy});
    }
  }
}
  marker->process_APPn[0] = get_interesting_appn;
  marker->process_APPn[14] = get_interesting_appn;
  /* Reset marker processing state */
  reset_marker_reader(cinfo);
}


/*
 * Control saving of COM and APPn markers into marker_list.
 */

#ifdef SAVE_MARKERS_SUPPORTED

GLOBAL(void)
jpeg_save_markers (j_decompress_ptr cinfo, int marker_code,
		   unsigned int length_limit)
{
  my_marker_ptr marker = (my_marker_ptr) cinfo->marker;
  long maxlength;
  jpeg_marker_parser_method processor;

  /* Length limit mustn't be larger than what we can allocate
   * (should only be a concern in a 16-bit environment).
   */
  maxlength = cinfo->mem->max_alloc_chunk - SIZEOF(struct jpeg_marker_struct);
  if (((long) length_limit) > maxlength)
    length_limit = (unsigned int) maxlength;

  /* Choose processor routine to use.
   * APP0/APP14 have special requirements.

  if (marker_code == (int) M_COM) {
    marker->process_COM = processor;
    marker->length_limit_COM = length_limit;
  } else if (marker_code >= (int) M_APP0 && marker_code <= (int) M_APP15) {
    marker->process_APPn[marker_code - (int) M_APP0] = processor;
    marker->length_limit_APPn[marker_code - (int) M_APP0] = length_limit;
  } else
    ERREXIT1(cinfo, JERR_UNKNOWN_MARKER, marker_code);
}

#endif /* SAVE_MARKERS_SUPPORTED */


/*
 * Install a special processing method for COM or APPn markers.
 */

GLOBAL(void)
jpeg_set_marker_processor (j_decompress_ptr cinfo, int marker_code,
			   jpeg_marker_parser_method routine)
{
  my_marker_ptr marker = (my_marker_ptr) cinfo->marker;

  if (marker_code == (int) M_COM)
    marker->process_COM = routine;
  else if (marker_code >= (int) M_APP0 && marker_code <= (int) M_APP15)
    marker->process_APPn[marker_code - (int) M_APP0] = routine;
  else
    ERREXIT1(cinfo, JERR_UNKNOWN_MARKER, marker_code);
}
