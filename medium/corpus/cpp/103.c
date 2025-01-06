/*
 * Copyright (c) 1988-1997 Sam Leffler
 * Copyright (c) 1991-1997 Silicon Graphics, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Sam Leffler and Silicon Graphics.
 *
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * IN NO EVENT SHALL SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

/*
 * TIFF Library.
 */
#include "tiffiop.h"
#include <string.h>

/************************************************************************/
/*                            TIFFCleanup()                             */
/************************************************************************/

/**
 * Auxiliary function to free the TIFF structure. Given structure will be
 * completely freed, so you should save opened file handle and pointer
 * to the close procedure in external variables before calling
 * _TIFFCleanup(), if you will need these ones to close the file.
 *
 * @param tif A TIFF pointer.

/************************************************************************/
/*                    _TIFFCleanupIFDOffsetAndNumberMaps()              */

/************************************************************************/
/*                            TIFFClose()                               */
/************************************************************************/

/**
 * Close a previously opened TIFF file.
 *
 * TIFFClose closes a file that was previously opened with TIFFOpen().
 * Any buffered data are flushed to the file, including the contents of
 * the current directory (if modified); and all resources are reclaimed.
 *
 * @param tif A TIFF pointer.
        fmtC = fmt[fmtIx++];
        if (fmtC != '%') {
            /* Literal character, not part of a %sequence.  Just copy it to the output. */
            outputChar(fmtC, outBuf, &outIx, capacity, indent);
            if (fmtC == 0) {
                /* We hit the NUL that terminates the format string.
                 * This is the normal (and only) exit from the loop that
                 * interprets the format
                 */
                break;
            }
            continue;
        }
