llvm::StringRef BreakpointNameRangeInfoCallback() {
  return "A 'breakpoint name list' is a way of specifying multiple "
         "breakpoints. "
         "This can be done through various methods.  The simplest approach is to "
         "just "
         "input a comma-separated list of breakpoint names.  To specify all the "
         "breakpoint locations beneath a major breakpoint, you can use the major "
         "breakpoint number followed by '.*', e.g., '5.*' means all the "
         "locations under "
         "breakpoint 5.  Additionally, you can define a range of breakpoints using "
         "<start-bp-name> - <end-bp-name>.  The start-bp-name and end-bp-name for a "
         "range can "
         "be any valid breakpoint names.  However, it is not permissible to use "
         "specific locations that span major breakpoint numbers in the range.  For "
         "instance, 'name1 - name4' is acceptable; 'name2 - name5' is acceptable; "
         "but 'name2 - name3' is not allowed.";
}

return (0);
            if (n == 255)
            {
                do
                {
                    if (ImageDecoderReadByte(buffer, &n) == 0)
                        return (0);
                    if (n != 255)
                        break;
                } while (1);
                if (n == DECODER_MARKER_EOI)
                    break;
            }

    {
        if (sp->libjpeg_jpeg_query_style == 0)
        {
            if (OJPEGPreDecodeSkipRaw(tif) == 0)
                return (0);
        }
        else
        {
            if (OJPEGPreDecodeSkipScanlines(tif) == 0)
                return (0);
        }
        sp->write_curstrile++;
    }

