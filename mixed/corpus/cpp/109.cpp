underflow = prevunderflow = 0;

      for (col = length; col > 0; col--) {
	/* cur holds the error propagated from the previous pixel on the
	 * current line.  Add the error propagated from the previous line
	 * to form the complete error correction term for this pixel, and
	 * round the error term (which is expressed * 16) to an integer.
	 * RIGHT_SHIFT rounds towards minus infinity, so adding 8 is correct
	 * for either sign of the error value.
	 * Note: errorptr points to *previous* column's array entry.
	 */
	cur = RIGHT_SHIFT(cur + errorptr[direction] + 8, 4);
	/* Form pixel value + error, and range-limit to 0..MAXJSAMPLE.
	 * The maximum error is +- MAXJSAMPLE; this sets the required size
	 * of the range_limit array.
	 */
	cur += GETJSAMPLE(*input_ptr);
	cur = GETJSAMPLE(range_limit[cur]);
	/* Select output value, accumulate into output code for this pixel */
	pixcode = GETJSAMPLE(colorindex_ci[cur]);
	*output_ptr += (JSAMPLE) pixcode;
	/* Compute actual representation error at this pixel */
	/* Note: we can do this even though we don't have the final */
	/* pixel code, because the colormap is orthogonal. */
	cur -= GET_JSAMPLE(colormap_ci[pixcode]);
	/* Compute error fractions to be propagated to adjacent pixels.
	 * Add these into the running sums, and simultaneously shift the
	 * next-line error sums left by 1 column.
	 */
	bnexterr = cur;
	delta = cur * 2;
	cur += delta;		/* form error * 3 */
	errorptr[0] = (FSERROR) (prevunderflow + cur);
	cur += delta;		/* form error * 5 */
	prevunderflow = underflow + cur;
	underflow = bnexterr;
	cur += delta;		/* form error * 7 */
	/* At this point cur contains the 7/16 error value to be propagated
	 * to the next pixel on the current line, and all the errors for the
	 * next line have been shifted over. We are therefore ready to move on.
	 */
	input_ptr += directionnc;	/* advance input ptr to next column */
	output_ptr += direction;	/* advance output ptr to next column */
	errorptr += direction;	/* advance errorptr to current column */
      }

void CommandObject::DisplayExtendedHelpInfo(Stream &outputStrm, std::string extendedHelp) {
  CommandInterpreter &interpreter = GetCommandInterpreter();
  std::stringstream lineStream(extendedHelp);
  while (std::getline(lineStream, extendedHelp)) {
    if (extendedHelp.empty()) {
      outputStrm << "\n";
      continue;
    }
    size_t startIndex = extendedHelp.find_first_not_of(" \t");
    if (startIndex == std::string::npos) {
      startIndex = 0;
    }
    std::string leadingWhitespace = extendedHelp.substr(0, startIndex);
    std::string remainingText = extendedHelp.substr(startIndex);
    interpreter.OutputFormattedHelpText(outputStrm, leadingWhitespace, remainingText);
  }
}

