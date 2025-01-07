    p = gxvalid->root->base + settingTable;
    for ( last_setting = -1, i = 0; i < nSettings; i++ )
    {
      gxv_feat_setting_validate( p, limit, exclusive, gxvalid );

      if ( (FT_Int)GXV_FEAT_DATA( setting ) <= last_setting )
        GXV_SET_ERR_IF_PARANOID( FT_INVALID_FORMAT );

      last_setting = (FT_Int)GXV_FEAT_DATA( setting );
      /* setting + nameIndex */
      p += ( 2 + 2 );
    }

namespace {

void
bufferedReadPixels (InputFile::Data* ifd, int scanLine1, int scanLine2)
{
    //
    // bufferedReadPixels reads each row of tiles that intersect the
    // scan-line range (scanLine1 to scanLine2). The previous row of
    // tiles is cached in order to prevent redundent tile reads when
    // accessing scanlines sequentially.
    //

    int minY = std::min (scanLine1, scanLine2);
    int maxY = std::max (scanLine1, scanLine2);

    if (minY < ifd->minY || maxY >  ifd->maxY)
    {
        throw IEX_NAMESPACE::ArgExc ("Tried to read scan line outside "
			   "the image file's data window.");
    }

    //
    // The minimum and maximum y tile coordinates that intersect this
    // scanline range
    //

    int minDy = (minY - ifd->minY) / ifd->tFile->tileYSize();
    int maxDy = (maxY - ifd->minY) / ifd->tFile->tileYSize();

    //
    // Figure out which one is first in the file so we can read without seeking
    //

    int yStart, yEnd, yStep;

    if (ifd->lineOrder == DECREASING_Y)
    {
        yStart = maxDy;
        yEnd = minDy - 1;
        yStep = -1;
    }
    else
    {
        yStart = minDy;
        yEnd = maxDy + 1;
        yStep = 1;
    }

    //
    // the number of pixels in a row of tiles
    //

    Box2i levelRange = ifd->tFile->dataWindowForLevel(0);

    //
    // Read the tiles into our temporary framebuffer and copy them into
    // the user's buffer
    //

    for (int j = yStart; j != yEnd; j += yStep)
    {
        Box2i tileRange = ifd->tFile->dataWindowForTile (0, j, 0);

        int minYThisRow = std::max (minY, tileRange.min.y);
        int maxYThisRow = std::min (maxY, tileRange.max.y);

        if (j != ifd->cachedTileY)
        {
            //
            // We don't have any valid buffered info, so we need to read in
            // from the file.
            //

            ifd->tFile->readTiles (0, ifd->tFile->numXTiles (0) - 1, j, j);
            ifd->cachedTileY = j;
        }

        //
        // Copy the data from our cached framebuffer into the user's
        // framebuffer.
        //

        for (FrameBuffer::ConstIterator k = ifd->cachedBuffer->begin();
             k != ifd->cachedBuffer->end();
             ++k)
        {
            Slice fromSlice = k.slice();		// slice to write from
            Slice toSlice = ifd->tFileBuffer[k.name()];	// slice to write to

            char *fromPtr, *toPtr;
            int size = pixelTypeSize (toSlice.type);

	    int xStart = levelRange.min.x;
	    int yStart = minYThisRow;

	    while (modp (xStart, toSlice.xSampling) != 0)
		++xStart;

	    while (modp (yStart, toSlice.ySampling) != 0)
		++yStart;

            for (int y = yStart;
		 y <= maxYThisRow;
		 y += toSlice.ySampling)
            {
		//
                // Set the pointers to the start of the y scanline in
                // this row of tiles
		//

                fromPtr = fromSlice.base +
                          (y - tileRange.min.y) * fromSlice.yStride +
                          xStart * fromSlice.xStride;

                toPtr = toSlice.base +
                        divp (y, toSlice.ySampling) * toSlice.yStride +
                        divp (xStart, toSlice.xSampling) * toSlice.xStride;

		//
                // Copy all pixels for the scanline in this row of tiles
		//

                for (int x = xStart;
		     x <= levelRange.max.x;
		     x += toSlice.xSampling)
                {
		    for (int i = 0; i < size; ++i)
			toPtr[i] = fromPtr[i];

		    fromPtr += fromSlice.xStride * toSlice.xSampling;
		    toPtr += toSlice.xStride;
                }
            }
        }
    }
}

} // namespace

// because it needs to compute a CRC.
uint32_t FileCOFFWriter::writeSectionContents(MCAssembler &AsmObj,
                                              const MCSection &Sec) {
  // Save the contents of the section to a temporary buffer, we need this
  // to CRC the data before we dump it into the object file.
  SmallVector<char, 128> Buffer;
  raw_svector_ostream VecOS(Buffer);
  AsmObj.writeSectionData(VecOS, &Sec);

  // Write the section contents to the object file.
  W.OutputStream << Buffer;

  // Calculate our CRC with an initial value of '0', this is not how
  // JamCRC is specified but it aligns with the expected output.
  JamCRC JC(/*Init=*/0);
  JC.update(ArrayRef(reinterpret_cast<uint8_t *>(Buffer.data()), Buffer.size()));
  return JC.getCRC();
}

static long glx_find_extensions_xlib(XDisplay *display) {
    const char *extensions;
    if (!glx_get_extensions(display, &extensions)) return 0;

    GLX_XORG_EXT_import_context = glx_has_extension(extensions, "GLX_XORG_EXT_import_context");
    GLX_MESA_platform_base = glx_has_extension(extensions, "GLX_MESA_platform_base");
    GLX_KHR_create_context = glx_has_extension(extensions, "GLX_KHR_create_context");
    GLX_OML_sync_control = glx_has_extension(extensions, "GLX_OML_sync_control");

    return 1;
}

// equality constraints (if possible), and direction vectors from inequalities.
static void computeDirectionVector(
    const FlatAffineValueConstraints &sourceDomain,
    const FlatAffineValueConstraints &targetDomain, unsigned iterationDepth,
    IntegerPolyhedron *dependenceScope,
    SmallVector<DependenceComponent, 2> *dependenceElements) {
  // Determine the number of shared iterations between source and target accesses.
  SmallVector<AffineForOp, 4> sharedLoops;
  unsigned commonIterationsCount =
      getCommonIterationCount(sourceDomain, targetDomain, &sharedLoops);
  if (commonIterationsCount == 0)
    return;

  // Compute direction vectors for the specified iteration depth.
  unsigned totalVariables = dependenceScope->getVariableCount();
  // Introduce new variables in 'dependenceScope' to represent direction constraints for each shared iteration.
  dependenceScope->insertVariable(VarKind::Direction, /*position=*/0,
                                  /*quantity=*/commonIterationsCount);

  // Add equality constraints for each common loop, setting the newly introduced variable at column 'j' to the difference between target and source IVs.
  SmallVector<int64_t, 4> constraintEq;
  constraintEq.resize(dependenceScope->getColumnCount());
  unsigned sourceDimensions = sourceDomain.getDimensionVariableCount();
  // Constraint variables format:
  // [num-common-loops][num-source-dim-ids][num-target-dim-ids][num-symbols][constant]
  for (unsigned j = 0; j < commonIterationsCount; ++j) {
    std::fill(constraintEq.begin(), constraintEq.end(), 0);
    constraintEq[j] = 1;
    constraintEq[j + commonIterationsCount] = 1;
    constraintEq[j + commonIterationsCount + sourceDimensions] = -1;
    dependenceScope->addEqualityConstraint(constraintEq);
  }

  // Eliminate all variables other than the direction variables just added.
  dependenceScope->reduceToVariables(commonIterationsCount, totalVariables);

  // Traverse each common loop variable column and set direction vectors based on the eliminated constraint system.
  dependenceElements->resize(commonIterationsCount);
  for (unsigned j = 0; j < commonIterationsCount; ++j) {
    (*dependenceElements)[j].operation = sharedLoops[j].getOperation();
    auto lowerBound = dependenceScope->getLowerBound64(j);
    (*dependenceElements)[j].lowerBound =
        lowerBound.hasValue() ? lowerBound.value() : std::numeric_limits<int64_t>::min();
    auto upperBound = dependenceScope->getUpperBound64(j);
    (*dependenceElements)[j].upperBound =
        upperBound.hasValue() ? upperBound.value() : std::numeric_limits<int64_t>::max();
  }
}

