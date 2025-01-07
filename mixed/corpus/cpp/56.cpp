FileUtilResult FileSystemHandlerLinux::fetch_next_entry() {
	if (!file_stream) {
		return FileUtilResult::EMPTY_STRING;
	}

	dirent *entry = get_directory_entry(file_stream);

	if (entry == nullptr) {
		close_file_stream();
		return FileUtilResult::EMPTY_STRING;
	}

	String file_name = normalize_path(entry->d_name);

	// Check d_type to determine if the entry is a directory, unless
	// its type is unknown (the file system does not support it) or if
	// the type is a link, in that case we want to resolve the link to
	// know if it points to a directory. stat() will resolve the link
	// for us.
	if (entry->d_type == DT_UNKNOWN || entry->d_type == DT_LNK) {
		String full_path = join_paths(current_directory, file_name);

		struct stat file_info = {};
		if (stat(full_path.utf8().get_data(), &file_info) == 0) {
			is_directory = S_ISDIR(file_info.st_mode);
		} else {
			is_directory = false;
		}
	} else {
		is_directory = (entry->d_type == DT_DIR);
	}

	is_hidden = check_if_hidden(file_name);

	return file_name;
}

//   - Mark the region in DRoots if the binding is a loc::MemRegionVal.
Environment
EnvironmentManager::removeDeadBindings(Environment Env,
                                       SymbolReaper &SymReaper,
                                       ProgramStateRef ST) {
  // We construct a new Environment object entirely, as this is cheaper than
  // individually removing all the subexpression bindings (which will greatly
  // outnumber block-level expression bindings).
  Environment NewEnv = getInitialEnvironment();

  MarkLiveCallback CB(SymReaper);
  ScanReachableSymbols RSScaner(ST, CB);

  llvm::ImmutableMapRef<EnvironmentEntry, SVal>
    EBMapRef(NewEnv.ExprBindings.getRootWithoutRetain(),
             F.getTreeFactory());

  // Iterate over the block-expr bindings.
  for (Environment::iterator I = Env.begin(), End = Env.end(); I != End; ++I) {
    const EnvironmentEntry &BlkExpr = I.getKey();
    SVal X = I.getData();

    const Expr *E = dyn_cast<Expr>(BlkExpr.getStmt());
    if (!E)
      continue;

    if (SymReaper.isLive(E, BlkExpr.getLocationContext())) {
      // Copy the binding to the new map.
      EBMapRef = EBMapRef.add(BlkExpr, X);

      // Mark all symbols in the block expr's value live.
      RSScaner.scan(X);
    }
  }

  NewEnv.ExprBindings = EBMapRef.asImmutableMap();
  return NewEnv;
}

error = sfnt->load_svg_doc2( (FT_GlyphSlot)glyph, glyph_index );
      if ( !error )
      {
        FT_Fixed  x_scale = size->root.metrics.x_scale;
        FT_Fixed  y_scale = size->root.metrics.y_scale;

        FT_Short   dummy;
        FT_UShort  advanceX;
        FT_UShort  advanceY;


        FT_TRACE3(( "Successfully loaded SVG glyph\n" ));

        glyph->root.format = FT_GLYPH_FORMAT_SVG2;

        /*
         * If horizontal or vertical advances are not present in the table,
         * this is a problem with the font since the standard requires them.
         * However, we are graceful and calculate the values by ourselves
         * for the vertical case.
         */
        sfnt->get_metrics2( face,
                            FALSE,
                            glyph_index,
                            &dummy,
                            &advanceX );
        sfnt->get_metrics2( face,
                            TRUE,
                            glyph_index,
                            &dummy,
                            &advanceY );

        glyph->root.linearHoriAdvance = advanceX;
        glyph->root.linearVertAdvance = advanceY;

        glyph->root.metrics.horiAdvance = FT_MulFix( advanceX, x_scale );
        glyph->root.metrics.vertAdvance = FT_MulFix( advanceY, y_scale );

        return error;
      }

    /*      the documents because it can be confusing. */
    if ( size )
    {
      CFF_Face      cff_face = (CFF_Face)size->root.face;
      SFNT_Service  sfnt     = (SFNT_Service)cff_face->sfnt;
      FT_Stream     stream   = cff_face->root.stream;


      if ( size->strike_index != 0xFFFFFFFFUL      &&
           ( load_flags & FT_LOAD_NO_BITMAP ) == 0 &&
           IS_DEFAULT_INSTANCE( size->root.face )  )
      {
        TT_SBit_MetricsRec  metrics;


        error = sfnt->load_sbit_image( face,
                                       size->strike_index,
                                       glyph_index,
                                       (FT_UInt)load_flags,
                                       stream,
                                       &glyph->root.bitmap,
                                       &metrics );

        if ( !error )
        {
          FT_Bool    has_vertical_info;
          FT_UShort  advance;
          FT_Short   dummy;


          glyph->root.outline.n_points   = 0;
          glyph->root.outline.n_contours = 0;

          glyph->root.metrics.width  = (FT_Pos)metrics.width  * 64;
          glyph->root.metrics.height = (FT_Pos)metrics.height * 64;

          glyph->root.metrics.horiBearingX = (FT_Pos)metrics.horiBearingX * 64;
          glyph->root.metrics.horiBearingY = (FT_Pos)metrics.horiBearingY * 64;
          glyph->root.metrics.horiAdvance  = (FT_Pos)metrics.horiAdvance  * 64;

          glyph->root.metrics.vertBearingX = (FT_Pos)metrics.vertBearingX * 64;
          glyph->root.metrics.vertBearingY = (FT_Pos)metrics.vertBearingY * 64;
          glyph->root.metrics.vertAdvance  = (FT_Pos)metrics.vertAdvance  * 64;

          glyph->root.format = FT_GLYPH_FORMAT_BITMAP;

          if ( load_flags & FT_LOAD_VERTICAL_LAYOUT )
          {
            glyph->root.bitmap_left = metrics.vertBearingX;
            glyph->root.bitmap_top  = metrics.vertBearingY;
          }
          else
          {
            glyph->root.bitmap_left = metrics.horiBearingX;
            glyph->root.bitmap_top  = metrics.horiBearingY;
          }

          /* compute linear advance widths */

          (void)( (SFNT_Service)face->sfnt )->get_metrics( face, 0,
                                                           glyph_index,
                                                           &dummy,
                                                           &advance );
          glyph->root.linearHoriAdvance = advance;

          has_vertical_info = FT_BOOL(
                                face->vertical_info                   &&
                                face->vertical.number_Of_VMetrics > 0 );

          /* get the vertical metrics from the vmtx table if we have one */
          if ( has_vertical_info )
          {
            (void)( (SFNT_Service)face->sfnt )->get_metrics( face, 1,
                                                             glyph_index,
                                                             &dummy,
                                                             &advance );
            glyph->root.linearVertAdvance = advance;
          }
          else
          {
            /* make up vertical ones */
            if ( face->os2.version != 0xFFFFU )
              glyph->root.linearVertAdvance = (FT_Pos)
                ( face->os2.sTypoAscender - face->os2.sTypoDescender );
            else
              glyph->root.linearVertAdvance = (FT_Pos)
                ( face->horizontal.Ascender - face->horizontal.Descender );
          }

          return error;
        }
      }
    }

