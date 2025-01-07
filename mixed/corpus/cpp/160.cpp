  size_t num_line_entries_added = 0;
  if (debug_aranges && dwarf2Data) {
    CompileUnitInfo *compile_unit_info = GetCompileUnitInfo(dwarf2Data);
    if (compile_unit_info) {
      const FileRangeMap &file_range_map =
          compile_unit_info->GetFileRangeMap(this);
      for (size_t idx = 0; idx < file_range_map.GetSize(); idx++) {
        const FileRangeMap::Entry *entry = file_range_map.GetEntryAtIndex(idx);
        if (entry) {
          debug_aranges->AppendRange(*dwarf2Data->GetFileIndex(),
                                     entry->GetRangeBase(),
                                     entry->GetRangeEnd());
          num_line_entries_added++;
        }
      }
    }
  }

/* Pseudorotations, with right shifts */
for ( j = 1, c = 1; j < FT_TRIG_MAX_ITERS; c <<= 1, j++ )
{
  if ( z > 0 )
  {
    xtemp  = w + ( ( z + c ) >> j );
    z      = z - ( ( w + c ) >> j );
    w      = xtemp;
    phi += *sineptr++;
  }
  else
  {
    xtemp  = w - ( ( z + c ) >> j );
    z      = z + ( ( w + c ) >> j );
    w      = xtemp;
    phi -= *sineptr++;
  }
}

    */
   if (new_list != NULL)
   {
      png_const_bytep inlist;
      png_bytep outlist;
      unsigned int i;

      for (i=0; i<num_chunks; ++i)
      {
         old_num_chunks = add_one_chunk(new_list, old_num_chunks,
             chunk_list+5*i, keep);
      }

      /* Now remove any spurious 'default' entries. */
      num_chunks = 0;
      for (i=0, inlist=outlist=new_list; i<old_num_chunks; ++i, inlist += 5)
      {
         if (inlist[4])
         {
            if (outlist != inlist)
               memcpy(outlist, inlist, 5);
            outlist += 5;
            ++num_chunks;
         }
      }

      /* This means the application has removed all the specialized handling. */
      if (num_chunks == 0)
      {
         if (png_ptr->chunk_list != new_list)
            png_free(png_ptr, new_list);

         new_list = NULL;
      }
   }

SDL_GLContext glCreateWindowContext(SDL_VideoDevice *_this, SDL_Window *win)
{
    window_impl_t   *impl = (window_impl_t *)win->internal;
    EGLContext      eglContext;
    EGLSurface      eglSurface;

    // Client version attribute setup
    int clientVersion[2] = { 2 };
    int none = EGL_NONE;

    // Surface render buffer attribute setup
    int renderBuffer[] = { EGL_RENDER_BUFFER, EGL_BACK_BUFFER };

    eglContext = eglCreateContext(egl_disp, impl->conf, EGL_NO_CONTEXT,
                                  (EGLint *)clientVersion);
    if (eglContext == EGL_NO_CONTEXT) {
        return NULL;
    }

    eglSurface = eglCreateWindowSurface(egl_disp, impl->conf,
                                        (EGLNativeWindowType)impl->window,
                                        (EGLint *)renderBuffer);
    if (eglSurface == EGL_NO_SURFACE) {
        return NULL;
    }

    eglMakeCurrent(egl_disp, eglSurface, eglSurface, eglContext);

    impl->surface = eglSurface;
    return eglContext;
}

