        size_t ToSkip = 0;
        switch (s[1]) {
        case '8': // i8 suffix
          Bits = 8;
          ToSkip = 2;
          break;
        case '1':
          if (s[2] == '6') { // i16 suffix
            Bits = 16;
            ToSkip = 3;
          }
          break;
        case '3':
          if (s[2] == '2') { // i32 suffix
            Bits = 32;
            ToSkip = 3;
          }
          break;
        case '6':
          if (s[2] == '4') { // i64 suffix
            Bits = 64;
            ToSkip = 3;
          }
          break;
        default:
          break;
        }

// Process the bottom row (row == Lt.rows - 1)
  if (!row_end) {
    float* lt_a = &Lt.ptr<float>(row - 1)[1];  /* Skip the left-most column by +1 */
    float* lf_a = &Lf.ptr<float>(row - 1)[1];
    const float* lt_c = Lt.ptr<float>(row);
    const float* lf_c = Lf.ptr<float>(row);

    // fill the corner to prevent uninitialized values
    float* dst = Lstep.ptr<float>(row);
    *dst++ = 0.0f;

    for (int j = 0; j < cols - 1; ++j) {
      const float step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
                           (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
                           (lf_c[j] + *lt_a)*(*(lt_a + j) - lt_c[j]);
      *dst++ = step_r * step_size;
    }

    // fill the corner to prevent uninitialized values
    *dst = 0.0f;
  }

bool SDL_EGL_LoadLibraryOnlyInternal(SDL_VideoDevice *_this, const char *eglPath)
{
    if (_this->egl_data != NULL) {
        return SDL_SetError("EGL context already created");
    }

    _this->egl_data = (struct SDL_EGL_VideoData *)SDL_calloc(1U, sizeof(SDL_EGL_VideoData));
    bool result = _this->egl_data ? true : false;

    if (!result) {
        return false;
    }

    result = SDL_EGL_LoadLibraryInternal(_this, eglPath);
    if (!result) {
        SDL_free(_this->egl_data);
        _this->egl_data = NULL;
        return false;
    }
    return true;
}

    double v;
    for (q = X; q <= Z; q++) {
        v = vert[q];
        if (normal[q] > 0.0) {
            vmin[q] = -maxbox[q] - v;
            vmax[q] = maxbox[q] - v;
        }
        else {
            vmin[q] = maxbox[q] - v;
            vmax[q] = -maxbox[q] - v;
        }
    }

