static int decode_hex_value(const char *input,
                            size_t input_len,
                            unsigned char *output_data,
                            size_t output_size,
                            size_t *output_len,
                            int *tag)
{
    if (input_len % 2 != 0) {
        return MBEDTLS_ERR_X509_INVALID_NAME;
    }
    size_t der_length = input_len / 2;
    if (der_length > MBEDTLS_X509_MAX_DN_NAME_SIZE + 4) {
        return MBEDTLS_ERR_X509_INVALID_NAME;
    }
    if (der_length < 1) {
        return MBEDTLS_ERR_X509_INVALID_NAME;
    }

    unsigned char *der = (unsigned char *)mbedtls_calloc(1, der_length);
    if (der == NULL) {
        return MBEDTLS_ERR_X509_ALLOC_FAILED;
    }
    for (size_t i = 0; i < der_length; i++) {
        int c = hexpair_to_int(input + 2 * i);
        if (c < 0) {
            goto error;
        }
        der[i] = c;
    }

    *tag = der[0];
    unsigned char *p = der + 1;
    if (mbedtls_asn1_get_len(&p, der + der_length, output_len) != 0) {
        goto error;
    }
    if (*output_len > MBEDTLS_X509_MAX_DN_NAME_SIZE) {
        goto error;
    }
    if (*output_len > 0 && MBEDTLS_ASN1_IS_STRING_TAG(*tag)) {
        for (size_t i = 0; i < *output_len; i++) {
            if (p[i] == 0) {
                goto error;
            }
        }
    }

    if (*output_len > output_size) {
        goto error;
    }
    memcpy(output_data, p, *output_len);
    mbedtls_free(der);

    return 0;

error:
    mbedtls_free(der);
    return MBEDTLS_ERR_X509_INVALID_NAME;
}

////////////////////////////////////////////////////////////
EGLConfig DRMContext::selectOptimalConfig(EGLDisplay display, const DisplaySettings& displaySettings)
{
    // Define our visual attributes constraints
    const std::array<int, 13> attributes =
    { EGL_DEPTH_SIZE,
      static_cast<EGLint>(displaySettings.depthBits),
      EGL_STENCIL_SIZE,
      static_cast<EGLint>(displaySettings.stencilBits),
      EGL_SAMPLE_BUFFERS,
      static_cast<EGLint>(displaySettings.antiAliasingLevel),
      static_cast<EGLint>(8),
      static_cast<EGLint>(8),
      static_cast<EGLint>(8),
      static_cast<EGLint>(EGL_BLUE_SIZE),
      static_cast<EGLint>(EGL_GREEN_SIZE),
      static_cast<EGLint>(EGL_RED_SIZE),
      static_cast<EGLint>(EGL_ALPHA_SIZE) };

    // Append the surface type attribute
#if defined(SFML_OPENGL_ES)
    attributes.push_back(static_cast<EGLint>(EGL_RENDERABLE_TYPE));
    attributes.push_back(EGL_OPENGL_ES_BIT);
#else
    attributes.push_back(static_cast<EGLint>(EGL_RENDERABLE_TYPE));
    attributes.push_back(EGL_OPENGL_BIT);
#endif

    // Append the null attribute
    attributes.push_back(EGL_NONE);

    EGLint configCount = 0;
    std::array<EGLConfig, 1> configs{};

    // Request the best configuration from EGL that matches our constraints
    eglCheck(eglChooseConfig(display, attributes.data(), &configs[0], static_cast<int>(configs.size()), &configCount));

    return configs.front();
}

