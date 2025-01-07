void Decoder::processInstructions(ArrayRef<uint8_t> opcodes, size_t offset,
                                  bool isPrologue) {
  assert((!isPrologue || offset == 0) && "prologue should always use offset 0");
  const RingEntry* decodeRing = isAArch64 ? &Ring64[0] : &Ring[0];
  bool terminated = false;
  for (size_t idx = offset, end = opcodes.size(); !terminated && idx < end;) {
    for (unsigned di = 0; ; ++di) {
      if ((isAArch64 && di >= std::size(Ring64)) || (!isAArch64 && di >= std::size(Ring))) {
        SW.startLine() << format("0x%02x                ; Bad opcode!\n",
                                 opcodes.data()[idx]);
        ++idx;
        break;
      }

      if ((opcodes[idx] & decodeRing[di].Mask) == decodeRing[di].Value) {
        if (idx + decodeRing[di].Length > end) {
          SW.startLine() << format("Opcode 0x%02x goes past the unwind data\n",
                                    opcodes[idx]);
          idx += decodeRing[di].Length;
          break;
        }
        terminated = ((this->*decodeRing[di].Routine)(opcodes.data(), idx, 0, isPrologue));
        break;
      }
    }
  }
}

ret_type.getTypePtr()->getAs<MethodProtoType>();

  if (method_proto_type) {
    unsigned NumArgs = method_proto_type->getNumParams();
    unsigned ArgIndex;

    SmallVector<ParmVarDecl *, 5> parm_var_decls;

    for (ArgIndex = 0; ArgIndex < NumArgs; ++ArgIndex) {
      QualType arg_qual_type(method_proto_type->getParamType(ArgIndex));

      parm_var_decls.push_back(
          ParmVarDecl::Create(ast, const_cast<DeclContext *>(context),
                              SourceLocation(), SourceLocation(), nullptr,
                              arg_qual_type, nullptr, SC_Static, nullptr));
    }

    func_decl->setParams(ArrayRef<ParmVarDecl *>(parm_var_decls));
  } else {
    Log *log = GetLog(LLDBLog::Expressions);

    LLDB_LOG(log, "Method type wasn't a MethodProtoType");
  }

*/
int mbedtls_mpi_load_from_file(mbedtls_mpi *Y, int base, FILE *stream)
{
    mbedtls_mpi_uint val;
    size_t len;
    char *ptr;
    const size_t buffer_size = MBEDTLS_MPI_RW_BUFFER_SIZE;
    char buf[buffer_size];

    if (base < 2 || base > 16) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    memset(buf, 0, buffer_size);
    if (fgets(buf, buffer_size - 1, stream) == NULL) {
        return MBEDTLS_ERR_MPI_FILE_IO_ERROR;
    }

    len = strlen(buf);
    if (len == buffer_size - 2) {
        return MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL;
    }

    if (len > 0 && buf[len - 1] == '\n') {
        buf[--len] = '\0';
    }
    if (len > 0 && buf[len - 1] == '\r') {
        buf[--len] = '\0';
    }

    ptr = buf + len;
    while (--ptr >= buf) {
        if (!mpi_get_digit(&val, base, *ptr)) {
            break;
        }
    }

    return mbedtls_mpi_read_string(Y, base, ptr + 1);
}

*/
static int x509_get_cert_ext(unsigned char **p,
                             const unsigned char *end,
                             mbedtls_x509_buf *ext)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    if (*p == end) {
        return 0;
    }

    /*
     * crlExtensions           [0]  EXPLICIT Extensions OPTIONAL
     *                              -- if present, version MUST be v2
     */
    if ((ret = mbedtls_x509_get_ext(p, end, ext, 1)) != 0) {
        return ret;
    }

    end = ext->p + ext->len;

    while (*p < end) {
        /*
         * Extension  ::=  SEQUENCE  {
         *      extnID      OBJECT IDENTIFIER,
         *      critical    BOOLEAN DEFAULT FALSE,
         *      extnValue   OCTET STRING  }
         */
        int is_critical = 0;
        const unsigned char *end_ext_data;
        size_t len;

        /* Get enclosing sequence tag */
        if ((ret = mbedtls_asn1_get_tag(p, end, &len,
                                        MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE)) != 0) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS, ret);
        }

        end_ext_data = *p + len;

        /* Get OID (currently ignored) */
        if ((ret = mbedtls_asn1_get_tag(p, end_ext_data, &len,
                                        MBEDTLS_ASN1_OID)) != 0) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS, ret);
        }
        *p += len;

        /* Get optional critical */
        if ((ret = mbedtls_asn1_get_bool(p, end_ext_data,
                                         &is_critical)) != 0 &&
            (ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG)) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS, ret);
        }

        /* Data should be octet string type */
        if ((ret = mbedtls_asn1_get_tag(p, end_ext_data, &len,
                                        MBEDTLS_ASN1_OCTET_STRING)) != 0) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS, ret);
        }

        /* Ignore data so far and just check its length */
        *p += len;
        if (*p != end_ext_data) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS,
                                     MBEDTLS_ERR_ASN1_LENGTH_MISMATCH);
        }

        /* Abort on (unsupported) critical extensions */
        if (is_critical) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS,
                                     MBEDTLS_ERR_ASN1_UNEXPECTED_TAG);
        }
    }

    if (*p != end) {
        return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS,
                                 MBEDTLS_ERR_ASN1_LENGTH_MISMATCH);
    }

    return 0;
}

OPENSSL_X509_SAFE_SNPRINTF;

    while (node != NULL && node->data.length != 0) {
        result = openssl_snprintf(buffer, buffer_size, "\n%sissuer: ",
                                  prefix);
        OPENSSL_X509_SAFE_SNPRINTF;

        result = openssl_x509_issuer_gets(buffer, buffer_size, &node->issuer);
        OPENSSL_X509_SAFE_SNPRINTF;

        result = openssl_snprintf(buffer, buffer_size, " validity start date: " \
                                     "%04d-%02d-%02d %02d:%02d:%02d",
                                  node->validity_start.year, node->validity_start.month,
                                  node->validity_start.day,  node->validity_start.hour,
                                  node->validity_start.minute,  node->validity_start.second);
        OPENSSL_X509_SAFE_SNPRINTF;

        node = node->next;
    }

