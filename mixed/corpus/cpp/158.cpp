    LLVMFuzzerInitialize(&argc, &argv);
  for (int i = 1; i < argc; i++) {
    fprintf(stderr, "Running: %s\n", argv[i]);
    FILE *f = fopen(argv[i], "r");
    assert(f);
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char*)malloc(len);
    size_t n_read = fread(buf, 1, len, f);
    fclose(f);
    assert(n_read == len);
    LLVMFuzzerTestOneInput(buf, len);
    free(buf);
    fprintf(stderr, "Done:    %s: (%zd bytes)\n", argv[i], n_read);
  }

/// gettok - Return the next token from standard input.
static int gettok() {
  static int LastChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = getchar();

  if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    while (isalnum((LastChar = getchar())))
      IdentifierStr += LastChar;

    if (IdentifierStr == "def") return tok_def;
    if (IdentifierStr == "extern") return tok_extern;
    if (IdentifierStr == "if") return tok_if;
    if (IdentifierStr == "then") return tok_then;
    if (IdentifierStr == "else") return tok_else;
    if (IdentifierStr == "for") return tok_for;
    if (IdentifierStr == "in") return tok_in;
    if (IdentifierStr == "binary") return tok_binary;
    if (IdentifierStr == "unary") return tok_unary;
    if (IdentifierStr == "var") return tok_var;
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') {   // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), 0);
    return tok_number;
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return gettok();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
}

void ProcessInternal(const uint8_t* base,
                     const FieldMetadata* field_metadata_table,
                     int32_t num_fields, io::CodedOutputStream* output) {
  SpecialSerializer func = nullptr;
  for (int i = 0; i < num_fields; ++i) {
    const FieldMetadata& metadata = field_metadata_table[i];
    const uint8_t* ptr = base + metadata.offset;
    switch (metadata.type) {
      case WireFormatLite::TYPE_DOUBLE:
        OneOfFieldHelper<double>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_FLOAT:
        OneOfFieldHelper<float>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_INT64:
        OneOfFieldHelper<int64_t>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_UINT64:
        OneOfFieldHelper<uint64_t>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_INT32:
        OneOfFieldHelper<int32_t>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_FIXED64:
        OneOfFieldHelper<uint64_t>::FixedSerialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_FIXED32:
        OneOfFieldHelper<uint32_t>::FixedSerialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_BOOL:
        OneOfFieldHelper<bool>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_STRING:
        OneOfFieldHelper<std::string>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_GROUP:
        func = reinterpret_cast<SpecialSerializer>(
            const_cast<void*>(metadata.ptr));
        func(base, metadata.offset, metadata.tag,
             metadata.has_offset, output);
        break;
      case WireFormatLite::TYPE_MESSAGE:
        OneOfFieldHelper<Message>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_BYTES:
        OneOfFieldHelper<std::vector<uint8_t>>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_UINT32:
        OneOfFieldHelper<uint32_t>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_ENUM:
        OneOfFieldHelper<int32_t>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_SFIXED32:
        OneOfFieldHelper<int32_t>::FixedSerialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_SFIXED64:
        OneOfFieldHelper<int64_t>::FixedSerialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_SINT32:
        OneOfFieldHelper<int32_t>::Serialize(ptr, metadata, output);
        break;
      case WireFormatLite::TYPE_SINT64:
        OneOfFieldHelper<int64_t>::Serialize(ptr, metadata, output);
        break;
      case FieldMetadata::kInlinedType:
        func = reinterpret_cast<SpecialSerializer>(
            const_cast<void*>(metadata.ptr));
        func(base, metadata.offset, metadata.tag,
             metadata.has_offset, output);
        break;
      default:
        // __builtin_unreachable()
        SerializeNotImplemented(metadata.type);
    }
  }
}

memcopy(dstOuter, src, srcroi.height*elemSize);

        if( boolMode )
        {
            const float* isrc = (float*)src;
            float* idstOuter = (float*)dstOuter;
            for( k = 0; k < bottom; k++ )
                idstOuter[k - bottom] = isrc[map[k]];
            for( k = 0; k < top; k++ )
                idstOuter[k + srcroi.height] = isrc[map[k + bottom]];
        }

void calculatePCAFeatures(const cv::Mat& inputData, cv::Mat& meanResult,
                          std::vector<cv::Mat>& eigenvectorsResult,
                          std::vector<double>& eigenvaluesResult,
                          double varianceThreshold)
{
    CV_INSTRUMENT_REGION();

    PCA pca;
    pca(inputData, meanResult, 0, varianceThreshold);
    meanResult.copyTo(meanResult);
    for (size_t i = 0; i < pca.eigenvectors.size(); ++i) {
        eigenvectorsResult.push_back(pca.eigenvectors.row(i));
    }
    eigenvaluesResult.assign(pca.eigenvalues.begin(), pca.eigenvalues.end());
}

