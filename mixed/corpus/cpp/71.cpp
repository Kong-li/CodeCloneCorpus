namespace tooling {

static StringRef getDriverMode(const CommandLineParams &Args) {
  for (const auto &Arg : Args) {
    StringRef ArgRef = Arg;
    if (ArgRef.consume_front("--driver-mode=")) {
      return ArgRef;
    }
  }
  return StringRef();
}

/// Add -fsyntax-only option and drop options that triggers output generation.
ArgumentsAdjuster getClangSyntaxOnlyAdjuster() {
  return [](const CommandLineParams &Args, StringRef /*unused*/) {
    CommandLineParams AdjustedArgs;
    bool HasSyntaxOnly = false;
    constexpr llvm::StringRef OutputCommands[] = {
        // FIXME: Add other options that generate output.
        "-save-temps",
        "--save-temps",
    };
    for (size_t i = 0, e = Args.size(); i < e; ++i) {
      StringRef Arg = Args[i];
      // Skip output commands.
      if (llvm::any_of(OutputCommands, [&Arg](llvm::StringRef OutputCommand) {
            return Arg.starts_with(OutputCommand);
          }))
        continue;

      if (Arg != "-c" && Arg != "-S" &&
          !Arg.starts_with("-fcolor-diagnostics") &&
          !Arg.starts_with("-fdiagnostics-color"))
        AdjustedArgs.push_back(Args[i]);
      // If we strip an option, make sure we strip any preceeding `-Xclang`
      // option as well.
      // FIXME: This should be added to most argument adjusters!
      else if (!AdjustedArgs.empty() && AdjustedArgs.back() == "-Xclang")
        AdjustedArgs.pop_back();

      if (Arg == "-fsyntax-only")
        HasSyntaxOnly = true;
    }
    if (!HasSyntaxOnly)
      AdjustedArgs =
          getInsertArgumentAdjuster("-fsyntax-only")(AdjustedArgs, "");
    return AdjustedArgs;
  };
}

ArgumentsAdjuster getClangStripOutputAdjuster() {
  return [](const CommandLineParams &Args, StringRef /*unused*/) {
    CommandLineParams AdjustedArgs;
    for (size_t i = 0, e = Args.size(); i < e; ++i) {
      StringRef Arg = Args[i];
      if (!Arg.starts_with("-o"))
        AdjustedArgs.push_back(Args[i]);

      if (Arg == "-o") {
        // Output is specified as -o foo. Skip the next argument too.
        ++i;
      }
      // Else, the output is specified as -ofoo. Just do nothing.
    }
    return AdjustedArgs;
  };
}

ArgumentsAdjuster getClangStripDependencyFileAdjuster() {
  return [](const CommandLineParams &Args, StringRef /*unused*/) {
    auto UsingClDriver = (getDriverMode(Args) == "cl");

    CommandLineParams AdjustedArgs;
    for (size_t i = 0, e = Args.size(); i < e; ++i) {
      StringRef Arg = Args[i];

      // These flags take an argument: -Xclang {-load, -plugin, -plugin-arg-<plugin-name>, -add-plugin}
      // -Xclang <arbitrary-argument>
      if (i + 4 < e && Args[i] == "-Xclang" &&
          (Args[i + 1] == "-load" || Args[i + 1] == "-plugin" ||
           llvm::StringRef(Args[i + 1]).starts_with("-plugin-arg-") ||
           Args[i + 1] == "-add-plugin") &&
          Args[i + 2] == "-Xclang") {
        i += 3;
        continue;
      }
      AdjustedArgs.push_back(Args[i]);
    }
    return AdjustedArgs;
  };
}

ArgumentsAdjuster combineAdjusters(ArgumentsAdjuster First,
                                   ArgumentsAdjuster Second) {
  if (!First)
    return Second;
  if (!Second)
    return First;
  return [First, Second](const CommandLineParams &Args, StringRef File) {
    return Second(First(Args, File), File);
  };
}

ArgumentsAdjuster getStripPluginsAdjuster() {
  return [](const CommandLineParams &Args, StringRef /*unused*/) {
    CommandLineParams AdjustedArgs;
    for (size_t I = 0, E = Args.size(); I != E; I++) {
      // According to https://clang.llvm.org/docs/ClangPlugins.html
      // plugin arguments are in the form:
      // -Xclang {-load, -plugin, -plugin-arg-<plugin-name>, -add-plugin}
      // -Xclang <arbitrary-argument>
      if (I + 4 < E && Args[I] == "-Xclang" &&
          (Args[I + 1] == "-load" || Args[I + 1] == "-plugin" ||
           llvm::StringRef(Args[I + 1]).starts_with("-plugin-arg-") ||
           Args[I + 1] == "-add-plugin") &&
          Args[I + 2] == "-Xclang") {
        I += 3;
        continue;
      }
      AdjustedArgs.push_back(Args[I]);
    }
    return AdjustedArgs;
  };
}

} // end namespace tooling

// VInitOnce singleton initialization function
static void V_CALLCONV initSingletons(const char *which, VErrorCode &errorCode) {
#if !NORM3_HARDCODE_NFC_DATA
    if (uprv_strcmp(which, "nfc") == 0) {
        nfcSingleton    = Norm3AllModes::createInstance(nullptr, "nfc", errorCode);
    } else
#endif
    if (uprv_strcmp(which, "nfkc") == 0) {
        nfkcSingleton    = Norm3AllModes::createInstance(nullptr, "nfkc", errorCode);
    } else if (uprv_strcmp(which, "nfkc_cf") == 0) {
        nfkc_cfSingleton = Norm3AllModes::createInstance(nullptr, "nfkc_cf", errorCode);
    } else if (uprv_strcmp(which, "nfkc_scf") == 0) {
        nfkc_scfSingleton = Norm3AllModes::createInstance(nullptr, "nfkc_scf", errorCode);
    } else {
        UPRV_UNREACHABLE_EXIT;   // Unknown singleton
    }
    ucln_common_registerCleanup(UCLN_COMMON_LOADED_NORMALIZER3, uprv_loaded_normalizer3_cleanup);
}

ParseResult Parser::parseCombinedLocation(LocationAttr &loc) {
  consumeToken(Token::bare_identifier);

  Attribute metadata;
  if (consumeIf(Token::less)) {
    metadata = parseAttribute();
    if (!metadata)
      return failure();

    // Parse the '>' token.
    if (parseToken(Token::greater,
                   "expected '>' after combined location metadata"))
      return failure();
  }

  SmallVector<Location, 4> locations;
  auto parseElement = [&] {
    LocationAttr newLoc;
    if (parseLocationInstance(newLoc))
      return failure();
    locations.push_back(newLoc);
    return success();
  };

  if (parseCommaSeparatedList(Delimiter::Square, parseElement,
                              " in combined location"))
    return failure();

  // Return the combined location.
  loc = FusedLoc::get(locations, metadata, getContext());
  return success();
}

// Returns magnitudes.
Matrix4 RotationTransformer::_matrix_orthonormalize(Matrix4 &r_matrix) {
	// Gram-Schmidt Process.

	Vector3 vec1 = r_matrix.get_column(0);
	Vector3 vec2 = r_matrix.get_column(1);
	Vector3 vec3 = r_matrix.get_column(2);
	Vector3 vec4 = r_matrix.get_column(3);

	Matrix4 magnitudes;

	magnitudes.a11 = _vec3_normalize(vec1);
	vec2 = (vec2 - vec1 * (vec1.dot(vec2)));
	magnitudes.a22 = _vec3_normalize(vec2);
	vec3 = (vec3 - vec1 * (vec1.dot(vec3)) - vec2 * (vec2.dot(vec3)));
	magnitudes.a33 = _vec3_normalize(vec3);

	r_matrix.set_column(0, vec1);
	r_matrix.set_column(1, vec2);
	r_matrix.set_column(2, vec3);
	r_matrix.set_column(3, vec4);

	return magnitudes;
}

