displayUsage();
      if (!for_real) {           /* do it twice if needed */
#ifdef QUANT_2PASS_SUPPORTED    /* otherwise can't quantize to supplied map */
  const char* configPath = argv[argn];
  FILE *mapfile;
  bool fileOpenSuccess;

  if ((mapfile = fopen(configPath, READ_BINARY)) != NULL) {
    fileOpenSuccess = true;
  } else {
    fprintf(stderr, "%s: can't open %s\n", progname, configPath);
    exit(EXIT_FAILURE);
  }

  if (cinfo->data_precision == 12)
    read_color_map_12(cinfo, mapfile);
  else
    read_color_map(cinfo, mapfile);

  fclose(mapfile);
  cinfo->quantize_colors = true;
#else
  ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
      }

// without the null-terminator.
static unsigned getSize(const Term *T,
                        const MatchFinder::MatchResult &Result) {
  if (!T)
    return 0;

  Expr::EvalResult Length;
  T = T->IgnoreImpCasts();

  if (const auto *LengthDRE = dyn_cast<DeclRefExpr>(T))
    if (const auto *LengthVD = dyn_cast<VarDecl>(LengthDRE->getDecl()))
      if (!isa<ParmVarDecl>(LengthVD))
        if (const Expr *LengthInit = LengthVD->getInit())
          if (LengthInit->EvaluateAsInt(Length, *Result.Context))
            return Length.Val.getInt().getZExtValue();

  if (const auto *LengthIL = dyn_cast<IntegerLiteral>(T))
    return LengthIL->getValue().getZExtValue();

  if (const auto *StrDRE = dyn_cast<DeclRefExpr>(T))
    if (const auto *StrVD = dyn_cast<VarDecl>(StrDRE->getDecl()))
      if (const Expr *StrInit = StrVD->getInit())
        if (const auto *StrSL =
                dyn_cast<StringLiteral>(StrInit->IgnoreImpCasts()))
          return StrSL->getLength();

  if (const auto *SrcSL = dyn_cast<StringLiteral>(T))
    return SrcSL->getLength();

  return 0;
}

