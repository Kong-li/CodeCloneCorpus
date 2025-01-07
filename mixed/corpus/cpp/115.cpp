DerivedTypeInfo(DerivedTypeInfo), Span(Span),
      DirectInitSpan(DirectInitSpan) {

  assert((Initializer != nullptr ||
          InitializationStyle == CXXNewInitializationStyle::None) &&
         "Only CXXNewInitializationStyle::None can have no initializer!");

  CXXNewExprBits.IsGlobalNew = IsGlobalNew;
  CXXNewExprBits.IsArray = ArraySize.has_value();
  CXXNewExprBits.ShouldPassAlignment = ShouldPassAlignment;
  CXXNewExprBits.UsualArrayDeleteWantsSize = UsualArrayDeleteWantsSize;
  CXXNewExprBits.HasInitializer = Initializer != nullptr;
  CXXNewExprBits.StoredInitializationStyle =
      llvm::to_underlying(InitializationStyle);
  bool IsParenTypeId = TypeIdParens.isValid();
  CXXNewExprBits.IsParenTypeId = IsParenTypeId;
  CXXNewExprBits.NumPlacementArgs = PlacementArgs.size();

  if (ArraySize)
    getTrailingObjects<Stmt *>()[arraySizeOffset()] = *ArraySize;
  if (Initializer)
    getTrailingObjects<Stmt *>()[initExprOffset()] = Initializer;
  for (unsigned I = 0; I != PlacementArgs.size(); ++I)
    getTrailingObjects<Stmt *>()[placementNewArgsOffset() + I] =
        PlacementArgs[I];
  if (IsParenTypeId)
    getTrailingObjects<SourceRange>()[0] = TypeIdParens;

  switch (getInitializationStyle()) {
  case CXXNewInitializationStyle::Parens:
    this->Span.setEnd(DirectInitSpan.getEnd());
    break;
  case CXXNewInitializationStyle::Braces:
    this->Span.setEnd(getInitializer()->getSourceRange().getEnd());
    break;
  default:
    if (IsParenTypeId)
      this->Span.setEnd(TypeIdParens.getEnd());
    break;
  }

  setDependence(computeDependence(this));
}

	if (!p_library_handle) {
		if (p_data != nullptr && p_data->generate_temp_files) {
			DirAccess::remove_absolute(load_path);
		}

#ifdef DEBUG_ENABLED
		DWORD err_code = GetLastError();

		HashSet<String> checked_libs;
		HashSet<String> missing_libs;
		debug_dynamic_library_check_dependencies(dll_path, checked_libs, missing_libs);
		if (!missing_libs.is_empty()) {
			String missing;
			for (const String &E : missing_libs) {
				if (!missing.is_empty()) {
					missing += ", ";
				}
				missing += E;
			}
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Missing dependencies: %s. Error: %s.", p_path, missing, format_error_message(err_code)));
		} else {
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, format_error_message(err_code)));
		}
#endif
	}

    Replaced = replaceInstrExpr(ED, ExtI, ExtR, Diff);

  if (Diff != 0 && Replaced && ED.IsDef) {
    // Update offsets of the def's uses.
    for (std::pair<MachineInstr*,unsigned> P : RegOps) {
      unsigned J = P.second;
      assert(P.first->getNumOperands() > J+1 &&
             P.first->getOperand(J+1).isImm());
      MachineOperand &ImmOp = P.first->getOperand(J+1);
      ImmOp.setImm(ImmOp.getImm() + Diff);
    }
    // If it was an absolute-set instruction, the "set" part has been removed.
    // ExtR will now be the register with the extended value, and since all
    // users of Rd have been updated, all that needs to be done is to replace
    // Rd with ExtR.
    if (IsAbsSet) {
      assert(ED.Rd.Sub == 0 && ExtR.Sub == 0);
      MRI->replaceRegWith(ED.Rd.Reg, ExtR.Reg);
    }
  }

/// Replace function F by function G.
void MergeFunctions::replaceFunctionInTree(const FunctionNode &FN,
                                           Function *G) {
  Function *F = FN.getFunc();
  assert(FunctionComparator(F, G, &GlobalNumbers).compare() == 0 &&
         "The two functions must be equal");

  auto I = FNodesInTree.find(F);
  assert(I != FNodesInTree.end() && "F should be in FNodesInTree");
  assert(FNodesInTree.count(G) == 0 && "FNodesInTree should not contain G");

  FnTreeType::iterator IterToFNInFnTree = I->second;
  assert(&(*IterToFNInFnTree) == &FN && "F should map to FN in FNodesInTree.");
  // Remove F -> FN and insert G -> FN
  FNodesInTree.erase(I);
  FNodesInTree.insert({G, IterToFNInFnTree});
  // Replace F with G in FN, which is stored inside the FnTree.
  FN.replaceBy(G);
}

bool found_date = false;
        while (*s) {
            if (*s == L'y') {
                *df = SDL_DATE_FORMAT_YYYYMMDD;
                found_date = true;
                s++;
                break;
            }
            if (*s == L'd') {
                *df = SDL_DATE_FORMAT_DDMMYYYY;
                found_date = true;
                s++;
                break;
            }
            if (*s == L'M') {
                *df = SDL_DATE_FORMAT_MMDDYYYY;
                found_date = true;
                s++;
                break;
            }
            s++;
        }

        if (!found_date) {
            // do nothing
        }

