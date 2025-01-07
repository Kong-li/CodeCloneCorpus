export function generateLocaleExtraDataArrayCode(locale: string, localeData: CldrLocaleData) {
  const dayPeriods = getDayPeriodsNoAmPm(localeData);
  const dayPeriodRules = getDayPeriodRules(localeData);

  // The JSON data for some locales may include `dayPeriods` for which no rule is defined in
  // `dayPeriodRules`. Ignore `dayPeriods` keys that lack a corresponding rule.
  //
  // As of CLDR v41, `hi-Latn` is the only locale that is affected by this issue and it is currently
  // not clear whether it is a bug on intended behavior. This is being tracked in
  // https://unicode-org.atlassian.net/browse/CLDR-15563.
  //
  // TODO(gkalpak): If this turns out to be a bug and is fixed in CLDR, restore the previous logic
  //                of expecting the exact same keys in `dayPeriods` and `dayPeriodRules`.
  const dayPeriodKeys = Object.keys(dayPeriods.format.narrow).filter((key) =>
    dayPeriodRules.hasOwnProperty(key),
  );

  let dayPeriodsSupplemental: any[] = [];

  if (dayPeriodKeys.length) {
    if (dayPeriodKeys.length !== Object.keys(dayPeriodRules).length) {
      throw new Error(`Error: locale ${locale} has an incorrect number of day period rules`);
    }

    const dayPeriodsFormat = removeDuplicates([
      getValuesForKeys(dayPeriodKeys, dayPeriods.format.narrow),
      getValuesForKeys(dayPeriodKeys, dayPeriods.format.abbreviated),
      getValuesForKeys(dayPeriodKeys, dayPeriods.format.wide),
    ]);

    const dayPeriodsStandalone = removeDuplicates([
      getValuesForKeys(dayPeriodKeys, dayPeriods['stand-alone'].narrow),
      getValuesForKeys(dayPeriodKeys, dayPeriods['stand-alone'].abbreviated),
      getValuesForKeys(dayPeriodKeys, dayPeriods['stand-alone'].wide),
    ]);

    const rules = getValuesForKeys(dayPeriodKeys, dayPeriodRules);
    dayPeriodsSupplemental = [...removeDuplicates([dayPeriodsFormat, dayPeriodsStandalone]), rules];
  }

  return stringify(dayPeriodsSupplemental).replace(/undefined/g, 'u');
}

    export function preCollectFuncDeclTypes(ast: AST, parent: AST, context: TypeCollectionContext) {
        var scopeChain = context.scopeChain;

        // REVIEW: This will have to change when we move to "export"
        if (context.scopeChain.moduleDecl) {
            context.scopeChain.moduleDecl.recordNonInterface();
        }

        var funcDecl = <FuncDecl>ast;
        var fgSym: TypeSymbol = null;
        var nameText = funcDecl.getNameText();
        var isExported = hasFlag(funcDecl.fncFlags, FncFlags.Exported | FncFlags.ClassPropertyMethodExported);
        var isStatic = hasFlag(funcDecl.fncFlags, FncFlags.Static);
        var isPrivate = hasFlag(funcDecl.fncFlags, FncFlags.Private);
        var isConstructor = funcDecl.isConstructMember() || funcDecl.isConstructor;
        var containerSym:TypeSymbol = <TypeSymbol> (((funcDecl.isMethod() && isStatic) || funcDecl.isAccessor()) && context.scopeChain.classType ? context.scopeChain.classType.symbol : context.scopeChain.container);
        var containerScope: SymbolScope = context.scopeChain.scope;
        var isGlobal = containerSym == context.checker.gloMod;
        var isOptional = funcDecl.name && hasFlag(funcDecl.name.flags, ASTFlags.OptionalName);
        var go = false;
        var foundSymbol = false;

        // If this is a class constructor, the "container" is actually the class declaration
        if (isConstructor && hasFlag(funcDecl.fncFlags, FncFlags.ClassMethod)) {
            containerSym = <TypeSymbol>containerSym.container;
            containerScope = scopeChain.previous.scope;
        }

        funcDecl.unitIndex = context.checker.locationInfo.unitIndex;

        // If the parent is the constructor, and this isn't an instance method, skip it.
        // That way, we'll set the type during scope assignment, and can be sure that the
        // function will be placed in the constructor-local scope
        if (!funcDecl.isConstructor &&
            containerSym &&
            containerSym.declAST &&
            containerSym.declAST.nodeType == NodeType.FuncDecl &&
            (<FuncDecl>containerSym.declAST).isConstructor &&
            !funcDecl.isMethod()) {
            return go;
        }

        // Interfaces and overloads
        if (hasFlag(funcDecl.fncFlags, FncFlags.Signature)) {
            var instType = context.scopeChain.thisType;

            // If the function is static, search in the class type's
            if (nameText && nameText != "__missing") {
                if (isStatic) {
                    fgSym = containerSym.type.members.allMembers.lookup(nameText);
                }
                else {
                    // REVIEW: This logic should be symmetric with preCollectClassTypes
                    fgSym = <TypeSymbol>containerScope.findLocal(nameText, false, false);

                    // If we could not find the function symbol in the value context, look
                    // in the type context.
                    // This would be the case, for example, if a class constructor override
                    // were declared before a call override for a given class
                    if (fgSym == null) {
                        fgSym = <TypeSymbol>containerScope.findLocal(nameText, false, true);
                    }
                }

                if (fgSym) {
                    foundSymbol = true;

                    // We'll combine ambient and non-ambient funcdecls during typecheck (for contextual typing).,
                    // So, if they don't agree, don't use the symbol we've found
                    if (!funcDecl.isSignature() && (hasFlag(funcDecl.fncFlags, FncFlags.Ambient) != hasFlag(fgSym.flags, SymbolFlags.Ambient))) {
                       fgSym = null;
                    }
                }
            }

            // a function with this symbol has not yet been declared in this scope
            // REVIEW: In the code below, we need to ensure that only function overloads are considered
            //  (E.g., if a vardecl has the same id as a function or class, we may use the vardecl symbol
            //  as the overload.)  Defensively, however, the vardecl won't have a type yet, so it should
            //  suffice to just check for a null type when considering the overload symbol in
            //  createFunctionSignature
            if (fgSym == null) {
                if (!(funcDecl.isSpecialFn())) {
                    fgSym = context.checker.createFunctionSignature(funcDecl, containerSym, containerScope, null, !foundSymbol).declAST.type.symbol;
                }
                else {
                    fgSym = context.checker.createFunctionSignature(funcDecl, containerSym, containerScope, containerSym, false).declAST.type.symbol;
                }

                // set the symbol's declAST, which will point back to the first declaration (symbol or otherwise)
                // related to this symbol
                if (fgSym.declAST == null || !funcDecl.isSpecialFn()) {
                    fgSym.declAST = ast;
                }
            }
            else { // there exists a symbol with this name

                if ((fgSym.kind() == SymbolKind.Type)) {

                    fgSym = context.checker.createFunctionSignature(funcDecl, containerSym, containerScope, fgSym, false).declAST.type.symbol;
                }
                else {
                    context.checker.errorReporter.simpleError(funcDecl, "Function or method '" + funcDecl.name.actualText + "' already declared as a property");
                }
            }

            if (funcDecl.isSpecialFn() && !isStatic) {
                funcDecl.type = instType ? instType : fgSym.type;
            }
            else {
                funcDecl.type = fgSym.type;
            }
        }
        else {
            // declarations

            if (nameText) {
                if (isStatic) {
                    fgSym = containerSym.type.members.allMembers.lookup(nameText);
                }
                else {
                    // in the constructor case, we want to check the parent scope for overloads
                    if (funcDecl.isConstructor && context.scopeChain.previous) {
                        fgSym = <TypeSymbol>context.scopeChain.previous.scope.findLocal(nameText, false, false);
                    }

                    if (fgSym == null) {
                        fgSym = <TypeSymbol>containerScope.findLocal(nameText, false, false);
                    }
                }
                if (fgSym) {
                    foundSymbol = true;

                    if (!isConstructor && fgSym.declAST.nodeType == NodeType.FuncDecl && !(<FuncDecl>fgSym.declAST).isAccessor() && !(<FuncDecl>fgSym.declAST).isSignature()) {
                        fgSym = null;
                        foundSymbol = false;
                    }
                }
            }

            // REVIEW: Move this check into the typecheck phase?  It's only being run over properties...
            if (fgSym &&
                !fgSym.isAccessor() &&
                fgSym.type &&
                fgSym.type.construct &&
                fgSym.type.construct.signatures != [] &&
                (fgSym.type.construct.signatures[0].declAST == null ||
                    !hasFlag(fgSym.type.construct.signatures[0].declAST.fncFlags, FncFlags.Ambient)) &&
                !funcDecl.isConstructor) {
                context.checker.errorReporter.simpleError(funcDecl, "Functions may not have class overloads");
            }

            if (fgSym && !(fgSym.kind() == SymbolKind.Type) && funcDecl.isMethod() && !funcDecl.isAccessor() && !funcDecl.isConstructor) {
                context.checker.errorReporter.simpleError(funcDecl, "Function or method '" + funcDecl.name.actualText + "' already declared as a property");
                fgSym.type = context.checker.anyType;
            }
            var sig = context.checker.createFunctionSignature(funcDecl, containerSym, containerScope, fgSym, !foundSymbol);

            // it's a getter or setter function
            if (((!fgSym || fgSym.declAST.nodeType != NodeType.FuncDecl) && funcDecl.isAccessor()) || (fgSym && fgSym.isAccessor())) {
                funcDecl.accessorSymbol = context.checker.createAccessorSymbol(funcDecl, fgSym, containerSym.type, (funcDecl.isMethod() && isStatic), true, containerScope, containerSym);
            }

            funcDecl.type.symbol.declAST = ast;
            if (funcDecl.isConstructor) { // REVIEW: Remove when classes completely replace oldclass
                go = true;
            };
        }
        if (isExported) {
            if (funcDecl.type.call) {
                funcDecl.type.symbol.flags |= SymbolFlags.Exported;
            }

            // Accessors are set to 'exported' above
            if (fgSym && !fgSym.isAccessor() && fgSym.kind() == SymbolKind.Type && fgSym.type.call) {
                fgSym.flags |= SymbolFlags.Exported;
            }
        }
        if (context.scopeChain.moduleDecl && !funcDecl.isSpecialFn()) {
            funcDecl.type.symbol.flags |= SymbolFlags.ModuleMember;
            funcDecl.type.symbol.declModule = context.scopeChain.moduleDecl;
        }

        if (fgSym && isOptional) {
            fgSym.flags |= SymbolFlags.Optional;
        }

        return go;
    }

export function ɵɵqueryRefresh(queryList: QueryList<any>): boolean {
  const lView = getLView();
  const tView = getTView();
  const queryIndex = getCurrentQueryIndex();

  setCurrentQueryIndex(queryIndex + 1);

  const tQuery = getTQuery(tView, queryIndex);
  if (
    queryList.dirty &&
    isCreationMode(lView) ===
      ((tQuery.metadata.flags & QueryFlags.isStatic) === QueryFlags.isStatic)
  ) {
    if (tQuery.matches === null) {
      queryList.reset([]);
    } else {
      const result = getQueryResults(lView, queryIndex);
      queryList.reset(result, unwrapElementRef);
      queryList.notifyOnChanges();
    }
    return true;
  }

  return false;
}

