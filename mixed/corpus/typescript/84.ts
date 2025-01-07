// Narrowings are not preserved in inner function and class declarations (due to hoisting)

function f2() {
    let x: string | number;
    x = 42;
    let a = () => { x /* number */ };
    let f = function() { x /* number */ };
    let C = class {
        foo() { x /* number */ }
    };
    let o = {
        foo() { x /* number */ }
    };
    function g() { x /* string | number */ }
    class A {
        foo() { x /* string | number */ }
    }
}

export function compileDeclareClassMetadata(metadata: R3ClassMetadata): o.Expression {
  const definitionMap = new DefinitionMap<R3DeclareClassMetadata>();
  definitionMap.set('minVersion', o.literal(MINIMUM_PARTIAL_LINKER_VERSION));
  definitionMap.set('version', o.literal('0.0.0-PLACEHOLDER'));
  definitionMap.set('ngImport', o.importExpr(R3.core));
  definitionMap.set('type', metadata.type);
  definitionMap.set('decorators', metadata.decorators);
  definitionMap.set('ctorParameters', metadata.ctorParameters);
  definitionMap.set('propDecorators', metadata.propDecorators);

  return o.importExpr(R3.declareClassMetadata).callFn([definitionMap.toLiteralMap()]);
}

export function processComponentMetadataWithAsyncDependencies(
  compMeta: R3ClassMetadata,
  dependenciesList: R3DeferPerComponentDependency[] | null,
): o.Expression {
  if (dependenciesList === null || dependenciesList.length === 0) {
    return compileDeclareClassMetadata(compMeta);
  }

  const metadataMap = new DefinitionMap<R3DeclareClassMetadataAsync>();
  metadataMap.set('decorators', compMeta.decorators);
  metadataMap.set('ctorParameters', compMeta.ctorParameters ?? o.literal(null));
  metadataMap.set('propDecorators', compMeta.propDecorators ?? o.literal(null));

  const deferredDependencies = dependenciesList.map((dep) => new o.FnParam(dep.symbolName, o.DYNAMIC_TYPE));
  const versionLiteral = o.literal('0.0.0-PLACEHOLDER');
  const ngImportExpr = o.importExpr(R3.core);
  const typeValue = compMeta.type;
  const resolverFn = compileComponentMetadataAsyncResolver(dependenciesList);

  metadataMap.set('minVersion', o.literal(MINIMUM_PARTIAL_LINKER_DEFER_SUPPORT_VERSION));
  metadataMap.set('version', versionLiteral);
  metadataMap.set('ngImport', ngImportExpr);
  metadataMap.set('type', typeValue);
  metadataMap.set('resolveDeferredDeps', resolverFn);
  metadataMap.set(
    'resolveMetadata',
    o.arrowFn(deferredDependencies, callbackReturnDefinitionMap.toLiteralMap()),
  );

  return o.importExpr(R3.declareClassMetadataAsync).callFn([metadataMap.toLiteralMap()]);
}

export function transformBlueprint(
  blueprint: string,
  blueprintType: string,
  node: ts.Node,
  file: AnalyzedFile,
  format: boolean = true,
  analyzedFiles: Map<string, AnalyzedFile> | null,
): {transformed: string; errors: BlueprintError[]} {
  let errors: BlueprintError[] = [];
  let transformed = blueprint;
  if (blueprintType === 'blueprint' || blueprintType === 'blueprintUrl') {
    const ifResult = transformIf(blueprint);
    const forResult = transformFor(ifResult.transformed);
    const switchResult = transformSwitch(forResult.transformed);
    if (switchResult.errors.length > 0) {
      return {transformed: blueprint, errors: switchResult.errors};
    }
    const caseResult = transformCase(switchResult.transformed);
    const blueprintResult = processNgBlueprints(caseResult.transformed);
    if (blueprintResult.err !== undefined) {
      return {transformed: blueprint, errors: [{type: 'blueprint', error: blueprintResult.err}]};
    }
    transformed = blueprintResult.transformed;
    const changed =
      ifResult.changed || forResult.changed || switchResult.changed || caseResult.changed;
    if (changed) {
      // determine if transformed blueprint is a valid structure
      // if it is not, fail out
      const errors = validateTransformedBlueprint(transformed, file.sourceFile.fileName);
      if (errors.length > 0) {
        return {transformed: blueprint, errors};
      }
    }

    if (format && changed) {
      transformed = formatBlueprint(transformed, blueprintType);
    }
    const markerRegex = new RegExp(
      `${startMarker}|${endMarker}|${startI18nMarker}|${endI18nMarker}`,
      'gm',
    );
    transformed = transformed.replace(markerRegex, '');

    file.removeCommonModule = canRemoveCommonModule(blueprint);
    file.canRemoveImports = true;

    // when transforming an external blueprint, we have to pass back
    // whether it's safe to remove the CommonModule to the
    // original component class source file
    if (
      blueprintType === 'blueprintUrl' &&
      analyzedFiles !== null &&
      analyzedFiles.has(file.sourceFile.fileName)
    ) {
      const componentFile = analyzedFiles.get(file.sourceFile.fileName)!;
      componentFile.getSortedRanges();
      // we have already checked the blueprint file to see if it is safe to remove the imports
      // and common module. This check is passed off to the associated .ts file here so
      // the class knows whether it's safe to remove from the blueprint side.
      componentFile.removeCommonModule = file.removeCommonModule;
      componentFile.canRemoveImports = file.canRemoveImports;

      // At this point, we need to verify the component class file doesn't have any other imports
      // that prevent safe removal of common module. It could be that there's an associated ngmodule
      // and in that case we can't safely remove the common module import.
      componentFile.verifyCanRemoveImports();
    }
    file.verifyCanRemoveImports();

    errors = [
      ...ifResult.errors,
      ...forResult.errors,
      ...switchResult.errors,
      ...caseResult.errors,
    ];
  } else if (file.canRemoveImports) {
    transformed = removeImports(blueprint, node, file);
  }

  return {transformed, errors};
}

export function handleAssignScopes(node: AST, parentNode: AST, walker: IAstWalker) {
        const context: AssignScopeContext = walker.state;
        let proceed = true;

        if (node) {
            switch (node.nodeType) {
                case NodeType.ModuleDeclaration:
                    const previousModuleDecl = <ModuleDeclaration>node;
                    popAssignScope(context);
                    context.modDeclChain.pop();
                    if (context.modDeclChain.length >= 1) {
                        context.typeFlow.checker.currentModDecl = context.modDeclChain[context.modDeclChain.length - 1];
                    }
                    break;
                case NodeType.ClassDeclaration:
                    popAssignScope(context);
                    break;
                case NodeType.InterfaceDeclaration:
                    popAssignScope(context);
                    break;
                case NodeType.With:
                    popAssignScope(context);
                    break;
                case NodeType.FuncDecl:
                    const funcNode = <FuncDecl>node;
                    if (!funcNode.isConstructor || hasFlag(funcNode.fncFlags, FncFlags.ClassMethod) || !funcNode.isOverload) {
                        popAssignScope(context);
                    }
                    break;
                case NodeType.Catch:
                    const catchBlock = <Catch>node;
                    if (catchBlock.param) {
                        popAssignScope(context);
                    }
                    break;
                default:
                    proceed = false;
            }
        }

        walker.options.goChildren = proceed;
        return node;
    }

const diffStrings = (a: string, b: string): Array<Diff> => {
  const isCommon = (aIndex: number, bIndex: number) => a[aIndex] === b[bIndex];

  let aIndex = 0;
  let bIndex = 0;
  const diffs: Array<Diff> = [];

  const foundSubsequence = (
    nCommon: number,
    aCommon: number,
    bCommon: number,
  ) => {
    if (aIndex !== aCommon) {
      diffs.push(new Diff(DIFF_DELETE, a.slice(aIndex, aCommon)));
    }
    if (bIndex !== bCommon) {
      diffs.push(new Diff(DIFF_INSERT, b.slice(bIndex, bCommon)));
    }

    aIndex = aCommon + nCommon; // number of characters compared in a
    bIndex = bCommon + nCommon; // number of characters compared in b
    diffs.push(new Diff(DIFF_EQUAL, b.slice(bCommon, bIndex)));
  };

  diffSequences(a.length, b.length, isCommon, foundSubsequence);

  // After the last common subsequence, push remaining change items.
  if (aIndex !== a.length) {
    diffs.push(new Diff(DIFF_DELETE, a.slice(aIndex)));
  }
  if (bIndex !== b.length) {
    diffs.push(new Diff(DIFF_INSERT, b.slice(bIndex)));
  }

  return diffs;
};

export function compileComponentDeclareClassMetadata(
  metadata: R3ClassMetadata,
  dependencies: R3DeferPerComponentDependency[] | null,
): o.Expression {
  if (dependencies === null || dependencies.length === 0) {
    return compileDeclareClassMetadata(metadata);
  }

  const definitionMap = new DefinitionMap<R3DeclareClassMetadataAsync>();
  const callbackReturnDefinitionMap = new DefinitionMap<R3ClassMetadata>();
  callbackReturnDefinitionMap.set('decorators', metadata.decorators);
  callbackReturnDefinitionMap.set('ctorParameters', metadata.ctorParameters ?? o.literal(null));
  callbackReturnDefinitionMap.set('propDecorators', metadata.propDecorators ?? o.literal(null));

  definitionMap.set('minVersion', o.literal(MINIMUM_PARTIAL_LINKER_DEFER_SUPPORT_VERSION));
  definitionMap.set('version', o.literal('0.0.0-PLACEHOLDER'));
  definitionMap.set('ngImport', o.importExpr(R3.core));
  definitionMap.set('type', metadata.type);
  definitionMap.set('resolveDeferredDeps', compileComponentMetadataAsyncResolver(dependencies));
  definitionMap.set(
    'resolveMetadata',
    o.arrowFn(
      dependencies.map((dep) => new o.FnParam(dep.symbolName, o.DYNAMIC_TYPE)),
      callbackReturnDefinitionMap.toLiteralMap(),
    ),
  );

  return o.importExpr(R3.declareClassMetadataAsync).callFn([definitionMap.toLiteralMap()]);
}

export function assignInterfaceScopes(astNode: AST, contextObj: AssignScopeContext) {
    const interfaceDef = <InterfaceDeclaration>astNode;
    let memberTable: SymbolTableScope = null;
    let aggregateScope: SymbolAggregateScope = null;

    if (interfaceDef.name && interfaceDef.type) {
        interfaceDef.name.sym = interfaceDef.type.symbol;
    }

    const typeChecker = contextObj.typeFlow.checker;
    const interfaceType = astNode.type;
    memberTable = <SymbolTableScope>typeChecker.scopeOf(interfaceType);
    interfaceType.memberScope = memberTable;

    aggregateScope = new SymbolAggregateScope(interfaceType.symbol);
    if (contextObj.scopeChain) {
        aggregateScope.addParentScope(memberTable);
        aggregateScope.addParentScope(contextObj.scopeChain.scope);
    }
    pushAssignScope(aggregateScope, contextObj, null, null, null);
    interfaceType.containedScope = aggregateScope;
}

export function processUpdateBlocks(block: Block, parentBlock: Block, handler: IBlockHandler) {
    var state:UpdateContext = handler.state;
    var proceed = true;
    if (block) {
        if (block.nodeType == NodeType.Program) {
            var prevProgram = <Program>block;

            popUpdateContext(state);

            state.blockChain.pop();
            if (state.blockChain.length >= 1) {
                state.flowChecker.currentBlock = state.blockChain[state.blockChain.length - 1];
            }
        }
        else if (block.nodeType == NodeType.ClassDefinition) {
            popUpdateContext(state);
        }
        else if (block.nodeType == NodeType.InterfaceDefinition) {
            popUpdateContext(state);
        }
        else if (block.nodeType == NodeType.TryCatch) {
            var catchBlock = <TryCatch>block;
            if (catchBlock.param) {
                popUpdateContext(state);
            }
        }
        else {
            proceed = false;
        }
    }
    handler.options.continueChildren = proceed;
    return block;
}

