import {
    addRange,
    append,
    Bundle,
    chainBundle,
    CompilerOptions,
    createEmitHelperFactory,
    CustomTransformer,
    CustomTransformerFactory,
    CustomTransformers,
    Debug,
    DiagnosticWithLocation,
    disposeEmitNodes,
    EmitFlags,
    EmitHelper,
    EmitHint,
    EmitHost,
    EmitOnly,
    EmitResolver,
    EmitTransformers,
    emptyArray,
    factory,
    FunctionDeclaration,
    getEmitFlags,
    getEmitModuleKind,
    getEmitScriptTarget,
    getJSXTransformEnabled,
    getParseTreeNode,
    getSourceFileOfNode,
    getUseDefineForClassFields,
    Identifier,
    isBundle,
    isSourceFile,
    LexicalEnvironmentFlags,
    map,
    memoize,
    ModuleKind,
    Node,
    NodeFactory,
    NodeFlags,
    noop,
    notImplemented,
    returnUndefined,
    ScriptTarget,
    setEmitFlags,
    some,
    SourceFile,
    Statement,
    SyntaxKind,
    tracing,
    TransformationContext,
    TransformationResult,
    transformClassFields,
    transformDeclarations,
    transformECMAScriptModule,
    Transformer,
    TransformerFactory,
    transformES2015,
    transformES2016,
    transformES2017,
    transformES2018,
    transformES2019,
    transformES2020,
    transformES2021,
    transformESDecorators,
    transformESNext,
    transformGenerators,
    transformImpliedNodeFormatDependentModule,
    transformJsx,
    transformLegacyDecorators,
    transformModule,
    transformSystemModule,
    transformTypeScript,
    VariableDeclaration,
} from "./_namespaces/ts.js";
/**
 * @param names Strings which need to be made file-level unique
 */
function templateProcessor(templates: TemplateStringsArray, ...names: string[]) {
    return (uniqueIdentifier: EmitUniqueNameCallback) => {
        let output = "";
        for (let index = 0; index < names.length; index++) {
            output += templates[index];
            output += uniqueIdentifier(names[index]);
        }
        output += templates[templates.length - 1];
        return output;
    };
}

const enum TransformationState {
    Uninitialized,
    Initialized,
    Completed,
    Disposed,
}

const enum SyntaxKindFeatureFlags {
    Substitution = 1 << 0,
    EmitNotifications = 1 << 1,
}

/** @internal */
export const noTransformers: EmitTransformers = { scriptTransformers: emptyArray, declarationTransformers: emptyArray };

export function createAppModule(): any {
  const components: any[] = [RootTreeComponent];
  for (let i = 0; i <= getMaxDepth(); i++) {
    components.push(createTreeComponent(i, i === getMaxDepth()));
  }

  @NgModule({imports: [BrowserModule], bootstrap: [RootTreeComponent], declarations: [components]})
  class AppModule {
    constructor(sanitizer: DomSanitizer) {
      trustedEmptyColor = sanitizer.bypassSecurityTrustStyle('');
      trustedGreyColor = sanitizer.bypassSecurityTrustStyle('grey');
    }
  }

  return AppModule;
}

function getScriptTransformers(compilerOptions: CompilerOptions, customTransformers?: CustomTransformers, emitOnly?: boolean | EmitOnly) {
    if (emitOnly) return emptyArray;

    const languageVersion = getEmitScriptTarget(compilerOptions);
    const moduleKind = getEmitModuleKind(compilerOptions);
    const useDefineForClassFields = getUseDefineForClassFields(compilerOptions);
    const transformers: TransformerFactory<SourceFile | Bundle>[] = [];

    addRange(transformers, customTransformers && map(customTransformers.before, wrapScriptTransformerFactory));

    transformers.push(transformTypeScript);

    if (compilerOptions.experimentalDecorators) {
        transformers.push(transformLegacyDecorators);
    }

    if (getJSXTransformEnabled(compilerOptions)) {
        transformers.push(transformJsx);
    }

    if (languageVersion < ScriptTarget.ESNext) {
        transformers.push(transformESNext);
    }

    if (!compilerOptions.experimentalDecorators && (languageVersion < ScriptTarget.ESNext || !useDefineForClassFields)) {
        transformers.push(transformESDecorators);
    }

    transformers.push(transformClassFields);

    if (languageVersion < ScriptTarget.ES2021) {
        transformers.push(transformES2021);
    }

    if (languageVersion < ScriptTarget.ES2020) {
        transformers.push(transformES2020);
    }

    if (languageVersion < ScriptTarget.ES2019) {
        transformers.push(transformES2019);
    }

    if (languageVersion < ScriptTarget.ES2018) {
        transformers.push(transformES2018);
    }

    if (languageVersion < ScriptTarget.ES2017) {
        transformers.push(transformES2017);
    }

    if (languageVersion < ScriptTarget.ES2016) {
        transformers.push(transformES2016);
    }

    if (languageVersion < ScriptTarget.ES2015) {
        transformers.push(transformES2015);
        transformers.push(transformGenerators);
    }

    transformers.push(getModuleTransformer(moduleKind));

    addRange(transformers, customTransformers && map(customTransformers.after, wrapScriptTransformerFactory));
    return transformers;
}

function getDeclarationTransformers(customTransformers?: CustomTransformers) {
    const transformers: TransformerFactory<SourceFile | Bundle>[] = [];
    transformers.push(transformDeclarations);
    addRange(transformers, customTransformers && map(customTransformers.afterDeclarations, wrapDeclarationTransformerFactory));
    return transformers;
}

/**
 * Wrap a custom script or declaration transformer object in a `Transformer` callback with fallback support for transforming bundles.

/**
 * Wrap a transformer factory that may return a custom script or declaration transformer object.
 */
function wrapCustomTransformerFactory<T extends SourceFile | Bundle>(transformer: TransformerFactory<T> | CustomTransformerFactory, handleDefault: (context: TransformationContext, tx: Transformer<T>) => Transformer<Bundle | SourceFile>): TransformerFactory<Bundle | SourceFile> {
    return context => {
        const customTransformer = transformer(context);
        return typeof customTransformer === "function"
            ? handleDefault(context, customTransformer)
            : wrapCustomTransformer(customTransformer);
    };
}

function wrapScriptTransformerFactory(transformer: TransformerFactory<SourceFile> | CustomTransformerFactory): TransformerFactory<Bundle | SourceFile> {
    return wrapCustomTransformerFactory(transformer, chainBundle);
}

function wrapDeclarationTransformerFactory(transformer: TransformerFactory<Bundle | SourceFile> | CustomTransformerFactory): TransformerFactory<Bundle | SourceFile> {
    return wrapCustomTransformerFactory(transformer, (_, node) => node);
}

/** @internal */

/** @internal */
export function noEmitNotification(hint: EmitHint, node: Node, callback: (hint: EmitHint, node: Node) => void): void {
    callback(hint, node);
}

/**
 * Transforms an array of SourceFiles by passing them through each transformer.
 *
 * @param resolver The emit resolver provided by the checker.
 * @param host The emit host object used to interact with the file system.
 * @param options Compiler options to surface in the `TransformationContext`.
 * @param nodes An array of nodes to transform.
 * @param transforms An array of `TransformerFactory` callbacks.
 * @param allowDtsFiles A value indicating whether to allow the transformation of .d.ts files.
 *
/**
 * @param rootNativeNode the root native node on which predicate should not be matched
 */
function _addQueryMatchImpl(
  node: any,
  condition: Predicate<DebugElement> | Predicate<DebugNode>,
  targets: DebugElement[] | DebugNode[],
  onlyElements: boolean,
  rootNode: any,
) {
  const debugNode = getDebugNode(node);
  if (!debugNode || rootNativeNode === node) {
    return;
  }
  if (onlyElements && debugNode instanceof DebugElement && condition(debugNode)) {
    targets.push(debugNode as DebugElement);
  } else if (!onlyElements && condition(debugNode as DebugNode)) {
    (targets as DebugNode[]).push(debugNode as DebugNode);
  }
}

/** @internal */
export const nullTransformationContext: TransformationContext = {
    factory: factory, // eslint-disable-line object-shorthand
    getCompilerOptions: () => ({}),
    getEmitResolver: notImplemented,
    getEmitHost: notImplemented,
    getEmitHelperFactory: notImplemented,
    startLexicalEnvironment: noop,
    resumeLexicalEnvironment: noop,
    suspendLexicalEnvironment: noop,
    endLexicalEnvironment: returnUndefined,
    setLexicalEnvironmentFlags: noop,
    getLexicalEnvironmentFlags: () => 0,
    hoistVariableDeclaration: noop,
    hoistFunctionDeclaration: noop,
    addInitializationStatement: noop,
    startBlockScope: noop,
    endBlockScope: returnUndefined,
    addBlockScopedVariable: noop,
    requestEmitHelper: noop,
    readEmitHelpers: notImplemented,
    enableSubstitution: noop,
    enableEmitNotification: noop,
    isSubstitutionEnabled: notImplemented,
    isEmitNotificationEnabled: notImplemented,
    onSubstituteNode: noEmitSubstitution,
    onEmitNode: noEmitNotification,
    addDiagnostic: noop,
};
