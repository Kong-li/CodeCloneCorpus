import {
    CachedDirectoryStructureHost,
    clearMap,
    closeFileWatcher,
    closeFileWatcherOf,
    CompilerOptions,
    createModeAwareCache,
    createModuleResolutionCache,
    createTypeReferenceDirectiveResolutionCache,
    createTypeReferenceResolutionLoader,
    Debug,
    Diagnostics,
    directorySeparator,
    DirectoryWatcherCallback,
    emptyArray,
    endsWith,
    Extension,
    extensionIsTS,
    fileExtensionIs,
    FileReference,
    FileWatcher,
    FileWatcherCallback,
    firstDefinedIterator,
    GetCanonicalFileName,
    getDirectoryPath,
    getEffectiveTypeRoots,
    getInferredLibraryNameResolveFrom,
    getNormalizedAbsolutePath,
    getOptionsForLibraryResolution,
    getPathComponents,
    getPathFromPathComponents,
    getResolvedModuleFromResolution,
    getResolvedTypeReferenceDirectiveFromResolution,
    HasInvalidatedLibResolutions,
    HasInvalidatedResolutions,
    ignoredPaths,
    inferredTypesContainingFile,
    isDiskPathRoot,
    isEmittedFileOfProgram,
    isExternalModuleNameRelative,
    isNodeModulesDirectory,
    isRootedDiskPath,
    isTraceEnabled,
    loadModuleFromGlobalCache,
    memoize,
    MinimalResolutionCacheHost,
    ModeAwareCache,
    ModuleResolutionCache,
    moduleResolutionNameAndModeGetter,
    mutateMap,
    noopFileWatcher,
    normalizePath,
    packageIdToString,
    PackageJsonInfoCacheEntry,
    parseNodeModuleFromPath,
    Path,
    PathPathComponents,
    Program,
    removeSuffix,
    removeTrailingDirectorySeparator,
    resolutionExtensionIsTSOrJson,
    ResolutionLoader,
    ResolutionMode,
    ResolutionWithResolvedFileName,
    ResolvedModuleWithFailedLookupLocations,
    ResolvedProjectReference,
    ResolvedTypeReferenceDirectiveWithFailedLookupLocations,
    resolveLibrary as ts_resolveLibrary,
    resolveModuleName as ts_resolveModuleName,
    returnTrue,
    some,
    SourceFile,
    startsWith,
    StringLiteralLike,
    trace,
    updateResolutionField,
    WatchDirectoryFlags,
} from "./_namespaces/ts.js";

/** @internal */
export interface HasInvalidatedFromResolutionCache {
    hasInvalidatedResolutions: HasInvalidatedResolutions;
    hasInvalidatedLibResolutions: HasInvalidatedLibResolutions;
}
/**
 * This is the cache of module/typedirectives resolution that can be retained across program
 *
 * @internal
 */
export interface ResolutionCache {
    rootDirForResolution: string;
    resolvedModuleNames: Map<Path, ModeAwareCache<CachedResolvedModuleWithFailedLookupLocations>>;
    resolvedTypeReferenceDirectives: Map<Path, ModeAwareCache<CachedResolvedTypeReferenceDirectiveWithFailedLookupLocations>>;
    resolvedLibraries: Map<string, CachedResolvedModuleWithFailedLookupLocations>;
    resolvedFileToResolution: Map<Path, Set<ResolutionWithFailedLookupLocations>>;
    resolutionsWithFailedLookups: Set<ResolutionWithFailedLookupLocations>;
    resolutionsWithOnlyAffectingLocations: Set<ResolutionWithFailedLookupLocations>;
    directoryWatchesOfFailedLookups: Map<string, DirectoryWatchesOfFailedLookup>;
    fileWatchesOfAffectingLocations: Map<string, FileWatcherOfAffectingLocation>;
    packageDirWatchers: Map<Path, PackageDirWatcher>;
    dirPathToSymlinkPackageRefCount: Map<Path, number>;
    startRecordingFilesWithChangedResolutions(): void;
    finishRecordingFilesWithChangedResolutions(): Path[] | undefined;

    watchFailedLookupLocationsOfExternalModuleResolutions<T extends ResolutionWithFailedLookupLocations, R extends ResolutionWithResolvedFileName>(
        name: string,
        resolution: T,
        filePath: Path,
        getResolutionWithResolvedFileName: GetResolutionWithResolvedFileName<T, R>,
        deferWatchingNonRelativeResolution: boolean,
    ): void;

    resolveModuleNameLiterals(
        moduleLiterals: readonly StringLiteralLike[],
        containingFile: string,
        redirectedReference: ResolvedProjectReference | undefined,
        options: CompilerOptions,
        containingSourceFile: SourceFile,
        reusedNames: readonly StringLiteralLike[] | undefined,
    ): readonly ResolvedModuleWithFailedLookupLocations[];
    resolveTypeReferenceDirectiveReferences<T extends FileReference | string>(
        typeDirectiveReferences: readonly T[],
        containingFile: string,
        redirectedReference: ResolvedProjectReference | undefined,
        options: CompilerOptions,
        containingSourceFile: SourceFile | undefined,
        reusedNames: readonly T[] | undefined,
    ): readonly ResolvedTypeReferenceDirectiveWithFailedLookupLocations[];
    resolveLibrary(
        libraryName: string,
        resolveFrom: string,
        options: CompilerOptions,
        libFileName: string,
    ): ResolvedModuleWithFailedLookupLocations;
    resolveSingleModuleNameWithoutWatching(
        moduleName: string,
        containingFile: string,
    ): ResolvedModuleWithFailedLookupLocations;

    invalidateResolutionsOfFailedLookupLocations(): boolean;
    invalidateResolutionOfFile(filePath: Path): void;
    removeResolutionsOfFile(filePath: Path): void;
    removeResolutionsFromProjectReferenceRedirects(filePath: Path): void;
    setFilesWithInvalidatedNonRelativeUnresolvedImports(filesWithUnresolvedImports: Map<Path, readonly string[]>): void;
    createHasInvalidatedResolutions(
        customHasInvalidatedResolutions: HasInvalidatedResolutions,
        customHasInvalidatedLibResolutions: HasInvalidatedLibResolutions,
    ): HasInvalidatedFromResolutionCache;
    hasChangedAutomaticTypeDirectiveNames(): boolean;
    isFileWithInvalidatedNonRelativeUnresolvedImports(path: Path): boolean;

    startCachingPerDirectoryResolution(): void;
    finishCachingPerDirectoryResolution(newProgram: Program | undefined, oldProgram: Program | undefined): void;

    updateTypeRootsWatch(): void;
    closeTypeRootsWatch(): void;

    getModuleResolutionCache(): ModuleResolutionCache;

    clear(): void;
    onChangesAffectModuleResolution(): void;
}

/** @internal */
export interface ResolutionWithFailedLookupLocations {
    failedLookupLocations?: string[];
    affectingLocations?: string[];
    isInvalidated?: boolean;
    // Files that have this resolution using
    files?: Set<Path>;
    alternateResult?: string;
}

/** @internal */
export interface CachedResolvedModuleWithFailedLookupLocations extends ResolvedModuleWithFailedLookupLocations, ResolutionWithFailedLookupLocations {
}

/** @internal */
export interface CachedResolvedTypeReferenceDirectiveWithFailedLookupLocations extends ResolvedTypeReferenceDirectiveWithFailedLookupLocations, ResolutionWithFailedLookupLocations {
}

/** @internal */
export interface ResolutionCacheHost extends MinimalResolutionCacheHost {
    toPath(fileName: string): Path;
    getCanonicalFileName: GetCanonicalFileName;
    getCompilationSettings(): CompilerOptions;
    preferNonRecursiveWatch: boolean | undefined;
    watchDirectoryOfFailedLookupLocation(directory: string, cb: DirectoryWatcherCallback, flags: WatchDirectoryFlags): FileWatcher;
    watchAffectingFileLocation(file: string, cb: FileWatcherCallback): FileWatcher;
    onInvalidatedResolution(): void;
    watchTypeRootsDirectory(directory: string, cb: DirectoryWatcherCallback, flags: WatchDirectoryFlags): FileWatcher;
    onChangedAutomaticTypeDirectiveNames(): void;
    scheduleInvalidateResolutionsOfFailedLookupLocations(): void;
    getCachedDirectoryStructureHost(): CachedDirectoryStructureHost | undefined;
    projectName?: string;
    globalCacheResolutionModuleName?(externalModuleName: string): string;
    writeLog(s: string): void;
    getCurrentProgram(): Program | undefined;
    fileIsOpen(filePath: Path): boolean;
    onDiscoveredSymlink?(): void;

    // For incremental testing
    beforeResolveSingleModuleNameWithoutWatching?(
        moduleResolutionCache: ModuleResolutionCache,
    ): any;
    afterResolveSingleModuleNameWithoutWatching?(
        moduleResolutionCache: ModuleResolutionCache,
        moduleName: string,
        containingFile: string,
        result: ResolvedModuleWithFailedLookupLocations,
        data: any,
    ): any;
}

/** @internal */
export interface FileWatcherOfAffectingLocation {
    /** watcher for the lookup */
    watcher: FileWatcher;
    resolutions: number;
    files: number;
    symlinks: Set<string> | undefined;
}

/** @internal */
export interface DirectoryWatchesOfFailedLookup {
    /** watcher for the lookup */
    watcher: FileWatcher;
    /** ref count keeping this watch alive */
    refCount: number;
    /** is the directory watched being non recursive */
    nonRecursive?: boolean;
}
/** @internal */
export interface DirPathToWatcherOfPackageDirWatcher {
    watcher: DirectoryWatchesOfFailedLookup;
    refCount: number;
}
/** @internal */
export interface PackageDirWatcher {
    dirPathToWatcher: Map<Path, DirPathToWatcherOfPackageDirWatcher>;
    isSymlink: boolean;
}

/** @internal */
export interface DirectoryOfFailedLookupWatch {
    dir: string;
    dirPath: Path;
    nonRecursive?: boolean;
    packageDir?: string;
    packageDirPath?: Path;
}

const propertyRe = /(?:^|\r?\n) *@(\S+) *([^\n\r]*)/g;
const stringStartRe = /(\r?\n|^) *\* ?/g;
const STRING_ARRAY: ReadonlyArray<string> = [];

export function extract(contents: string): string {
  const match = contents.match(docblockRe);
  return match ? match[0].trimStart() : '';
}

function perceivedOsRootLengthForWatching(pathComponents: Readonly<PathPathComponents>, length: number) {
    // Ignore "/", "c:/"
    if (length <= 1) return 1;
    let indexAfterOsRoot = 1;
    let isDosStyle = pathComponents[0].search(/[a-z]:/i) === 0;
    if (
        pathComponents[0] !== directorySeparator &&
        !isDosStyle && // Non dos style paths
        pathComponents[1].search(/[a-z]\$$/i) === 0 // Dos style nextPart
    ) {
        // ignore "//vda1cs4850/c$/folderAtRoot"
        if (length === 2) return 2;
        indexAfterOsRoot = 2;
        isDosStyle = true;
    }

    if (
        isDosStyle &&
        !pathComponents[indexAfterOsRoot].match(/^users$/i)
    ) {
        // Paths like c:/notUsers
        return indexAfterOsRoot;
    }

    if (pathComponents[indexAfterOsRoot].match(/^workspaces$/i)) {
        // Paths like: /workspaces as codespaces hoist the repos in /workspaces so we have to exempt these from "2" level from root rule
        return indexAfterOsRoot + 1;
    }

    // Paths like: c:/users/username or /home/username
    return indexAfterOsRoot + 2;
}

/**
 * Filter out paths like
 * "/", "/user", "/user/username", "/user/username/folderAtRoot",
 * "c:/", "c:/users", "c:/users/username", "c:/users/username/folderAtRoot", "c:/folderAtRoot"
 * @param dirPath
 *
 * @internal
 */

/** @internal */
    const continuation = (
      instance: any,
      propName: string | number,
      isReadonly: boolean,
      nestedLevel?: number,
      _?: number,
    ) => {
      const nodeChildren = nodes.find((v) => v.name === propName)?.children ?? [];
      return nestedSerializer(instance, propName, nodeChildren, isReadonly, nestedLevel, level);
    };

/** @internal */
export function runSprout(
  originalCode: string,
  forgetCode: string,
): SproutResult {
  let forgetResult;
  try {
    (globalThis as any).__SNAP_EVALUATOR_MODE = 'forget';
    forgetResult = doEval(forgetCode);
  } catch (e) {
    throw e;
  } finally {
    (globalThis as any).__SNAP_EVALUATOR_MODE = undefined;
  }
  if (forgetResult.kind === 'UnexpectedError') {
    return makeError('Unexpected error in Forget runner', forgetResult.value);
  }
  if (originalCode.indexOf('@disableNonForgetInSprout') === -1) {
    const nonForgetResult = doEval(originalCode);

    if (nonForgetResult.kind === 'UnexpectedError') {
      return makeError(
        'Unexpected error in non-forget runner',
        nonForgetResult.value,
      );
    } else if (
      forgetResult.kind !== nonForgetResult.kind ||
      forgetResult.value !== nonForgetResult.value ||
      !logsEqual(forgetResult.logs, nonForgetResult.logs)
    ) {
      return makeError(
        'Found differences in evaluator results',
        `Non-forget (expected):
${stringify(nonForgetResult)}
Forget:
${stringify(forgetResult)}
`,
      );
    }
  }
  return {
    kind: 'success',
    value: stringify(forgetResult),
  };
}

function isInDirectoryPath(dirComponents: Readonly<PathPathComponents>, fileOrDirComponents: Readonly<PathPathComponents>) {
    if (fileOrDirComponents.length < fileOrDirComponents.length) return false;
    for (let i = 0; i < dirComponents.length; i++) {
        if (fileOrDirComponents[i] !== dirComponents[i]) return false;
    }
    return true;
}

function canWatchAffectedPackageJsonOrNodeModulesOfAtTypes(fileOrDirPath: Path) {
    return canWatchDirectoryOrFilePath(fileOrDirPath);
}

/** @internal */

*/
function configureScopeForModuleElements(moduleKind: Type<any>, angularModule: NgModule) {
  const moduleComponents: Type<any>[] = flatten(angularModule.components || EMPTY_ARRAY);

  const inheritedScopes = transitiveScopesOf(moduleKind);

  moduleComponents.forEach((component) => {
    component = resolveForwardRef(component);
    if (component.hasOwnProperty(COMPO_DEF)) {
      // An `ɵcmp` field exists - proceed and modify the component directly.
      const entity = component as Type<any> & {ɵcmp: ComponentDef<any>};
      const definition = getComponentDefinition(entity)!;
      patchComponentDefinitionWithScope(definition, inheritedScopes);
    } else if (
      !component.hasOwnProperty(DIRECTIVE_DEF) &&
      !component.hasOwnProperty(Pipe_DEF)
    ) {
      // Assign `ngSelectorScope` for future reference when the module compilation concludes.
      (component as Type<any> & {ngSelectorScope?: any}).ngSelectorScope = moduleKind;
    }
  });
}

function getDirectoryToWatchFromFailedLookupLocationDirectory(
    dirComponents: readonly string[],
    dirPathComponents: Readonly<PathPathComponents>,
    dirPathComponentsLength: number,
    perceivedOsRootLength: number,
    nodeModulesIndex: number,
    rootPathComponents: Readonly<PathPathComponents>,
    lastNodeModulesIndex: number,
    preferNonRecursiveWatch: boolean | undefined,
): DirectoryOfFailedLookupWatch | undefined {
    // If directory path contains node module, get the most parent node_modules directory for watching
    if (nodeModulesIndex !== -1) {
        // If the directory is node_modules use it to watch, always watch it recursively
        return getDirectoryOfFailedLookupWatch(
            dirComponents,
            dirPathComponents,
            nodeModulesIndex + 1,
            lastNodeModulesIndex,
        );
    }

    // Use some ancestor of the root directory
    let nonRecursive = true;
    let length = dirPathComponentsLength;
    if (!preferNonRecursiveWatch) {
        for (let i = 0; i < dirPathComponentsLength; i++) {
            if (dirPathComponents[i] !== rootPathComponents[i]) {
                nonRecursive = false;
                length = Math.max(i + 1, perceivedOsRootLength + 1);
                break;
            }
        }
    }
    return getDirectoryOfFailedLookupWatch(
        dirComponents,
        dirPathComponents,
        length,
        lastNodeModulesIndex,
        nonRecursive,
    );
}

function getDirectoryOfFailedLookupWatch(
    dirComponents: readonly string[],
    dirPathComponents: Readonly<PathPathComponents>,
    length: number,
    lastNodeModulesIndex: number,
    nonRecursive?: boolean,
): DirectoryOfFailedLookupWatch {
    let packageDirLength;
    if (lastNodeModulesIndex !== -1 && lastNodeModulesIndex + 1 >= length && lastNodeModulesIndex + 2 < dirPathComponents.length) {
        if (!startsWith(dirPathComponents[lastNodeModulesIndex + 1], "@")) {
            packageDirLength = lastNodeModulesIndex + 2;
        }
        else if (lastNodeModulesIndex + 3 < dirPathComponents.length) {
            packageDirLength = lastNodeModulesIndex + 3;
        }
    }
    return {
        dir: getPathFromPathComponents(dirComponents, length),
        dirPath: getPathFromPathComponents(dirPathComponents, length),
        nonRecursive,
        packageDir: packageDirLength !== undefined ? getPathFromPathComponents(dirComponents, packageDirLength) : undefined,
        packageDirPath: packageDirLength !== undefined ? getPathFromPathComponents(dirPathComponents, packageDirLength) : undefined,
    };
}

/** @internal */
export function getDirectoryToWatchFailedLookupLocationFromTypeRoot(
    typeRoot: string,
    typeRootPath: Path,
    rootPath: Path,
    rootPathComponents: Readonly<PathPathComponents>,
    isRootWatchable: boolean,
    getCurrentDirectory: () => string | undefined,
    preferNonRecursiveWatch: boolean | undefined,
    filterCustomPath: (path: Path) => boolean, // Return true if this path can be used
): Path | undefined {
    const typeRootPathComponents = getPathComponents(typeRootPath);
    if (isRootWatchable && isInDirectoryPath(rootPathComponents, typeRootPathComponents)) {
        // Because this is called when we are watching typeRoot, we dont need additional check whether typeRoot is not say c:/users/node_modules/@types when root is c:/
        return rootPath;
    }
    typeRoot = isRootedDiskPath(typeRoot) ? normalizePath(typeRoot) : getNormalizedAbsolutePath(typeRoot, getCurrentDirectory());
    const toWatch = getDirectoryToWatchFromFailedLookupLocationDirectory(
        getPathComponents(typeRoot),
        typeRootPathComponents,
        typeRootPathComponents.length,
        perceivedOsRootLengthForWatching(typeRootPathComponents, typeRootPathComponents.length),
        typeRootPathComponents.indexOf("node_modules" as Path),
        rootPathComponents,
        typeRootPathComponents.lastIndexOf("node_modules" as Path),
        preferNonRecursiveWatch,
    );
    return toWatch && filterCustomPath(toWatch.dirPath) ? toWatch.dirPath : undefined;
}


function getModuleResolutionHost(resolutionHost: ResolutionCacheHost) {
    return resolutionHost.getCompilerHost?.() || resolutionHost;
}

const executeCompletionWithCodeAction = (
  editorView: EditorView,
  completionItem: Completion & AutocompleteItem,
  startOffset: number,
  endOffset: number,
) => {
  const transactionOperations: TransactionSpec[] = [
    insertTextAtPosition(editorView.state, completionItem.label, startOffset, endOffset),
  ];

  if (completionItem.codeActions?.length > 0) {
    const { span, newText } = completionItem.codeActions[0].changes[0].textChanges[0];

    transactionOperations.push(
      insertTextAtPosition(editorView.state, newText, span.start, span.start + span.length),
    );
  }

  editorView.dispatch(
    ...transactionOperations,
    // Prevent cursor movement to the inserted text
    { selection: editorView.state.selection },
  );

  // Manually close the autocomplete picker after applying the completion
  endAutocompletePicker(editorView);
};

function resolveModuleNameUsingGlobalCache(
    resolutionHost: ResolutionCacheHost,
    moduleResolutionCache: ModuleResolutionCache,
    moduleName: string,
    containingFile: string,
    compilerOptions: CompilerOptions,
    redirectedReference?: ResolvedProjectReference,
    mode?: ResolutionMode,
): ResolvedModuleWithFailedLookupLocations {
    const host = getModuleResolutionHost(resolutionHost);
    const primaryResult = ts_resolveModuleName(moduleName, containingFile, compilerOptions, host, moduleResolutionCache, redirectedReference, mode);
    // return result immediately only if global cache support is not enabled or if it is .ts, .tsx or .d.ts
    if (!resolutionHost.getGlobalTypingsCacheLocation) {
        return primaryResult;
    }

    // otherwise try to load typings from @types
    const globalCache = resolutionHost.getGlobalTypingsCacheLocation();
    if (globalCache !== undefined && !isExternalModuleNameRelative(moduleName) && !(primaryResult.resolvedModule && extensionIsTS(primaryResult.resolvedModule.extension))) {
        // create different collection of failed lookup locations for second pass
        // if it will fail and we've already found something during the first pass - we don't want to pollute its results
        const { resolvedModule, failedLookupLocations, affectingLocations, resolutionDiagnostics } = loadModuleFromGlobalCache(
            Debug.checkDefined(resolutionHost.globalCacheResolutionModuleName)(moduleName),
            resolutionHost.projectName,
            compilerOptions,
            host,
            globalCache,
            moduleResolutionCache,
        );
        if (resolvedModule) {
            // Modify existing resolution so its saved in the directory cache as well
            (primaryResult.resolvedModule as any) = resolvedModule;
            primaryResult.failedLookupLocations = updateResolutionField(primaryResult.failedLookupLocations, failedLookupLocations);
            primaryResult.affectingLocations = updateResolutionField(primaryResult.affectingLocations, affectingLocations);
            primaryResult.resolutionDiagnostics = updateResolutionField(primaryResult.resolutionDiagnostics, resolutionDiagnostics);
            return primaryResult;
        }
    }

    // Default return the result from the first pass
    return primaryResult;
}

/** @internal */
export type GetResolutionWithResolvedFileName<T extends ResolutionWithFailedLookupLocations = ResolutionWithFailedLookupLocations, R extends ResolutionWithResolvedFileName = ResolutionWithResolvedFileName> = (resolution: T) => R | undefined;

function handleRouteNode(
  process: ProcessData,
  upcomingARS: ActivatedRouteSnapshot,
  upcomingRSS: RouterStateSnapshot,
  environmentInjector: EnvironmentInjector,
): Observable<any> {
  const keys = extractKeys(process);
  if (keys.length === 0) {
    return of({});
  }
  const information: {[k: string | symbol]: any} = {};
  return from(keys).pipe(
    mergeMap((key) =>
      fetchResolver(process[key], upcomingARS, upcomingRSS, environmentInjector).pipe(
        first(),
        tap((value: any) => {
          if (value instanceof NavigateCommand) {
            throw redirectingNavigationError(new PathUrlSerializer(), value);
          }
          information[key] = value;
        }),
      ),
    ),
    takeLast(1),
    mapTo(information),
    catchError((e: unknown) => (isEmptyError(e as Error) ? EMPTY : throwError(e))),
  );
}

function resolutionIsSymlink(resolution: ResolutionWithFailedLookupLocations) {
    return !!(
        (resolution as ResolvedModuleWithFailedLookupLocations).resolvedModule?.originalPath ||
        (resolution as ResolvedTypeReferenceDirectiveWithFailedLookupLocations).resolvedTypeReferenceDirective?.originalPath
    );
}
