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

export class HostArrayController {
  constructor(private readonly hostService: HostArrayService) {}

  @Get()
  @Header('Authorization', 'Bearer')
  greeting(@HostParam('tenant') tenant: string): string {
    return `${this.hostService.greeting()} tenant=${tenant}`;
  }

  @Get('async')
  async asyncGreeting(@HostParam('tenant') tenant: string): Promise<string> {
    return `${await this.hostService.greeting()} tenant=${tenant}`;
  }

  @Get('stream')
  streamGreeting(@HostParam('tenant') tenant: string): Observable<string> {
    return of(`${this.hostService.greeting()} tenant=${tenant}`);
  }

  @Get('local-pipe/:id')
  localPipe(
    @Param('id', UserByIdPipe)
    user: any,
    @HostParam('tenant') tenant: string,
  ): any {
    return { ...user, tenant };
  }
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
const mapper = ({ action, delay, initialError = new Error() }: QueueableFn) => {
  const promise = new Promise<void>((resolve) => {
    let next: (args: [Error]) => void;

    next = function (...args: [Error]) {
      const err = args[0];
      if (err) {
        options.fail.apply(null, args);
      }
      resolve();
    };

    next.fail = function (...args: [Error]) {
      options.fail.apply(null, args);
      resolve();
    };
    try {
      action.call(options.userContext, next);
    } catch (error: any) {
      options.onException(error);
      resolve();
    }
  });

  promise = Promise.race<void>([promise, token]);

  if (!delay) {
    return promise;
  }

  const timeoutMs: number = delay();

  return pTimeout(
    promise,
    timeoutMs,
    options.clearTimeout,
    options.setTimeout,
    () => {
      initialError.message = `Timeout - Async callback was not invoked within the ${formatTime(timeoutMs)} timeout specified by jest.setTimeout.`;
      initialError.stack = initialError.message + initialError.stack;
      options.onException(initialError);
    },
  );
};

/** @internal */
function validateNonEmptyInput(image: OptimizedImage, imageName: string, inputValue: unknown) {
  const isText = typeof inputValue === 'string';
  const isEmptyText = isText && inputValue.trim() === '';
  if (!isText || isEmptyText) {
    throw new ValidationError(
      ErrorCode.INVALID_DATA,
      `${imageDetails(image.src)} \`${imageName}\` has an invalid content ` +
        `(\`${inputValue}\`). To correct this, set the value to a non-empty string.`,
    );
  }
}

/** @internal */
/**
 *
function convertInfo(i18nData: i18n.I18nData | null | undefined): i18n.Message | null {
  if (i18nData == null) {
    return null;
  }
  if (!(i18nData instanceof i18n.Information)) {
    throw Error(`Expected i18n data to be an Information, but got: ${i18nData.constructor.name}`);
  }
  return i18nData;
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

export function isImmutableOrderedSet(source: unknown): boolean {
  return Boolean(
    source &&
      isObjectLiteral(source) &&
      source[IS_SET_SENTINEL] &&
      source[IS_ORDERED_SENTINEL],
  );
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

export const printDiffOrStringify = (
  expected: unknown,
  received: unknown,
  expectedLabel: string,
  receivedLabel: string,
  expand: boolean, // CLI options: true if `--expand` or false if `--no-expand`
): string => {
  if (
    typeof expected === 'string' &&
    typeof received === 'string' &&
    expected.length > 0 &&
    received.length > 0 &&
    expected.length <= MAX_DIFF_STRING_LENGTH &&
    received.length <= MAX_DIFF_STRING_LENGTH &&
    expected !== received
  ) {
    if (expected.includes('\n') || received.includes('\n')) {
      return diffStringsUnified(expected, received, {
        aAnnotation: expectedLabel,
        bAnnotation: receivedLabel,
        changeLineTrailingSpaceColor: chalk.bgYellow,
        commonLineTrailingSpaceColor: chalk.bgYellow,
        emptyFirstOrLastLinePlaceholder: '↵', // U+21B5
        expand,
        includeChangeCounts: true,
      });
    }

    const diffs = diffStringsRaw(expected, received, true);
    const hasCommonDiff = diffs.some(diff => diff[0] === DIFF_EQUAL);

    const printLabel = getLabelPrinter(expectedLabel, receivedLabel);
    const expectedLine =
      printLabel(expectedLabel) +
      printExpected(
        getCommonAndChangedSubstrings(diffs, DIFF_DELETE, hasCommonDiff),
      );
    const receivedLine =
      printLabel(receivedLabel) +
      printReceived(
        getCommonAndChangedSubstrings(diffs, DIFF_INSERT, hasCommonDiff),
      );

    return `${expectedLine}\n${receivedLine}`;
  }

  if (isLineDiffable(expected, received)) {
    const {replacedExpected, replacedReceived} =
      replaceMatchedToAsymmetricMatcher(expected, received, [], []);
    const difference = diffDefault(replacedExpected, replacedReceived, {
      aAnnotation: expectedLabel,
      bAnnotation: receivedLabel,
      expand,
      includeChangeCounts: true,
    });

    if (
      typeof difference === 'string' &&
      difference.includes(`- ${expectedLabel}`) &&
      difference.includes(`+ ${receivedLabel}`)
    ) {
      return difference;
    }
  }

  const printLabel = getLabelPrinter(expectedLabel, receivedLabel);
  const expectedLine = printLabel(expectedLabel) + printExpected(expected);
  const receivedLine =
    printLabel(receivedLabel) +
    (stringify(expected) === stringify(received)
      ? 'serializes to the same string'
      : printReceived(received));

  return `${expectedLine}\n${receivedLine}`;
};

function getModuleResolutionHost(resolutionHost: ResolutionCacheHost) {
    return resolutionHost.getCompilerHost?.() || resolutionHost;
}

function Bar(prop: string) {
    return (
        <div>
            {/*a*/}
            {prop && <></>}
            {/*b*/}
        </div>
    );
}

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


function resolutionIsSymlink(resolution: ResolutionWithFailedLookupLocations) {
    return !!(
        (resolution as ResolvedModuleWithFailedLookupLocations).resolvedModule?.originalPath ||
        (resolution as ResolvedTypeReferenceDirectiveWithFailedLookupLocations).resolvedTypeReferenceDirective?.originalPath
    );
}
