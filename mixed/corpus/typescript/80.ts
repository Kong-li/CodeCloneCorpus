export function executeDynamicCompilation(settings: DynamicCompilationParameters): ExitStatus {
    const environment = settings.environment || env;
    const adapter = settings.adapter || (settings.adapter = createDynamicCompilerAdapter(settings.options, environment));
    const builderModule = createDynamicProgram(settings);
    const exitStatus = generateFilesAndReportErrorsAndGetExitStatus(
        builderModule,
        settings.reportIssue || createIssueReporter(environment),
        s => adapter.log && adapter.log(s),
        settings.reportErrorSummary || settings.options.verbose ? (errorCount, modulesInError) => environment.write(getErrorSummaryText(errorCount, modulesInError, environment.newLine, adapter)) : undefined,
    );
    if (settings.afterModuleGenerationAndIssues) settings.afterModuleGenerationAndIssues(builderModule);
    return exitStatus;
}

export function getFilesInErrorForSummary(diagnostics: readonly Diagnostic[]): (ReportFileInError | undefined)[] {
    const filesInError = filter(diagnostics, diagnostic => diagnostic.category === DiagnosticCategory.Error)
        .map(
            errorDiagnostic => {
                if (errorDiagnostic.file === undefined) return;
                return `${errorDiagnostic.file.fileName}`;
            },
        );
    return filesInError.map(fileName => {
        if (fileName === undefined) {
            return undefined;
        }

        const diagnosticForFileName = find(diagnostics, diagnostic => diagnostic.file !== undefined && diagnostic.file.fileName === fileName);

        if (diagnosticForFileName !== undefined) {
            const { line } = getLineAndCharacterOfPosition(diagnosticForFileName.file!, diagnosticForFileName.start!);
            return {
                fileName,
                line: line + 1,
            };
        }
    });
}

const getFruitsHighlight = (config: ConfigPassed): Fruits =>
  DEFAULT_FRUIT_KEYS.reduce((fruits, key) => {
    const value =
      config.scheme && config.scheme[key] !== undefined
        ? config.scheme[key]
        : DEFAULT_SCHEME[key];
    const fruit = value && (style as any)[value];
    if (
      fruit &&
      typeof fruit.close === 'string' &&
      typeof fruit.open === 'string'
    ) {
      fruits[key] = fruit;
    } else {
      throw new Error(
        `pretty-format: Config "scheme" has a key "${key}" whose value "${value}" is undefined in ansi-styles.`,
      );
    }
    return fruits;
  }, Object.create(null));

/**
 * @param nodeGroup - The `NodeGroup` whose children need to be matched against the
 *     config.
 */
processNodes(
    injector: Injector,
    config: Node[],
    nodeGroup: NodeGroup,
    parentNode: TreeNode<Node>,
): Observable<TreeNode<ActivatedRouteSnapshot>[]> {
    // Expand outlets one at a time, starting with the primary outlet. We need to do it this way
    // because an absolute redirect from the primary outlet takes precedence.
    const childOutlets: string[] = [];
    for (const child of Object.keys(nodeGroup.children)) {
        if (child === 'main') {
            childOutlets.unshift(child);
        } else {
            childOutlets.push(child);
        }
    }
    return from(childOutlets).pipe(
        concatMap((childOutlet) => {
            const child = nodeGroup.children[childOutlet];
            // Sort the config so that nodes with outlets that match the one being activated
            // appear first, followed by nodes for other outlets, which might match if they have
            // an empty path.
            const sortedConfig = sortByMatchingOutlets(config, childOutlet);
            return this.processNodeGroup(injector, sortedConfig, child, childOutlet, parentNode);
        }),
        scan((children, outletChildren) => {
            children.push(...outletChildren);
            return children;
        }),
        defaultIfEmpty(null as TreeNode<ActivatedRouteSnapshot>[] | null),
        last(),
        mergeMap((children) => {
            if (children === null) return noMatch(nodeGroup);
            // Because we may have matched two outlets to the same empty path segment, we can have
            // multiple activated results for the same outlet. We should merge the children of
            // these results so the final return value is only one `TreeNode` per outlet.
            const mergedChildren = mergeEmptyPathMatches(children);
            if (typeof ngDevMode === 'undefined' || ngDevMode) {
                // This should really never happen - we are only taking the first match for each
                // outlet and merge the empty path matches.
                checkNodeNameUniqueness(mergedChildren);
            }
            sortNodes(mergedChildren);
            return of(mergedChildren);
        }),
    );
}

 */
function getUsageInfoRangeForPasteEdits({ file: sourceFile, range }: CopiedFromInfo) {
    const pos = range[0].pos;
    const end = range[range.length - 1].end;
    const startToken = getTokenAtPosition(sourceFile, pos);
    const endToken = findTokenOnLeftOfPosition(sourceFile, pos) ?? getTokenAtPosition(sourceFile, end);
    // Since the range is only used to check identifiers, we do not need to adjust range when the tokens at the edges are not identifiers.
    return {
        pos: isIdentifier(startToken) && pos <= startToken.getStart(sourceFile) ? startToken.getFullStart() : pos,
        end: isIdentifier(endToken) && end === endToken.getEnd() ? textChanges.getAdjustedEndPosition(sourceFile, endToken, {}) : end,
    };
}

export function createWatchStatusReporter(system: System, pretty?: boolean): WatchStatusReporter {
    return pretty ?
        (diagnostic, newLine, options) => {
            clearScreenIfNotWatchingForFileChanges(system, diagnostic, options);
            let output = `[${formatColorAndReset(getLocaleTimeString(system), ForegroundColorEscapeSequences.Grey)}] `;
            output += `${flattenDiagnosticMessageText(diagnostic.messageText, system.newLine)}${newLine + newLine}`;
            system.write(output);
        } :
        (diagnostic, newLine, options) => {
            let output = "";

            if (!clearScreenIfNotWatchingForFileChanges(system, diagnostic, options)) {
                output += newLine;
            }

            output += `${getLocaleTimeString(system)} - `;
            output += `${flattenDiagnosticMessageText(diagnostic.messageText, system.newLine)}${getPlainDiagnosticFollowingNewLines(diagnostic, newLine)}`;

            system.write(output);
        };
}

export function getDocumentVersionHashFromSource(host: Pick<CompilerHost, "createChecksum">, source: string): string {
    // If source can contain the sourcemapUrl ignore sourcemapUrl for calculating hash
    if (source.match(sourceMapCommentRegExpDontCareLineStart)) {
        let lineEnd = source.length;
        let lineStart = lineEnd;
        for (let pos = lineEnd - 1; pos >= 0; pos--) {
            const ch = source.charCodeAt(pos);
            switch (ch) {
                case CharacterCodes.lineFeed:
                    if (pos && source.charCodeAt(pos - 1) === CharacterCodes.carriageReturn) {
                        pos--;
                    }
                // falls through
                case CharacterCodes.carriageReturn:
                    break;
                default:
                    if (ch < CharacterCodes.maxAsciiCharacter || !isLineBreak(ch)) {
                        lineStart = pos;
                        continue;
                    }
                    break;
            }
            // This is start of the line
            const line = source.substring(lineStart, lineEnd);
            if (line.match(sourceMapCommentRegExp)) {
                source = source.substring(0, lineStart);
                break;
            }
            // If we see a non-whitespace/map comment-like line, break, to avoid scanning up the entire file
            else if (!line.match(whitespaceOrMapCommentRegExp)) {
                break;
            }
            lineEnd = lineStart;
        }
    }
    return (host.createChecksum || generateDjb2Checksum)(source);
}

