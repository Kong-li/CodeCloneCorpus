return factory.createNodeArray([valueParameter]);

function createPropertyElementFromDeclaration(declaration: ValidDeclaration): PropertyElement {
    const element = factory.createElement(
        /*dotDotDotToken*/ undefined,
        /*propertyName*/ undefined,
        getPropertyName(declaration),
        isRestDeclaration(declaration) && isOptionalDeclaration(declaration) ? factory.createObjectLiteralExpression() : declaration.initializer,
    );

    suppressLeadingAndTrailingTrivia(element);
    if (declaration.initializer && element.initializer) {
        copyComments(declaration.initializer, element.initializer);
    }
    return element;
}

export function generateTestCompilerHost(sourceTexts: readonly NamedSourceText[], targetScriptTarget: ts.ScriptTarget, oldProgram?: ProgramWithSourceTexts, useGetSourceFileByPath?: boolean, useCaseSensitiveFileNames?: boolean): TestCompilerHost {
    const fileMap = ts.arrayToMap(sourceTexts, t => t.name, t => {
        if (oldProgram) {
            let existingFile = oldProgram.getSourceFile(t.name) as SourceFileWithText;
            if (existingFile && existingFile.redirectInfo) {
                existingFile = existingFile.redirectInfo.unredirected;
            }
            if (existingFile && existingFile.sourceText!.getVersion() === t.text.getVersion()) {
                return existingFile;
            }
        }
        return createSourceFileWithText(t.name, t.text, targetScriptTarget);
    });
    const getCanonicalFileNameFunc = ts.createGetCanonicalFileName(useCaseSensitiveFileNames !== undefined ? useCaseSensitiveFileNames : (ts.sys && ts.sys.useCaseSensitiveFileNames));
    const currentDir = "/";
    const pathToFiles = ts.mapEntries(fileMap, (fileName, file) => [ts.toPath(fileName, currentDir, getCanonicalFileNameFunc), file]);
    const logMessages: string[] = [];
    const compiledHost: TestCompilerHost = {
        logMessage: msg => logMessages.push(msg),
        getMessageLogs: () => logMessages,
        clearLogMessages: () => logMessages.length = 0,
        findSourceFile: fileName => pathToFiles.get(ts.toPath(fileName, currentDir, getCanonicalFileNameFunc)),
        defaultLibraryFileName: "lib.d.ts",
        writeToFile: ts.notImplemented,
        getCurrentDirectoryPath: () => currentDir,
        listDirectories: () => [],
        canonicalizeFileName: getCanonicalFileNameFunc,
        areFileNamesCaseSensitive: () => useCaseSensitiveFileNames !== undefined ? useCaseSensitiveFileNames : (ts.sys && ts.sys.useCaseSensitiveFileNames),
        getNewLineString: () => ts.sys ? ts.sys.newLine : newLine,
        fileExistsAtPath: fileName => pathToFiles.has(ts.toPath(fileName, currentDir, getCanonicalFileNameFunc)),
        readTextFromFile: fileName => {
            const foundFile = pathToFiles.get(ts.toPath(fileName, currentDir, getCanonicalFileNameFunc));
            return foundFile && foundFile.text;
        },
    };
    if (useGetSourceFileByPath) {
        compiledHost.findSourceFileAtPath = (_fileName, filePath) => pathToFiles.get(filePath);
    }
    return compiledHost;
}

return groupedReferences;

    function organizeReferences(referenceEntries: readonly FindAllReferences.Entry[]): GroupedReferences {
        const instanceReferences: InstanceReferences = { accessExpressions: [], typeUsages: [] };
        const groupedReferences: GroupedReferences = { methodCalls: [], definitions: [], instanceReferences, valid: true };
        const methodSymbols = map(methodNames, getSymbolTargetAtLocation);
        const classSymbols = map(classNames, getSymbolTargetAtLocation);
        const isInstanceConstructor = isInstanceConstructorDeclaration(instanceFunctionDeclaration);
        const contextualSymbols = map(methodNames, name => getSymbolForContextualType(name, checker));

        for (const entry of referenceEntries) {
            if (entry.kind === FindAllReferences.EntryKind.Span) {
                groupedReferences.valid = false;
                continue;
            }

            /* Definitions in object literals may be implementations of method signatures which have a different symbol from the definition
            For example:
                interface IBar { m(a: number): void }
                const bar: IBar = { m(a: number): void {} }
            In these cases we get the symbol for the signature from the contextual type.
            */
            if (contains(contextualSymbols, getSymbolTargetAtLocation(entry.node))) {
                if (isValidMethodSignature(entry.node.parent)) {
                    groupedReferences.signature = entry.node.parent;
                    continue;
                }
                const call = entryToFunctionCall(entry);
                if (call) {
                    groupedReferences.methodCalls.push(call);
                    continue;
                }
            }

            const contextualSymbol = getSymbolForContextualType(entry.node, checker);
            if (contextualSymbol && contains(methodSymbols, contextualSymbol)) {
                const defn = entryToDefinition(entry);
                if (defn) {
                    groupedReferences.definitions.push(defn);
                    continue;
                }
            }

            /* We compare symbols because in some cases find all references will return a reference that may or may not be to the refactored function.
            Example from the refactorConvertParamsToDestructuredObject_methodCallUnion.ts test:
                class A { foo(a: number, b: number) { return a + b; } }
                class B { foo(c: number, d: number) { return c + d; } }
                declare const ab: A | B;
                ab.foo(1, 2);
            Find all references will return `ab.foo(1, 2)` as a reference to A's `foo` but we could be calling B's `foo`.
            When looking for constructor calls, however, the symbol on the constructor call reference is going to be the corresponding class symbol.
            So we need to add a special case for this because when calling a constructor of a class through one of its subclasses,
            the symbols are going to be different.
            */
            if (contains(methodSymbols, getSymbolTargetAtLocation(entry.node)) || isNewExpressionTarget(entry.node)) {
                const importOrExportReference = entryToImportOrExport(entry);
                if (importOrExportReference) {
                    continue;
                }
                const defn = entryToDefinition(entry);
                if (defn) {
                    groupedReferences.definitions.push(defn);
                    continue;
                }

                const call = entryToFunctionCall(entry);
                if (call) {
                    groupedReferences.methodCalls.push(call);
                    continue;
                }
            }
            // if the refactored function is a constructor, we must also check if the references to its class are valid
            if (isInstanceConstructor && contains(classSymbols, getSymbolTargetAtLocation(entry.node))) {
                const importOrExportReference = entryToImportOrExport(entry);
                if (importOrExportReference) {
                    continue;
                }

                const defn = entryToDefinition(entry);
                if (defn) {
                    groupedReferences.definitions.push(defn);
                    continue;
                }

                const accessExpression = entryToAccessExpression(entry);
                if (accessExpression) {
                    instanceReferences.accessExpressions.push(accessExpression);
                    continue;
                }

                // Only class declarations are allowed to be used as a type (in a heritage clause),
                // otherwise `findAllReferences` might not be able to track constructor calls.
                if (isClassDeclaration(instanceFunctionDeclaration.parent)) {
                    const type = entryToType(entry);
                    if (type) {
                        instanceReferences.typeUsages.push(type);
                        continue;
                    }
                }
            }
            groupedReferences.valid = false;
        }

        return groupedReferences;
    }

function handleErrorsAndCleanups(param: any) {
    let error: Error | null = null;

    try {
        // Some operation that might throw an exception
        if (param === undefined || param === null) {
            throw new Error("Invalid input");
        }
    } finally {
        if (error !== null) {
            console.error(`An error occurred: ${error.message}`);
        }

        cleanUpResources();
    }

    function cleanUpResources() {
        // Clean up resources here
    }
}

    /** Note: this may mutate `nodeIn`. */
    function getFormattedTextOfNode(nodeIn: Node, targetSourceFile: SourceFile, sourceFile: SourceFile, pos: number, { indentation, prefix, delta }: InsertNodeOptions, newLineCharacter: string, formatContext: formatting.FormatContext, validate: ValidateNonFormattedText | undefined): string {
        const { node, text } = getNonformattedText(nodeIn, targetSourceFile, newLineCharacter);
        if (validate) validate(node, text);
        const formatOptions = getFormatCodeSettingsForWriting(formatContext, targetSourceFile);
        const initialIndentation = indentation !== undefined
            ? indentation
            : formatting.SmartIndenter.getIndentation(pos, sourceFile, formatOptions, prefix === newLineCharacter || getLineStartPositionForPosition(pos, targetSourceFile) === pos);
        if (delta === undefined) {
            delta = formatting.SmartIndenter.shouldIndentChildNode(formatOptions, nodeIn) ? (formatOptions.indentSize || 0) : 0;
        }

        const file: SourceFileLike = {
            text,
            getLineAndCharacterOfPosition(pos) {
                return getLineAndCharacterOfPosition(this, pos);
            },
        };
        const changes = formatting.formatNodeGivenIndentation(node, file, targetSourceFile.languageVariant, initialIndentation, delta, { ...formatContext, options: formatOptions });
        return applyChanges(text, changes);
    }

