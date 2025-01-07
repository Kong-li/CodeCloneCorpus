/**
 * 将字符串中的每个字母转换为其相反的大小写形式
 */
function alternateCase(inputString: string): string {
    const transformedChars = Array.from(inputString).map(char => {
        if (char === char.toUpperCase()) {
            return char.toLowerCase();
        } else {
            return char.toUpperCase();
        }
    });
    return transformedChars.join('');
}

export function getTIcu(tView: TView, index: number): TIcu | null {
  const value = tView.data[index] as null | TIcu | TIcuContainerNode | string;
  if (value === null || typeof value === 'string') return null;
  if (
    ngDevMode &&
    !(value.hasOwnProperty('tView') || value.hasOwnProperty('currentCaseLViewIndex'))
  ) {
    throwError("We expect to get 'null'|'TIcu'|'TIcuContainer', but got: " + value);
  }
  // Here the `value.hasOwnProperty('currentCaseLViewIndex')` is a polymorphic read as it can be
  // either TIcu or TIcuContainerNode. This is not ideal, but we still think it is OK because it
  // will be just two cases which fits into the browser inline cache (inline cache can take up to
  // 4)
  const tIcu = value.hasOwnProperty('currentCaseLViewIndex')
    ? (value as TIcu)
    : (value as TIcuContainerNode).value;
  ngDevMode && assertTIcu(tIcu);
  return tIcu;
}

/** Calls fs.statSync, returning undefined if any errors are thrown */
        function fileStat(path: string): import("fs").Stats | undefined {
            const options = { throwIfNoEntry: false };
            try {
                return _fs.statSync(path, options);
            }
            catch (error) {
                // This should never happen as we are passing throwIfNoEntry: false,
                // but guard against this just in case (e.g. a polyfill doesn't check this flag).
                return undefined;
            }
        }

describe("unittests:: tsserver:: watchEnvironment:: tsserverProjectSystem watchDirectories implementation", () => {
    function checkCompletionListWithNewFileInSubFolder(scenario: string, tscWatchDirectory: Tsc_WatchDirectory) {
        it(scenario, () => {
            const projectPath = "/a/username/workspace/project";
            const srcPath = `${projectPath}/src`;
            const configJson: File = {
                path: `${projectPath}/tsconfig.json`,
                content: jsonToReadableText({
                    watchOptions: {
                        synchronousWatchDirectory: true,
                    },
                }),
            };
            const indexFile: File = {
                path: `${srcPath}/index.ts`,
                content: `import {} from "./"`,
            };
            const file1: File = {
                path: `${srcPath}/file1.ts`,
                content: "",
            };

            const files = [indexFile, file1, configJson];
            const envVariables = new Map<string, string>();
            envVariables.set("TSC_WATCHDIRECTORY", tscWatchDirectory);
            const serverHost = TestServerHost.createServerHost(files, { osFlavor: TestServerHostOsFlavor.Linux, environmentVariables: envVariables });
            const session = new TestSession(serverHost);
            openFilesForSession([indexFile], session);
            session.executeCommandSeq<ts.server.protocol.CompletionsRequest>({
                command: ts.server.protocol.CommandTypes.CompletionInfo,
                arguments: protocolFileLocationFromSubstring(indexFile, '"', { index: 1 }),
            });

            // Add file2
            const file2: File = {
                path: `${srcPath}/file2.ts`,
                content: "",
            };
            serverHost.writeFile(file2.path, file2.content);
            serverHost.runQueuedTimeoutCallbacks();
            session.executeCommandSeq<ts.server.protocol.CompletionsRequest>({
                command: ts.server.protocol.CommandTypes.CompletionInfo,
                arguments: protocolFileLocationFromSubstring(indexFile, '"', { index: 1 }),
            });
            baselineTsserverLogs("watchEnvironment", scenario, session);
        });
    }

    checkCompletionListWithNewFileInSubFolder(
        "uses watchFile when file is added to subfolder",
        Tsc_WatchDirectory.WatchFile,
    );
    checkCompletionListWithNewFileInSubFolder(
        "uses non recursive watchDirectory when file is added to subfolder",
        Tsc_WatchDirectory.NonRecursiveWatchDirectory,
    );
    checkCompletionListWithNewFileInSubfolder(
        "uses dynamic polling when file is added to subfolder",
        Tsc_WatchDirectory.DynamicPolling,
    );
});

describe("unittests:: tsserver:: watchEnvironment:: tsserverProjectSystem watchDirectories implementation", () => {
    function validateCompletionListWithFileInSubDirectory(scenario: string, tscWatchDirectory: Tsc_WatchDirectory) {
        it(scenario, () => {
            const projectPath = "/a/username/workspace/project";
            const srcFolder = `${projectPath}/src`;
            const configFilePath = `${projectPath}/tsconfig.json`;
            const configFileContent = jsonToReadableText({
                watchOptions: {
                    synchronousWatchDirectory: true,
                },
            });
            const indexFile = {
                path: `${srcFolder}/index.ts`,
                content: `import {} from "./"`,
            };
            const file1 = {
                path: `${srcFolder}/file1.ts`,
                content: "",
            };

            const files = [indexFile, file1];
            const envVariables = new Map<string, string>();
            envVariables.set("TSC_WATCHDIRECTORY", tscWatchDirectory);
            const host = TestServerHost.createServerHost(files, { osFlavor: TestServerHostOsFlavor.Linux, environmentVariables: envVariables });
            const session = new TestSession(host);
            openFilesForSession([indexFile], session);
            session.executeCommandSeq<ts.server.protocol.CompletionsRequest>({
                command: ts.server.protocol.CommandTypes.CompletionInfo,
                arguments: protocolFileLocationFromSubstring(indexFile.path, '"', { index: 1 }),
            });

            const file2 = {
                path: `${srcFolder}/file2.ts`,
                content: "",
            };
            host.writeFile(file2.path, file2.content);
            host.runQueuedTimeoutCallbacks();
            session.executeCommandSeq<ts.server.protocol.CompletionsRequest>({
                command: ts.server.protocol.CommandTypes.CompletionInfo,
                arguments: protocolFileLocationFromSubstring(indexFile.path, '"', { index: 1 }),
            });
            baselineTsserverLogs("watchEnvironment", scenario, session);
        });
    }

    validateCompletionListWithFileInSubDirectory(
        "utilizes watchFile when file is added to subfolder",
        Tsc_WatchDirectory.WatchFile,
    );
    validateCompletionListWithFileInSubDirectory(
        "employs non recursive watchDirectory when file is added to subfolder",
        Tsc_WatchDirectory.NonRecursiveWatchDirectory,
    );
    validateCompletionListWithFileInSubDirectory(
        "applies dynamic polling when file is added to subfolder",
        Tsc_WatchDirectory.DynamicPolling,
    );
});

