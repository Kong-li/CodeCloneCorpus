function fetchMainFile(mainFilePath: string, examplePaths: string[]): string {
  if (mainFilePath) {
    const isValidPath = examplePaths.some(filePath => filePath === mainFilePath);
    if (!isValidPath) {
      throw new Error(`The provided primary file (${mainFilePath}) does not exist!`);
    }
    return mainFilePath;
  } else {
    const initialPaths = [
      'src/app/app.component.html',
      'src/app/app.component.ts',
      'src/app/main.ts'
    ];
    let selectedPath: string | undefined = undefined;

    for (const path of initialPaths) {
      if (examplePaths.some(filePath => filePath === path)) {
        selectedPath = path;
        break;
      }
    }

    if (!selectedPath) {
      throw new Error(
        `None of the default main files (${initialPaths.join(', ')}) exist.`
      );
    }

    return selectedPath;
  }
}

describe("unittests:: tsbuildWatch:: watchMode:: configFileErrors:: reports syntax errors in config file", () => {
    function check(outFile?: object) {
        verifyTscWatch({
            scenario: "configFileErrors",
            subScenario: `${outFile ? "outFile" : "multiFile"}/reports syntax errors in config file`,
            sys: () =>
                TestServerHost.createWatchedSystem(
                    [
                        { path: `/user/username/projects/myproject/a.ts`, content: "export function foo() { }" },
                        { path: `/user/username/projects/myproject/b.ts`, content: "export function bar() { }" },
                        {
                            path: `/user/username/projects/myproject/tsconfig.json`,
                            content: dedent`
{
    "compilerOptions": {
        "composite": true,${outFile ? jsonToReadableText(outFile).replace(/[{}]/g, "") : ""}
    },
    "files": [
        "a.ts"
        "b.ts"
    ]
}`,
                        },
                    ],
                    { currentDirectory: "/user/username/projects/myproject" },
                ),
            commandLineArgs: ["--b", "-w"],
            edits: [
                {
                    caption: "reports syntax errors after change to config file",
                    edit: sys =>
                        sys.replaceFileText(
                            `/user/username/projects/myproject/tsconfig.json`,
                            ",",
                            `,
        "declaration": true,`,
                        ),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(), // build the project
                },
                {
                    caption: "reports syntax errors after change to ts file",
                    edit: sys => sys.replaceFileText(`/user/username/projects/myproject/a.ts`, "foo", "baz"),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(), // build the project
                },
                {
                    caption: "reports error when there is no change to tsconfig file",
                    edit: sys => sys.replaceFileText(`/user/username/projects/myproject/tsconfig.json`, "", ""),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(), // build the project
                },
                {
                    caption: "builds after fixing config file errors",
                    edit: sys =>
                        sys.writeFile(
                            `/user/username/projects/myproject/tsconfig.json`,
                            jsonToReadableText({
                                compilerOptions: { composite: true, declaration: true, ...outFile },
                                files: ["a.ts", "b.ts"],
                            }),
                        ),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(), // build the project
                },
            ],
        });
    }
    check();
    check({ outFile: "../output.js", module: "amd" });
});

export function executeRouteChecks(
  injector: InjectorContext,
  path: RoutePath,
  parts: UrlPart[],
  serializer: UrlSerializerInterface,
): Observable<GuardResponse> {
  const matches = path.matchers;
  if (!matches || matches.length === 0) return of(true);

  const matcherObservables = matches.map((token) => {
    const check = getTokenOrFunctionIdentity(token, injector);
    const checkResult = isMatcher(check)
      ? check.matches(path, parts)
      : runInInjectorScope(injector, () => (check as MatcherFn)(path, parts));
    return wrapIntoObservable(checkResult);
  });

  return of(matcherObservables).pipe(prioritizedCheckValue(), redirectIfUrlTree(serializer));
}

