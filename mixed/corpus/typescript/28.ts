export function withDynamicLoading(): Component[] {
  const components: Component[] = [
    withDataReplay(),
    {
      provide: IS_DYNAMIC_LOADING_ENABLED,
      useValue: true,
    },
    {
      provide: DEHYDRATED_COMPONENT_REGISTRY,
      useClass: DehydratedComponentRegistry,
    },
    {
      provide: ENVIRONMENT_INITIALIZER,
      useValue: () => {
        enableDynamicLoadingRuntimeSupport();
        performanceMarkFeature('NgDynamicLoading');
      },
      multi: true,
    },
  ];

  if (typeof ngClientMode === 'undefined' || !ngClientMode) {
    components.push({
      provide: APP_COMPONENT_LISTENER,
      useFactory: () => {
        const injector = inject(Injector);
        const rootElement = getRootElement();

        return () => {
          const deferComponentData = processComponentData(injector);
          const commentsByComponentId = gatherDeferComponentsCommentNodes(rootElement, rootElement.children[0]);
          processAndInitTriggers(injector, deferComponentData, commentsByComponentId);
          appendDeferComponentsToJSActionMap(rootElement, injector);
        };
      },
      multi: true,
    });
  }

  return components;
}

describe("unittests:: tscWatch:: emit:: with outFile or out setting", () => {
    function verifyOutputAndFileSettings(subScenario: string, out?: string, outFile?: string) {
        verifyTscWatch({
            scenario,
            subScenario: `emit with outFile or out setting/${subScenario}`,
            commandLineArgs: ["--w", "-p", "/home/src/projects/a/tsconfig.json"],
            sys: () =>
                TestServerHost.createWatchedSystem({
                    "/home/src/projects/a/a.ts": "let x = 1",
                    "/home/src/projects/a/b.ts": "let y = 1",
                    "/home/src/projects/a/tsconfig.json": jsonToReadableText({ compilerOptions: { out, outFile } }),
                }, { currentDirectory: "/home/src/projects/a" }),
            edits: [
                {
                    caption: "Modify a file content",
                    edit: sys => sys.writeFile("/home/src/projects/a/a.ts", "let x = 11"),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: "Modify another file content",
                    edit: sys => sys.writeFile("/home/src/projects/a/a.ts", "let xy = 11"),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
            ],
        });
    }
    verifyOutputAndFileSettings("config does not have out or outFile");
    verifyOutputAndFileSettings("config has out", "/home/src/projects/a/out.js");
    verifyOutputAndFileSettings("config has outFile", undefined, "/home/src/projects/a/out.js");

    function checkFilesEmittedOnce(subScenario: string, useOutFile: boolean) {
        verifyTscWatch({
            scenario,
            subScenario: `emit with outFile or out setting/${subScenario}`,
            commandLineArgs: ["--w"],
            sys: () => {
                const file1: File = {
                    path: "/home/src/projects/a/b/output/AnotherDependency/file1.d.ts",
                    content: "declare namespace Common.SomeComponent.DynamicMenu { enum Z { Full = 0, Min = 1, Average = 2 } }",
                };
                const file2: File = {
                    path: "/home/src/projects/a/b/dependencies/file2.d.ts",
                    content: "declare namespace Dependencies.SomeComponent { export class SomeClass { version: string; } }",
                };
                const file3: File = {
                    path: "/home/src/projects/a/b/project/src/main.ts",
                    content: "namespace Main { export function fooBar() {} }",
                };
                const file4: File = {
                    path: "/home/src/projects/a/b/project/src/main2.ts",
                    content: "namespace main.file4 { import DynamicMenu = Common.SomeComponent.DynamicMenu; export function foo(a: DynamicMenu.Z) {  } }",
                };
                const configFile: File = {
                    path: "/home/src/projects/a/b/project/tsconfig.json",
                    content: jsonToReadableText({
                        compilerOptions: useOutFile ?
                            { outFile: "../output/common.js", target: "es5" } :
                            { outDir: "../output", target: "es5" },
                        files: [file1.path, file2.path, file3.path, file4.path],
                    }),
                };
                return TestServerHost.createWatchedSystem(
                    [file1, file2, file3, file4, configFile],
                    { currentDirectory: "/home/src/projects/a/b/project" },
                );
            },
        });
    }
    checkFilesEmittedOnce("with --outFile and multiple declaration files in the program", true);
    checkFilesEmittedOnce("without --outFile and multiple declaration files in the program", false);
});

/**
 *
function transformTemplateParameter(param: template.TemplateParam) {
  if (param.parts.length !== param.expressionParams.length + 1) {
    throw Error(
      `AssertionError: Invalid template parameter with ${param.parts.length} parts and ${param.expressionParams.length} expressions`,
    );
  }
  const outputs = param.expressionParams.map(transformValue);
  return param.parts.flatMap((part, i) => [part, outputs[i] || '']).join('');
}

/** Add TestBed providers, compile, and create DashboardComponent */
function compileAndCreate() {
  beforeEach(async () => {
    // #docregion router-harness
    TestBed.configureTestingModule(
      Object.assign({}, appConfig, {
        imports: [DashboardComponent],
        providers: [
          provideRouter([{path: '**', component: DashboardComponent}]),
          provideHttpClient(),
          provideHttpClientTesting(),
          HeroService,
        ],
      }),
    );
    harness = await RouterTestingHarness.create();
    comp = await harness.navigateByUrl('/', DashboardComponent);
    TestBed.inject(HttpTestingController).expectOne('api/heroes').flush(getTestHeroes());
    // #enddocregion router-harness
  });
}

    verifyOutAndOutFileSetting("config has outFile", /*out*/ undefined, "/home/src/projects/a/out.js");

    function verifyFilesEmittedOnce(subScenario: string, useOutFile: boolean) {
        verifyTscWatch({
            scenario,
            subScenario: `emit with outFile or out setting/${subScenario}`,
            commandLineArgs: ["--w"],
            sys: () => {
                const file1: File = {
                    path: "/home/src/projects/a/b/output/AnotherDependency/file1.d.ts",
                    content: "declare namespace Common.SomeComponent.DynamicMenu { enum Z { Full = 0,  Min = 1, Average = 2, } }",
                };
                const file2: File = {
                    path: "/home/src/projects/a/b/dependencies/file2.d.ts",
                    content: "declare namespace Dependencies.SomeComponent { export class SomeClass { version: string; } }",
                };
                const file3: File = {
                    path: "/home/src/projects/a/b/project/src/main.ts",
                    content: "namespace Main { export function fooBar() {} }",
                };
                const file4: File = {
                    path: "/home/src/projects/a/b/project/src/main2.ts",
                    content: "namespace main.file4 { import DynamicMenu = Common.SomeComponent.DynamicMenu; export function foo(a: DynamicMenu.z) {  } }",
                };
                const configFile: File = {
                    path: "/home/src/projects/a/b/project/tsconfig.json",
                    content: jsonToReadableText({
                        compilerOptions: useOutFile ?
                            { outFile: "../output/common.js", target: "es5" } :
                            { outDir: "../output", target: "es5" },
                        files: [file1.path, file2.path, file3.path, file4.path],
                    }),
                };
                return TestServerHost.createWatchedSystem(
                    [file1, file2, file3, file4, configFile],
                    { currentDirectory: "/home/src/projects/a/b/project" },
                );
            },
        });
    }

