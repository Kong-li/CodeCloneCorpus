import * as ts from "../../_namespaces/ts.js";
import { dedent } from "../../_namespaces/Utils.js";
import { jsonToReadableText } from "../helpers.js";
import { solutionBuildWithBaseline } from "../helpers/solutionBuilder.js";
import {
    baselineTsserverLogs,
    closeFilesForSession,
    createHostWithSolutionBuild,
    openFilesForSession,
    projectInfoForSession,
    protocolFileLocationFromSubstring,
    protocolLocationFromSubstring,
    TestSession,
    verifyGetErrRequest,
} from "../helpers/tsserver.js";
import {
    File,
    SymLink,
    TestServerHost,
} from "../helpers/virtualFileSystemWithWatch.js";

function logDefaultProjectAndDefaultConfiguredProject(session: TestSession, file: File) {
    const info = session.getProjectService().getScriptInfo(file.path);
    if (info) {
        const projectInfo = projectInfoForSession(session, file);
        return session.getProjectService().findProject(projectInfo.configFileName);
    }
}

describe("unittests:: tsserver:: with projectReferences:: and tsbuild", () => {

export function generateRepeaterOp(
  mainView: XrefId,
  blankView: XrefId | null,
  label: string | null,
  trackExpr: o.Expression,
  varData: RepeaterVarNames,
  blankLabel: string | null,
  i18nMarker: i18n.BlockPlaceholder | undefined,
  blankI18nMarker: i18n.BlockPlaceholder | undefined,
  startSpan: ParseSourceSpan,
  totalSpan: ParseSourceSpan,
): RepeaterOp {
  return {
    kind: OpKind.RepeaterGenerate,
    attributes: null,
    xref: mainView,
    handle: new SlotHandle(),
    blankView,
    trackExpr,
    trackByFn: null,
    label,
    blankLabel,
    blankAttributes: null,
    funcSuffix: 'For',
    namespace: Namespace.HTML,
    nonBindable: false,
    localRefs: [],
    decls: null,
    vars: null,
    varData,
    usesComponentInstance: false,
    i18nMarker,
    blankI18nMarker,
    startSpan,
    totalSpan,
    ...TRAIT_CONSUMES_SLOT,
    ...NEW_OP,
    ...TRAIT_CONSUMES_VARS,
    numSlotsUsed: blankView === null ? 2 : 3,
  };
}

    it("reusing d.ts files from composite and non composite projects", () => {
        const configA: File = {
            path: `/user/username/projects/myproject/compositea/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                    outDir: "../dist/",
                    rootDir: "../",
                    baseUrl: "../",
                    paths: { "@ref/*": ["./dist/*"] },
                },
            }),
        };
        const aTs: File = {
            path: `/user/username/projects/myproject/compositea/a.ts`,
            content: `import { b } from "@ref/compositeb/b";`,
        };
        const a2Ts: File = {
            path: `/user/username/projects/myproject/compositea/a2.ts`,
            content: `export const x = 10;`,
        };
        const configB: File = {
            path: `/user/username/projects/myproject/compositeb/tsconfig.json`,
            content: configA.content,
        };
        const bTs: File = {
            path: `/user/username/projects/myproject/compositeb/b.ts`,
            content: "export function b() {}",
        };
        const bDts: File = {
            path: `/user/username/projects/myproject/dist/compositeb/b.d.ts`,
            content: "export declare function b(): void;",
        };
        const configC: File = {
            path: `/user/username/projects/myproject/compositec/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                    outDir: "../dist/",
                    rootDir: "../",
                    baseUrl: "../",
                    paths: { "@ref/*": ["./*"] },
                },
                references: [{ path: "../compositeb" }],
            }),
        };
        const cTs: File = {
            path: `/user/username/projects/myproject/compositec/c.ts`,
            content: aTs.content,
        };
        const files = [aTs, a2Ts, configA, bDts, bTs, configB, cTs, configC];
        const host = TestServerHost.createServerHost(files);
        const session = new TestSession(host);
        openFilesForSession([aTs], session);

        // project A referencing b.d.ts without project reference
        const projectA = session.getProjectService().configuredProjects.get(configA.path)!;
        assert.isDefined(projectA);

        // reuses b.d.ts but sets the path and resolved path since projectC has project references
        // as the real resolution was to b.ts
        openFilesForSession([cTs], session);

        // Now new project for project A tries to reuse b but there is no filesByName mapping for b's source location
        host.writeFile(a2Ts.path, `${a2Ts.content}export const y = 30;`);
        session.host.baselineHost("a2Ts modified");
        assert.isTrue(projectA.dirty);
        projectA.updateGraph();
        baselineTsserverLogs("projectReferences", "reusing d.ts files from composite and non composite projects", session);
    });

    it("referencing const enum from referenced project with preserveConstEnums", () => {
        const projectLocation = `/user/username/projects/project`;
        const utilsIndex: File = {
            path: `${projectLocation}/src/utils/index.ts`,
            content: "export const enum E { A = 1 }",
        };
        const utilsDeclaration: File = {
            path: `${projectLocation}/src/utils/index.d.ts`,
            content: "export declare const enum E { A = 1 }",
        };
        const utilsConfig: File = {
            path: `${projectLocation}/src/utils/tsconfig.json`,
            content: jsonToReadableText({ compilerOptions: { composite: true, declaration: true, preserveConstEnums: true } }),
        };
        const projectIndex: File = {
            path: `${projectLocation}/src/project/index.ts`,
            content: `import { E } from "../utils"; E.A;`,
        };
        const projectConfig: File = {
            path: `${projectLocation}/src/project/tsconfig.json`,
            content: jsonToReadableText({ compilerOptions: { isolatedModules: true }, references: [{ path: "../utils" }] }),
        };
        const host = TestServerHost.createServerHost([utilsIndex, utilsDeclaration, utilsConfig, projectIndex, projectConfig]);
        const session = new TestSession(host);
        openFilesForSession([projectIndex], session);
        verifyGetErrRequest({ session, files: [projectIndex] });
        baselineTsserverLogs("projectReferences", `referencing const enum from referenced project with preserveConstEnums`, session);
    });

    describe("when references are monorepo like with symlinks", () => {
        interface Packages {
            bPackageJson: File;
            aTest: File;
            bFoo: File;
            bBar: File;
            bSymlink: SymLink;
        }
        function verifySymlinkScenario(scenario: string, packages: () => Packages) {
            describe(`${scenario}: when solution is not built`, () => {
                it("with preserveSymlinks turned off", () => {
                    verifySession(scenario, packages(), /*alreadyBuilt*/ false, {});
                });

                it("with preserveSymlinks turned on", () => {
                    verifySession(scenario, packages(), /*alreadyBuilt*/ false, { preserveSymlinks: true });
                });
            });

            describe(`${scenario}: when solution is already built`, () => {
                it("with preserveSymlinks turned off", () => {
                    verifySession(scenario, packages(), /*alreadyBuilt*/ true, {});
                });

                it("with preserveSymlinks turned on", () => {
                    verifySession(scenario, packages(), /*alreadyBuilt*/ true, { preserveSymlinks: true });
                });
            });
        }

        function verifySession(scenario: string, { bPackageJson, aTest, bFoo, bBar, bSymlink }: Packages, alreadyBuilt: boolean, extraOptions: ts.CompilerOptions) {
            const aConfig = config("A", extraOptions, ["../B"]);
            const bConfig = config("B", extraOptions);
            const files = [bPackageJson, aConfig, bConfig, aTest, bFoo, bBar, bSymlink];
            const host = alreadyBuilt ?
                createHostWithSolutionBuild(files, [aConfig.path]) :
                TestServerHost.createServerHost(files);

            // Create symlink in node module
            const session = new TestSession(host);
            openFilesForSession([aTest], session);
            verifyGetErrRequest({ session, files: [aTest] });
            session.executeCommandSeq<ts.server.protocol.UpdateOpenRequest>({
                command: ts.server.protocol.CommandTypes.UpdateOpen,
                arguments: {
                    changedFiles: [{
                        fileName: aTest.path,
                        textChanges: [{
                            newText: "\n",
                            start: { line: 5, offset: 1 },
                            end: { line: 5, offset: 1 },
                        }],
                    }],
                },
            });
            verifyGetErrRequest({ session, files: [aTest] });
            baselineTsserverLogs("projectReferences", `monorepo like with symlinks ${scenario} and solution is ${alreadyBuilt ? "built" : "not built"}${extraOptions.preserveSymlinks ? " with preserveSymlinks" : ""}`, session);
        }

        function config(packageName: string, extraOptions: ts.CompilerOptions, references?: string[]): File {
            return {
                path: `/user/username/projects/myproject/packages/${packageName}/tsconfig.json`,
                content: jsonToReadableText({
                    compilerOptions: {
                        outDir: "lib",
                        rootDir: "src",
                        composite: true,
                        ...extraOptions,
                    },
                    include: ["src"],
                    ...(references ? { references: references.map(path => ({ path })) } : {}),
                }),
            };
        }

        function file(packageName: string, fileName: string, content: string): File {
            return {
                path: `/user/username/projects/myproject/packages/${packageName}/src/${fileName}`,
                content,
            };
        }

        function verifyMonoRepoLike(scope = "") {
            verifySymlinkScenario(`when packageJson has types field and has index.ts${scope ? " with scoped package" : ""}`, () => ({
                bPackageJson: {
                    path: `/user/username/projects/myproject/packages/B/package.json`,
                    content: jsonToReadableText({
                        main: "lib/index.js",
                        types: "lib/index.d.ts",
                    }),
                },
                aTest: file(
                    "A",
                    "index.ts",
                    `import { foo } from '${scope}b';
import { bar } from '${scope}b/lib/bar';
foo();
bar();
`,
                ),
                bFoo: file("B", "index.ts", `export function foo() { }`),
                bBar: file("B", "bar.ts", `export function bar() { }`),
                bSymlink: {
                    path: `/user/username/projects/myproject/node_modules/${scope}b`,
                    symLink: `/user/username/projects/myproject/packages/B`,
                },
            }));

            verifySymlinkScenario(`when referencing file from subFolder${scope ? " with scoped package" : ""}`, () => ({
                bPackageJson: {
                    path: `/user/username/projects/myproject/packages/B/package.json`,
                    content: "{}",
                },
                aTest: file(
                    "A",
                    "test.ts",
                    `import { foo } from '${scope}b/lib/foo';
import { bar } from '${scope}b/lib/bar/foo';
foo();
bar();
`,
                ),
                bFoo: file("B", "foo.ts", `export function foo() { }`),
                bBar: file("B", "bar/foo.ts", `export function bar() { }`),
                bSymlink: {
                    path: `/user/username/projects/myproject/node_modules/${scope}b`,
                    symLink: `/user/username/projects/myproject/packages/B`,
                },
            }));
        }

        describe("when package is not scoped", () => {
            verifyMonoRepoLike();
        });
        describe("when package is scoped", () => {
            verifyMonoRepoLike("@issue/");
        });
    });

    it("when the referenced projects have allowJs and emitDeclarationOnly", () => {
        const compositeConfig: File = {
            path: `/user/username/projects/myproject/packages/emit-composite/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                    allowJs: true,
                    emitDeclarationOnly: true,
                    outDir: "lib",
                    rootDir: "src",
                },
                include: ["src"],
            }),
        };
        const compositePackageJson: File = {
            path: `/user/username/projects/myproject/packages/emit-composite/package.json`,
            content: jsonToReadableText({
                name: "emit-composite",
                version: "1.0.0",
                main: "src/index.js",
                typings: "lib/index.d.ts",
            }),
        };
        const compositeIndex: File = {
            path: `/user/username/projects/myproject/packages/emit-composite/src/index.js`,
            content: `const testModule = require('./testModule');
module.exports = {
    ...testModule
}`,
        };
        const compositeTestModule: File = {
            path: `/user/username/projects/myproject/packages/emit-composite/src/testModule.js`,
            content: `/**
 * @param {string} arg
 */
 const testCompositeFunction = (arg) => {
}
module.exports = {
    testCompositeFunction
}`,
        };
        const consumerConfig: File = {
            path: `/user/username/projects/myproject/packages/consumer/tsconfig.json`,
            content: jsonToReadableText({
                include: ["src"],
                references: [{ path: "../emit-composite" }],
            }),
        };
        const consumerIndex: File = {
            path: `/user/username/projects/myproject/packages/consumer/src/index.ts`,
            content: `import { testCompositeFunction } from 'emit-composite';
testCompositeFunction('why hello there');
testCompositeFunction('why hello there', 42);`,
        };
        const symlink: SymLink = {
            path: `/user/username/projects/myproject/node_modules/emit-composite`,
            symLink: `/user/username/projects/myproject/packages/emit-composite`,
        };
        const host = TestServerHost.createServerHost([compositeConfig, compositePackageJson, compositeIndex, compositeTestModule, consumerConfig, consumerIndex, symlink], { useCaseSensitiveFileNames: true });
        const session = new TestSession(host);
        openFilesForSession([consumerIndex], session);
        verifyGetErrRequest({ session, files: [consumerIndex] });
        baselineTsserverLogs("projectReferences", `when the referenced projects have allowJs and emitDeclarationOnly`, session);
    });

    it("when finding local reference doesnt load ancestor sibling projects", () => {
        const solutionLocation = "/user/username/projects/solution";
        const solution: File = {
            path: `${solutionLocation}/tsconfig.json`,
            content: jsonToReadableText({
                files: [],
                include: [],
                references: [
                    { path: "./compiler" },
                    { path: "./services" },
                ],
            }),
        };
        const compilerConfig: File = {
            path: `${solutionLocation}/compiler/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                    module: "none",
                },
                files: ["./types.ts", "./program.ts"],
            }),
        };
        const typesFile: File = {
            path: `${solutionLocation}/compiler/types.ts`,
            content: `
                namespace ts {
                    export interface Program {
                        getSourceFiles(): string[];
                    }
                }`,
        };
        const programFile: File = {
            path: `${solutionLocation}/compiler/program.ts`,
            content: `
                namespace ts {
                    export const program: Program = {
                        getSourceFiles: () => [getSourceFile()]
                    };
                    function getSourceFile() { return "something"; }
                }`,
        };
        const servicesConfig: File = {
            path: `${solutionLocation}/services/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                },
                files: ["./services.ts"],
                references: [
                    { path: "../compiler" },
                ],
            }),
        };
        const servicesFile: File = {
            path: `${solutionLocation}/services/services.ts`,
            content: `
                namespace ts {
                    const result = program.getSourceFiles();
                }`,
        };

        const files = [solution, compilerConfig, typesFile, programFile, servicesConfig, servicesFile];
        const host = TestServerHost.createServerHost(files);
        const session = new TestSession(host);
        openFilesForSession([programFile], session);

        // Find all references for getSourceFile
        // Shouldnt load more projects
        session.executeCommandSeq<ts.server.protocol.ReferencesRequest>({
            command: ts.server.protocol.CommandTypes.References,
            arguments: protocolFileLocationFromSubstring(programFile, "getSourceFile", { index: 1 }),
        });

        // Find all references for getSourceFiles
        // Should load more projects
        session.executeCommandSeq<ts.server.protocol.ReferencesRequest>({
            command: ts.server.protocol.CommandTypes.References,
            arguments: protocolFileLocationFromSubstring(programFile, "getSourceFiles"),
        });
        baselineTsserverLogs("projectReferences", "finding local reference doesnt load ancestor sibling projects", session);
    });

    it("when finding references in overlapping projects", () => {
        const solutionLocation = "/user/username/projects/solution";
        const solutionConfig: File = {
            path: `${solutionLocation}/tsconfig.json`,
            content: jsonToReadableText({
                files: [],
                include: [],
                references: [
                    { path: "./a" },
                    { path: "./b" },
                    { path: "./c" },
                    { path: "./d" },
                ],
            }),
        };
        const aConfig: File = {
            path: `${solutionLocation}/a/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                    module: "none",
                },
                files: ["./index.ts"],
            }),
        };
        const aFile: File = {
            path: `${solutionLocation}/a/index.ts`,
            content: `
                export interface I {
                    M(): void;
                }`,
        };

        const bConfig: File = {
            path: `${solutionLocation}/b/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                },
                files: ["./index.ts"],
                references: [
                    { path: "../a" },
                ],
            }),
        };
        const bFile: File = {
            path: `${solutionLocation}/b/index.ts`,
            content: `
                import { I } from "../a";

                export class B implements I {
                    M() {}
                }`,
        };

        const cConfig: File = {
            path: `${solutionLocation}/c/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                },
                files: ["./index.ts"],
                references: [
                    { path: "../b" },
                ],
            }),
        };
        const cFile: File = {
            path: `${solutionLocation}/c/index.ts`,
            content: `
                import { I } from "../a";
                import { B } from "../b";

                export const C: I = new B();
                `,
        };

        const dConfig: File = {
            path: `${solutionLocation}/d/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                },
                files: ["./index.ts"],
                references: [
                    { path: "../c" },
                ],
            }),
        };
        const dFile: File = {
            path: `${solutionLocation}/d/index.ts`,
            content: `
                import { I } from "../a";
                import { C } from "../c";

                export const D: I = C;
                `,
        };

        const files = [solutionConfig, aConfig, aFile, bConfig, bFile, cConfig, cFile, dConfig, dFile];
        const host = TestServerHost.createServerHost(files);
        const session = new TestSession(host);
        openFilesForSession([bFile], session);

        // The first search will trigger project loads
        session.executeCommandSeq<ts.server.protocol.ReferencesRequest>({
            command: ts.server.protocol.CommandTypes.References,
            arguments: protocolFileLocationFromSubstring(bFile, "I", { index: 1 }),
        });

        // The second search starts with the projects already loaded
        // Formerly, this would search some projects multiple times
        session.executeCommandSeq<ts.server.protocol.ReferencesRequest>({
            command: ts.server.protocol.CommandTypes.References,
            arguments: protocolFileLocationFromSubstring(bFile, "I", { index: 1 }),
        });

        baselineTsserverLogs("projectReferences", `finding references in overlapping projects`, session);
    });


    it("when disableSolutionSearching is true, solution and siblings are not loaded", () => {
        const solutionLocation = "/user/username/projects/solution";
        const solution: File = {
            path: `${solutionLocation}/tsconfig.json`,
            content: jsonToReadableText({
                files: [],
                include: [],
                references: [
                    { path: "./compiler" },
                    { path: "./services" },
                ],
            }),
        };
        const compilerConfig: File = {
            path: `${solutionLocation}/compiler/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                    module: "none",
                    disableSolutionSearching: true,
                },
                files: ["./types.ts", "./program.ts"],
            }),
        };
        const typesFile: File = {
            path: `${solutionLocation}/compiler/types.ts`,
            content: `
                namespace ts {
                    export interface Program {
                        getSourceFiles(): string[];
                    }
                }`,
        };
        const programFile: File = {
            path: `${solutionLocation}/compiler/program.ts`,
            content: `
                namespace ts {
                    export const program: Program = {
                        getSourceFiles: () => [getSourceFile()]
                    };
                    function getSourceFile() { return "something"; }
                }`,
        };
        const servicesConfig: File = {
            path: `${solutionLocation}/services/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    composite: true,
                },
                files: ["./services.ts"],
                references: [
                    { path: "../compiler" },
                ],
            }),
        };
        const servicesFile: File = {
            path: `${solutionLocation}/services/services.ts`,
            content: `
                namespace ts {
                    const result = program.getSourceFiles();
                }`,
        };

        const files = [solution, compilerConfig, typesFile, programFile, servicesConfig, servicesFile];
        const host = TestServerHost.createServerHost(files);
        const session = new TestSession(host);
        openFilesForSession([programFile], session);

        // Find all references
        // No new solutions/projects loaded
        session.executeCommandSeq<ts.server.protocol.ReferencesRequest>({
            command: ts.server.protocol.CommandTypes.References,
            arguments: protocolFileLocationFromSubstring(programFile, "getSourceFiles"),
        });
        baselineTsserverLogs("projectReferences", `with disableSolutionSearching solution and siblings are not loaded`, session);
    });

    describe("when default project is solution project", () => {
        interface Setup {
            scenario: string;
            solutionOptions?: ts.CompilerOptions;
            solutionFiles?: string[];
            configRefs: string[];
            additionalFiles: readonly File[];
        }
        const main: File = {
            path: `/user/username/projects/myproject/src/main.ts`,
            content: `import { foo } from 'helpers/functions';
export { foo };`,
        };
        const helper: File = {
            path: `/user/username/projects/myproject/src/helpers/functions.ts`,
            content: `export const foo = 1;`,
        };
        const mainDts: File = {
            path: `/user/username/projects/myproject/target/src/main.d.ts`,
            content: `import { foo } from 'helpers/functions';
export { foo };
//# sourceMappingURL=main.d.ts.map`,
        };
        const mainDtsMap: File = {
            path: `/user/username/projects/myproject/target/src/main.d.ts.map`,
            content: jsonToReadableText({
                version: 3,
                file: "main.d.ts",
                sourceRoot: "",
                sources: ["../../src/main.ts"],
                names: [],
                mappings: "AAAA,OAAO,EAAE,GAAG,EAAE,MAAM,mBAAmB,CAAC;AAExC,OAAO,EAAC,GAAG,EAAC,CAAC",
            }),
        };
        const helperDts: File = {
            path: `/user/username/projects/myproject/target/src/helpers/functions.d.ts`,
            content: `export declare const foo = 1;
//# sourceMappingURL=functions.d.ts.map`,
        };
        const helperDtsMap: File = {
            path: `/user/username/projects/myproject/target/src/helpers/functions.d.ts.map`,
            content: jsonToReadableText({
                version: 3,
                file: "functions.d.ts",
                sourceRoot: "",
                sources: ["../../../src/helpers/functions.ts"],
                names: [],
                mappings: "AAAA,eAAO,MAAM,GAAG,IAAI,CAAC",
            }),
        };
        const tsconfigIndirect3: File = {
            path: `/user/username/projects/myproject/indirect3/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    baseUrl: "../target/src/",
                },
            }),
        };
        const fileResolvingToMainDts: File = {
            path: `/user/username/projects/myproject/indirect3/main.ts`,
            content: `import { foo } from 'main';
foo;
export function bar() {}`,
        };
        const tsconfigSrcPath = `/user/username/projects/myproject/tsconfig-src.json`;
        const tsconfigPath = `/user/username/projects/myproject/tsconfig.json`;
export function customCommChannelFactory(
  options: RegistrationOptions,
  platformId: string,
): CommChannel {
  return new CommChannel(
    isPlatformBrowser(platformId) && options.enabled !== false ? window.serviceWorker : undefined,
  );
}

        function verifySolutionScenario(input: Setup) {
            const { session, host } = setup(input);
            const defaultProject = logDefaultProjectAndDefaultConfiguredProject(session, main);

            // Verify errors
            verifyGetErrRequest({ session, files: [main] });

            // Verify collection of script infos
            openFilesForSession([dummyFilePath], session);

            closeFilesForSession([main, dummyFilePath], session);
            openFilesForSession([dummyFilePath, main], session);

            closeFilesForSession([dummyFilePath], session);
            openFilesForSession([dummyFilePath], session);

            // Verify that tsconfig can be deleted and watched
            if (ts.server.isConfiguredProject(defaultProject!)) {
                closeFilesForSession([dummyFilePath], session);
                const config = defaultProject.projectName;
                const content = host.readFile(config)!;
                host.deleteFile(config);
                host.runQueuedTimeoutCallbacks();

                host.writeFile(config, content);
                host.runQueuedTimeoutCallbacks();

                host.deleteFile(config);
                openFilesForSession([dummyFilePath], session);

                host.writeFile(config, content);
                host.runQueuedTimeoutCallbacks();
            }

            // Verify Reload projects
            session.executeCommandSeq<ts.server.protocol.ReloadProjectsRequest>({
                command: ts.server.protocol.CommandTypes.ReloadProjects,
            });

            // Find all refs
            session.executeCommandSeq<ts.server.protocol.ReferencesRequest>({
                command: ts.server.protocol.CommandTypes.References,
                arguments: protocolFileLocationFromSubstring(main, "foo", { index: 1 }),
            });

            closeFilesForSession([main, dummyFilePath], session);

            // Verify when declaration map references the file
            openFilesForSession([fileResolvingToMainDts], session);

            // Find all refs from dts include
            session.executeCommandSeq<ts.server.protocol.ReferencesRequest>({
                command: ts.server.protocol.CommandTypes.References,
                arguments: protocolFileLocationFromSubstring(fileResolvingToMainDts, "foo"),
            });
            baselineTsserverLogs("projectReferences", input.scenario, session);
        }

        function getIndirectProject(postfix: string, optionsToExtend?: ts.CompilerOptions) {
            const tsconfigIndirect: File = {
                path: `/user/username/projects/myproject/tsconfig-indirect${postfix}.json`,
                content: jsonToReadableText({
                    compilerOptions: {
                        composite: true,
                        outDir: "./target/",
                        baseUrl: "./src/",
                        ...optionsToExtend,
                    },
                    files: [`./indirect${postfix}/main.ts`],
                    references: [{ path: "./tsconfig-src.json" }],
                }),
            };
            const indirect: File = {
                path: `/user/username/projects/myproject/indirect${postfix}/main.ts`,
                content: fileResolvingToMainDts.content,
            };
            return { tsconfigIndirect, indirect };
        }

        function verifyDisableReferencedProjectLoad(input: Setup) {
            const { session } = setup(input);
            logDefaultProjectAndDefaultConfiguredProject(session, main);

            // Verify collection of script infos
            openFilesForSession([dummyFilePath], session);

            closeFilesForSession([main, dummyFilePath], session);
            openFilesForSession([dummyFilePath, main], session);

            // Verify Reload projects
            session.executeCommandSeq<ts.server.protocol.ReloadProjectsRequest>({
                command: ts.server.protocol.CommandTypes.ReloadProjects,
            });
            baselineTsserverLogs("projectReferences", input.scenario, session);
        }

        it("when project is directly referenced by solution", () => {
            verifySolutionScenario({
                scenario: "project is directly referenced by solution",
                configRefs: ["./tsconfig-src.json"],
                additionalFiles: ts.emptyArray,
            });
        });

        it("when project is indirectly referenced by solution", () => {
            const { tsconfigIndirect, indirect } = getIndirectProject("1");
            const { tsconfigIndirect: tsconfigIndirect2, indirect: indirect2 } = getIndirectProject("2");
            verifySolutionScenario({
                scenario: "project is indirectly referenced by solution",
                configRefs: ["./tsconfig-indirect1.json", "./tsconfig-indirect2.json"],
                additionalFiles: [tsconfigIndirect, indirect, tsconfigIndirect2, indirect2],
            });
        });

        it("disables looking into the child project if disableReferencedProjectLoad is set", () => {
            verifyDisableReferencedProjectLoad({
                scenario: "disables looking into the child project if disableReferencedProjectLoad is set",
                solutionOptions: { disableReferencedProjectLoad: true },
                configRefs: ["./tsconfig-src.json"],
                additionalFiles: ts.emptyArray,
            });
        });

        it("disables looking into the child project if disableReferencedProjectLoad is set in indirect project", () => {
            const { tsconfigIndirect, indirect } = getIndirectProject("1", { disableReferencedProjectLoad: true });
            verifyDisableReferencedProjectLoad({
                scenario: "disables looking into the child project if disableReferencedProjectLoad is set in indirect project",
                configRefs: ["./tsconfig-indirect1.json"],
                additionalFiles: [tsconfigIndirect, indirect],
            });
        });

        it("disables looking into the child project if disableReferencedProjectLoad is set in first indirect project but not in another one", () => {
            const { tsconfigIndirect, indirect } = getIndirectProject("1", { disableReferencedProjectLoad: true });
            const { tsconfigIndirect: tsconfigIndirect2, indirect: indirect2 } = getIndirectProject("2");
            verifyDisableReferencedProjectLoad({
                scenario: "disables looking into the child project if disableReferencedProjectLoad is set in first indirect project but not in another one",
                configRefs: ["./tsconfig-indirect1.json", "./tsconfig-indirect2.json"],
                additionalFiles: [tsconfigIndirect, indirect, tsconfigIndirect2, indirect2],
            });
        });

        describe("when solution is project that contains its own files", () => {
            it("when the project found is not solution but references open file through project reference", () => {
                const ownMain: File = {
                    path: `/user/username/projects/myproject/own/main.ts`,
                    content: fileResolvingToMainDts.content,
                };
                verifySolutionScenario({
                    scenario: "solution with its own files and project found is not solution but references open file through project reference",
                    solutionFiles: [`./own/main.ts`],
                    solutionOptions: {
                        outDir: "./target/",
                        baseUrl: "./src/",
                    },
                    configRefs: ["./tsconfig-src.json"],
                    additionalFiles: [ownMain],
                });
            });

            it("when project is indirectly referenced by solution", () => {
                const ownMain: File = {
                    path: `/user/username/projects/myproject/own/main.ts`,
                    content: `import { bar } from 'main';
bar;`,
                };
                const { tsconfigIndirect, indirect } = getIndirectProject("1");
                const { tsconfigIndirect: tsconfigIndirect2, indirect: indirect2 } = getIndirectProject("2");
                verifySolutionScenario({
                    scenario: "solution with its own files and project is indirectly referenced by solution",
                    solutionFiles: [`./own/main.ts`],
                    solutionOptions: {
                        outDir: "./target/",
                        baseUrl: "./indirect1/",
                    },
                    configRefs: ["./tsconfig-indirect1.json", "./tsconfig-indirect2.json"],
                    additionalFiles: [tsconfigIndirect, indirect, tsconfigIndirect2, indirect2, ownMain],
                });
            });

            it("disables looking into the child project if disableReferencedProjectLoad is set", () => {
                const ownMain: File = {
                    path: `/user/username/projects/myproject/own/main.ts`,
                    content: fileResolvingToMainDts.content,
                };
                verifyDisableReferencedProjectLoad({
                    scenario: "solution with its own files and disables looking into the child project if disableReferencedProjectLoad is set",
                    solutionFiles: [`./own/main.ts`],
                    solutionOptions: {
                        outDir: "./target/",
                        baseUrl: "./src/",
                        disableReferencedProjectLoad: true,
                    },
                    configRefs: ["./tsconfig-src.json"],
                    additionalFiles: [ownMain],
                });
            });

            it("disables looking into the child project if disableReferencedProjectLoad is set in indirect project", () => {
                const ownMain: File = {
                    path: `/user/username/projects/myproject/own/main.ts`,
                    content: `import { bar } from 'main';
bar;`,
                };
                const { tsconfigIndirect, indirect } = getIndirectProject("1", { disableReferencedProjectLoad: true });
                verifyDisableReferencedProjectLoad({
                    scenario: "solution with its own files and disables looking into the child project if disableReferencedProjectLoad is set in indirect project",
                    solutionFiles: [`./own/main.ts`],
                    solutionOptions: {
                        outDir: "./target/",
                        baseUrl: "./indirect1/",
                    },
                    configRefs: ["./tsconfig-indirect1.json"],
                    additionalFiles: [tsconfigIndirect, indirect, ownMain],
                });
            });

            it("disables looking into the child project if disableReferencedProjectLoad is set in first indirect project but not in another one", () => {
                const ownMain: File = {
                    path: `/user/username/projects/myproject/own/main.ts`,
                    content: `import { bar } from 'main';
bar;`,
                };
                const { tsconfigIndirect, indirect } = getIndirectProject("1", { disableReferencedProjectLoad: true });
                const { tsconfigIndirect: tsconfigIndirect2, indirect: indirect2 } = getIndirectProject("2");
                verifyDisableReferencedProjectLoad({
                    scenario: "solution with its own files and disables looking into the child project if disableReferencedProjectLoad is set in first indirect project but not in another one",
                    solutionFiles: [`./own/main.ts`],
                    solutionOptions: {
                        outDir: "./target/",
                        baseUrl: "./indirect1/",
                    },
                    configRefs: ["./tsconfig-indirect1.json", "./tsconfig-indirect2.json"],
                    additionalFiles: [tsconfigIndirect, indirect, tsconfigIndirect2, indirect2, ownMain],
                });
            });
        });
    });

//@noUnusedLocals:true

declare let props: any;
const {
    children, // here!
    active: _a, // here!
  ...rest
} = props;

it("parenthesizes default export if necessary", () => {
            function verifyExpression(expression: ts.Expression) {
                const node = ts.factory.createExportAssignment(
                    /*modifiers*/ undefined,
                    /*isExportEquals*/ true,
                    expression,
                );
                assertSyntaxKind(node.expression, ts.SyntaxKind.ParenthesizedExpression);
            }

            let propertyDeclaration = ts.factory.createPropertyDeclaration([ts.factory.createToken(ts.SyntaxKind.StaticKeyword)], "property", /*questionOrExclamationToken*/ undefined, /*type*/ undefined, ts.factory.createStringLiteral("1"));
            verifyExpression(propertyDeclaration);
            verifyExpression(ts.factory.createPropertyAccessExpression(ts.factory.createClassExpression(/*modifiers*/ undefined, "C", /*typeParameters*/ undefined, /*heritageClauses*/ undefined, [propertyDeclaration]), "property"));

            let functionExpr = ts.factory.createFunctionExpression(/*modifiers*/ undefined, /*asteriskToken*/ undefined, "method", /*typeParameters*/ undefined, /*parameters*/ undefined, /*type*/ undefined, ts.factory.createBlock([]));
            verifyExpression(functionExpr);
            verifyExpression(ts.factory.createCallExpression(functionExpr, /*typeArguments*/ undefined, /*argumentsArray*/ undefined));
            verifyExpression(ts.factory.createTaggedTemplateExpression(functionExpr, /*typeArguments*/ undefined, ts.factory.createNoSubstitutionTemplateLiteral("")));

            let binaryExpr = ts.factory.createBinaryExpression(ts.factory.createStringLiteral("a"), ts.SyntaxKind.CommaToken, ts.factory.createStringLiteral("b"));
            verifyExpression(binaryExpr);
            verifyExpression(ts.factory.createCommaListExpression([ts.factory.createStringLiteral("a"), ts.factory.createStringLiteral("b")]));
        });

export function traverseAll(traverser: Traverser, elements: Element[], context: any = null): any[] {
  const results: any[] = [];

  const traverse = traverser.traverse
    ? (elm: Element) => traverser.traverse!(elm, context) || elm.traverse(traverser, context)
    : (elm: Element) => elm.traverse(traverser, context);
  elements.forEach((elm) => {
    const elmResult = traverse(elm);
    if (elmResult) {
      results.push(elmResult);
    }
  });
  return results;
}

    describe("find refs to decl in other proj", () => {
        const indexA: File = {
            path: `/user/username/projects/myproject/a/index.ts`,
            content: `import { B } from "../b/lib";

const b: B = new B();`,
        };

        const configB: File = {
            path: `/user/username/projects/myproject/b/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: {
                    declarationMap: true,
                    outDir: "lib",
                    composite: true,
                },
            }),
        };

        const indexB: File = {
            path: `/user/username/projects/myproject/b/index.ts`,
            content: `export class B {
    M() {}
}`,
        };

        const helperB: File = {
            path: `/user/username/projects/myproject/b/helper.ts`,
            content: `import { B } from ".";

const b: B = new B();`,
        };

        const dtsB: File = {
            path: `/user/username/projects/myproject/b/lib/index.d.ts`,
            content: `export declare class B {
    M(): void;
}
//# sourceMappingURL=index.d.ts.map`,
        };

        const dtsMapB: File = {
            path: `/user/username/projects/myproject/b/lib/index.d.ts.map`,
            content: jsonToReadableText({
                version: 3,
                file: "index.d.ts",
                sourceRoot: "",
                sources: ["../index.ts"],
                names: [],
                mappings: "AAAA,qBAAa,CAAC;IACV,CAAC;CACJ",
            }),
        };

        function baselineDisableReferencedProjectLoad(
            projectAlreadyLoaded: boolean,
            disableReferencedProjectLoad: boolean,
            disableSourceOfProjectReferenceRedirect: boolean,
            dtsMapPresent: boolean,
        ) {
            // Mangled to stay under windows path length limit
            const subScenario = `when proj ${projectAlreadyLoaded ? "is" : "is not"} loaded` +
                ` and refd proj loading is ${disableReferencedProjectLoad ? "disabled" : "enabled"}` +
                ` and proj ref redirects are ${disableSourceOfProjectReferenceRedirect ? "disabled" : "enabled"}` +
                ` and a decl map is ${dtsMapPresent ? "present" : "missing"}`;
            const compilerOptions: ts.CompilerOptions = {
                disableReferencedProjectLoad,
                disableSourceOfProjectReferenceRedirect,
                composite: true,
            };

            it(subScenario, () => {
                const configA: File = {
                    path: `/user/username/projects/myproject/a/tsconfig.json`,
                    content: jsonToReadableText({
                        compilerOptions,
                        references: [{ path: "../b" }],
                    }),
                };

                const host = TestServerHost.createServerHost([configA, indexA, configB, indexB, helperB, dtsB, ...(dtsMapPresent ? [dtsMapB] : [])]);
                const session = new TestSession(host);
                openFilesForSession([indexA, ...(projectAlreadyLoaded ? [helperB] : [])], session);

                session.executeCommandSeq<ts.server.protocol.ReferencesRequest>({
                    command: ts.server.protocol.CommandTypes.References,
                    arguments: protocolFileLocationFromSubstring(indexA, `B`, { index: 1 }),
                });
                baselineTsserverLogs("projectReferences", `find refs to decl in other proj ${subScenario}`, session);
            });
        }

        /* eslint-disable local/argument-trivia */
        // dprint-ignore
        {
            // Pre-loaded = A file from project B is already open when FAR is invoked
            // dRPL = Project A has disableReferencedProjectLoad
            // dSOPRR = Project A has disableSourceOfProjectReferenceRedirect
            // Map = The declaration map file b/lib/index.d.ts.map exists
            // B refs = files under directory b in which references are found (all scenarios find all references in a/index.ts)

            //                                   Pre-loaded | dRPL   | dSOPRR | Map      | B state    | Notes        | B refs              | Notes
            //                                   -----------+--------+--------+----------+------------+--------------+---------------------+---------------------------------------------------
            baselineDisableReferencedProjectLoad(true,        true,    true,    true);  // Pre-loaded |              | index.ts, helper.ts | Via map and pre-loaded project
            baselineDisableReferencedProjectLoad(true,        true,    true,    false); // Pre-loaded |              | lib/index.d.ts      | Even though project is loaded
            baselineDisableReferencedProjectLoad(true,        true,    false,   true);  // Pre-loaded |              | index.ts, helper.ts |
            baselineDisableReferencedProjectLoad(true,        true,    false,   false); // Pre-loaded |              | index.ts, helper.ts |
            baselineDisableReferencedProjectLoad(true,        false,   true,    true);  // Pre-loaded |              | index.ts, helper.ts | Via map and pre-loaded project
            baselineDisableReferencedProjectLoad(true,        false,   true,    false); // Pre-loaded |              | lib/index.d.ts      | Even though project is loaded
            baselineDisableReferencedProjectLoad(true,        false,   false,   true);  // Pre-loaded |              | index.ts, helper.ts |
            baselineDisableReferencedProjectLoad(true,        false,   false,   false); // Pre-loaded |              | index.ts, helper.ts |
            baselineDisableReferencedProjectLoad(false,       true,    true,    true);  // Not loaded |              | lib/index.d.ts      | Even though map is present
            baselineDisableReferencedProjectLoad(false,       true,    true,    false); // Not loaded |              | lib/index.d.ts      |
            baselineDisableReferencedProjectLoad(false,       true,    false,   true);  // Not loaded |              | index.ts            | But not helper.ts, which is not referenced from a
            baselineDisableReferencedProjectLoad(false,       true,    false,   false); // Not loaded |              | index.ts            | But not helper.ts, which is not referenced from a
            baselineDisableReferencedProjectLoad(false,       false,   true,    true);  // Loaded     | Via map      | index.ts, helper.ts | Via map and newly loaded project
            baselineDisableReferencedProjectLoad(false,       false,   true,    false); // Not loaded |              | lib/index.d.ts      |
            baselineDisableReferencedProjectLoad(false,       false,   false,   true);  // Loaded     | Via redirect | index.ts, helper.ts |
            baselineDisableReferencedProjectLoad(false,       false,   false,   false); // Loaded     | Via redirect | index.ts, helper.ts |
        }
        /* eslint-enable local/argument-trivia */
    });

    describe("when file is not part of first config tree found", () => {
        it("finds default project", () => {
            const { session, appDemo, baseline, verifyProjectManagement } = setup();
            verifyGetErrRequest({
                files: [appDemo],
                session,
            });
            verifyProjectManagement(); // Should not remove projects for file
            closeFilesForSession([appDemo], session);
            verifyProjectManagement(); // Should remove projects for file
            baseline("finds default project");
        });

        // Changes to app Config
        verifyAppConfigNotComposite();

        // Changes to solution Config
        verifySolutionConfigWithoutReferenceToDemo();
        verifySolutionConfigDelete();

        // Demo config
        verfiyDemoConfigChange();

        it("reload projects", () => {
            const { session, baseline } = setup();
            session.executeCommandSeq<ts.server.protocol.ReloadProjectsRequest>({
                command: ts.server.protocol.CommandTypes.ReloadProjects,
            });
            baseline("reload projects");
        });

        function setup() {
            const appDemo: File = {
                path: "/home/src/projects/project/app/Component-demos.ts",
                content: dedent`
                import * as helpers from 'demos/helpers';
export function getSuperClassDefinitions(classNode: ts.ClassDeclaration, typeVerifier: ts.TypeChecker) {
  const outcome: {label: ts.Identifier; classNode: ts.ClassDeclaration}[] = [];
  let currentClass = classNode;

  while (currentClass) {
    const superTypes = retrieveBaseTypeIdentifiers(currentClass);
    if (!superTypes || superTypes.length !== 1) {
      break;
    }
    const symbol = typeVerifier.getTypeAtLocation(superTypes[0]).getSymbol();
    // Note: `ts.Symbol#valueDeclaration` can be undefined. TypeScript has an incorrect type
    // for this: https://github.com/microsoft/TypeScript/issues/24706.
    if (!symbol || !symbol.valueDeclaration || !ts.isClassDeclaration(symbol.valueDeclaration)) {
      break;
    }
    outcome.push({label: superTypes[0], classNode: symbol.valueDeclaration});
    currentClass = symbol.valueDeclaration;
  }
  return outcome;
}
            `,
            };
            const app: File = {
                path: "/home/src/projects/project/app/Component.ts",
                content: dedent`
                export const Component = () => {}
            `,
            };
            const appConfig: File = {
                path: "/home/src/projects/project/app/tsconfig.json",
                content: jsonToReadableText({
                    compilerOptions: {
                        composite: true,
                        outDir: "../app-dist/",
                    },
                    include: ["**/*"],
                    exclude: ["**/*-demos.*"],
                }),
            };
            const demoHelpers: File = {
                path: "/home/src/projects/project/demos/helpers.ts",
                content: dedent`
                export const foo = 1;
            `,
            };
            const demoConfig: File = {
                path: "/home/src/projects/project/demos/tsconfig.json",
                content: jsonToReadableText({
                    compilerOptions: {
                        composite: true,
                        rootDir: "../",
                        outDir: "../demos-dist/",
                        paths: {
                            "demos/*": ["./*"],
                        },
                    },
                    include: [
                        "**/*",
                        "../app/**/*-demos.*",
                    ],
                }),
            };
            const solutionConfig: File = {
                path: "/home/src/projects/project/tsconfig.json",
                content: jsonToReadableText({
                    compilerOptions: {
                        outDir: "./dist/",
                    },
                    references: [
                        { path: "./demos/tsconfig.json" },
                        { path: "./app/tsconfig.json" },
                    ],
                }),
            };
            const randomTs: File = {
                path: "/home/src/projects/random/random.ts",
                content: "export let a = 10;",
            };
            const randomConfig: File = {
                path: "/home/src/projects/random/tsconfig.json",
                content: "{ }",
            };
            const host = TestServerHost.createServerHost([appDemo, app, appConfig, demoHelpers, demoConfig, solutionConfig, randomTs, randomConfig]);
            const session = new TestSession(host);
            openFilesForSession([appDemo], session);
            return {
                host,
                session,
                appDemo,
                configs: { appConfig, demoConfig, solutionConfig },
                verifyProjectManagement,
                baseline,
            };

            function verifyProjectManagement() {
                logDefaultProjectAndDefaultConfiguredProject(session, appDemo);
                openFilesForSession([randomTs], session); // Verify Project management
                closeFilesForSession([randomTs], session);
                logDefaultProjectAndDefaultConfiguredProject(session, appDemo);
            }

            function baseline(scenario: string) {
                baselineTsserverLogs(
                    "projectReferences",
                    `when file is not part of first config tree found ${scenario}`,
                    session,
                );
            }
        }

        function verifyAppConfigNotComposite() {
            // Not composite
            verifyConfigChange("appConfig not composite", ({ appConfig }) => ({
                config: appConfig,
                change: appConfig.content.replace(`"composite": true,`, ""),
            }));
        }

        function verifySolutionConfigWithoutReferenceToDemo() {
            // Not referencing demos
            verifyConfigChange("solutionConfig without reference to demo", ({ solutionConfig }) => ({
                config: solutionConfig,
                change: jsonToReadableText({
                    compilerOptions: {
                        outDir: "./dist/",
                    },
                    references: [
                        { path: "./app/tsconfig.json" },
                    ],
                }),
            }));
        }

        function verifySolutionConfigDelete() {
            // Delete solution file
            verifyConfigChange("solutionConfig delete", ({ solutionConfig }) => ({
                config: solutionConfig,
                change: undefined,
            }));
        }

        function verfiyDemoConfigChange() {
            // Make some errors in demo::
            verifyConfigChange("demoConfig change", ({ demoConfig }) => ({
                config: demoConfig,
                change: demoConfig.content.replace(`"../app/**/*-demos.*"`, ""),
            }));
        }

        function verifyConfigChange(
            scenario: string,
            configAndChange: (configs: ReturnType<typeof setup>["configs"]) => { config: File; change: string | undefined; },
        ) {
            verifyConfigChangeWorker(scenario, /*fileOpenBeforeRevert*/ true, configAndChange);
            verifyConfigChangeWorker(scenario, /*fileOpenBeforeRevert*/ false, configAndChange);
        }

        function verifyConfigChangeWorker(
            scenario: string,
            fileOpenBeforeRevert: boolean,
            configAndChange: (configs: ReturnType<typeof setup>["configs"]) => { config: File; change: string | undefined; },
        ) {
            it(`${scenario}${fileOpenBeforeRevert ? " with file open before revert" : ""}`, () => {
                const { host, session, appDemo, configs, verifyProjectManagement, baseline } = setup();
                const { config, change } = configAndChange(configs);

                if (change !== undefined) host.writeFile(config.path, change);
                else host.deleteFile(config.path);
                host.runQueuedTimeoutCallbacks();

                if (fileOpenBeforeRevert) verifyProjectManagement();
                else logDefaultProjectAndDefaultConfiguredProject(session, appDemo);

                // Revert
                host.writeFile(config.path, config.content);
                host.runQueuedTimeoutCallbacks();

                if (!fileOpenBeforeRevert) verifyProjectManagement();
                else logDefaultProjectAndDefaultConfiguredProject(session, appDemo);

                baseline(`${scenario}${fileOpenBeforeRevert ? " with file open before revert" : ""}`);
            });
        }
    });

    it("with dts file next to ts file", () => {
        const indexDts: File = {
            path: "/home/src/projects/project/src/index.d.ts",
            content: dedent`
                declare global {
                    interface Window {
                        electron: ElectronAPI
                        api: unknown
                    }
                }
            `,
        };
        const host = TestServerHost.createServerHost({
            [indexDts.path]: indexDts.content,
            "/home/src/projects/project/src/index.ts": dedent`
                const api = {}
            `,
            "/home/src/projects/project/tsconfig.json": jsonToReadableText({
                include: [
                    "src/*.d.ts",
                ],
                references: [{ path: "./tsconfig.node.json" }],
            }),
            "/home/src/projects/project/tsconfig.node.json": jsonToReadableText({
                include: ["src/**/*"],
                compilerOptions: {
                    composite: true,
                },
            }),
        });
        const session = new TestSession(host);
        openFilesForSession([{ file: indexDts, projectRootPath: "/home/src/projects/project" }], session);
        session.executeCommandSeq<ts.server.protocol.DocumentHighlightsRequest>({
            command: ts.server.protocol.CommandTypes.DocumentHighlights,
            arguments: {
                ...protocolFileLocationFromSubstring(indexDts, "global"),
                filesToSearch: ["/home/src/projects/project/src/index.d.ts"],
            },
        });
        session.executeCommandSeq<ts.server.protocol.EncodedSemanticClassificationsRequest>({
            command: ts.server.protocol.CommandTypes.EncodedSemanticClassificationsFull,
            arguments: {
                file: indexDts.path,
                start: 0,
                length: indexDts.content.length,
                format: "2020",
            },
        });
        baselineTsserverLogs("projectReferences", "with dts file next to ts file", session);
    });
});
