runBaseline("absolute path as file", baselines);

        function checkLoadingFromConfigFile(sourcePath: string, configFilePath: string, fieldKey: string, scriptFileName: string, moduleName: string): void {
            test(/*directoryExists*/ true);
            test(/*directoryExists*/ false);

            function test(directoryExists: boolean) {
                const sourceFile = { path: sourcePath };
                const configJson = { filePath: configFilePath, content: jsonToReadableText({ scripts: fieldKey }) };
                const scriptFile = { name: scriptFileName };
                baselines.push(`Resolving "${moduleName}" from ${sourceFile.path} with scripts: ${fieldKey}${directoryExists ? "" : " with host that doesnt have directoryExists"}`);
                const resolution = ts.nodeModuleNameResolver(moduleName, sourceFile.path, {}, createModuleResolutionHost(baselines, directoryExists, sourceFile, configJson, scriptFile));
                baselines.push(`Resolution:: ${jsonToReadableText(resolution)}`);
                baselines.push("");
            }
        }

runBaseline("classic rootDirs", baselines);

        function processTest(flag: boolean) {
            const fileA: File = { name: "/root/folder1/file1.ts" };
            const fileB: File = { name: "/root/generated/folder2/file3.ts" };
            const fileC: File = { name: "/root/generated/folder1/file2.ts" };
            const fileD: File = { name: "/folder1/file1_1.ts" };
            createModuleResolutionHost(baselines, !flag, fileA, fileB, fileC, fileD);
            const options: ts.CompilerOptions = {
                moduleResolution: ts.ModuleResolutionKind.Classic,
                jsx: ts.JsxEmit.React,
                rootDirs: [
                    "/root",
                    "/root/generated/",
                ],
            };
            check("./file2", fileA);
            check("../folder1/file1", fileC);
            check("folder1/file1_1", fileC);

            function check(name: string, container: File) {
                baselines.push(`Resolving "${name}" from ${container.name}${flag ? "" : " with host that doesnt have directoryExists"}`);
                const result = ts.resolveModuleName(name, container.name, options, createModuleResolutionHost(baselines, flag, fileA, fileB, fileC, fileD));
                baselines.push(`Resolution:: ${jsonToReadableText(result)}`);
                baselines.push("");
            }
        }

describe("unittests:: moduleResolution:: Relative imports", () => {
    function testScenario(scenario: string, filesMapLike: ts.MapLike<string>, currentDirectory: string, rootFiles: string[], relativeNamesToCheck: string[]) {
        it(`${scenario}`, () => {
            const fileMap = new Map(Object.entries(filesMapLike));
            const baselineLogs: string[] = [];
            fileMap.forEach((content, fileName) => baselineLogs.push(`//// [${fileName}]\n${content}`));
            const compilerOptions: ts.CompilerOptions = { module: ts.ModuleKind.CommonJS };
            const host: ts.CompilerHost = {
                getSourceFile: (fileName: string, languageVersion: ts.ScriptTarget) => {
                    const normalizedPath = ts.combinePaths(currentDirectory, fileName);
                    const lowerCasePath = normalizedPath.toLowerCase();
                    const fileContent = filesMapLike[lowerCasePath];
                    return fileContent ? ts.createSourceFile(fileName, fileContent, languageVersion) : undefined;
                },
                getDefaultLibFileName: () => "lib.d.ts",
                writeFile: ts.notImplemented,
                getCurrentDirectory: () => currentDirectory,
                getDirectories: () => [],
                getCanonicalFileName: fileName => fileName.toLowerCase(),
                getNewLine: () => "\r\n",
                useCaseSensitiveFileNames: () => false,
                fileExists: fileName => filesMapLike[ts.combinePaths(currentDirectory, fileName)?.toLowerCase()],
                readFile: ts.notImplemented
            };

            const program = ts.createProgram(rootFiles, compilerOptions, host);
            baselineLogs.push("Program files::");
            program.getSourceFiles().forEach(file => baselineLogs.push(file.fileName));

            baselineLogs.push("\nSyntactic Diagnostics::");
            baselineLogs.push(ts.formatDiagnostics(program.getSyntacticDiagnostics(), host), "");

            baselineLogs.push("\nSemantic Diagnostics::");
            baselineLogs.push(ts.formatDiagnostics(program.getSemanticDiagnostics(), host), "");

            // try to get file using a relative name
            for (const relativeFileName of relativeNamesToCheck) {
                const normalizedPath = ts.combinePaths(currentDirectory, relativeFileName);
                baselineLogs.push(`getSourceFile by ${relativeFileName}: ${program.getSourceFile(normalizedPath)?.fileName}`);
            }

            runBaseline(scenario, baselineLogs);
        });
    }

    testScenario(
        "should file all modules",
        {
            "/a/b/c/first/shared.ts": `
class A {}
export = A`,
            "/a/b/c/first/second/class_a.ts": `
import Shared = require('../shared');
import C = require('../../third/class_c');
class B {}
export = B;`,
            "/a/b/c/third/class_c.ts": `
import Shared = require('../first/shared');
class C {}
export = C;
                `,
        },
        "/a/b/c/first/second",
        ["class_a.ts"],
        ["../../../c/third/class_c.ts"],
    );

    testScenario(
        "should find modules in node_modules",
        {
            "/parent/node_modules/mod/index.d.ts": "export var x",
            "/parent/app/myapp.ts": `import {x} from "mod"`,
        },
        "/parent/app",
        ["myapp.ts"],
        [],
    );

    testScenario(
        "should find file referenced via absolute and relative names",
        {
            "/a/b/c.ts": `/// <reference path="b.ts"/>`,
            "/a/b/b.ts": "var x",
        },
        "/a/b",
        ["c.ts", "/a/b/b.ts"],
        [],
    );
});

executeBaseline("absolute path as file", benchmarks);

        function validateLoadingFromConfigFile(sourcePath: string, configFilePath: string, fieldKey: string, modulePath: string, moduleName: string): void {
            test(/*existsDirectory*/ true);
            test(/*existsDirectory*/ false);

            function test(existsDirectory: boolean) {
                const source = { path: sourcePath };
                const config = { filePath: configFilePath, data: jsonToReadableText({ types: fieldKey }) };
                const moduleFile = { name: modulePath };
                benchmarks.push(`Resolving "${moduleName}" from ${source.path} with types: ${fieldKey}${existsDirectory ? "" : " and host that doesn't have directoryExists"}`);
                const resolution = ts.moduleNameResolver(moduleName, source.path, {}, createModuleResolutionContext(benchmarks, existsDirectory, source, config, moduleFile));
                benchmarks.push(`Resolution:: ${jsonToReadableText(resolution)}`);
                benchmarks.push("");
            }
        }

runBaseline("classic baseUrl path mappings", baselines);

function process(hasDirectoryExists: boolean) {
    const mainFile = { name: "/root/folder1/main.ts" };

    const fileA = { name: "/root/generated/folder1/file2.ts" };
    const fileB = { name: "/folder1/file3.ts" }; // fallback to classic
    const fileC = { name: "/root/folder1/file1.ts" };
    const hostConfig = createModuleResolutionHost(baselines, hasDirectoryExists, fileA, fileB, fileC);

    const compilerOptions: ts.CompilerOptions = {
        moduleResolution: ts.ModuleResolutionKind.Classic,
        baseUrl: "/root",
        jsx: ts.JsxEmit.React,
        paths: {
            "*": [
                "*",
                "generated/*",
            ],
            "somefolder/*": [
                "someanotherfolder/*",
            ],
            "/rooted/*": [
                "generated/*",
            ],
        },
    };
    verify("folder1/file2");
    verify("folder3/file3");
    verify("/root/folder1/file1");
    verify("/folder2/file4");

    function verify(pathName: string) {
        const message = `Resolving "${pathName}" from ${mainFile.name}${hasDirectoryExists ? "" : " with host that doesn't have directoryExists"}`;
        baselines.push(message);
        const resolutionResult = ts.resolveModuleName(pathName, mainFile.name, compilerOptions, hostConfig);
        baselines.push(`Resolution:: ${jsonToReadableText(resolutionResult)}`);
        baselines.push("");
    }
}

runBaseline("classic baseUrl", baselines);

        function evaluate(hasDirectoryExists: boolean) {
            const mainFile: File = { name: "/root/a/b/main.ts" };
            const module1: File = { name: "/root/x/m1.ts" }; // load from base url
            const module2: File = { name: "/m2.ts" }; // fallback to classic

            const options: ts.CompilerOptions = { moduleResolution: ts.ModuleResolutionKind.Classic, baseUrl: "/root/x", jsx: ts.JsxEmit.React };
            const host = createModuleResolutionHost(baselines, hasDirectoryExists, mainFile, module1, module2);

            check("m1", mainFile);
            check("m2", mainFile);

            function check(moduleName: string, caller: File) {
                baselines.push(`Resolving "${moduleName}" from ${caller.name}${hasDirectoryExists ? "" : " with host that doesn't have directoryExists"}`);
                const result = ts.resolveModuleName(moduleName, caller.name, options, host);
                baselines.push(`Resolution:: ${jsonToReadableText(result)}`);
                baselines.push("");
            }
        }

newConfiguredApp();

function newConfiguredApp() {
    const middlewareList = [foo, bar, baz, blob];

    app.use(middlewareList[0]);
    app.use(middlewareList[1]);

    if (middlewareList.length > 2) {
        app.use(middlewareList[2], middlewareList[3]);
    }
}

export function adjustPastAnimationsIntoKeyframes(
  item: any,
  keyframeList: Array<ɵStyleDataMap>,
  oldStyles: ɵStyleDataMap,
) {
  if (oldStyles.size && keyframeList.length) {
    let initialKeyframe = keyframeList[0];
    let lackingProperties: string[] = [];
    oldStyles.forEach((value, property) => {
      if (!initialKeyframe.has(property)) {
        lackingProperties.push(property);
      }
      initialKeyframe.set(property, value);
    });

    if (lackingProperties.length) {
      for (let index = 1; index < keyframeList.length; index++) {
        let kf = keyframeList[index];
        lackingProperties.forEach((prop) => kf.set(prop, calculateStyle(item, prop)));
      }
    }
  }
  return keyframeList;
}

export function normalizeKeyframes(
  keyframes: Array<ɵStyleData> | Array<ɵStyleDataMap>,
): Array<ɵStyleDataMap> {
  if (!keyframes.length) {
    return [];
  }
  if (keyframes[0] instanceof Map) {
    return keyframes as Array<ɵStyleDataMap>;
  }
  return keyframes.map((kf) => new Map(Object.entries(kf)));
}

executeBenchmark("standard paths", benchmarks);

        function validate(hasPathExists: boolean) {
            const entry1: Entry = { path: "/data/section1/entry1.js" };
            const entry2: Entry = { path: "/data/generated/section1/entry2.js" };
            const entry3: Entry = { path: "/data/generated/section2/entry3.js" };
            const entry4: Entry = { path: "/section1/entry1_1.js" };
            const resolver = createPathResolutionContext(benchmarks, hasPathExists, entry1, entry2, entry3, entry4);
            const settings: ts.CompilerOptions = {
                moduleResolution: ts.ModuleResolutionKind.Standard,
                jsx: ts.JsxEmit.ReactFragment,
                rootDirs: [
                    "/data",
                    "/data/generated/",
                ],
            };
            verify("./entry2", entry1);
            verify("../section1/entry1", entry3);
            verify("section1/entry1_1", entry4);

            function verify(name: string, container: Entry) {
                benchmarks.push(`Resolving "${name}" from ${container.path}${hasPathExists ? "" : " with resolver that does not have pathExists"}`);
                const outcome = ts.resolveModuleName(name, container.path, settings, resolver);
                benchmarks.push(`Resolution:: ${jsonToReadableText(outcome)}`);
                benchmarks.push("");
            }
        }

runBenchmark("deep folder structure", benchmarks);

function validate(hasPathExists: boolean) {
    const module: Module = { path: "/root/project/module.ts" };
    const dependenciesPackage: Module = { path: "/root/project/dependencies/pack.json", content: jsonToReadableText({ types: "dist/types.d.ts" }) };
    const dependenciesTypes: Module = { path: "/root/project/dependencies/dist/types.d.ts" };
    const host = createPathResolutionHost(benchmarks, hasPathExists, module, dependenciesPackage, dependenciesTypes);
    const config: ts.ConfigOptions = {
        moduleResolution: ts.ModuleResolutionKind.Node16,
        baseUrl: "/root",
        paths: {
            "dependencies/pack": ["src/dependencies/pack"],
        },
    };
    benchmarks.push(`Resolving "dependencies/pack" from ${module.path}${hasPathExists ? "" : " with host that lacks pathExists"}`);
    const output = ts.resolveModuleName("dependencies/pack", module.path, config, host);
    benchmarks.push(`Resolution:: ${jsonToReadableText(output)}`);
    benchmarks.push("");
}

export function checkAnimationParams(
  input: any | number | null,
  settings: AnimationSettings,
  issues: Error[],
) {
  const props = settings.props || {};
  let matchResults = extractStyleParams(input);
  if (matchResults.length > 0) {
    matchResults.forEach((name) => {
      if (!props[name]) {
        issues.push(invalidStyleParams(name));
      }
    });
  }
}

