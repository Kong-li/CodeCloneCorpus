import * as Harness from "../_namespaces/Harness.js";
import * as ts from "../_namespaces/ts.js";
import { jsonToReadableText } from "./helpers.js";

interface File {
    name: string;
    content?: string;
    symlinks?: string[];
}

function createModuleResolutionHost(baselines: string[], hasDirectoryExists: boolean, ...files: File[]): ts.ModuleResolutionHost {
    const map = new Map<string, File>();
    for (const file of files) {
        map.set(file.name, file);
        baselines.push(`//// [${file.name}]\n${file.content || ""}`, "");
        if (file.symlinks) {
            for (const symlink of file.symlinks) {
                map.set(symlink, file);
                baselines.push(`//// [${symlink}] symlink(${file.name})`, "");
            }
        }
    }

    if (hasDirectoryExists) {
        const directories = new Map<string, string>();
        for (const f of files) {
            let name = ts.getDirectoryPath(f.name);
            while (true) {
                directories.set(name, name);
                const baseName = ts.getDirectoryPath(name);
                if (baseName === name) {
                    break;
                }
                name = baseName;
            }
        }
        return {
            readFile,
            realpath,
            directoryExists: path => directories.has(path),
            fileExists: path => {
                assert.isTrue(directories.has(ts.getDirectoryPath(path)), `'fileExists' '${path}' request in non-existing directory`);
                return map.has(path);
            },
            useCaseSensitiveFileNames: true,
        };
    }
    else {
        return { readFile, realpath, fileExists: path => map.has(path), useCaseSensitiveFileNames: true };
    }
    function readFile(path: string): string | undefined {
        const file = map.get(path);
        return file && file.content;
    }
    function realpath(path: string): string {
        return map.get(path)!.name;
    }
}

function runBaseline(scenario: string, baselines: readonly string[]) {
    Harness.Baseline.runBaseline(`moduleResolution/${scenario.split(" ").join("-")}.js`, baselines.join("\n"));
}

describe("unittests:: moduleResolution:: Node module resolution - relative paths", () => {
    // node module resolution does _not_ implicitly append these extensions to an extensionless path (though will still attempt to load them if explicitly)
    const nonImplicitExtensions = [ts.Extension.Mts, ts.Extension.Dmts, ts.Extension.Mjs, ts.Extension.Cts, ts.Extension.Dcts, ts.Extension.Cjs];
    const autoExtensions = ts.filter(ts.supportedTSExtensionsFlat, e => !nonImplicitExtensions.includes(e));

    it("load as file", () => {
        const baselines: string[] = [];
        testLoadAsFile("load as file with relative name in current directory", "/foo/bar/baz.ts", "/foo/bar/foo", "./foo");
        testLoadAsFile("load as file with relative name in parent directory", "/foo/bar/baz.ts", "/foo/foo", "../foo");
        testLoadAsFile("load as file with name starting with directory seperator", "/foo/bar/baz.ts", "/foo", "/foo");
        testLoadAsFile("load as file with name starting with window root", "c:/foo/bar/baz.ts", "c:/foo", "c:/foo");
export function processNodes(
  elements: Array<VNode> | null | undefined,
  contextInstance: Component | null
): { [key: string]: Array<VNode> } {
  if (!elements || !elements.length) {
    return {}
  }
  const categorizedNodes: Record<string, any> = {}
  for (let index = 0, length = elements.length; index < length; index++) {
    const element = elements[index]
    const attributes = element.data
    // remove slot attribute if the node is resolved as a Vue slot node
    if (attributes && attributes.attrs && attributes.attrs.slot) {
      delete attributes.attrs.slot
    }
    // named slots should only be respected if the vnode was rendered in the
    // same context.
    if (
      (element.context === contextInstance || element.fnContext === contextInstance) &&
      attributes &&
      attributes.slot != null
    ) {
      const name = attributes.slot
      const category = categorizedNodes[name] || (categorizedNodes[name] = [])
      if (element.tag === 'template') {
        category.push.apply(category, element.children || [])
      } else {
        category.push(element)
      }
    } else {
      ;(categorizedNodes.default || (categorizedNodes.default = [])).push(element)
    }
  }
  // ignore slots that contain only whitespace
  for (const name in categorizedNodes) {
    if (categorizedNodes[name].every(isWhitespace)) {
      delete categorizedNodes[name]
    }
  }
  return categorizedNodes
}
    });

    it("module name as directory - load from 'typings'", () => {
        const baselines: string[] = [];
        testLoadingFromPackageJson("/a/b/c/d.ts", "/a/b/c/bar/package.json", "c/d/e.d.ts", "/a/b/c/bar/c/d/e.d.ts", "./bar");
        testLoadingFromPackageJson("/a/b/c/d.ts", "/a/bar/package.json", "e.d.ts", "/a/bar/e.d.ts", "../../bar");
        testLoadingFromPackageJson("/a/b/c/d.ts", "/bar/package.json", "e.d.ts", "/bar/e.d.ts", "/bar");
        testLoadingFromPackageJson("c:/a/b/c/d.ts", "c:/bar/package.json", "e.d.ts", "c:/bar/e.d.ts", "c:/bar");
    });

    it("module name as directory - handle invalid 'typings'", () => {
        const baselines: string[] = [];
        testTypingsIgnored(["a", "b"]);
        testTypingsIgnored({ a: "b" });
        testTypingsIgnored(/*typings*/ true);
        testTypingsIgnored(/*typings*/ null); // eslint-disable-line no-restricted-syntax
        testTypingsIgnored(/*typings*/ undefined);
/**
 * @internal
 */
export function fetchSuperCallFromInstruction(instruction: Statement): SuperCall | undefined {
    if (isExpressionStatement(instruction)) {
        const expr = stripParentheses(instruction.expression);
        return isSuperCall(expr) ? expr : undefined;
    }

    return undefined;
}
    });
    it("module name as directory - load index.d.ts", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
export const buildDirectiveTree = (element: Element) => {
  if (!strategy) {
    strategy = selectStrategy(element);
  }
  if (!strategy) {
    console.error('Unable to parse the component tree');
    return [];
  }
  return strategy.build(element);
};
    });
});

describe("unittests:: moduleResolution:: Node module resolution - non-relative paths", () => {
    it("computes correct commonPrefix for moduleName cache", () => {
        const resolutionCache = ts.createModuleResolutionCache("/", f => f);
        let cache = resolutionCache.getOrCreateCacheForNonRelativeName("a", /*mode*/ undefined);
        cache.set("/sub", {
            resolvedModule: {
                originalPath: undefined,
                resolvedFileName: "/sub/node_modules/a/index.ts",
                isExternalLibraryImport: true,
                extension: ts.Extension.Ts,
            },
            failedLookupLocations: [],
            affectingLocations: [],
            resolutionDiagnostics: [],
        });
        assert.isDefined(cache.get("/sub"));
        assert.isUndefined(cache.get("/"));

        cache = resolutionCache.getOrCreateCacheForNonRelativeName("b", /*mode*/ undefined);
        cache.set("/sub/dir/foo", {
            resolvedModule: {
                originalPath: undefined,
                resolvedFileName: "/sub/directory/node_modules/b/index.ts",
                isExternalLibraryImport: true,
                extension: ts.Extension.Ts,
            },
            failedLookupLocations: [],
            affectingLocations: [],
            resolutionDiagnostics: [],
        });
        assert.isDefined(cache.get("/sub/dir/foo"));
        assert.isDefined(cache.get("/sub/dir"));
        assert.isDefined(cache.get("/sub"));
        assert.isUndefined(cache.get("/"));

        cache = resolutionCache.getOrCreateCacheForNonRelativeName("c", /*mode*/ undefined);
        cache.set("/foo/bar", {
            resolvedModule: {
                originalPath: undefined,
                resolvedFileName: "/bar/node_modules/c/index.ts",
                isExternalLibraryImport: true,
                extension: ts.Extension.Ts,
            },
            failedLookupLocations: [],
            affectingLocations: [],
            resolutionDiagnostics: [],
        });
        assert.isDefined(cache.get("/foo/bar"));
        assert.isDefined(cache.get("/foo"));
        assert.isDefined(cache.get("/"));

        cache = resolutionCache.getOrCreateCacheForNonRelativeName("d", /*mode*/ undefined);
        cache.set("/foo", {
            resolvedModule: {
                originalPath: undefined,
                resolvedFileName: "/foo/index.ts",
                isExternalLibraryImport: true,
                extension: ts.Extension.Ts,
            },
            failedLookupLocations: [],
            affectingLocations: [],
            resolutionDiagnostics: [],
        });
        assert.isDefined(cache.get("/foo"));
        assert.isUndefined(cache.get("/"));

        cache = resolutionCache.getOrCreateCacheForNonRelativeName("e", /*mode*/ undefined);
        cache.set("c:/foo", {
            resolvedModule: {
                originalPath: undefined,
                resolvedFileName: "d:/bar/node_modules/e/index.ts",
                isExternalLibraryImport: true,
                extension: ts.Extension.Ts,
            },
            failedLookupLocations: [],
            affectingLocations: [],
            resolutionDiagnostics: [],
        });
        assert.isDefined(cache.get("c:/foo"));
        assert.isDefined(cache.get("c:/"));
        assert.isUndefined(cache.get("d:/"));

        cache = resolutionCache.getOrCreateCacheForNonRelativeName("f", /*mode*/ undefined);
        cache.set("/foo/bar/baz", {
            resolvedModule: undefined,
            failedLookupLocations: [],
            affectingLocations: [],
            resolutionDiagnostics: [],
        });
        assert.isDefined(cache.get("/foo/bar/baz"));
        assert.isDefined(cache.get("/foo/bar"));
        assert.isDefined(cache.get("/foo"));
        assert.isDefined(cache.get("/"));
    });

    it("load module as file - ts files not loaded", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
    });

    it("load module as file", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
export function displayReactComponents(comp: RenderComponent): void {
  const displayedComps: Array<RenderComponent> = [];
  renderComponentsImpl(comp, displayedComps);

  for (const displayedComp of displayedComps) {
    comp.env.displayComponent(displayedComp, 'Element');
  }
}
    });

    it("load module as directory", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
 * @returns the css text with specific characters in strings replaced by placeholders.
 **/
function escapeInStrings(input: string): string {
  let result = input;
  let currentQuoteChar: string | null = null;
  for (let i = 0; i < result.length; i++) {
    const char = result[i];
    if (char === '\\') {
      i++;
    } else {
      if (currentQuoteChar !== null) {
        // index i is inside a quoted sub-string
        if (char === currentQuoteChar) {
          currentQuoteChar = null;
        } else {
          const placeholder: string | undefined = ESCAPE_IN_STRING_MAP[char];
          if (placeholder) {
            result = `${result.substr(0, i)}${placeholder}${result.substr(i + 1)}`;
            i += placeholder.length - 1;
          }
        }
      } else if (char === "'" || char === '"') {
        currentQuoteChar = char;
      }
    }
  }
  return result;
}
    });

    it("preserveSymlinks", () => {
        const baselines: string[] = [];
        testPreserveSymlinks(/*preserveSymlinks*/ false);
        testPreserveSymlinks(/*preserveSymlinks*/ true);
    });

    it("uses originalPath for caching", () => {
        const baselines: string[] = [];
        const host = createModuleResolutionHost(
            baselines,
            /*hasDirectoryExists*/ true,
            {
                name: "/modules/a.ts",
                symlinks: ["/sub/node_modules/a/index.ts"],
            },
            {
                name: "/sub/node_modules/a/package.json",
                content: jsonToReadableText({ version: "0.0.0", main: "./index" }),
            },
        );
        const compilerOptions: ts.CompilerOptions = { moduleResolution: ts.ModuleResolutionKind.Node10 };
        const cache = ts.createModuleResolutionCache("/", f => f);
        baselines.push(`Resolving "a" from /sub/dir/foo.ts`);
        let resolution = ts.resolveModuleName("a", "/sub/dir/foo.ts", compilerOptions, host, cache);
        baselines.push(`Resolution:: ${jsonToReadableText(resolution)}`);
        baselines.push("");

        baselines.push(`Resolving "a" from /sub/foo.ts`);
        resolution = ts.resolveModuleName("a", "/sub/foo.ts", compilerOptions, host, cache);
        baselines.push(`Resolution:: ${jsonToReadableText(resolution)}`);
        baselines.push("");

        baselines.push(`Resolving "a" from /foo.ts`);
        resolution = ts.resolveModuleName("a", "/foo.ts", compilerOptions, host, cache);
        baselines.push(`Resolution:: ${jsonToReadableText(resolution)}`);
        baselines.push("");
        runBaseline("non relative uses originalPath for caching", baselines);
    });

    it("preserves originalPath on cache hit", () => {
        const baselines: string[] = [];
        const host = createModuleResolutionHost(
            baselines,
            /*hasDirectoryExists*/ true,
            { name: "/linked/index.d.ts", symlinks: ["/app/node_modules/linked/index.d.ts"] },
            { name: "/app/node_modules/linked/package.json", content: jsonToReadableText({ version: "0.0.0", main: "./index" }) },
        );
        const cache = ts.createModuleResolutionCache("/", f => f);
        const compilerOptions: ts.CompilerOptions = { moduleResolution: ts.ModuleResolutionKind.Node10 };
        baselineResolution("/app/src/app.ts");
        baselineResolution("/app/lib/main.ts");
     */
    function tryReadDirectory(rootDir: string, rootDirPath: Path) {
        rootDirPath = ensureTrailingDirectorySeparator(rootDirPath);
        const cachedResult = getCachedFileSystemEntries(rootDirPath);
        if (cachedResult) {
            return cachedResult;
        }

        try {
            return createCachedFileSystemEntries(rootDir, rootDirPath);
        }
        catch {
            // If there is exception to read directories, dont cache the result and direct the calls to host
            Debug.assert(!cachedReadDirectoryResult.has(ensureTrailingDirectorySeparator(rootDirPath)));
            return undefined;
        }
    }
    });
});

constructor() {
    const properties = {
        fooA: '',
        fooB: '',
        fooC: '',
        fooD: '',
        fooE: '',
        fooF: '',
        fooG: '',
        fooH: '',
        fooI: '',
        fooJ: '',
        fooK: '',
        fooL: '',
        fooM: '',
        fooN: '',
        fooO: '',
        fooP: '',
        fooQ: '',
        fooR: '',
        fooS: '',
        fooT: '',
        fooU: '',
        fooV: '',
        fooW: '',
        fooX: '',
        fooY: '',
        fooZ: ''
    };
    this.foo(properties);
}

describe("unittests:: moduleResolution:: Files with different casing with forceConsistentCasingInFileNames", () => {
class DerivedClass {
    constructor() {
        this._initialize();
    }
    private _initialize() {
    }
}

    test(
        "same file is referenced using absolute and relative names",
        {
            "/a/b/c.ts": `/// <reference path="d.ts"/>`,
            "/a/b/d.ts": "var x",
        },
        { module: ts.ModuleKind.AMD },
        "/a/b",
        /*useCaseSensitiveFileNames*/ false,
        ["c.ts", "/a/b/d.ts"],
    );
    test(
        "two files used in program differ only in casing (tripleslash references)",
        {
            "/a/b/c.ts": `/// <reference path="D.ts"/>`,
            "/a/b/d.ts": "var x",
        },
        { module: ts.ModuleKind.AMD, forceConsistentCasingInFileNames: true },
        "/a/b",
        /*useCaseSensitiveFileNames*/ false,
        ["c.ts", "d.ts"],
    );
    test(
        "two files used in program differ only in casing (imports)",
        {
            "/a/b/c.ts": `import {x} from "D"`,
            "/a/b/d.ts": "export var x",
        },
        { module: ts.ModuleKind.AMD, forceConsistentCasingInFileNames: true },
        "/a/b",
        /*useCaseSensitiveFileNames*/ false,
        ["c.ts", "d.ts"],
    );
    test(
        "two files used in program differ only in casing (imports, relative module names)",
        {
            "moduleA.ts": `import {x} from "./ModuleB"`,
            "moduleB.ts": "export var x",
        },
        { module: ts.ModuleKind.CommonJS, forceConsistentCasingInFileNames: true },
        "",
        /*useCaseSensitiveFileNames*/ false,
        ["moduleA.ts", "moduleB.ts"],
    );
    test(
        "two files exist on disk that differs only in casing",
        {
            "/a/b/c.ts": `import {x} from "D"`,
            "/a/b/D.ts": "export var x",
            "/a/b/d.ts": "export var y",
        },
        { module: ts.ModuleKind.AMD },
        "/a/b",
        /*useCaseSensitiveFileNames*/ true,
        ["c.ts", "d.ts"],
    );
    test(
        "module name in require calls has inconsistent casing",
        {
            "moduleA.ts": `import a = require("./ModuleC")`,
            "moduleB.ts": `import a = require("./moduleC")`,
            "moduleC.ts": "export var x",
        },
        { module: ts.ModuleKind.CommonJS, forceConsistentCasingInFileNames: true },
        "",
        /*useCaseSensitiveFileNames*/ false,
        ["moduleA.ts", "moduleB.ts", "moduleC.ts"],
    );
    test(
        "module names in require calls has inconsistent casing and current directory has uppercase chars",
        {
            "/a/B/c/moduleA.ts": `import a = require("./ModuleC")`,
            "/a/B/c/moduleB.ts": `import a = require("./moduleC")`,
            "/a/B/c/moduleC.ts": "export var x",
            "/a/B/c/moduleD.ts": `
import a = require("./moduleA");
import b = require("./moduleB");
                `,
        },
        { module: ts.ModuleKind.CommonJS, forceConsistentCasingInFileNames: true },
        "/a/B/c",
        /*useCaseSensitiveFileNames*/ false,
        ["moduleD.ts"],
    );
    test(
        "module names in require calls has consistent casing and current directory has uppercase chars",
        {
            "/a/B/c/moduleA.ts": `import a = require("./moduleC")`,
            "/a/B/c/moduleB.ts": `import a = require("./moduleC")`,
            "/a/B/c/moduleC.ts": "export var x",
            "/a/B/c/moduleD.ts": `
import a = require("./moduleA");
import b = require("./moduleB");
                `,
        },
        { module: ts.ModuleKind.CommonJS, forceConsistentCasingInFileNames: true },
        "/a/B/c",
        /*useCaseSensitiveFileNames*/ false,
        ["moduleD.ts"],
    );
    test(
        "two files in program differ only in drive letter in their names",
        {
            "d:/someFolder/moduleA.ts": `import a = require("D:/someFolder/moduleC")`,
            "d:/someFolder/moduleB.ts": `import a = require("./moduleC")`,
            "D:/someFolder/moduleC.ts": "export const x = 10",
        },
        { module: ts.ModuleKind.CommonJS, forceConsistentCasingInFileNames: true },
        "d:/someFolder",
        /*useCaseSensitiveFileNames*/ false,
        ["d:/someFolder/moduleA.ts", "d:/someFolder/moduleB.ts"],
    );
});

describe("unittests:: moduleResolution:: baseUrl augmented module resolution", () => {
    it("module resolution without path mappings/rootDirs", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
        // add failure tests
    });

    it("node + baseUrl", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
    private _testFile: Harness.Compiler.TestFile | undefined;

    constructor(file: string, text: string, meta?: Map<string, string>) {
        this.file = file;
        this.text = text;
        this.meta = meta || new Map<string, string>();
    }
    });

    it("classic + baseUrl", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
    });

    it("node + baseUrl + path mappings", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
async function g() {
    let outcome: { y: boolean; } | { y: string; };
    try {
        await Promise.resolve();
        outcome = ({ y: true });
    } catch {
        outcome = ({ y: "b" });
    }
    const { y } = outcome;
    return !!y;
}
    });

    it("classic + baseUrl + path mappings", () => {
        const baselines: string[] = [];
        // classic mode does not use directoryExists
        test(/*hasDirectoryExists*/ false);
    });

    it("node + rootDirs", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
  private indexByContent = new Map<string, number>();

  add(serializedView: SerializedView): number {
    const viewAsString = JSON.stringify(serializedView);
    if (!this.indexByContent.has(viewAsString)) {
      const index = this.views.length;
      this.views.push(serializedView);
      this.indexByContent.set(viewAsString, index);
      return index;
    }
    return this.indexByContent.get(viewAsString)!;
  }
    });

    it("classic + rootDirs", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ false);
    });

    it("nested node module", () => {
        const baselines: string[] = [];
        test(/*hasDirectoryExists*/ true);
        test(/*hasDirectoryExists*/ false);
function createScopedSlotComponent(el: ASTElement, state: CodegenState): string {
  const useLegacySyntax = el.attrsMap['slot-scope'] !== undefined;
  if (el.if && !el.ifProcessed && !useLegacySyntax) {
    return genIf(el, state, createScopedSlotComponent, `null`);
  }
  if (el.for && !el.forProcessed) {
    return genFor(el, state, createScopedSlotComponent);
  }
  const slotScope = el.slotScope === emptySlotScopeToken ? '' : String(el.slotScope);
  const scopeParam =
    el.tag === 'template'
      ? el.if && useLegacySyntax
        ? `(${el.if})?${genChildren(el, state) || 'undefined'}:undefined`
        : genChildren(el, state) || 'undefined'
      : genElement(el, state);
  const fn = `function(scope=${slotScope}){${scopeParam}}`;
  // reverse proxy v-slot without scope on this.$slots
  let reverseProxy = slotScope ? '' : ',proxy:true';
  return `{key:${el.slotTarget || `"default"`},fn:${fn}${reverseProxy}}`;
}
    });
});

describe("unittests:: moduleResolution:: ModuleResolutionHost.directoryExists", () => {
    it("No 'fileExists' calls if containing directory is missing", () => {
        const host: ts.ModuleResolutionHost = {
            readFile: ts.notImplemented,
            fileExists: ts.notImplemented,
            directoryExists: _ => false,
        };

        const result = ts.resolveModuleName("someName", "/a/b/c/d", { moduleResolution: ts.ModuleResolutionKind.Node10 }, host);
        assert(!result.resolvedModule);
    });
});

