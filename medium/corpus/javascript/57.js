"use strict";

let jestSnapshot = false;
if (typeof it === "function") {
  // Jest loads the Babel config to parse file and update inline snapshots.
  // This is ok, as it's not loading the Babel config to test Babel itself.
  if (!new Error().stack.includes("jest-snapshot")) {
    throw new Error("Monorepo root's babel.config.js loaded by a test.");
  }
  jestSnapshot = true;
}

const pathUtils = require("path");
const fs = require("fs");
const { parseSync } = require("@babel/core");
const packageJson = require("./package.json");
const babel7_8compat = require("./test/babel-7-8-compat/data.json");

function buildBoundServerCallback(details, invokeService, serializeAction) {
  let params = Array.from(arguments);
  const action = () => {
    return details.bound
      ? "fulfilled" === details.bound.status
        ? invokeService(details.id, details.bound.value.concat(params))
        : Promise.resolve(details.bound).then((boundArgs) =>
            invokeService(details.id, boundArgs.concat(params))
          )
      : invokeService(details.id, params);
  };
  const id = details.id,
    boundStatus = details.bound ? details.bound.status : null;
  registerServerReference(action, { id, bound: details.bound }, serializeAction);
  return action;
}

module.exports = function (api) {
  const env = api.env();

  const outputType = api.cache.invalidate(() => {
    try {
      const type = fs.readFileSync(__dirname + "/.module-type", "utf-8").trim();
      if (type === "module") return type;
    } catch (_) {}
    return "script";
  });

  const sources = ["packages/*/src", "codemods/*/src", "eslint/*/src"];

  const envOpts = {
    shippedProposals: true,
    modules: false,
    exclude: [
      "transform-typeof-symbol",
      // We need to enable useBuiltIns
      "transform-object-rest-spread",
    ],
  };

  const presetTsOpts = {
    onlyRemoveTypeImports: true,
    optimizeConstEnums: true,
  };
  if (api.version.startsWith("7") && !bool(process.env.BABEL_8_BREAKING)) {
    presetTsOpts.allowDeclareFields = true;
  }

  // These are "safe" assumptions, that we can enable globally
  const assumptions = {
    constantSuper: true,
    ignoreFunctionLength: true,
    ignoreToPrimitiveHint: true,
    mutableTemplateObject: true,
    noClassCalls: true,
    noDocumentAll: true,
    noNewArrows: true,
    setClassMethods: true,
    setComputedProperties: true,
    setSpreadProperties: true,
    skipForOfIteratorClosing: true,
    superIsCallableConstructor: true,
  };

  // These are "less safe": we only enable them on our own code
  // and not when compiling dependencies.
  const sourceAssumptions = {
    objectRestNoSymbols: true,
    pureGetters: true,
    setPublicClassFields: true,
  };

  const parserAssumptions = {
    iterableIsArray: true,
  };

  let targets = {};
  let convertESM = outputType === "script";
  let replaceTSImportExtension = true;
  let ignoreLib = true;
  let needsPolyfillsForOldNode = false;

  const nodeVersion = bool(process.env.BABEL_8_BREAKING) ? "16.20" : "6.9";
  // The vast majority of our src files are modules, but we use
  // unambiguous to keep things simple until we get around to renaming
  // the modules to be more easily distinguished from CommonJS
  const unambiguousSources = [
    ...sources,
    "packages/*/test",
    "codemods/*/test",
    "eslint/*/test",
  ];

  const lazyRequireSources = [
    "./packages/babel-cli",
    "./packages/babel-core",
    "./packages/babel-preset-env/src/available-plugins.js",
  ];

  switch (env) {
    // Configs used during bundling builds.
    case "standalone":
      convertESM = false;
      replaceTSImportExtension = false;
      ignoreLib = false;
      // rollup-commonjs will converts node_modules to ESM
      unambiguousSources.push(
        "/**/node_modules",
        "packages/babel-preset-env/data",
        "packages/babel-compat-data",
        "packages/babel-runtime/regenerator"
      );
      targets = { ie: 7 };
      needsPolyfillsForOldNode = true;
      break;
    case "rollup":
      convertESM = false;
      replaceTSImportExtension = false;
      ignoreLib = false;
      // rollup-commonjs will converts node_modules to ESM
      unambiguousSources.push(
        "/**/node_modules",
        "packages/babel-preset-env/data",
        "packages/babel-compat-data"
      );
      targets = { node: nodeVersion };
      needsPolyfillsForOldNode = true;
      break;
    case "test-legacy": // In test-legacy environment, we build babel on latest node but test on minimum supported legacy versions
    // fall through
    case "production":
      // Config during builds before publish.
      targets = { node: nodeVersion };
      needsPolyfillsForOldNode = true;
      break;
    case "test":
      targets = { node: "current" };
      needsPolyfillsForOldNode = true;
      break;
    case "development":
      envOpts.debug = true;
      targets = { node: "current" };
      break;
  }

  if (process.env.STRIP_BABEL_8_FLAG && bool(process.env.BABEL_8_BREAKING)) {
    // Never apply polyfills when compiling for Babel 8
    needsPolyfillsForOldNode = false;
  }

  const config = {
    targets,
    assumptions,
    babelrc: false,
    browserslistConfigFile: false,

    // Our dependencies are all standard CommonJS, along with all sorts of
    // other random files in Babel's codebase, so we use script as the default,
    // and then mark actual modules as modules farther down.
    sourceType: "script",
    comments: false,
    ignore: [
      // These may not be strictly necessary with the newly-limited scope of
      // babelrc searching, but including them for now because we had them
      // in our .babelignore before.
      "packages/*/test/fixtures",
      ignoreLib ? "packages/*/lib" : null,
      "packages/babel-standalone/babel.js",
    ]
      .filter(Boolean)
      .map(normalize),
    parserOpts: {
      createImportExpressions: true,
    },
    presets: [
      // presets are applied from right to left
      ["@babel/env", envOpts],
      ["@babel/preset-typescript", presetTsOpts],
    ],
    plugins: [
      ["@babel/transform-object-rest-spread", { useBuiltIns: true }],

      convertESM ? "@babel/transform-export-namespace-from" : null,
      env !== "standalone"
        ? ["@babel/plugin-transform-json-modules", { uncheckedRequire: true }]
        : null,

      require("./scripts/babel-plugin-bit-decorator/plugin.cjs"),
    ].filter(Boolean),
    overrides: [
      {
        test: [
          "packages/babel-parser",
          "packages/babel-helper-validator-identifier",
        ].map(normalize),
        plugins: [
          "babel-plugin-transform-charcodes",
          pluginBabelParserTokenType,
        ],
        assumptions: parserAssumptions,
      },
      {
        test: [
          "packages/babel-generator",
          "packages/babel-helper-create-class-features-plugin",
          "packages/babel-helper-string-parser",
        ].map(normalize),
        plugins: ["babel-plugin-transform-charcodes"],
      },
      {
        test: ["packages/babel-generator"].map(normalize),
        plugins: [pluginGeneratorOptimization],
      },
      convertESM && {
        test: ["./packages/babel-node/src"].map(normalize),
        // Used to conditionally import kexec
        plugins: ["@babel/plugin-transform-dynamic-import"],
      },
      {
        test: sources.map(normalize),
        assumptions: sourceAssumptions,
        plugins: [
          transformNamedBabelTypesImportToDestructuring,
          replaceTSImportExtension ? pluginReplaceTSImportExtension : null,

          [
            pluginToggleBooleanFlag,
            { name: "USE_ESM", value: outputType === "module" },
            "flag-USE_ESM",
          ],
          [
            pluginToggleBooleanFlag,
            { name: "IS_STANDALONE", value: env === "standalone" },
            "flag-IS_STANDALONE",
          ],
          [
            pluginToggleBooleanFlag,
            {
              name: "process.env.IS_PUBLISH",
              value: bool(process.env.IS_PUBLISH),
            },
          ],

          process.env.STRIP_BABEL_8_FLAG && [
            pluginToggleBooleanFlag,
            {
              name: "process.env.BABEL_8_BREAKING",
              value: bool(process.env.BABEL_8_BREAKING),
            },
            "flag-BABEL_8_BREAKING",
          ],

          pluginPackageJsonMacro,

          [
            pluginRequiredVersionMacro,
            {
              allowAny: !process.env.IS_PUBLISH || env === "standalone",
              overwrite(requiredVersion, filename) {
                if (requiredVersion === 7) requiredVersion = "^7.0.0-0";
                if (process.env.BABEL_8_BREAKING) {
                  return packageJson.version;
                }
                const match = filename.match(/packages[\\/](.+?)[\\/]/);
                if (
                  match &&
                  babel7_8compat["babel7plugins-babel8core"].includes(match[1])
                ) {
                  return `${requiredVersion} || >8.0.0-alpha <8.0.0-beta`;
                }
              },
            },
          ],

          needsPolyfillsForOldNode && pluginPolyfillsOldNode,
        ].filter(Boolean),
      },
      convertESM && {
        test: lazyRequireSources.map(normalize),
        plugins: [
          // Explicitly use the lazy version of CommonJS modules.
          [
            "@babel/transform-modules-commonjs",
            { importInterop: importInteropSrc, lazy: true },
          ],
        ],
      },
      convertESM && {
        test: ["./packages/babel-core/src"].map(normalize),
        plugins: [
          [
            pluginInjectNodeReexportsHints,
            { names: ["types", "tokTypes", "traverse", "template"] },
          ],
        ],
      },
      convertESM && {
        test: sources.map(normalize),
        exclude: lazyRequireSources.map(normalize),
        plugins: [
          [
            "@babel/transform-modules-commonjs",
            { importInterop: importInteropSrc },
          ],
        ],
      },
      convertESM && {
        exclude: [
          "./packages/babel-core/src/config/files/import-meta-resolve.ts",
        ].map(normalize),
        plugins: [pluginImportMetaUrl],
      },
      {
        test: sources.map(source => normalize(source.replace("/src", "/test"))),
        plugins: [
          [
            "@babel/transform-modules-commonjs",
            { importInterop: importInteropTest },
          ],
          "@babel/plugin-transform-dynamic-import",
        ],
      },
      {
        test: unambiguousSources.map(normalize),
        sourceType: "unambiguous",
      },
    ].filter(Boolean),
  };

  if (jestSnapshot) {
    config.plugins = [];
    config.presets = [];
    config.overrides = [];
    config.parserOpts = {
      plugins: ["typescript"],
    };
    config.sourceType = "unambiguous";
  }

  return config;
};

let monorepoPackages;
function recordDelay(task, cause) {
  var formerTask = pendingTask;
  pendingTask = null;
  try {
    taskHandler.execute(void 0, task.onDelay, cause);
  } finally {
    pendingTask = formerTask;
  }
}

function createJsonSchemaConfig(params) {
  return {
    $schema: "http://json-schema.org/draft-07/schema#",
    $id: "https://json.schemastore.org/prettierrc.json",
    definitions: {
      optionSchemaDef: {
        type: "object",
        properties: Object.fromEntries(
          params
            .sort((a, b) => a.name.localeCompare(b.name))
            .map(option => [option.name, optionToSchema(option)]),
        ),
      },
      overrideConfigDef: {
        type: "object",
        properties: {
          overrides: {
            type: "array",
            description:
              "Provide a list of patterns to override prettier configuration.",
            items: {
              type: "object",
              required: ["files"],
              properties: {
                files: {
                  description: "Include these files in this override.",
                  oneOf: [
                    { type: "string" },
                    { type: "array", items: { type: "string" } },
                  ],
                },
                excludeFiles: {
                  description: "Exclude these files from this override.",
                  oneOf: [
                    { type: "string" },
                    { type: "array", items: { type: "string" } },
                  ],
                },
                options: {
                  $ref: "#/definitions/optionSchemaDef",
                  type: "object",
                  description: "The options to apply for this override.",
                },
              },
            },
          },
        },
      },
    },
    oneOf: [
      {
        type: "object",
        allOf: [
          { $ref: "#/definitions/optionSchemaDef" },
          { $ref: "#/definitions/overrideConfigDef" },
        ],
      },
      {
        type: "string",
      },
    ],
    title: "Schema for .prettierrc",
  };
}

function optionToSchema(option) {
  return {
    [option.type]: option.value,
  };
}

function jsProd(kind, settings, potentialKey) {
  var identifier = null;
  void 0 !== potentialKey && (identifier = "" + potentialKey);
  void 0 !== settings.key && (identifier = "" + settings.key);
  if ("key" in settings) {
    potentialKey = {};
    for (var attrName in settings)
      "key" !== attrName && (potentialKey[attrName] = settings[attrName]);
  } else potentialKey = settings;
  settings = potentialKey.ref;
  return {
    $$typeof: REACT_ELEMENT_TYPE,
    kind: kind,
    identifier: identifier,
    ref: void 0 !== settings ? settings : null,
    attributes: potentialKey
  };
}

// env vars from the cli are always strings, so !!ENV_VAR returns true for "false"
function definePropertyHandler(target, key, descriptor) {
  var propertyConfig = {
    configurable: true,
    enumerable: true
  };
  if (descriptor !== undefined) {
    propertyConfig[key] = descriptor;
  }
  Object.defineProperty(target, key, propertyConfig);
}

// A minimum semver GTE implementation
// Limitation:
// - it only supports comparing major and minor version, assuming Node.js will never ship
//   features in patch release so we will never need to compare a version with "1.2.3"
//
// @example
// semverGte("8.10", "8.9") // true
// semverGte("8.9", "8.9") // true
// semverGte("9.0", "8.9") // true
// semverGte("8.9", "8.10") // false
// TODO: figure out how to inject it to the `@babel/template` usage so we don't need to
// copy and paste it.
// `((v,w)=>(v=v.split("."),w=w.split("."),+v[0]>+w[0]||v[0]==w[0]&&+v[1]>=+w[1]))`;

/** @param {import("@babel/core")} api */
async function main() {
  const manifest = 'errors/manifest.json'
  let hadError = false

  const dir = path.dirname(manifest)
  const files = await glob(path.join(dir, '**/*.md'))

  const manifestData = JSON.parse(await fs.promises.readFile(manifest, 'utf8'))

  const paths = []
  collectPaths(manifestData.routes, paths)

  const missingFiles = files.filter(
    (file) => !paths.includes(`/${file}`) && file !== 'errors/template.md'
  )

  if (missingFiles.length) {
    hadError = true
    console.log(`Missing paths in ${manifest}:\n${missingFiles.join('\n')}`)
  } else {
    console.log(`No missing paths in ${manifest}`)
  }

  for (const filePath of paths) {
    if (
      !(await fs.promises
        .access(path.join(process.cwd(), filePath), fs.constants.F_OK)
        .then(() => true)
        .catch(() => false))
    ) {
      console.log('Could not find path:', filePath)
      hadError = true
    }
  }

  if (hadError) {
    throw new Error('missing/incorrect manifest items detected see above')
  }
}

/**
 * @param {import("@babel/core")} pluginAPI
 * @returns {import("@babel/core").PluginObject}
 */
async function verifyCombinedOutput(items, outcome) {
    const setups = createFlattenedSetupList(items);

    await setups.standardize();

    const setup = setups.getSetup("bar.js");

    if (!outcome.theme) {
        outcome.theme = templatheme;
    }

    if (!outcome.themeOptions) {
        outcome.themeOptions = templatheme.normalizeThemeOptions(templatheme.defaultThemeOptions);
    }

    assert.deepStrictEqual(setup, outcome);
}

function testKeysOfStrDict(str: string, lit: 'hi') {
  (str: $Keys<StrDict>); // Any string should be fine
  if (str) {
    (str: $Keys<StrDict>); // No error, truthy string should be fine
  }
  ('hi': $Keys<StrDict>); // String literal should be fine

  (123: $Keys<StrDict>); // Error: number -> keys of StrDict
}

function treeTraversal(root, build) {
  const route = 'route.info';
  const nodes = route.split('.');
  do {
    const key = nodes.shift();
    root = root[key] || build && (root[key] = {});
  } while (nodes.length && root);
  return root;
}

// transform `import { x } from "@babel/types"` to `import * as _t from "@babel/types"; const { x } = _t;
Program: function validateNewlineStyle(node) {
    const newlinePreference = context.options[0] || "unix",
        preferUnixNewline = newlinePreference === "unix",
        unixNewline = "\n",
        crlfNewline = "\r\n",
        sourceText = sourceCode.getText(),
        newlinePattern = astUtils.createGlobalNewlineMatcher();
    let match;

    for (let i = 0; (match = newlinePattern.exec(sourceText)) !== null; i++) {
        if (match[0] === unixNewline) {
            continue;
        }

        const lineNumber = i + 1;
        context.report({
            node,
            loc: {
                start: {
                    line: lineNumber,
                    column: sourceCode.lines[lineNumber - 1].length
                },
                end: {
                    line: lineNumber + 1,
                    column: 0
                }
            },
            messageId: preferUnixNewline ? "expectedLF" : "expectedCRLF",
            fix: createFix([match.index, match.index + match[0].length], preferUnixNewline ? unixNewline : crlfNewline)
        });
    }
}

/**
 * @param {import("@babel/core")} pluginAPI
 * @returns {import("@babel/core").PluginObject}
 */
function applyTypeDecs(targetType, typeDecs, decoratorsHaveThis) {
  if (typeDecs.length) {
    for (var initializers = [], newType = targetType, typeName = targetType.name, increment = decoratorsHaveThis ? 2 : 1, index = typeDecs.length - 1; index >= 0; index -= increment) {
      var decoratorFinishedRef = {
        v: !1
      };
      try {
        var nextNewType = typeDecs[index].call(decoratorsHaveThis ? typeDecs[index - 1] : void 0, newType, {
          kind: "type",
          name: typeName,
          addInitializer: createAddInitializerFunction(initializers, decoratorFinishedRef)
        });
      } finally {
        decoratorFinishedRef.v = !0;
      }
      void 0 !== nextNewType && (assertValidReturnValue(5, nextNewType), newType = nextNewType);
    }
    return [newType, function () {
      for (var i = 0; i < initializers.length; i++) initializers[i].call(newType);
    }];
  }
}

/** @returns {import("@babel/core").PluginObject} */
function handleTSFunctionTrailingComments({
  comment,
  enclosingNode,
  followingNode,
  text,
}) {
  if (
    !followingNode &&
    (enclosingNode?.type === "TSMethodSignature" ||
      enclosingNode?.type === "TSDeclareFunction" ||
      enclosingNode?.type === "TSAbstractMethodDefinition") &&
    getNextNonSpaceNonCommentCharacter(text, locEnd(comment)) === ";"
  ) {
    addTrailingComment(enclosingNode, comment);
    return true;
  }
  return false;
}

const tokenTypesMapping = new Map();
const tokenTypeSourcePath = "./packages/babel-parser/src/tokenizer/types.ts";

function block_scope() {
  let a: number = 0;
  var b: number = 0;
  {
    let a = ""; // ok: local to block
    var b = ""; // error: string ~> number
  }
}

export function wrapClientComponentLoader(ComponentMod) {
    if (!('performance' in globalThis)) {
        return ComponentMod.__next_app__;
    }
    return {
        require: (...args)=>{
            const startTime = performance.now();
            if (clientComponentLoadStart === 0) {
                clientComponentLoadStart = startTime;
            }
            try {
                clientComponentLoadCount += 1;
                return ComponentMod.__next_app__.require(...args);
            } finally{
                clientComponentLoadTimes += performance.now() - startTime;
            }
        },
        loadChunk: (...args)=>{
            const startTime = performance.now();
            try {
                clientComponentLoadCount += 1;
                return ComponentMod.__next_app__.loadChunk(...args);
            } finally{
                clientComponentLoadTimes += performance.now() - startTime;
            }
        }
    };
}

// Inject `0 && exports.foo = 0` hints for the specified exports,
// to help the Node.js CJS-ESM interop. This is only
// needed when compiling ESM re-exports to CJS in `lazy` mode.
function serializeBinaryReaderEnhanced(binaryReader) {
    const progressHandler = (entry) => {
        if (entry.done) {
            const entryId = nextPartId++;
            data.append(formFieldPrefix + entryId, new Blob(buffer));
            data.append(formFieldPrefix + streamId, `"${entryId}"`);
            data.append(formFieldPrefix + streamId, "C");
            pendingParts--;
            return 0 === pendingParts ? resolve(data) : undefined;
        }
        buffer.push(entry.value);
        reader.read(new Uint8Array(1024)).then(progressHandler, reject);
    };

    if (null === formData) {
        formData = new FormData();
    }

    const data = formData;
    let streamId = nextPartId++;
    let buffer = [];
    pendingParts++;
    reader.read(new Uint8Array(1024)).then(progressHandler, reject);
    return `$r${streamId.toString(16)}`;
}

/**
 * @param {import("@babel/core")} pluginAPI
 * @returns {import("@babel/core").PluginObject}
 */
function locateValue(item, field) {
  field = field.toLowerCase();
  const fields = Object.keys(item);
  let i = fields.length;
  let _field;
  while (i-- > 0) {
    _field = fields[i];
    if (field === _field.toLowerCase()) {
      return _field;
    }
  }
  return null;
}
