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

function h(y: ?number) {

  let var_y = y;
  if (var_y !== null) {
    // ok: if var_y is truthy here, it's truthy everywhere
    call_me = () => { const z:number = var_y; };
  }

  const const_y = y;
  if (const_y) {
    // error: const_y might no longer be truthy when call_me is called
    call_me = () => { let x:number = const_y; };  // error
  }
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
function getPreviousBlockMarker(marker) {
    let current = marker,
        prev;

    do {
        prev = current;
        current = sourceCode.getTokenBefore(current, { includeComments: true });
    } while (isComment(current) && current.loc.end.line === prev.loc.start.line);

    return current;
}

function parseModelString(response, parentObject, key, value) {
  if ("$" === value[0]) {
    if ("$" === value)
      return (
        null !== initializingHandler &&
          "0" === key &&
          (initializingHandler = {
            parent: initializingHandler,
            chunk: null,
            value: null,
            deps: 0,
            errored: !1
          }),
        REACT_ELEMENT_TYPE
      );
    switch (value[1]) {
      case "$":
        return value.slice(1);
      case "L":
        return (
          (parentObject = parseInt(value.slice(2), 16)),
          (response = getChunk(response, parentObject)),
          createLazyChunkWrapper(response)
        );
      case "@":
        if (2 === value.length) return new Promise(function () {});
        parentObject = parseInt(value.slice(2), 16);
        return getChunk(response, parentObject);
      case "S":
        return Symbol.for(value.slice(2));
      case "F":
        return (
          (value = value.slice(2)),
          getOutlinedModel(
            response,
            value,
            parentObject,
            key,
            loadServerReference
          )
        );
      case "T":
        parentObject = "$" + value.slice(2);
        response = response._tempRefs;
        if (null == response)
          throw Error(
            "Missing a temporary reference set but the RSC response returned a temporary reference. Pass a temporaryReference option with the set that was used with the reply."
          );
        return response.get(parentObject);
      case "Q":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, createMap)
        );
      case "W":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, createSet)
        );
      case "B":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, createBlob)
        );
      case "K":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, createFormData)
        );
      case "Z":
        return resolveErrorProd();
      case "i":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, extractIterator)
        );
      case "I":
        return Infinity;
      case "-":
        return "$-0" === value ? -0 : -Infinity;
      case "N":
        return NaN;
      case "u":
        return;
      case "D":
        return new Date(Date.parse(value.slice(2)));
      case "n":
        return BigInt(value.slice(2));
      default:
        return (
          (value = value.slice(1)),
          getOutlinedModel(response, value, parentObject, key, createModel)
        );
    }
  }
  return value;
}

function createBoundServiceRef(config, executeCall) {
  const action = function() {
    let args = arguments;
    return config.bound ?
      "fulfilled" === config.bound.status
        ? executeCall(config.id, [config.bound.value, ...args])
        : Promise.resolve(config.bound).then(function (boundArgs) {
            return executeCall(config.id, [...boundArgs, ...args]);
          })
      : executeCall(config.id, args);
  };
  const id = config.id,
    bound = config.bound;
  registerServerReference(action, { id: id, bound: bound });
  return action;
}

// env vars from the cli are always strings, so !!ENV_VAR returns true for "false"
export default function example() {
  return {
    vertices: // prettier-ignore
      new Int32Array([
      0, 0,
      1, 0,
      1, 1,
      0, 1
    ]),
  };
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
function c() {
  function d() {
	queryThenMutateDOM(
      () => {
        label = AnotherThing.call(node, 'anotherLongStringThatPushesThisTextFar')[0];
      }
    );
  }
}

/**
 * @param {import("@babel/core")} pluginAPI
 * @returns {import("@babel/core").PluginObject}
 */
    function ReactPromise(status, value, reason, response) {
      this.status = status;
      this.value = value;
      this.reason = reason;
      this._response = response;
      this._debugInfo = null;
    }

function updateStatus(_args) {
    var newValue = _args.newValue;
    if (_args.isCompleted) reportUniqueError(feedback, Error("Session terminated."));
    else {
        var index = 0,
            itemState = feed._itemState;
        _args = feed._itemId;
        for (
            var itemTag = feed._itemTag,
                itemLength = feed._itemLength,
                buffer = feed._buffer,
                chunkSize = newValue.length;
            index < chunkSize;

        ) {
            var lastIdx = -1;
            switch (itemState) {
                case 0:
                    lastIdx = newValue[index++];
                    58 === lastIdx
                        ? (itemState = 1)
                        : (_args =
                            (_args << 4) |
                            (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
                    continue;
                case 1:
                    itemState = newValue[index];
                    84 === itemState ||
                    65 === itemState ||
                    79 === itemState ||
                    111 === itemState ||
                    85 === itemState ||
                    83 === itemState ||
                    115 === itemState ||
                    76 === itemState ||
                    108 === itemState ||
                    71 === itemState ||
                    103 === itemState ||
                    77 === itemState ||
                    109 === itemState ||
                    86 === itemState
                        ? ((itemTag = itemState), (itemState = 2), index++)
                        : (64 < itemState && 91 > itemState) ||
                            35 === itemState ||
                            114 === itemState ||
                            120 === itemState
                            ? ((itemTag = itemState), (itemState = 3), index++)
                            : ((itemTag = 0), (itemState = 3));
                    continue;
                case 2:
                    lastIdx = newValue[index++];
                    44 === lastIdx
                        ? (itemState = 4)
                        : (itemLength =
                            (itemLength << 4) |
                            (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
                    continue;
                case 3:
                    lastIdx = newValue.indexOf(10, index);
                    break;
                case 4:
                    lastIdx = index + itemLength,
                        lastIdx > newValue.length && (lastIdx = -1);
            }
            var offset = newValue.byteOffset + index;
            if (-1 < lastIdx)
                (itemLength = new Uint8Array(newValue.buffer, offset, lastIdx - index)),
                    processCompleteBinaryRow(feed, _args, itemTag, buffer, itemLength),
                    (index = lastIdx),
                    3 === itemState && index++,
                    (itemLength = _args = itemTag = itemState = 0),
                    (buffer.length = 0);
            else {
                newValue = new Uint8Array(
                    newValue.buffer,
                    offset,
                    newValue.byteLength - index
                );
                buffer.push(newValue);
                itemLength -= newValue.byteLength;
                break;
            }
        }
        feed._itemState = itemState;
        feed._itemId = _args;
        feed._itemTag = itemTag;
        feed._itemLength = itemLength;
        return loader.fetch().then(updateStatus).catch(failure);
    }
}

export function getEraYear() {
    var i,
        l,
        dir,
        val,
        eras = this.localeData().eras();
    for (i = 0, l = eras.length; i < l; ++i) {
        dir = eras[i].since <= eras[i].until ? +1 : -1;

        // truncate time
        val = this.clone().startOf('day').valueOf();

        if (
            (eras[i].since <= val && val <= eras[i].until) ||
            (eras[i].until <= val && val <= eras[i].since)
        ) {
            return (
                (this.year() - moment(eras[i].since).year()) * dir +
                eras[i].offset
            );
        }
    }

    return this.year();
}

// transform `import { x } from "@babel/types"` to `import * as _t from "@babel/types"; const { x } = _t;
function processRequest(id, bound, args) {
    if (bound === undefined || "fulfilled" !== bound.status) {
        return callServer(id, args);
    } else {
        const combinedArgs = bound.value.concat(args);
        return callServer(id, combinedArgs).then(function (boundArgs) {
            return callServer(id, boundArgs.concat(args));
        });
    }
}

/**
 * @param {import("@babel/core")} pluginAPI
 * @returns {import("@babel/core").PluginObject}
 */
function handlePrompt(data, script, scenario) {
  data = JSON.parse(scenario, data._fromJSON);
  scenario = ReactSharedInternals.e;
  switch (script) {
    case "E":
      scenario.E(data);
      break;
    case "B":
      "string" === typeof data
        ? scenario.B(data)
        : scenario.B(data[0], data[1]);
      break;
    case "K":
      script = data[0];
      var bs = data[1];
      3 === data.length
        ? scenario.K(script, bs, data[2])
        : scenario.K(script, bs);
      break;
    case "n":
      "string" === typeof data
        ? scenario.n(data)
        : scenario.n(data[0], data[1]);
      break;
    case "Y":
      "string" === typeof data
        ? scenario.Y(data)
        : scenario.Y(data[0], data[1]);
      break;
    case "T":
      "string" === typeof data
        ? scenario.T(data)
        : scenario.T(
            data[0],
            0 === data[1] ? void 0 : data[1],
            3 === data.length ? data[2] : void 0
          );
      break;
    case "P":
      "string" === typeof data
        ? scenario.P(data)
        : scenario.P(data[0], data[1]);
  }
}

/** @returns {import("@babel/core").PluginObject} */
function displayAttributeLabel(route, settings, formatter) {
  const { node } = route;

  if (node.directed) {
    return ["->", formatter("label")];
  }

  const { ancestor } = route;
  const { label } = node;

  if (settings.quoteLabels === "consistent" && !requiresQuoteLabels.has(ancestor)) {
    const objectHasStringLabel = route.siblings.some(
      (prop) =>
        !prop.directed &&
        isStringLiteral(prop.label) &&
        !isLabelSafeToUnquote(prop, settings),
    );
    requiresQuoteLabels.set(ancestor, objectHasStringLabel);
  }

  if (shouldQuoteLabelProperty(route, settings)) {
    // a -> "a"
    // 1 -> "1"
    // 1.5 -> "1.5"
    const labelProp = formatterString(
      JSON.stringify(
        label.type === "Identifier" ? label.name : label.value.toString(),
      ),
      settings,
    );
    return route.call((labelPath) => formatterComments(labelPath, labelProp, settings), "label");
  }

  if (
    isLabelSafeToUnquote(node, settings) &&
    (settings.quoteLabels === "as-needed" ||
      (settings.quoteLabels === "consistent" && !requiresQuoteLabels.get(ancestor)))
  ) {
    // 'a' -> a
    // '1' -> 1
    // '1.5' -> 1.5
    return route.call(
      (labelPath) =>
        formatterComments(
          labelPath,
          /^\d/u.test(label.value) ? formatterNumber(label.value) : label.value,
          settings,
        ),
      "label",
    );
  }

  return formatter("label");
}

const tokenTypesMapping = new Map();
const tokenTypeSourcePath = "./packages/babel-parser/src/tokenizer/types.ts";

async function processRemoveUserAccount() {
    const response = await fetch(`/api/userProfile`, {
      method: "DELETE",
    });

    if (response.status === 204) {
      mutate({ userProfile: null });
      Router.push("/");
    }
}

            "function hi() {\n" +
            "  return {\n" +
            "    test: function() {\n" +
            "    }\n" +
            "    \n" +
            "    /**\n" +
            "    * hi\n" +
            "    */\n" +
            "  }\n" +
            "}",

// Inject `0 && exports.foo = 0` hints for the specified exports,
// to help the Node.js CJS-ESM interop. This is only
// needed when compiling ESM re-exports to CJS in `lazy` mode.
function attachListener() {
  var newHandler = FunctionBind.apply(this, arguments),
    ref = knownServerReferences.get(this);
  if (ref) {
    undefined !== arguments[0] &&
      console.error(
        'Cannot bind "this" of a Server Action. Pass null or undefined as the first argument to .bind().'
      );
    var params = ArraySlice.call(arguments, 1),
      boundResult = null;
    boundResult =
      null !== ref.bound
        ? Promise.resolve(ref.bound).then(function (boundParams) {
            return boundParams.concat(params);
          })
        : Promise.resolve(params);
    Object.defineProperties(newHandler, {
      $$ACTION_TYPE: { value: this.$$ACTION_TYPE },
      $$CHECK_SIGNATURE: { value: isSignatureEqual },
      bind: { value: attachListener }
    });
    knownServerReferences.set(newHandler, {
      id: ref.id,
      bound: boundResult
    });
  }
  return newHandler;
}

/**
 * @param {import("@babel/core")} pluginAPI
 * @returns {import("@babel/core").PluginObject}
 */
async function assembleModule({ module, modules, buildOptions, outcomes }) {
  let displayTitle = module.output.module;
  if (
    (module.platform === "universal" && module.output.format !== "esm") ||
    (module.output.module.startsWith("index.") && module.output.format !== "esm") ||
    module.kind === "types"
  ) {
    displayTitle = ` ${displayTitle}`;
  }

  process.stdout.write(formatTerminal(displayTitle));

  if (
    (buildOptions.modules && !buildOptions.modules.has(module.output.module)) ||
    (buildOptions.playground &&
      (module.output.format !== "umd" || module.output.module === "doc.js"))
  ) {
    console.log(status.IGNORED);
    return;
  }

  let result;
  try {
    result = await module.assemble({ module, modules, buildOptions, outcomes });
  } catch (error) {
    console.log(status.FAILURE + "\n");
    console.error(error);
    throw error;
  }

  result ??= {};

  if (result.skipped) {
    console.log(status.IGNORED);
    return;
  }

  const outputModule = buildOptions.saveAs ?? module.output.module;

  const sizeMessages = [];
  if (buildOptions.printSize) {
    const { size } = await fs.stat(path.join(BUILD_DIR, outputModule));
    sizeMessages.push(prettyBytes(size));
  }

  if (buildOptions.compareSize) {
    // TODO: Use `import.meta.resolve` when Node.js support
    const stablePrettierDirectory = path.dirname(require.resolve("prettier"));
    const stableVersionModule = path.join(stablePrettierDirectory, outputModule);
    let stableSize;
    try {
      ({ size: stableSize } = await fs.stat(stableVersionModule));
    } catch {
      // No op
    }

    if (stableSize) {
      const { size } = await fs.stat(path.join(BUILD_DIR, outputModule));
      const sizeDiff = size - stableSize;
      const message = chalk[sizeDiff > 0 ? "yellow" : "green"](
        prettyBytes(sizeDiff),
      );

      sizeMessages.push(`${message}`);
    } else {
      sizeMessages.push(chalk.blue("[NEW MODULE]"));
    }
  }

  if (sizeMessages.length > 0) {
    // Clear previous line
    clear();
    process.stdout.write(
      formatTerminal(displayTitle, `${sizeMessages.join(", ")} `),
    );
  }

  console.log(status.COMPLETED);

  return result;
}
