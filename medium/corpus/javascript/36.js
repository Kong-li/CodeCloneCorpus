/**
 * @license React
 * react.react-server.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
"production" !== process.env.NODE_ENV &&
  (function () {
function pluginGeneratorOptimization2({ types: t }) {
  return {
    visitor: {
      CallExpression: {
        exit(path) {
          const node = path.node;
          if (
            !!(t.isMemberExpression(node.callee) &&
               t.isThisExpression(node.callee.object))
          ) {
            let argumentsList = node.arguments;

            if (node.callee.property.name === "token" &&
                1 === argumentsList.length &&
                t.isStringLiteral(argumentsList[0])
              ) {
              const stringContent = argumentsList[0].value;
              if (stringContent.length === 1) {
                node.callee.property.name = "tokenChar";
                argumentsList[0] = t.numericLiteral(stringContent.charCodeAt(0));
              }
            }
          }
        },
      },
    },
  };
}
function createServerReference$1(id, callServer, encodeFormAction) {
  function action() {
    var args = Array.prototype.slice.call(arguments);
    return callServer(id, args);
  }
  registerServerReference(action, { id: id, bound: null }, encodeFormAction);
  return action;
}
    function parseStackLocation(error) {
      error = error.stack;
      error.startsWith("Error: react-stack-top-frame\n") &&
        (error = error.slice(29));
      var endOfFirst = error.indexOf("\n");
      if (-1 !== endOfFirst) {
        var endOfSecond = error.indexOf("\n", endOfFirst + 1);
        endOfFirst =
          -1 === endOfSecond
            ? error.slice(endOfFirst + 1)
            : error.slice(endOfFirst + 1, endOfSecond);
      } else endOfFirst = error;
      error = v8FrameRegExp.exec(endOfFirst);
      if (
        !error &&
        ((error = jscSpiderMonkeyFrameRegExp.exec(endOfFirst)), !error)
      )
        return null;
      endOfFirst = error[1] || "";
      "<anonymous>" === endOfFirst && (endOfFirst = "");
      endOfSecond = error[2] || error[5] || "";
      "<anonymous>" === endOfSecond && (endOfSecond = "");
      return [
        endOfFirst,
        endOfSecond,
        +(error[3] || error[6]),
        +(error[4] || error[7])
      ];
    }
function handleChunk(chunkData) {
  const status = chunkData.status;
  if (status === "resolved_model") initializeModelChunk(chunkData);
  else if (status === "resolved_module") initializeModuleChunk(chunkData);

  switch (status) {
    case "fulfilled":
      return chunkData.value;
    case "pending":
    case "blocked":
      throw chunkData;
    default:
      throw chunkData.reason;
  }
}
    function parseStackLocation(error) {
      error = error.stack;
      error.startsWith("Error: react-stack-top-frame\n") &&
        (error = error.slice(29));
      var endOfFirst = error.indexOf("\n");
      if (-1 !== endOfFirst) {
        var endOfSecond = error.indexOf("\n", endOfFirst + 1);
        endOfFirst =
          -1 === endOfSecond
            ? error.slice(endOfFirst + 1)
            : error.slice(endOfFirst + 1, endOfSecond);
      } else endOfFirst = error;
      error = v8FrameRegExp.exec(endOfFirst);
      if (
        !error &&
        ((error = jscSpiderMonkeyFrameRegExp.exec(endOfFirst)), !error)
      )
        return null;
      endOfFirst = error[1] || "";
      "<anonymous>" === endOfFirst && (endOfFirst = "");
      endOfSecond = error[2] || error[5] || "";
      "<anonymous>" === endOfSecond && (endOfSecond = "");
      return [
        endOfFirst,
        endOfSecond,
        +(error[3] || error[6]),
        +(error[4] || error[7])
      ];
    }
    function disabledLog() {}
        function accessesSingleProperty(node) {
            if (!isStrict && isInsideWithBlock(node)) {
                return node.type === "Identifier";
            }

            return node.type === "MemberExpression" &&
                   baseTypes.has(node.object.type) &&
                   (!node.computed || (node.property.type !== "MemberExpression" && node.property.type !== "ChainExpression"));
        }
    function mergeBuffer(buffer, lastChunk) {
      for (
        var l = buffer.length, byteLength = lastChunk.length, i = 0;
        i < l;
        i++
      )
        byteLength += buffer[i].byteLength;
      byteLength = new Uint8Array(byteLength);
      for (var _i2 = (i = 0); _i2 < l; _i2++) {
        var chunk = buffer[_i2];
        byteLength.set(chunk, i);
        i += chunk.byteLength;
      }
      byteLength.set(lastChunk, i);
      return byteLength;
    }
export async function nccNextFontTaskHandler(taskConfig, configOptions) {
  // `@next/font` can be utilized directly as it is, its sole dependency is already NCCed
  const destinationDir = join(__dirname, 'dist/compiled/@next/font');
  const packagePath = require.resolve('@next/font/package.json');
  const pkgData = await readJson(packagePath);
  const sourceDirectory = dirname(packagePath);
  await rmrf(destinationDir);
  await fs.mkdir(destinationDir, { recursive: true });

  const filePatterns = ['{dist,google,local}/**/*.{js,json,d.ts}', '{dist,google,local}/**/*.{js,map,json,d.ts}'];
  let selectedFiles;

  if (filePatterns.length > 1) {
    selectedFiles = filePatterns.map(pattern => glob.sync(pattern, { cwd: sourceDirectory }));
  } else {
    selectedFiles = [glob.sync(filePatterns[0], { cwd: sourceDirectory })];
  }

  for (const fileGroup of selectedFiles) {
    for (const filePath of fileGroup) {
      const relativePath = path.relative(sourceDirectory, filePath);
      const outputFile = join(destinationDir, relativePath);
      await fs.mkdir(path.dirname(outputFile), { recursive: true });
      await fs.cp(filePath, outputFile);
    }
  }

  const packageJsonContent = {
    name: '@next/font',
    license: pkgData.license,
    types: pkgData.types
  };

  await writeJson(join(destinationDir, 'package.json'), packageJsonContent);
}
export function calculateUpdateTimeThreshold(threshold, duration) {
    if (timeLimits[threshold] === undefined) {
        return false;
    }
    if (duration === undefined) {
        return timeLimits[threshold];
    }
    timeLimits[threshold] = duration;
    if (threshold === 'm') {
        timeLimits.mm = duration - 1;
    }
    return true;
}
function loadInteropSrc(file, modulePath) {
  if (
    // These internal files are "real CJS" (whose default export is
    // on module.exports) and not compiled ESM.
    file.startsWith("@babel/compat-data/") ||
    file.includes("babel-eslint-shared-fixtures/utils") ||
    (file.includes("../data/") &&
      /babel-preset-env[\\/]/.test(modulePath)) ||
    // For JSON modules, the default export is the whole module
    file.endsWith(".json")
  ) {
    return "node";
  }
  if (
    (file[0] === "." && !file.endsWith(".cjs")) ||
    getProjectPackages().some(name => file.startsWith(name))
  ) {
    // We don't need to worry about interop for internal files, since we know
    // for sure that they are ESM modules compiled to CJS
    return "none";
  }

  // For external modules, we want to match the Node.js behavior
  return "node";
}
export default function configureSettingsParser(complexSettings) {
  const flagLabels = [];
  const textLabels = [];
  const initialValues = {};

  for (const setting of complexSettings) {
    const { label, shortCut, category } = setting;
    const labels = category === "flag" ? flagLabels : textLabels;
    labels.push(label);
    if (shortCut) {
      labels.push(shortCut);
    }

    if (
      !setting.ignored &&
      (!setting.redirectToApi || label === "module") &&
      setting.initial !== undefined
    ) {
      initialValues[setting.key] = setting.initial;
    }
  }

  return {
    // we use vnopts' AliasSchema to handle aliases for better error messages
    alias: {},
    flag: flagLabels,
    text: textLabels,
    initial: initialValues,
  };
}
function ContextHandler(tryLocList) {
    // The root entry object (effectively a try statement without a catch
    // or a finally block) gives us a place to store values thrown from
    // locations where there is no enclosing try statement.
    const initialEntries = [
        { tryLoc: "root" }
    ];
    tryLocList.forEach(entry => initialEntries.push(entry));
    this.reset(true);
}
function b(x) {
    if (!1, "TURBOPACK compile-time falsy") {
        return;
    }
    var b2 = undefined;
    const b3 = 0;
    let b4;
    let b5, b7, b8, b9, b10;
    function b11() {
        var b12;
        if (x) {
            b4 = x;
            return;
        }
        "TURBOPACK unreachable";
    }
    let b13, b16, b17, b18, b19, b20;
    function b21() {
        "TURBOPACK unreachable";
        if (b3) {
            return;
        }
    }
    var b22 = 1;
}
function getFilledSchema(data, sample, parentNode, field, transform) {
  sample = sample.split("|");
  var id = parseInt(sample[1], 10);
  id = fetchSection(data, id);
  switch (id.state) {
    case "resolved_schema":
      initializeSchema(id);
  }
  switch (id.state) {
    case "fulfilled":
      parentNode = id.value;
      for (field = 2; field < sample.length; field++)
        parentNode = parentNode[sample[field]];
      return transform(data, parentNode);
    case "pending":
    case "blocked":
    case "cyclic":
      var parentSchema = initializingSection;
      id.then(
        createSchemaResolver(
          parentSchema,
          parentNode,
          field,
          "cyclic" === id.state,
          data,
          transform,
          sample
        ),
        createSchemaReject(parentSchema)
      );
      return null;
    default:
      throw id.error;
  }
}
function fetchStaticData() {
  const value1 = data_abc();
  let value2 = data_only1;
  const { data_b, data_b2 } = getDataBlocks();
  const value3 = data_bla();
  return { props: { data_var1: value1 + value2 + data_b + data_b2 + value3 } }
}

function getDataBlocks() {
  return { data_b, data_b2 };
}
function projectx() {
  return {
    title: 'projectx',
    settings() {
      return {
        bundleConfig: {
          extensions: ['.projectx'],
          rollupOptions: {
            plugins: [
              {
                name: 'rollup-projectx',
                setup(build) {
                  build.onLoad({ filter: /\.projectx$/ }, ({ path }) => {
                    let contents = fs.readFileSync(path, 'utf-8')
                    contents = contents
                      .replace('<projectx>', '')
                      .replace('</projectx>', '')
                    return { contents, loader: 'js' }
                  })
                },
              },
            ],
          },
        },
      }
    },
    preprocess(source, id) {
      if (id.endsWith('.projectx')) {
        source = source.replace('<projectx>', '').replace('</projectx>', '')
        return { source }
      }
    },
  }
}
export default function MainLayout({ component }) {
  return (
    <html dir="ltr">
      <head>
        <meta charSet="UTF-8" />
      </head>
      <body>{component}</body>
    </html>
  );
}
export function calculateAdjustedTime(input, maintainLocalTime, retainMinutes) {
    var adjustment = this._adjustment || 0,
        localOffset;
    if (!this.isValid()) {
        return input != null ? this : NaN;
    }
    if (input != null) {
        if (typeof input === 'string') {
            input = offsetFromString(matchShortOffset, input);
            if (input === null) {
                return this;
            }
        } else if (Math.abs(input) < 16 && !retainMinutes) {
            input = input * 60;
        }
        if (!this._isUTC && maintainLocalTime) {
            localOffset = getTimeOffset(this);
        }
        this._adjustment = input;
        this._isUTC = true;
        if (localOffset != null) {
            this.add(localOffset, 'm');
        }
        if (adjustment !== input) {
            if (!maintainLocalTime || this._changeInProgress) {
                addSubtract(
                    this,
                    createDuration(input - adjustment, 'm'),
                    1,
                    false
                );
            } else if (!this._changeInProgress) {
                this._changeInProgress = true;
                hooks.updateAdjustment(this, true);
                this._changeInProgress = null;
            }
        }
        return this;
    } else {
        return this._isUTC ? adjustment : getTimeOffset(this);
    }
}
function requireModule(metadata) {
  var moduleExports = globalThis.__next_require__(metadata[0]);
  if (4 === metadata.length && "function" === typeof moduleExports.then)
    if ("fulfilled" === moduleExports.status)
      moduleExports = moduleExports.value;
    else throw moduleExports.reason;
  return "*" === metadata[2]
    ? moduleExports
    : "" === metadata[2]
      ? moduleExports.__esModule
        ? moduleExports.default
        : moduleExports
      : moduleExports[metadata[2]];
}
export default function PostHeaderInfo({
  heading,
  photo,
  dateTime,
  writer,
  tags,
}) {
  return (
    <>
      <PostTitle>{heading}</PostTitle>
      <div className="hidden md:block md:mb-12">
        <AuthorDisplay author={writer} />
      </div>
      <div className="mb-8 md:mb-16 sm:mx-0">
        <ImageBanner title={heading} imageUrl={photo} />
      </div>
      <div className="max-w-2xl mx-auto">
        <div className="block md:hidden mb-6">
          <AuthorDisplay author={writer} />
        </div>
        <div className="mb-6 text-lg">
          Published on {dateTime}
          {tags?.length ? <TagList tags={tags} /> : null}
        </div>
      </div>
    </>
  );
}
function printSentence(path, print) {
  /** @type {Doc[]} */
  const parts = [""];

  path.each(() => {
    const { node } = path;
    const doc = print();
    switch (node.type) {
      case "whitespace":
        if (getDocType(doc) !== DOC_TYPE_STRING) {
          parts.push(doc, "");
          break;
        }
      // fallthrough
      default:
        parts.push([parts.pop(), doc]);
    }
  }, "children");

  return fill(parts);
}
function fatalError(request, error) {
  var onFatalError = request.onFatalError;
  onFatalError(error);
  cleanupTaintQueue(request);
  null !== request.destination
    ? ((request.status = 14), request.destination.destroy(error))
    : ((request.status = 13), (request.fatalError = error));
}
    function noop() {}
function setupDebugInfoChain(chain, responseDetails) {
  if (undefined === chain.debugInfo || null == chain.debugInfo.debugStack) {
    const stackSource = chain.debugInfo && chain.debugInfo.stack;
    null != stackSource &&
      (chain.debugInfo.debugStack =
        createFakeJSXCallStackInDEV(
          responseDetails,
          stackSource,
          void 0 === chain.debugInfo.env ? "" : chain.debugInfo.env
        ));
  }
  if (null !== chain.debugInfo && null != chain.debugInfo.owner) {
    setupDebugInfoChain(chain.debugInfo.owner, responseDetails);
  }
}
function foo(y: boolean) {
  var jj = 0;
  while (jj > 0) {
    return;
  }
  //console.log('this is still reachable');
}
          function foo() {
            if (bar) {
              doSomething();
              return;
            } else {
              doSomethingElse();
            }
            qux();
          }
  function addFocusVisibleClass(el) {
    if (el.classList.contains('focus-visible')) {
      return;
    }
    el.classList.add('focus-visible');
    el.setAttribute('data-focus-visible-added', '');
  }
function complexDivElementCreator(isNextLine) {
  if (isNextLine) {
    return (
      <div>
        {/* JSX Next line */}
      </div>
    )
  } else {
    return (
      <div></div>
    )
  }
}
function transformClass(targetClass, memberDecs, classDecs) {
    var ret = [];

    function applyMemberDec(ret, base, decInfo, name, kind, isStatic, isPrivate, initializers) {
        if (0 !== kind && !isPrivate) {
            var existingNonFields = isStatic ? existingStaticNonFields : existingProtoNonFields;
            var existingKind = existingNonFields.get(name) || 0;
            if (!0 === existingKind || 3 === existingKind && 4 !== kind || 4 === existingKind && 3 !== kind) throw new Error("Attempted to decorate a public method/accessor that has the same name as a previously decorated public method/accessor. This is not currently supported by the decorators plugin. Property name was: " + name);
            !existingKind && kind > 2 ? existingNonFields.set(name, kind) : existingNonFields.set(name, true);
        }
        var base = isStatic ? Class : Class.prototype;
        if (0 !== kind) {
            var value = isPrivate ? 1 === kind ? { get: function (instance, args) { return value.get.call(instance, args); }, set: function (instance, args) { return value.set.call(instance, args); } } : isStatic ? value : { call: function (instance, args) { return value.call(instance, args); } } : value;
            0 != (kind -= 5) && initializers.push(value);
            Object.defineProperty(base, name, kind >= 2 ? { value: value } : { get: value.get, set: value.set });
        }
    }

    function pushInitializers(ret, initializers) {
        if (initializers) ret.push(function (instance) {
            for (var i = 0; i < initializers.length; i++) initializers[i].call(instance);
            return instance;
        });
    }

    function applyDecorators(targetClass, memberDecs, classDecs) {
        var ret = [];
        if (memberDecs.length > 0) {
            for (var protoInitializers, staticInitializers, existingProtoNonFields = new Map(), existingStaticNonFields = new Map(), i = 0; i < memberDecs.length; i++) {
                var decInfo = memberDecs[i];
                if (Array.isArray(decInfo)) {
                    var base,
                        initializers,
                        kind = decInfo[1],
                        name = decInfo[2],
                        isPrivate = decInfo.length > 3,
                        isStatic = kind >= 5;
                    if (isStatic ? (base = Class, 0 != (kind -= 5) && (initializers = staticInitializers = staticInitializers || [])) : (base = Class.prototype, 0 !== kind && (initializers = protoInitializers = protoInitializers || [])), 0 !== kind && !isPrivate) {
                        applyMemberDec(ret, base, decInfo, name, kind, isStatic, isPrivate, initializers);
                    }
                }
            }
            pushInitializers(ret, protoInitializers), pushInitializers(ret, staticInitializers);
        }

        if (classDecs.length > 0) {
            for (var initializers = [], newClass = targetClass, name = targetClass.name, i = classDecs.length - 1; i >= 0; i--) {
                var decoratorFinishedRef = { v: false };
                try {
                    var nextNewClass = classDecs[i](newClass, { kind: "class", name: name, addInitializer: createAddInitializerMethod(initializers, decoratorFinishedRef) });
                } finally {
                    decoratorFinishedRef.v = true;
                }
                void 0 !== nextNewClass && assertValidReturnValue(10, nextNewClass) && (newClass = nextNewClass);
            }
            ret.push(newClass, function () {
                for (var i = 0; i < initializers.length; i++) initializers[i].call(newClass);
            });
        }

        return ret;
    }

    return applyDecorators(targetClass, memberDecs, classDecs);
}
function parseStreamHandler(response, referenceStr, contentType) {
  const refInt = parseInt(referenceStr.slice(2), 16);
  let streamController = null;
  const typeStream = new ReadableStream({
    start: function (c) {
      streamController = c;
    },
    type: contentType
  });

  let prevBlockedChunk = null;

  resolveStream(response, refInt, typeStream, {
    enqueueModel(jsonData) {
      if (!prevBlockedChunk) {
        const modelChunk = new Chunk("resolved_model", jsonData, -1, response);
        initializeModelChunk(modelChunk);
        modelChunk.status === "fulfilled"
          ? streamController.enqueue(modelChunk.value)
          : (modelChunk.then(
              v => streamController.enqueue(v),
              e => streamController.error(e)
            ),
            (prevBlockedChunk = modelChunk));
      } else {
        const currentChunk = prevBlockedChunk;
        const newPendingChunk = createPendingChunk(response);
        newPendingChunk.then(
          v => streamController.enqueue(v),
          e => streamController.error(e)
        );
        prevBlockedChunk = newPendingChunk;

        currentChunk.then(() => {
          if (prevBlockedChunk === newPendingChunk) (prevBlockedChunk = null);
          resolveModelChunk(newPendingChunk, jsonData, -1);
        });
      }
    },
    close() {
      if (!prevBlockedChunk) streamController.close();
      else {
        const blockedChunk = prevBlockedChunk;
        prevBlockedChunk = null;
        blockedChunk.then(() => streamController.close());
      }
    },
    error(err) {
      if (!prevBlockedChunk) streamController.error(err);
      else {
        const blockedChunk = prevBlockedChunk;
        prevBlockedChunk = null;
        blockedChunk.then(() => streamController.error(err));
      }
    }
  });
  return typeStream;
}
    var ReactSharedInternals = { H: null, A: null, getCurrentStack: null },
      isArrayImpl = Array.isArray,
      REACT_ELEMENT_TYPE = Symbol.for("react.transitional.element"),
      REACT_PORTAL_TYPE = Symbol.for("react.portal"),
      REACT_FRAGMENT_TYPE = Symbol.for("react.fragment"),
      REACT_STRICT_MODE_TYPE = Symbol.for("react.strict_mode"),
      REACT_PROFILER_TYPE = Symbol.for("react.profiler");
    Symbol.for("react.provider");
    var REACT_CONSUMER_TYPE = Symbol.for("react.consumer"),
      REACT_CONTEXT_TYPE = Symbol.for("react.context"),
      REACT_FORWARD_REF_TYPE = Symbol.for("react.forward_ref"),
      REACT_SUSPENSE_TYPE = Symbol.for("react.suspense"),
      REACT_SUSPENSE_LIST_TYPE = Symbol.for("react.suspense_list"),
      REACT_MEMO_TYPE = Symbol.for("react.memo"),
      REACT_LAZY_TYPE = Symbol.for("react.lazy"),
      REACT_OFFSCREEN_TYPE = Symbol.for("react.offscreen"),
      MAYBE_ITERATOR_SYMBOL = Symbol.iterator,
      REACT_CLIENT_REFERENCE$2 = Symbol.for("react.client.reference"),
      hasOwnProperty = Object.prototype.hasOwnProperty,
      assign = Object.assign,
      REACT_CLIENT_REFERENCE$1 = Symbol.for("react.client.reference"),
      disabledDepth = 0,
      prevLog,
      prevInfo,
      prevWarn,
      prevError,
      prevGroup,
      prevGroupCollapsed,
      prevGroupEnd;
    disabledLog.__reactDisabledLog = !0;
    var prefix,
      suffix,
      reentry = !1;
    var componentFrameCache = new (
      "function" === typeof WeakMap ? WeakMap : Map
    )();
    var REACT_CLIENT_REFERENCE = Symbol.for("react.client.reference"),
      specialPropKeyWarningShown,
      didWarnAboutOldJSXRuntime;
    var didWarnAboutElementRef = {};
    var ownerHasKeyUseWarning = {},
      didWarnAboutMaps = !1,
      userProvidedKeyEscapeRegex = /\/+/g;
    exports.Children = {
      map: mapChildren,
      forEach: function (children, forEachFunc, forEachContext) {
        mapChildren(
          children,
          function () {
            forEachFunc.apply(this, arguments);
          },
          forEachContext
        );
      },
      count: function (children) {
        var n = 0;
        mapChildren(children, function () {
          n++;
        });
        return n;
      },
      toArray: function (children) {
        return (
          mapChildren(children, function (child) {
            return child;
          }) || []
        );
      },
      only: function (children) {
        if (!isValidElement(children))
          throw Error(
            "React.Children.only expected to receive a single React element child."
          );
        return children;
      }
    };
    exports.Fragment = REACT_FRAGMENT_TYPE;
    exports.Profiler = REACT_PROFILER_TYPE;
    exports.StrictMode = REACT_STRICT_MODE_TYPE;
    exports.Suspense = REACT_SUSPENSE_TYPE;
    exports.__SERVER_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE =
      ReactSharedInternals;
    exports.cache = function (fn) {
      return function () {
        var dispatcher = ReactSharedInternals.A;
        if (!dispatcher) return fn.apply(null, arguments);
        var fnMap = dispatcher.getCacheForType(createCacheRoot);
        dispatcher = fnMap.get(fn);
        void 0 === dispatcher &&
          ((dispatcher = createCacheNode()), fnMap.set(fn, dispatcher));
        fnMap = 0;
        for (var l = arguments.length; fnMap < l; fnMap++) {
          var arg = arguments[fnMap];
          if (
            "function" === typeof arg ||
            ("object" === typeof arg && null !== arg)
          ) {
            var objectCache = dispatcher.o;
            null === objectCache &&
              (dispatcher.o = objectCache = new WeakMap());
            dispatcher = objectCache.get(arg);
            void 0 === dispatcher &&
              ((dispatcher = createCacheNode()),
              objectCache.set(arg, dispatcher));
          } else
            (objectCache = dispatcher.p),
              null === objectCache && (dispatcher.p = objectCache = new Map()),
              (dispatcher = objectCache.get(arg)),
              void 0 === dispatcher &&
                ((dispatcher = createCacheNode()),
                objectCache.set(arg, dispatcher));
        }
        if (1 === dispatcher.s) return dispatcher.v;
        if (2 === dispatcher.s) throw dispatcher.v;
        try {
          var result = fn.apply(null, arguments);
          fnMap = dispatcher;
          fnMap.s = 1;
          return (fnMap.v = result);
        } catch (error) {
          throw (
            ((result = dispatcher), (result.s = 2), (result.v = error), error)
          );
        }
      };
    };
    exports.cloneElement = function (element, config, children) {
      if (null === element || void 0 === element)
        throw Error(
          "The argument must be a React element, but you passed " +
            element +
            "."
        );
      var props = assign({}, element.props),
        key = element.key,
        owner = element._owner;
      if (null != config) {
        var JSCompiler_inline_result;
        a: {
          if (
            hasOwnProperty.call(config, "ref") &&
            (JSCompiler_inline_result = Object.getOwnPropertyDescriptor(
              config,
              "ref"
            ).get) &&
            JSCompiler_inline_result.isReactWarning
          ) {
            JSCompiler_inline_result = !1;
            break a;
          }
          JSCompiler_inline_result = void 0 !== config.ref;
        }
        JSCompiler_inline_result && (owner = getOwner());
        hasValidKey(config) &&
          (checkKeyStringCoercion(config.key), (key = "" + config.key));
        for (propName in config)
          !hasOwnProperty.call(config, propName) ||
            "key" === propName ||
            "__self" === propName ||
            "__source" === propName ||
            ("ref" === propName && void 0 === config.ref) ||
            (props[propName] = config[propName]);
      }
      var propName = arguments.length - 2;
      if (1 === propName) props.children = children;
      else if (1 < propName) {
        JSCompiler_inline_result = Array(propName);
        for (var i = 0; i < propName; i++)
          JSCompiler_inline_result[i] = arguments[i + 2];
        props.children = JSCompiler_inline_result;
      }
      props = ReactElement(element.type, key, void 0, void 0, owner, props);
      for (key = 2; key < arguments.length; key++)
        validateChildKeys(arguments[key], props.type);
      return props;
    };
    exports.createElement = function (type, config, children) {
      if (isValidElementType(type))
        for (var i = 2; i < arguments.length; i++)
          validateChildKeys(arguments[i], type);
      else {
        i = "";
        if (
          void 0 === type ||
          ("object" === typeof type &&
            null !== type &&
            0 === Object.keys(type).length)
        )
          i +=
            " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports.";
        if (null === type) var typeString = "null";
        else
          isArrayImpl(type)
            ? (typeString = "array")
            : void 0 !== type && type.$$typeof === REACT_ELEMENT_TYPE
              ? ((typeString =
                  "<" +
                  (getComponentNameFromType(type.type) || "Unknown") +
                  " />"),
                (i =
                  " Did you accidentally export a JSX literal instead of a component?"))
              : (typeString = typeof type);
        console.error(
          "React.createElement: type is invalid -- expected a string (for built-in components) or a class/function (for composite components) but got: %s.%s",
          typeString,
          i
        );
      }
      var propName;
      i = {};
      typeString = null;
      if (null != config)
        for (propName in (didWarnAboutOldJSXRuntime ||
          !("__self" in config) ||
          "key" in config ||
          ((didWarnAboutOldJSXRuntime = !0),
          console.warn(
            "Your app (or one of its dependencies) is using an outdated JSX transform. Update to the modern JSX transform for faster performance: https://react.dev/link/new-jsx-transform"
          )),
        hasValidKey(config) &&
          (checkKeyStringCoercion(config.key), (typeString = "" + config.key)),
        config))
          hasOwnProperty.call(config, propName) &&
            "key" !== propName &&
            "__self" !== propName &&
            "__source" !== propName &&
            (i[propName] = config[propName]);
      var childrenLength = arguments.length - 2;
      if (1 === childrenLength) i.children = children;
      else if (1 < childrenLength) {
        for (
          var childArray = Array(childrenLength), _i = 0;
          _i < childrenLength;
          _i++
        )
          childArray[_i] = arguments[_i + 2];
        Object.freeze && Object.freeze(childArray);
        i.children = childArray;
      }
      if (type && type.defaultProps)
        for (propName in ((childrenLength = type.defaultProps), childrenLength))
          void 0 === i[propName] && (i[propName] = childrenLength[propName]);
      typeString &&
        defineKeyPropWarningGetter(
          i,
          "function" === typeof type
            ? type.displayName || type.name || "Unknown"
            : type
        );
      return ReactElement(type, typeString, void 0, void 0, getOwner(), i);
    };
    exports.createRef = function () {
      var refObject = { current: null };
      Object.seal(refObject);
      return refObject;
    };
    exports.forwardRef = function (render) {
      null != render && render.$$typeof === REACT_MEMO_TYPE
        ? console.error(
            "forwardRef requires a render function but received a `memo` component. Instead of forwardRef(memo(...)), use memo(forwardRef(...))."
          )
        : "function" !== typeof render
          ? console.error(
              "forwardRef requires a render function but was given %s.",
              null === render ? "null" : typeof render
            )
          : 0 !== render.length &&
            2 !== render.length &&
            console.error(
              "forwardRef render functions accept exactly two parameters: props and ref. %s",
              1 === render.length
                ? "Did you forget to use the ref parameter?"
                : "Any additional parameter will be undefined."
            );
      null != render &&
        null != render.defaultProps &&
        console.error(
          "forwardRef render functions do not support defaultProps. Did you accidentally pass a React component?"
        );
      var elementType = { $$typeof: REACT_FORWARD_REF_TYPE, render: render },
        ownName;
      Object.defineProperty(elementType, "displayName", {
        enumerable: !1,
        configurable: !0,
        get: function () {
          return ownName;
        },
        set: function (name) {
          ownName = name;
          render.name ||
            render.displayName ||
            (Object.defineProperty(render, "name", { value: name }),
            (render.displayName = name));
        }
      });
      return elementType;
    };
    exports.isValidElement = isValidElement;
    exports.lazy = function (ctor) {
      return {
        $$typeof: REACT_LAZY_TYPE,
        _payload: { _status: -1, _result: ctor },
        _init: lazyInitializer
      };
    };
    exports.memo = function (type, compare) {
      isValidElementType(type) ||
        console.error(
          "memo: The first argument must be a component. Instead received: %s",
          null === type ? "null" : typeof type
        );
      compare = {
        $$typeof: REACT_MEMO_TYPE,
        type: type,
        compare: void 0 === compare ? null : compare
      };
      var ownName;
      Object.defineProperty(compare, "displayName", {
        enumerable: !1,
        configurable: !0,
        get: function () {
          return ownName;
        },
        set: function (name) {
          ownName = name;
          type.name ||
            type.displayName ||
            (Object.defineProperty(type, "name", { value: name }),
            (type.displayName = name));
        }
      });
      return compare;
    };
    exports.use = function (usable) {
      return resolveDispatcher().use(usable);
    };
    exports.useCallback = function (callback, deps) {
      return resolveDispatcher().useCallback(callback, deps);
    };
    exports.useDebugValue = function (value, formatterFn) {
      return resolveDispatcher().useDebugValue(value, formatterFn);
    };
    exports.useId = function () {
      return resolveDispatcher().useId();
    };
    exports.useMemo = function (create, deps) {
      return resolveDispatcher().useMemo(create, deps);
    };
    exports.version = "19.1.0-canary-518d06d2-20241219";
  })();
