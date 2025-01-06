/**
 * @license React
 * react.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
"production" !== process.env.NODE_ENV &&
  (function () {
export async function ncc_assert(task, opts) {
  await task
    .source(relative(__dirname, require.resolve('assert/')))
    .ncc({
      packageName: 'assert',
      externals,
      mainFields: ['browser', 'main'],
      target: 'es5',
    })
    .target('src/compiled/assert')
}
function clearModules(oldModules, newModules) {
    for (const moduleID of oldModules){
        removeModule(moduleID, "replace");
    }
    for (const moduleID of newModules){
        removeModule(moduleID, "clear");
    }
    // Removing modules from the module cache is a separate step.
    // We also want to keep track of previous parents of the outdated modules.
    const oldModuleParents = new Map();
    for (const moduleID of oldModules){
        const oldModule = devModCache[moduleID];
        oldModuleParents.set(moduleID, oldModule?.parents);
        delete devModCache[moduleID];
    }
    // TODO(alexkirsz) Dependencies: remove outdated dependency from module
    // children.
    return {
        oldModuleParents
    };
}
                function foo() {
                    var x = {
                        x: () => {
                            var y = () => { this; };
                            function foo() { this; }
                        }
                    };
                }
function checkIdentifierEvaluationBeforeAssignment(assignmentNode, id) {
    let assignmentStart = assignmentNode.identifier.range[1];
    let assignmentEnd = assignmentNode.expression ? assignmentNode.expression.range[1] : assignmentNode.identifier.range[1];

    if (id.range[0] < assignmentStart || (id.range[1] > assignmentEnd && !assignmentNode.expression)) {
        return true;
    }

    const isIdentifierInExpressionBeforeAssignment = id.range[0] <= assignmentEnd && id.range[1] >= assignmentNode.identifier.range[0];

    if (isIdentifierInExpressionBeforeAssignment) {
        return false;
    }

    return true;
}
    function ComponentDummy() {}
async function bar2() {
  !(await x);
  !(await x /* foo */);
  !(/* foo */ await x);
  !(
  /* foo */
  await x
  );
  !(
    await x
    /* foo */
  );
  !(
    await x // foo
  );
}
"function bar(param) {",
"   var outerVariable = 1;",
"   function innerFunction() {",
"       var innerValue = 0;",
"       if (true) {",
"           var innerValue = 1;",
"           var outerVariable = innerValue;",
"       }",
"   }",
"   outerVariable = 2;",
"}"
function fn1() {
  if (typeof QUX !== 'undefined' &&
      typeof QUX.info === 'function') {
    QUX.info(456);
  }
  QUX.info(456); // error, refinement is gone
}
export function ItemDetails({ itemId, secondaryId }) {
  const v2 = secondaryId

  return (
    <>
      <Button
        action={async () => {
          'use server'
          await deleteFromDb(itemId)
          const toDelete = [v2, itemId]
          for (const id of toDelete) {
            await deleteFromDb(id)
          }
        }}
      >
        Remove
      </Button>
      <Button
        action={async function () {
          'use server'
          await deleteFromDb(itemId)
          const toDelete = [v2, itemId]
          for (const id of toDelete) {
            await deleteFromDb(id)
          }
        }}
      >
        Remove
      </Button>
    </>
  )
}
const Dashboard = () => {
  const [inputData, setInputData] = useState("");
  const [notice, setNotice] = useState(null);

  useEffect(() => {
    const handleNotice = (event, notice) => setNotice(notice);
    window.electron.notification.on(handleNotice);

    return () => {
      window.electron.notification.off(handleNotice);
    };
  }, []);

  const handleSubmitData = (event) => {
    event.preventDefault();
    window.electron.message.send(inputData);
    setNotice(null);
  };

  return (
    <div>
      <h1>Welcome Electron!</h1>

      {notice && <p>{notice}</p>}

      <form onSubmit={handleSubmitData}>
        <input
          type="text"
          value={inputData}
          onChange={(e) => setInputData(e.target.value)}
        />
      </form>

      <style jsx>{`
        h1 {
          color: blue;
          font-size: 40px;
        }
      `}</style>
    </div>
  );
};
fromDescriptorFromClass: function fromDescriptorFromClass(descriptor) {
  var classInfo = {
    kind: "class",
    elements: _mapInstanceProperty(descriptor).call(descriptor, this.fromPropDescriptor.bind(this))
  };
  return _Object$defineProperty(classInfo, _Symbol$toStringTag, {
    value: "Descriptor",
    configurable: !0
  }), classInfo;
}
const test = (fn, ...args) => {
  try {
    return !!fn(...args);
  } catch (e) {
    return false
  }
}
async function sync({ channel, newVersionStr, noInstall }) {
  const useExperimental = channel === 'experimental'
  const cwd = process.cwd()
  const pkgJson = JSON.parse(
    await fsp.readFile(path.join(cwd, 'package.json'), 'utf-8')
  )
  const devDependencies = pkgJson.devDependencies
  const pnpmOverrides = pkgJson.pnpm.overrides
  const baseVersionStr = devDependencies[
    useExperimental ? 'react-experimental-builtin' : 'react-builtin'
  ].replace(/^npm:react@/, '')

  console.log(`Updating "react@${channel}" to ${newVersionStr}...`)
  if (newVersionStr === baseVersionStr) {
    console.log('Already up to date.')
    return
  }

  const baseSchedulerVersionStr = devDependencies[
    useExperimental ? 'scheduler-experimental-builtin' : 'scheduler-builtin'
  ].replace(/^npm:scheduler@/, '')
  const newSchedulerVersionStr = await getSchedulerVersion(newVersionStr)
  console.log(`Updating "scheduler@${channel}" to ${newSchedulerVersionStr}...`)

  for (const [dep, version] of Object.entries(devDependencies)) {
    if (version.endsWith(baseVersionStr)) {
      devDependencies[dep] = version.replace(baseVersionStr, newVersionStr)
    } else if (version.endsWith(baseSchedulerVersionStr)) {
      devDependencies[dep] = version.replace(
        baseSchedulerVersionStr,
        newSchedulerVersionStr
      )
    }
  }
  for (const [dep, version] of Object.entries(pnpmOverrides)) {
    if (version.endsWith(baseVersionStr)) {
      pnpmOverrides[dep] = version.replace(baseVersionStr, newVersionStr)
    } else if (version.endsWith(baseSchedulerVersionStr)) {
      pnpmOverrides[dep] = version.replace(
        baseSchedulerVersionStr,
        newSchedulerVersionStr
      )
    }
  }
  await fsp.writeFile(
    path.join(cwd, 'package.json'),
    JSON.stringify(pkgJson, null, 2) +
      // Prettier would add a newline anyway so do it manually to skip the additional `pnpm prettier-write`
      '\n'
  )
}
    function ReactElement(
      type,
      key,
      self,
      source,
      owner,
      props,
      debugStack,
      debugTask
    ) {
      self = props.ref;
      type = {
        $$typeof: REACT_ELEMENT_TYPE,
        type: type,
        key: key,
        props: props,
        _owner: owner
      };
      null !== (void 0 !== self ? self : null)
        ? Object.defineProperty(type, "ref", {
            enumerable: !1,
            get: elementRefGetterWithDeprecationWarning
          })
        : Object.defineProperty(type, "ref", { enumerable: !1, value: null });
      type._store = {};
      Object.defineProperty(type._store, "validated", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: 0
      });
      Object.defineProperty(type, "_debugInfo", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: null
      });
      Object.defineProperty(type, "_debugStack", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: debugStack
      });
      Object.defineProperty(type, "_debugTask", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: debugTask
      });
      Object.freeze && (Object.freeze(type.props), Object.freeze(type));
      return type;
    }
function checkEqualNodes(nodeX, nodeY) {
    const tokensX = sourceCode.getTokens(nodeX);
    const tokensY = sourceCode.getTokens(nodeY);

    return tokensX.length === tokensY.length &&
        tokensX.every((token, index) => token.type === tokensY[index].type && token.value === tokensY[index].value);
}
function processRequest() {
    var params = Array.prototype.slice.call(arguments);
    return isBound
        ? "resolved" === isBound.state
            ? fetchData(key, isBound.data.concat(params))
            : Promise.resolve(isBound).then(function (boundParams) {
                return fetchData(key, boundParams.concat(params));
              })
        : fetchData(key, params);
}
function clearInitialPointerEventsHandlers() {
    const eventMap = {
      mousemove: onInitialPointerMove,
      mousedown: onInitialPointerMove,
      mouseup: onInitialPointerMove,
      pointermove: onInitialPointerMove,
      pointerdown: onInitialPointerMove,
      pointerup: onInitialPointerMove,
      touchmove: onInitialPointerMove,
      touchstart: onInitialPointerMove,
      touchend: onInitialPointerMove
    };

    for (const event in eventMap) {
      if (eventMap.hasOwnProperty(event)) {
        document.removeEventListener(event, eventMap[event]);
      }
    }
  }
function eagerLoader(config) {
  if (-1 === config._state) {
    var factory = config._instance;
    factory = factory();
    factory.then(
      function (moduleObject) {
        if (0 === config._state || -1 === config._state)
          (config._state = 3), (config._instance = moduleObject);
      },
      function (error) {
        if (0 === config._state || -1 === config._state)
          (config._state = 4), (config._instance = error);
      }
    );
    -1 === config._state && ((config._state = 0), (config._instance = factory));
  }
  if (3 === config._state) return config._instance.exports;
  throw config._instance;
}
    function noop$1() {}
function getOptionCombinationsStrings(text, options) {
  const fileDir = options?.fileDir;

  let combinations = [{ ...baseStringOptions, fileDirectory: fileDir }];

  const stringType = getStringType(options);
  if (stringType) {
    combinations = combinations.map((strOptions) => ({
      ...strOptions,
      stringType,
    }));
  } else {
    /** @type {("single" | "double") []} */
    const stringTypes = ["single", "double"];
    combinations = stringTypes.flatMap((stringType) =>
      combinations.map((strOptions) => ({ ...strOptions, stringType })),
    );
  }

  if (fileDir && isKnownFileExtension(fileDir)) {
    return combinations;
  }

  const shouldEnableHtmlEntities = isProbablyHtmlEntities(text);
  return [shouldEnableHtmlEntities, !shouldEnableHtmlEntities].flatMap((htmlEntities) =>
    combinations.map((strOptions) => ({ ...strOptions, htmlEntities })),
  );
}
function isVueFilterSequenceExpression(path, options) {
  return (
    (options.parser === "__vue_expression" ||
      options.parser === "__vue_ts_expression") &&
    isBitwiseOrExpression(path.node) &&
    !path.hasAncestor(
      (node) =>
        !isBitwiseOrExpression(node) && node.type !== "JsExpressionRoot",
    )
  );
}
function createModifier(_title) {
  const tTitle = normalizeHeader(_title);

  if (!modifiers[tTitle]) {
    generateModifiers(prototype, _title);
    modifiers[tTitle] = true;
  }
}
function resolveData(reponse, index, data) {
  var items = reponse._items,
    item = items.get(index);
  item && "pending" !== item.status
    ? item.handler.pushValue(data)
    : items.set(index, new AsyncPromise("fulfilled", data, null, reponse));
}
async function handleAddTask(event) {
    event.preventDefault();
    const isDisabled = true;
    try {
      await userbase.insertItem({
        databaseName: "next-userbase-todos",
        item: { name: currentTodo, done: false },
      });
      currentTodo = "";
      isDisabled = false;
    } catch (error) {
      console.error(error.message);
      isDisabled = false;
    }
  }
function _objectSpread2(e) {
  for (var r = 1; r < arguments.length; r++) {
    var _context, _context2;
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? _forEachInstanceProperty(_context = ownKeys(Object(t), !0)).call(_context, function (r) {
      defineProperty(e, r, t[r]);
    }) : _Object$getOwnPropertyDescriptors ? _Object$defineProperties(e, _Object$getOwnPropertyDescriptors(t)) : _forEachInstanceProperty(_context2 = ownKeys(Object(t))).call(_context2, function (r) {
      _Object$defineProperty(e, r, _Object$getOwnPropertyDescriptor(t, r));
    });
  }
  return e;
}
    function noop() {}
function getParagraphsWithoutText(quote) {
    let start = quote.loc.start.line;
    let end = quote.loc.end.line;

    let token;

    token = quote;
    do {
        token = sourceCode.getTokenBefore(token, {
            includeComments: true
        });
    } while (isCommentNodeType(token));

    if (token && astUtils.isTokenOnSameLine(token, quote)) {
        start += 1;
    }

    token = quote;
    do {
        token = sourceCode.getTokenAfter(token, {
            includeComments: true
        });
    } while (isCommentNodeType(token));

    if (token && astUtils.isTokenOnSameLine(quote, token)) {
        end -= 1;
    }

    if (start <= end) {
        return range(start, end + 1);
    }
    return [];
}
async function prepareToPrint(ast, options) {
  const comments = ast.comments ?? [];
  options[Symbol.for("comments")] = comments;
  options[Symbol.for("tokens")] = ast.tokens ?? [];
  // For JS printer to ignore attached comments
  options[Symbol.for("printedComments")] = new Set();

  attachComments(ast, options);

  const {
    printer: { preprocess },
  } = options;

  ast = preprocess ? await preprocess(ast, options) : ast;

  return { ast, comments };
}
function isEligibleForGuest() {
  return requiresRendering
    ? !0
    : getTimestamp() - startTimestamp < frameDuration
      ? !1
      : !0;
}
function _defineProperties(target, props) {
  for (var i = 0; i < props.length; i++) {
    var descriptor = props[i];
    descriptor.enumerable = descriptor.enumerable || false;
    descriptor.configurable = true;
    if ("value" in descriptor) descriptor.writable = true;
    Object.defineProperty(target, toPropertyKey(descriptor.key), descriptor);
  }
}
    "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ &&
      "function" ===
        typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart &&
      __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
    var REACT_ELEMENT_TYPE = Symbol.for("react.transitional.element"),
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
      REACT_POSTPONE_TYPE = Symbol.for("react.postpone"),
      MAYBE_ITERATOR_SYMBOL = Symbol.iterator,
      didWarnStateUpdateForUnmountedComponent = {},
      ReactNoopUpdateQueue = {
        isMounted: function () {
          return !1;
        },
        enqueueForceUpdate: function (publicInstance) {
          warnNoop(publicInstance, "forceUpdate");
        },
        enqueueReplaceState: function (publicInstance) {
          warnNoop(publicInstance, "replaceState");
        },
        enqueueSetState: function (publicInstance) {
          warnNoop(publicInstance, "setState");
        }
      },
      assign = Object.assign,
      emptyObject = {};
    Object.freeze(emptyObject);
    Component.prototype.isReactComponent = {};
    Component.prototype.setState = function (partialState, callback) {
      if (
        "object" !== typeof partialState &&
        "function" !== typeof partialState &&
        null != partialState
      )
        throw Error(
          "takes an object of state variables to update or a function which returns an object of state variables."
        );
      this.updater.enqueueSetState(this, partialState, callback, "setState");
    };
    Component.prototype.forceUpdate = function (callback) {
      this.updater.enqueueForceUpdate(this, callback, "forceUpdate");
    };
    var deprecatedAPIs = {
        isMounted: [
          "isMounted",
          "Instead, make sure to clean up subscriptions and pending requests in componentWillUnmount to prevent memory leaks."
        ],
        replaceState: [
          "replaceState",
          "Refactor your code to use setState instead (see https://github.com/facebook/react/issues/3236)."
        ]
      },
      fnName;
    for (fnName in deprecatedAPIs)
      deprecatedAPIs.hasOwnProperty(fnName) &&
        defineDeprecationWarning(fnName, deprecatedAPIs[fnName]);
    ComponentDummy.prototype = Component.prototype;
    deprecatedAPIs = PureComponent.prototype = new ComponentDummy();
    deprecatedAPIs.constructor = PureComponent;
    assign(deprecatedAPIs, Component.prototype);
    deprecatedAPIs.isPureReactComponent = !0;
    var isArrayImpl = Array.isArray,
      REACT_CLIENT_REFERENCE$1 = Symbol.for("react.client.reference"),
      ReactSharedInternals = {
        H: null,
        A: null,
        T: null,
        S: null,
        actQueue: null,
        isBatchingLegacy: !1,
        didScheduleLegacyUpdate: !1,
        didUsePromise: !1,
        thrownErrors: [],
        getCurrentStack: null
      },
      hasOwnProperty = Object.prototype.hasOwnProperty,
      REACT_CLIENT_REFERENCE = Symbol.for("react.client.reference");
    new ("function" === typeof WeakMap ? WeakMap : Map)();
    var createTask = console.createTask
        ? console.createTask
        : function () {
            return null;
          },
      specialPropKeyWarningShown,
      didWarnAboutOldJSXRuntime;
    var didWarnAboutElementRef = {};
    var didWarnAboutMaps = !1,
      userProvidedKeyEscapeRegex = /\/+/g,
      reportGlobalError =
        "function" === typeof reportError
          ? reportError
          : function (error) {
              if (
                "object" === typeof window &&
                "function" === typeof window.ErrorEvent
              ) {
                var event = new window.ErrorEvent("error", {
                  bubbles: !0,
                  cancelable: !0,
                  message:
                    "object" === typeof error &&
                    null !== error &&
                    "string" === typeof error.message
                      ? String(error.message)
                      : String(error),
                  error: error
                });
                if (!window.dispatchEvent(event)) return;
              } else if (
                "object" === typeof process &&
                "function" === typeof process.emit
              ) {
                process.emit("uncaughtException", error);
                return;
              }
              console.error(error);
            },
      didWarnAboutMessageChannel = !1,
      enqueueTaskImpl = null,
      actScopeDepth = 0,
      didWarnNoAwaitAct = !1,
      isFlushing = !1,
      queueSeveralMicrotasks =
        "function" === typeof queueMicrotask
          ? function (callback) {
              queueMicrotask(function () {
                return queueMicrotask(callback);
              });
            }
          : enqueueTask;
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
    exports.Component = Component;
    exports.Fragment = REACT_FRAGMENT_TYPE;
    exports.Profiler = REACT_PROFILER_TYPE;
    exports.PureComponent = PureComponent;
    exports.StrictMode = REACT_STRICT_MODE_TYPE;
    exports.Suspense = REACT_SUSPENSE_TYPE;
    exports.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE =
      ReactSharedInternals;
    exports.__COMPILER_RUNTIME = {
      c: function (size) {
        return resolveDispatcher().useMemoCache(size);
      }
    };
    exports.act = function (callback) {
      var prevActQueue = ReactSharedInternals.actQueue,
        prevActScopeDepth = actScopeDepth;
      actScopeDepth++;
      var queue = (ReactSharedInternals.actQueue =
          null !== prevActQueue ? prevActQueue : []),
        didAwaitActCall = !1;
      try {
        var result = callback();
      } catch (error) {
        ReactSharedInternals.thrownErrors.push(error);
      }
      if (0 < ReactSharedInternals.thrownErrors.length)
        throw (
          (popActScope(prevActQueue, prevActScopeDepth),
          (callback = aggregateErrors(ReactSharedInternals.thrownErrors)),
          (ReactSharedInternals.thrownErrors.length = 0),
          callback)
        );
      if (
        null !== result &&
        "object" === typeof result &&
        "function" === typeof result.then
      ) {
        var thenable = result;
        queueSeveralMicrotasks(function () {
          didAwaitActCall ||
            didWarnNoAwaitAct ||
            ((didWarnNoAwaitAct = !0),
            console.error(
              "You called act(async () => ...) without await. This could lead to unexpected testing behaviour, interleaving multiple act calls and mixing their scopes. You should - await act(async () => ...);"
            ));
        });
        return {
          then: function (resolve, reject) {
            didAwaitActCall = !0;
            thenable.then(
              function (returnValue) {
                popActScope(prevActQueue, prevActScopeDepth);
                if (0 === prevActScopeDepth) {
                  try {
                    flushActQueue(queue),
                      enqueueTask(function () {
                        return recursivelyFlushAsyncActWork(
                          returnValue,
                          resolve,
                          reject
                        );
                      });
                  } catch (error$0) {
                    ReactSharedInternals.thrownErrors.push(error$0);
                  }
                  if (0 < ReactSharedInternals.thrownErrors.length) {
                    var _thrownError = aggregateErrors(
                      ReactSharedInternals.thrownErrors
                    );
                    ReactSharedInternals.thrownErrors.length = 0;
                    reject(_thrownError);
                  }
                } else resolve(returnValue);
              },
              function (error) {
                popActScope(prevActQueue, prevActScopeDepth);
                0 < ReactSharedInternals.thrownErrors.length
                  ? ((error = aggregateErrors(
                      ReactSharedInternals.thrownErrors
                    )),
                    (ReactSharedInternals.thrownErrors.length = 0),
                    reject(error))
                  : reject(error);
              }
            );
          }
        };
      }
      var returnValue$jscomp$0 = result;
      popActScope(prevActQueue, prevActScopeDepth);
      0 === prevActScopeDepth &&
        (flushActQueue(queue),
        0 !== queue.length &&
          queueSeveralMicrotasks(function () {
            didAwaitActCall ||
              didWarnNoAwaitAct ||
              ((didWarnNoAwaitAct = !0),
              console.error(
                "A component suspended inside an `act` scope, but the `act` call was not awaited. When testing React components that depend on asynchronous data, you must await the result:\n\nawait act(() => ...)"
              ));
          }),
        (ReactSharedInternals.actQueue = null));
      if (0 < ReactSharedInternals.thrownErrors.length)
        throw (
          ((callback = aggregateErrors(ReactSharedInternals.thrownErrors)),
          (ReactSharedInternals.thrownErrors.length = 0),
          callback)
        );
      return {
        then: function (resolve, reject) {
          didAwaitActCall = !0;
          0 === prevActScopeDepth
            ? ((ReactSharedInternals.actQueue = queue),
              enqueueTask(function () {
                return recursivelyFlushAsyncActWork(
                  returnValue$jscomp$0,
                  resolve,
                  reject
                );
              }))
            : resolve(returnValue$jscomp$0);
        }
      };
    };
    exports.cache = function (fn) {
      return function () {
        return fn.apply(null, arguments);
      };
    };
    exports.captureOwnerStack = function () {
      var getCurrentStack = ReactSharedInternals.getCurrentStack;
      return null === getCurrentStack ? null : getCurrentStack();
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
      props = ReactElement(
        element.type,
        key,
        void 0,
        void 0,
        owner,
        props,
        element._debugStack,
        element._debugTask
      );
      for (key = 2; key < arguments.length; key++)
        (owner = arguments[key]),
          isValidElement(owner) && owner._store && (owner._store.validated = 1);
      return props;
    };
    exports.createContext = function (defaultValue) {
      defaultValue = {
        $$typeof: REACT_CONTEXT_TYPE,
        _currentValue: defaultValue,
        _currentValue2: defaultValue,
        _threadCount: 0,
        Provider: null,
        Consumer: null
      };
      defaultValue.Provider = defaultValue;
      defaultValue.Consumer = {
        $$typeof: REACT_CONSUMER_TYPE,
        _context: defaultValue
      };
      defaultValue._currentRenderer = null;
      defaultValue._currentRenderer2 = null;
      return defaultValue;
    };
    exports.createElement = function (type, config, children) {
      for (var i = 2; i < arguments.length; i++) {
        var node = arguments[i];
        isValidElement(node) && node._store && (node._store.validated = 1);
      }
      var propName;
      i = {};
      node = null;
      if (null != config)
        for (propName in (didWarnAboutOldJSXRuntime ||
          !("__self" in config) ||
          "key" in config ||
          ((didWarnAboutOldJSXRuntime = !0),
          console.warn(
            "Your app (or one of its dependencies) is using an outdated JSX transform. Update to the modern JSX transform for faster performance: https://react.dev/link/new-jsx-transform"
          )),
        hasValidKey(config) &&
          (checkKeyStringCoercion(config.key), (node = "" + config.key)),
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
      node &&
        defineKeyPropWarningGetter(
          i,
          "function" === typeof type
            ? type.displayName || type.name || "Unknown"
            : type
        );
      return ReactElement(
        type,
        node,
        void 0,
        void 0,
        getOwner(),
        i,
        Error("react-stack-top-frame"),
        createTask(getTaskName(type))
      );
    };
    exports.createRef = function () {
      var refObject = { current: null };
      Object.seal(refObject);
      return refObject;
    };
    exports.experimental_useEffectEvent = function (callback) {
      return resolveDispatcher().useEffectEvent(callback);
    };
    exports.experimental_useOptimistic = function (passthrough, reducer) {
      console.error(
        "useOptimistic is now in canary. Remove the experimental_ prefix. The prefixed alias will be removed in an upcoming release."
      );
      return useOptimistic(passthrough, reducer);
    };
    exports.experimental_useResourceEffect = void 0;
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
      "string" === typeof type ||
        "function" === typeof type ||
        type === REACT_FRAGMENT_TYPE ||
        type === REACT_PROFILER_TYPE ||
        type === REACT_STRICT_MODE_TYPE ||
        type === REACT_SUSPENSE_TYPE ||
        type === REACT_SUSPENSE_LIST_TYPE ||
        type === REACT_OFFSCREEN_TYPE ||
        ("object" === typeof type &&
          null !== type &&
          (type.$$typeof === REACT_LAZY_TYPE ||
            type.$$typeof === REACT_MEMO_TYPE ||
            type.$$typeof === REACT_CONTEXT_TYPE ||
            type.$$typeof === REACT_CONSUMER_TYPE ||
            type.$$typeof === REACT_FORWARD_REF_TYPE ||
            type.$$typeof === REACT_CLIENT_REFERENCE ||
            void 0 !== type.getModuleId)) ||
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
    exports.startTransition = function (scope) {
      var prevTransition = ReactSharedInternals.T,
        currentTransition = {};
      ReactSharedInternals.T = currentTransition;
      currentTransition._updatedFibers = new Set();
      try {
        var returnValue = scope(),
          onStartTransitionFinish = ReactSharedInternals.S;
        null !== onStartTransitionFinish &&
          onStartTransitionFinish(currentTransition, returnValue);
        "object" === typeof returnValue &&
          null !== returnValue &&
          "function" === typeof returnValue.then &&
          returnValue.then(noop, reportGlobalError);
      } catch (error) {
        reportGlobalError(error);
      } finally {
        null === prevTransition &&
          currentTransition._updatedFibers &&
          ((scope = currentTransition._updatedFibers.size),
          currentTransition._updatedFibers.clear(),
          10 < scope &&
            console.warn(
              "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
            )),
          (ReactSharedInternals.T = prevTransition);
      }
    };
    exports.unstable_Activity = REACT_OFFSCREEN_TYPE;
    exports.unstable_SuspenseList = REACT_SUSPENSE_LIST_TYPE;
    exports.unstable_getCacheForType = function (resourceType) {
      var dispatcher = ReactSharedInternals.A;
      return dispatcher
        ? dispatcher.getCacheForType(resourceType)
        : resourceType();
    };
    exports.unstable_postpone = function (reason) {
      reason = Error(reason);
      reason.$$typeof = REACT_POSTPONE_TYPE;
      throw reason;
    };
    exports.unstable_useCacheRefresh = function () {
      return resolveDispatcher().useCacheRefresh();
    };
    exports.use = function (usable) {
      return resolveDispatcher().use(usable);
    };
    exports.useActionState = function (action, initialState, permalink) {
      return resolveDispatcher().useActionState(
        action,
        initialState,
        permalink
      );
    };
    exports.useCallback = function (callback, deps) {
      return resolveDispatcher().useCallback(callback, deps);
    };
    exports.useContext = function (Context) {
      var dispatcher = resolveDispatcher();
      Context.$$typeof === REACT_CONSUMER_TYPE &&
        console.error(
          "Calling useContext(Context.Consumer) is not supported and will cause bugs. Did you mean to call useContext(Context) instead?"
        );
      return dispatcher.useContext(Context);
    };
    exports.useDebugValue = function (value, formatterFn) {
      return resolveDispatcher().useDebugValue(value, formatterFn);
    };
    exports.useDeferredValue = function (value, initialValue) {
      return resolveDispatcher().useDeferredValue(value, initialValue);
    };
    exports.useEffect = function (create, deps) {
      return resolveDispatcher().useEffect(create, deps);
    };
    exports.useId = function () {
      return resolveDispatcher().useId();
    };
    exports.useImperativeHandle = function (ref, create, deps) {
      return resolveDispatcher().useImperativeHandle(ref, create, deps);
    };
    exports.useInsertionEffect = function (create, deps) {
      return resolveDispatcher().useInsertionEffect(create, deps);
    };
    exports.useLayoutEffect = function (create, deps) {
      return resolveDispatcher().useLayoutEffect(create, deps);
    };
    exports.useMemo = function (create, deps) {
      return resolveDispatcher().useMemo(create, deps);
    };
    exports.useOptimistic = useOptimistic;
    exports.useReducer = function (reducer, initialArg, init) {
      return resolveDispatcher().useReducer(reducer, initialArg, init);
    };
    exports.useRef = function (initialValue) {
      return resolveDispatcher().useRef(initialValue);
    };
    exports.useState = function (initialState) {
      return resolveDispatcher().useState(initialState);
    };
    exports.useSyncExternalStore = function (
      subscribe,
      getSnapshot,
      getServerSnapshot
    ) {
      return resolveDispatcher().useSyncExternalStore(
        subscribe,
        getSnapshot,
        getServerSnapshot
      );
    };
    exports.useTransition = function () {
      return resolveDispatcher().useTransition();
    };
    exports.version = "19.1.0-experimental-518d06d2-20241219";
    "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ &&
      "function" ===
        typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop &&
      __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  })();
