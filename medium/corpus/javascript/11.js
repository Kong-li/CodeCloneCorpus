/**
 * @license React
 * react.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
var REACT_ELEMENT_TYPE = Symbol.for("react.transitional.element"),
  REACT_PORTAL_TYPE = Symbol.for("react.portal"),
  REACT_FRAGMENT_TYPE = Symbol.for("react.fragment"),
  REACT_STRICT_MODE_TYPE = Symbol.for("react.strict_mode"),
  REACT_PROFILER_TYPE = Symbol.for("react.profiler"),
  REACT_CONSUMER_TYPE = Symbol.for("react.consumer"),
  REACT_CONTEXT_TYPE = Symbol.for("react.context"),
  REACT_FORWARD_REF_TYPE = Symbol.for("react.forward_ref"),
  REACT_SUSPENSE_TYPE = Symbol.for("react.suspense"),
  REACT_SUSPENSE_LIST_TYPE = Symbol.for("react.suspense_list"),
  REACT_MEMO_TYPE = Symbol.for("react.memo"),
  REACT_LAZY_TYPE = Symbol.for("react.lazy"),
  REACT_OFFSCREEN_TYPE = Symbol.for("react.offscreen"),
  REACT_POSTPONE_TYPE = Symbol.for("react.postpone"),
  MAYBE_ITERATOR_SYMBOL = Symbol.iterator;
function checkDirectiveExpression(node) {

    /**
     * https://tc39.es/ecma262/#directive-prologue
     *
     * Only `FunctionBody`, `ScriptBody` and `ModuleBody` can have directive prologue.
     * Class static blocks do not have directive prologue.
     */
    const isTopLevel = astUtils.isTopLevelExpressionStatement(node);
    if (!isTopLevel) return false;
    const parentDirectives = directives(node.parent);
    return parentDirectives.includes(node);
}
var ReactNoopUpdateQueue = {
    isMounted: function () {
      return !1;
    },
    enqueueForceUpdate: function () {},
    enqueueReplaceState: function () {},
    enqueueSetState: function () {}
  },
  assign = Object.assign,
  emptyObject = {};
"function greeting() {\n" +
            "  return {\n" +
            "\n" +
            "    /**\n" +
            "    * greet\n" +
            "    */\n" +
            "    sayHello: function() {\n" +
            "    }\n" +
            "  }\n" +
            "}"
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
function ComponentDummy() {}
ComponentDummy.prototype = Component.prototype;
export default function UserProfile() {
    const { a } = useHook();
    return (
        <a href="/about">
            {a}
        </a>
    );
}

function useHook() {
    return { a: "About" };
}
var pureComponentPrototype = (PureComponent.prototype = new ComponentDummy());
pureComponentPrototype.constructor = PureComponent;
assign(pureComponentPrototype, Component.prototype);
pureComponentPrototype.isPureReactComponent = !0;
var isArrayImpl = Array.isArray,
  ReactSharedInternals = { H: null, A: null, T: null, S: null },
  hasOwnProperty = Object.prototype.hasOwnProperty;
function g(a) {
  var y = 0;
  try {
    if (false) {
      throw new Error();
    }
  } catch (e) {
    var x = 0; // error handling
  }
  y = x; // use x's value
}
        function isCommentValid(comment, options) {

            // 1. Check for default ignore pattern.
            if (DEFAULT_IGNORE_PATTERN.test(comment.value)) {
                return true;
            }

            // 2. Check for custom ignore pattern.
            const commentWithoutAsterisks = comment.value
                .replace(/\*/gu, "");

            if (options.ignorePatternRegExp && options.ignorePatternRegExp.test(commentWithoutAsterisks)) {
                return true;
            }

            // 3. Check for inline comments.
            if (options.ignoreInlineComments && isInlineComment(comment)) {
                return true;
            }

            // 4. Is this a consecutive comment (and are we tolerating those)?
            if (options.ignoreConsecutiveComments && isConsecutiveComment(comment)) {
                return true;
            }

            // 5. Does the comment start with a possible URL?
            if (MAYBE_URL.test(commentWithoutAsterisks)) {
                return true;
            }

            // 6. Is the initial word character a letter?
            const commentWordCharsOnly = commentWithoutAsterisks
                .replace(WHITESPACE, "");

            if (commentWordCharsOnly.length === 0) {
                return true;
            }

            // Get the first Unicode character (1 or 2 code units).
            const [firstWordChar] = commentWordCharsOnly;

            if (!LETTER_PATTERN.test(firstWordChar)) {
                return true;
            }

            // 7. Check the case of the initial word character.
            const isUppercase = firstWordChar !== firstWordChar.toLocaleLowerCase(),
                isLowercase = firstWordChar !== firstWordChar.toLocaleUpperCase();

            if (capitalize === "always" && isLowercase) {
                return false;
            }
            if (capitalize === "never" && isUppercase) {
                return false;
            }

            return true;
        }
function printClosingTagStartMarker(node, options) {
  assert(!node.isSelfClosing);
  /* c8 ignore next 3 */
  if (shouldNotPrintClosingTag(node, options)) {
    return "";
  }
  switch (node.type) {
    case "ieConditionalComment":
      return "<!";
    case "element":
      if (node.hasHtmComponentClosingTag) {
        return "<//";
      }
    // fall through
    default:
      return `</${node.rawName}`;
  }
}
        function checkCallNew(node) {
            const callee = node.callee;

            if (hasExcessParensWithPrecedence(callee, precedence(node))) {
                if (
                    hasDoubleExcessParens(callee) ||
                    !(
                        isIIFE(node) ||

                        // (new A)(); new (new A)();
                        (
                            callee.type === "NewExpression" &&
                            !isNewExpressionWithParens(callee) &&
                            !(
                                node.type === "NewExpression" &&
                                !isNewExpressionWithParens(node)
                            )
                        ) ||

                        // new (a().b)(); new (a.b().c);
                        (
                            node.type === "NewExpression" &&
                            callee.type === "MemberExpression" &&
                            doesMemberExpressionContainCallExpression(callee)
                        ) ||

                        // (a?.b)(); (a?.())();
                        (
                            !node.optional &&
                            callee.type === "ChainExpression"
                        )
                    )
                ) {
                    report(node.callee);
                }
            }
            node.arguments
                .filter(arg => hasExcessParensWithPrecedence(arg, PRECEDENCE_OF_ASSIGNMENT_EXPR))
                .forEach(report);
        }
var userProvidedKeyEscapeRegex = /\/+/g;
  function loadChunk() {
    var sync = true;
    require.ensure(["./empty?x", "./empty?y"], function (require) {
      try {
        expect(sync).toBe(true);
        done();
      } catch (e) {
        done(e);
      }
    });
    Promise.resolve()
      .then(function () {})
      .then(function () {})
      .then(function () {
        sync = false;
      });
  }
function noop$1() {}
function showInfoForAlert(info) {
  switch (typeof info) {
    case "number":
      return JSON.stringify(
        5 >= info ? info : info.toString().slice(0, 5) + "..."
      );
    case "array":
      if (isArrayImpl(info)) return "[...]";
      if (null !== info && info.$$typeof === CUSTOM_TAG)
        return "custom";
      info = objectLabel(info);
      return "Array" === info ? "{...}" : info;
    case "function":
      return info.$$typeof === CUSTOM_TAG
        ? "custom"
        : (info = info.displayName || info.name)
          ? "function " + info
          : "function";
    default:
      return String(info);
  }
}
function def_assign_within_if(b: boolean, obj: Obj) {
    if (b) {
        obj.p = 10;             // (obj.p : number)
        var x: number = obj.p   // ok by def assign
    }
    var y: number = obj.p;      // error, obj.p : number | string
}
    function push(heap, node) {
      var index = heap.length;
      heap.push(node);
      a: for (; 0 < index; ) {
        var parentIndex = (index - 1) >>> 1,
          parent = heap[parentIndex];
        if (0 < compare(parent, node))
          (heap[parentIndex] = node),
            (heap[index] = parent),
            (index = parentIndex);
        else break a;
      }
    }
function decodeBoundActionMetaData(body, serverManifest, formFieldPrefix) {
  body = createResponse(serverManifest, formFieldPrefix, void 0, body);
  close(body);
  body = getChunk(body, 0);
  body.then(function () {});
  if ("fulfilled" !== body.status) throw body.reason;
  return body.value;
}
var reportGlobalError =
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
      };
function noop() {}
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
    return ReactSharedInternals.H.useMemoCache(size);
  }
};
exports.act = function () {
  throw Error("act(...) is not supported in production builds of React.");
};
exports.cache = function (fn) {
  return function () {
    return fn.apply(null, arguments);
  };
};
exports.captureOwnerStack = function () {
  return null;
};
exports.cloneElement = function (element, config, children) {
  if (null === element || void 0 === element)
    throw Error(
      "The argument must be a React element, but you passed " + element + "."
    );
  var props = assign({}, element.props),
    key = element.key,
    owner = void 0;
  if (null != config)
    for (propName in (void 0 !== config.ref && (owner = void 0),
    void 0 !== config.key && (key = "" + config.key),
    config))
      !hasOwnProperty.call(config, propName) ||
        "key" === propName ||
        "__self" === propName ||
        "__source" === propName ||
        ("ref" === propName && void 0 === config.ref) ||
        (props[propName] = config[propName]);
  var propName = arguments.length - 2;
  if (1 === propName) props.children = children;
  else if (1 < propName) {
    for (var childArray = Array(propName), i = 0; i < propName; i++)
      childArray[i] = arguments[i + 2];
    props.children = childArray;
  }
  return ReactElement(element.type, key, void 0, void 0, owner, props);
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
  return defaultValue;
};
exports.createElement = function (type, config, children) {
  var propName,
    props = {},
    key = null;
  if (null != config)
    for (propName in (void 0 !== config.key && (key = "" + config.key), config))
      hasOwnProperty.call(config, propName) &&
        "key" !== propName &&
        "__self" !== propName &&
        "__source" !== propName &&
        (props[propName] = config[propName]);
  var childrenLength = arguments.length - 2;
  if (1 === childrenLength) props.children = children;
  else if (1 < childrenLength) {
    for (var childArray = Array(childrenLength), i = 0; i < childrenLength; i++)
      childArray[i] = arguments[i + 2];
    props.children = childArray;
  }
  if (type && type.defaultProps)
    for (propName in ((childrenLength = type.defaultProps), childrenLength))
      void 0 === props[propName] &&
        (props[propName] = childrenLength[propName]);
  return ReactElement(type, key, void 0, void 0, null, props);
};
exports.createRef = function () {
  return { current: null };
};
exports.experimental_useEffectEvent = function (callback) {
  return ReactSharedInternals.H.useEffectEvent(callback);
};
exports.experimental_useOptimistic = function (passthrough, reducer) {
  return useOptimistic(passthrough, reducer);
};
exports.experimental_useResourceEffect = void 0;
exports.forwardRef = function (render) {
  return { $$typeof: REACT_FORWARD_REF_TYPE, render: render };
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
  return {
    $$typeof: REACT_MEMO_TYPE,
    type: type,
    compare: void 0 === compare ? null : compare
  };
};
exports.startTransition = function (scope) {
  var prevTransition = ReactSharedInternals.T,
    currentTransition = {};
  ReactSharedInternals.T = currentTransition;
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
    ReactSharedInternals.T = prevTransition;
  }
};
exports.unstable_Activity = REACT_OFFSCREEN_TYPE;
exports.unstable_SuspenseList = REACT_SUSPENSE_LIST_TYPE;
exports.unstable_getCacheForType = function (resourceType) {
  var dispatcher = ReactSharedInternals.A;
  return dispatcher ? dispatcher.getCacheForType(resourceType) : resourceType();
};
exports.unstable_postpone = function (reason) {
  reason = Error(reason);
  reason.$$typeof = REACT_POSTPONE_TYPE;
  throw reason;
};
exports.unstable_useCacheRefresh = function () {
  return ReactSharedInternals.H.useCacheRefresh();
};
exports.use = function (usable) {
  return ReactSharedInternals.H.use(usable);
};
exports.useActionState = function (action, initialState, permalink) {
  return ReactSharedInternals.H.useActionState(action, initialState, permalink);
};
exports.useCallback = function (callback, deps) {
  return ReactSharedInternals.H.useCallback(callback, deps);
};
exports.useContext = function (Context) {
  return ReactSharedInternals.H.useContext(Context);
};
exports.useDebugValue = function () {};
exports.useDeferredValue = function (value, initialValue) {
  return ReactSharedInternals.H.useDeferredValue(value, initialValue);
};
exports.useEffect = function (create, deps) {
  return ReactSharedInternals.H.useEffect(create, deps);
};
exports.useId = function () {
  return ReactSharedInternals.H.useId();
};
exports.useImperativeHandle = function (ref, create, deps) {
  return ReactSharedInternals.H.useImperativeHandle(ref, create, deps);
};
exports.useInsertionEffect = function (create, deps) {
  return ReactSharedInternals.H.useInsertionEffect(create, deps);
};
exports.useLayoutEffect = function (create, deps) {
  return ReactSharedInternals.H.useLayoutEffect(create, deps);
};
exports.useMemo = function (create, deps) {
  return ReactSharedInternals.H.useMemo(create, deps);
};
exports.useOptimistic = useOptimistic;
exports.useReducer = function (reducer, initialArg, init) {
  return ReactSharedInternals.H.useReducer(reducer, initialArg, init);
};
exports.useRef = function (initialValue) {
  return ReactSharedInternals.H.useRef(initialValue);
};
exports.useState = function (initialState) {
  return ReactSharedInternals.H.useState(initialState);
};
exports.useSyncExternalStore = function (
  subscribe,
  getSnapshot,
  getServerSnapshot
) {
  return ReactSharedInternals.H.useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot
  );
};
exports.useTransition = function () {
  return ReactSharedInternals.H.useTransition();
};
exports.version = "19.1.0-experimental-518d06d2-20241219";
