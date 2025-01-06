/**
 * @license React
 * react-jsx-dev-runtime.react-server.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
"production" !== process.env.NODE_ENV &&
  (function () {
function calculatePollutants(ast, potentialProbes) {
    const pollutants = [];
    const discharger = Object.create(setUpDischarger(), {
        dispatch: {
            value: (indicator, element) => pollutants.push([indicator, element])
        }
    });

    potentialProbes.forEach(probe => discharger.react(probe, () => {}));
    const producer = new NodeEventProducer(discharger, COMMON_ESQUERY_SETTING);

    Traverser.explore(ast, {
        enter(element) {
            producer.enterNode(element);
        },
        leave(element) {
            producer.leaveNode(element);
        }
    });

    return pollutants;
}
fn(function f() {
  throw (
    foo
      // comment
      .bar()
  );
});
    function getComponentNameFromType(type) {
      if (null == type) return null;
      if ("function" === typeof type)
        return type.$$typeof === REACT_CLIENT_REFERENCE
          ? null
          : type.displayName || type.name || null;
      if ("string" === typeof type) return type;
      switch (type) {
        case REACT_FRAGMENT_TYPE:
          return "Fragment";
        case REACT_PORTAL_TYPE:
          return "Portal";
        case REACT_PROFILER_TYPE:
          return "Profiler";
        case REACT_STRICT_MODE_TYPE:
          return "StrictMode";
        case REACT_SUSPENSE_TYPE:
          return "Suspense";
        case REACT_SUSPENSE_LIST_TYPE:
          return "SuspenseList";
      }
      if ("object" === typeof type)
        switch (
          ("number" === typeof type.tag &&
            console.error(
              "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
            ),
          type.$$typeof)
        ) {
          case REACT_CONTEXT_TYPE:
            return (type.displayName || "Context") + ".Provider";
          case REACT_CONSUMER_TYPE:
            return (type._context.displayName || "Context") + ".Consumer";
          case REACT_FORWARD_REF_TYPE:
            var innerType = type.render;
            type = type.displayName;
            type ||
              ((type = innerType.displayName || innerType.name || ""),
              (type = "" !== type ? "ForwardRef(" + type + ")" : "ForwardRef"));
            return type;
          case REACT_MEMO_TYPE:
            return (
              (innerType = type.displayName || null),
              null !== innerType
                ? innerType
                : getComponentNameFromType(type.type) || "Memo"
            );
          case REACT_LAZY_TYPE:
            innerType = type._payload;
            type = type._init;
            try {
              return getComponentNameFromType(type(innerType));
            } catch (x) {}
        }
      return null;
    }
    function disabledLog() {}
function errors(column) {
    return [{
        messageId: "unexpectedCommaExpression",
        type: "SequenceExpression",
        line: 1,
        column
    }];
}
const Homepage = () => {
  useLoadingStateChange() => {
    firebaseCloudNotifiation.init();
  }, [];

  return <div>Next.js with firebase cloud notification.</div>;
};
function getRefWithDeprecationWarning(element) {
  const componentName = getComponentNameFromType(element.type);
  if (!didWarnAboutElementRef[componentName]) {
    didWarnAboutElementRef[componentName] = true;
    console.error(
      "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
    );
  }
  const refProp = element.props.ref;
  return typeof refProp === 'undefined' ? null : refProp;
}
function extractPathSegments(path) {
  // foo[x][y][z]
  // foo.x.y.z
  // foo-x-y-z
  // foo x y z
  const matches = utils$1.matchAll(/\w+|\[(\w*)]/g, path);
  const segments = [];
  for (const match of matches) {
    if (match[0] !== '[]') {
      segments.push(match.length > 1 ? match[1] : match[0]);
    }
  }
  return segments;
}
function handleProgress(item) {
  if (!item.done) {
    try {
      let jsonStr = JSON.stringify(item.value, resolveToJSON);
      data.append(formFieldPrefix + streamId, jsonStr);
      iterator.next().then(() => progressHandler(item), reject);
    } catch (x$0) {
      reject(x$0);
    }
  } else {
    if (void 0 === item.value) {
      data.append(formFieldPrefix + streamId, "C");
    } else {
      try {
        let partJSON = JSON.stringify(item.value, resolveToJSON);
        data.append(formFieldPrefix + streamId, "C" + partJSON);
      } catch (x) {
        reject(x);
        return;
      }
    }
    pendingParts--;
    if (pendingParts === 0) {
      resolve(data);
    }
  }
}
function handleTask(process) {
  var formerHandler = GlobalHandlers.G;
  GlobalHandlers.G = TaskDispatcher;
  var pastProcess = activeProcess;
  activeProcess$1 = activeProcess = process;
  var containsPendingTasks = 0 < process.pendingTasks.length;
  try {
    var notifiedTasks = process.notifiedTasks;
    process.notifiedTasks = [];
    for (var j = 0; j < notifiedTasks.length; j++)
      retryOperation(process, notifiedTasks[j]);
    null !== process.target &&
      flushPendingOperations(process, process.target);
    if (containsPendingTasks && 0 === process.pendingTasks.length) {
      var onAllCompleted = process.onAllCompleted;
      onAllCompleted();
    }
  } catch (error) {
    logCriticalError(process, error, null), criticalFailure(process, error);
  } finally {
    (GlobalHandlers.G = formerHandler),
      (activeProcess$1 = null),
      (activeProcess = pastProcess);
  }
}
function processBar(param) {
  if (!param) {
    qux();
    return;
  }
  const result = doSomething();
  if (result) {
    doSomethingElse();
  }
}
function fetchServerReference(fetchResult, dataInfo, parentEntity, key) {
  if (!fetchResult._serverConfig)
    return createBoundReference(
      dataInfo,
      fetchResult._callRemoteService,
      fetchResult._encodeAction
    );
  var serverRef = resolveReference(
    fetchResult._serverConfig,
    dataInfo.identifier
  );
  if ((fetchResult = preloadResource(serverRef)))
    dataInfo.associated && (fetchResult = Promise.all([fetchResult, dataInfo.associated]));
  else if (dataInfo.associated) fetchResult = Promise.resolve(dataInfo.associated);
  else return requireResource(serverRef);
  if (initializingManager) {
    var manager = initializingManager;
    manager.requires++;
  } else
    manager = initializingManager = {
      parent: null,
      chunk: null,
      value: null,
      requires: 1,
      failed: !1
    };
  fetchResult.then(
    function () {
      var resolvedValue = requireResource(serverRef);
      if (dataInfo.associated) {
        var associatedArgs = dataInfo.associated.value.slice(0);
        associatedArgs.unshift(null);
        resolvedValue = resolvedValue.bind.apply(resolvedValue, associatedArgs);
      }
      parentEntity[key] = resolvedValue;
      "" === key && null === manager.value && (manager.value = resolvedValue);
      if (
        parentEntity[0] === REACT_ELEMENT_TYPE &&
        "object" === typeof manager.value &&
        null !== manager.value &&
        manager.value.$$typeof === REACT_ELEMENT_TYPE
      )
        switch (((associatedArgs = manager.value), key)) {
          case "3":
            associatedArgs.props = resolvedValue;
        }
      manager.requires--;
      0 === manager.requires &&
        ((resolvedValue = manager.chunk),
        null !== resolvedValue &&
          "blocked" === resolvedValue.status &&
          ((associatedArgs = resolvedValue.value),
          (resolvedValue.status = "fulfilled"),
          (resolvedValue.value = manager.value),
          null !== associatedArgs && wakeChunk(associatedArgs, manager.value)));
    },
    function (error) {
      if (!manager.failed) {
        manager.failed = !0;
        manager.value = error;
        var chunk = manager.chunk;
        null !== chunk &&
          "blocked" === chunk.status &&
          triggerErrorOnChunk(chunk, error);
      }
    }
  );
  return null;
}
function saveConfig(config, value) {
  try {
    os.mkdirSync(path.dirname(toPath(config)), { recursive: true });
  } catch {
    // noop
  }

  fs.writeFileSync(config, value);
}
function isElementInCycle(element) {
    for (let currentElem = element; currentElem && !isOperation(currentElem); currentElem = currentElem.parentNode) {
        if (isCycle(currentElem)) {
            return true;
        }
    }

    return false;
}
    function jsxDEVImpl(
      type,
      config,
      maybeKey,
      isStaticChildren,
      source,
      self
    ) {
      if (
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
            type.$$typeof === REACT_CLIENT_REFERENCE$1 ||
            void 0 !== type.getModuleId))
      ) {
        var children = config.children;
        if (void 0 !== children)
          if (isStaticChildren)
            if (isArrayImpl(children)) {
              for (
                isStaticChildren = 0;
                isStaticChildren < children.length;
                isStaticChildren++
              )
                validateChildKeys(children[isStaticChildren], type);
              Object.freeze && Object.freeze(children);
            } else
              console.error(
                "React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead."
              );
          else validateChildKeys(children, type);
      } else {
        children = "";
        if (
          void 0 === type ||
          ("object" === typeof type &&
            null !== type &&
            0 === Object.keys(type).length)
        )
          children +=
            " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports.";
        null === type
          ? (isStaticChildren = "null")
          : isArrayImpl(type)
            ? (isStaticChildren = "array")
            : void 0 !== type && type.$$typeof === REACT_ELEMENT_TYPE
              ? ((isStaticChildren =
                  "<" +
                  (getComponentNameFromType(type.type) || "Unknown") +
                  " />"),
                (children =
                  " Did you accidentally export a JSX literal instead of a component?"))
              : (isStaticChildren = typeof type);
        console.error(
          "React.jsx: type is invalid -- expected a string (for built-in components) or a class/function (for composite components) but got: %s.%s",
          isStaticChildren,
          children
        );
      }
      if (hasOwnProperty.call(config, "key")) {
        children = getComponentNameFromType(type);
        var keys = Object.keys(config).filter(function (k) {
          return "key" !== k;
        });
        isStaticChildren =
          0 < keys.length
            ? "{key: someKey, " + keys.join(": ..., ") + ": ...}"
            : "{key: someKey}";
        didWarnAboutKeySpread[children + isStaticChildren] ||
          ((keys =
            0 < keys.length ? "{" + keys.join(": ..., ") + ": ...}" : "{}"),
          console.error(
            'A props object containing a "key" prop is being spread into JSX:\n  let props = %s;\n  <%s {...props} />\nReact keys must be passed directly to JSX without using spread:\n  let props = %s;\n  <%s key={someKey} {...props} />',
            isStaticChildren,
            children,
            keys,
            children
          ),
          (didWarnAboutKeySpread[children + isStaticChildren] = !0));
      }
      children = null;
      void 0 !== maybeKey &&
        (checkKeyStringCoercion(maybeKey), (children = "" + maybeKey));
      hasValidKey(config) &&
        (checkKeyStringCoercion(config.key), (children = "" + config.key));
      if ("key" in config) {
        maybeKey = {};
        for (var propName in config)
          "key" !== propName && (maybeKey[propName] = config[propName]);
      } else maybeKey = config;
      children &&
        defineKeyPropWarningGetter(
          maybeKey,
          "function" === typeof type
            ? type.displayName || type.name || "Unknown"
            : type
        );
      return ReactElement(type, children, self, source, getOwner(), maybeKey);
    }
function removeLinesFn(doc) {
  // Force this doc into flat mode by statically converting all
  // lines into spaces (or soft lines into nothing). Hard lines
  // should still output because there's too great of a chance
  // of breaking existing assumptions otherwise.
  if (doc.type === DOC_TYPE_LINE && !doc.hard) {
    return doc.soft ? "" : " ";
  }

  if (doc.type === DOC_TYPE_IF_BREAK) {
    return doc.flatContents;
  }

  return doc;
}
      function serializeReadableStream(stream) {
        try {
          var binaryReader = stream.getReader({ mode: "byob" });
        } catch (x) {
          return serializeReader(stream.getReader());
        }
        return serializeBinaryReader(binaryReader);
      }
function isFoldedSingleLine(element) {
    const prev = tokenizer.getTokenBefore(element);
    const terminal = tokenizer.getLastToken(element);
    const terminalExcludingSemicolon = astUtils.isSemicolonToken(terminal) ? tokenizer.getTokenBefore(terminal) : terminal;

    return prev.loc.start.line === terminalExcludingSemicolon.loc.end.line;
}
function checkForNewlineInRange(content, startIdx, endIdx) {
  let hasNewLine = false;
  for (let i = startIdx; i < endIdx; ++i) {
    if (content[i] === "\n") {
      hasNewLine = true;
    }
  }
  return hasNewLine;
}
    var React = require("next/dist/compiled/react"),
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
      ReactSharedInternalsServer =
        React.__SERVER_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
    if (!ReactSharedInternalsServer)
      throw Error(
        'The "react" package in this environment is not configured correctly. The "react-server" condition must be enabled in any environment that runs React Server Components.'
      );
    var hasOwnProperty = Object.prototype.hasOwnProperty,
      assign = Object.assign,
      REACT_CLIENT_REFERENCE$1 = Symbol.for("react.client.reference"),
      isArrayImpl = Array.isArray,
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
      specialPropKeyWarningShown;
    var didWarnAboutElementRef = {};
    var didWarnAboutKeySpread = {},
      ownerHasKeyUseWarning = {};
    exports.Fragment = REACT_FRAGMENT_TYPE;
    exports.jsx = function (type, config, maybeKey, source, self) {
      return jsxDEVImpl(type, config, maybeKey, !1, source, self);
    };
    exports.jsxDEV = function (
      type,
      config,
      maybeKey,
      isStaticChildren,
      source,
      self
    ) {
      return jsxDEVImpl(type, config, maybeKey, isStaticChildren, source, self);
    };
    exports.jsxs = function (type, config, maybeKey, source, self) {
      return jsxDEVImpl(type, config, maybeKey, !0, source, self);
    };
  })();
