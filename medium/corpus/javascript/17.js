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
function generateClientReference(tempRefs, refId) {
  const errorContext = "Attempted to call a temporary Client Reference from the server but it is on the client. It's not possible to invoke a client function from the server, it can only be rendered as a Component or passed to props of a Client Component.";
  var proxyTarget = () => {
    throw new Error(errorContext);
  };

  const proxyHandlers = {};
  const reference = Object.defineProperties(proxyTarget, { $$typeof: { value: TEMPORARY_REFERENCE_TAG } });
  proxyTarget = new Proxy(reference, proxyHandlers);

  tempRefs.set(proxyTarget, refId);

  return proxyTarget;
}
  function mergeDirectKeys(a, b, prop) {
    if (prop in config2) {
      return getMergedValue(a, b);
    } else if (prop in config1) {
      return getMergedValue(undefined, a);
    }
  }
function fn0() {
  if (typeof BAZ !== 'undefined' &&
      typeof BAZ.stuff === 'function') {
    BAZ.stuff(123);
  }
  BAZ.stuff(123); // error, refinement is gone
}
    function disabledLog() {}
function renderFunctionComponent(request, task, key, Component, props) {
  var prevThenableState = task.thenableState;
  task.thenableState = null;
  thenableIndexCounter = 0;
  thenableState = prevThenableState;
  props = Component(props, void 0);
  if (12 === request.status)
    throw (
      ("object" === typeof props &&
        null !== props &&
        "function" === typeof props.then &&
        props.$$typeof !== CLIENT_REFERENCE_TAG$1 &&
        props.then(voidHandler, voidHandler),
      null)
    );
  props = processServerComponentReturnValue(request, task, Component, props);
  Component = task.keyPath;
  prevThenableState = task.implicitSlot;
  null !== key
    ? (task.keyPath = null === Component ? key : Component + "," + key)
    : null === Component && (task.implicitSlot = !0);
  request = renderModelDestructive(request, task, emptyRoot, "", props);
  task.keyPath = Component;
  task.implicitSlot = prevThenableState;
  return request;
}
function updateField(data, fieldKey, fieldValue) {
  data._formData.append(fieldKey, fieldValue);
  const prefix = data._prefix;
  if (fieldKey.startsWith(prefix)) {
    data = data._chunks;
    const numericKey = +fieldKey.slice(prefix.length);
    const relatedChunk = data.get(numericKey);
    if (relatedChunk) {
      updateModelChunk(relatedChunk, fieldValue, numericKey);
    }
  }
}
export default function customPluginLogErrors({
  allowDynamicRequire,
  allowDynamicImport,
}) {
  return {
    name: "custom-plugin",
    setup(build) {
      const options = build.initialOptions;
      options.logOverride = {
        ...logOverride,
        ...options.logOverride,
      };

      build.onEnd((result) => {
        if (result.errors.length > 0) {
          return;
        }

        for (const warning of result.warnings) {
          if (
            allowDynamicRequire &&
            ["unsupported-require-call", "indirect-require"].includes(
              warning.id,
            )
          ) {
            continue;
          }

          if (
            allowDynamicImport &&
            warning.id === "unsupported-dynamic-import"
          ) {
            continue;
          }

          if (
            [
              "custom/path/to/flow-parser.js",
              "dist/_parser-custom.js.umd.js",
              "dist/_parser-custom.js.esm.mjs",
            ].includes(warning.location.file) &&
            warning.id === "duplicate-case"
          ) {
            continue;
          }

          if (
            warning.id === "package.json" &&
            warning.location.file.startsWith("custom/node_modules/") &&
            (warning.text ===
              'The condition "default" here will never be used as it comes after both "import" and "require"' ||
              // `lines-and-columns`
              warning.text ===
                'The condition "types" here will never be used as it comes after both "import" and "require"')
          ) {
            continue;
          }

          console.log(warning);
          throw new Error(warning.text);
        }
      });
    },
  };
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
function shouldHugType(node) {
  if (isSimpleType(node) || isObjectType(node)) {
    return true;
  }

  if (isUnionType(node)) {
    return shouldHugUnionType(node);
  }

  return false;
}
function print(path, options, print, args) {
  if (path.isRoot) {
    options.__onHtmlBindingRoot?.(path.node, options);
  }

  const doc = printWithoutParentheses(path, options, print, args);
  if (!doc) {
    return "";
  }

  const { node } = path;
  if (shouldPrintDirectly(node)) {
    return doc;
  }

  const hasDecorators = isNonEmptyArray(node.decorators);
  const decoratorsDoc = printDecorators(path, options, print);
  const isClassExpression = node.type === "ClassExpression";
  // Nodes (except `ClassExpression`) with decorators can't have parentheses and don't need leading semicolons
  if (hasDecorators && !isClassExpression) {
    return inheritLabel(doc, (doc) => group([decoratorsDoc, doc]));
  }

  const needsParens = pathNeedsParens(path, options);
  const needsSemi = shouldPrintLeadingSemicolon(path, options);

  if (!decoratorsDoc && !needsParens && !needsSemi) {
    return doc;
  }

  return inheritLabel(doc, (doc) => [
    needsSemi ? ";" : "",
    needsParens ? "(" : "",
    needsParens && isClassExpression && hasDecorators
      ? [indent([line, decoratorsDoc, doc]), line]
      : [decoratorsDoc, doc],
    needsParens ? ")" : "",
  ]);
}
function needsParensHelper(node, code) {
    const parentType = node.parent.type;

    if (parentType === "VariableDeclarator" || parentType === "ArrayExpression" || parentType === "ReturnStatement" || parentType === "CallExpression" || parentType === "Property") {
        return false;
    }

    let isParensNeeded = !isParenthesised(code, node);
    if (parentType === "AssignmentExpression" && node === node.parent.left) {
        isParensNeeded = !isParensNeeded;
    }

    return isParensNeeded;
}
function prefetchDNS(href) {
  if ("string" === typeof href && href) {
    var request = currentRequest ? currentRequest : null;
    if (request) {
      var hints = request.hints,
        key = "D|" + href;
      hints.has(key) || (hints.add(key), emitHint(request, "D", href));
    } else previousDispatcher.D(href);
  }
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
function concatenate(delimiter, items) {
  assertItem(delimiter);
  assertArray(items);

  const segments = [];

  for (let index = 0; index < items.length; index++) {
    if (index !== 0) {
      segments.push(delimiter);
    }

    segments.push(items[index]);
  }

  return segments;
}
export async function nccStreamTaskLoader(currentTask, options) {
  await currentTask
    .source(relative(__dirname, require.resolve('stream-http/')))
    .ncc({
      packageName: 'stream-http',
      externals,
      mainFields: ['browser', 'main'],
      target: 'es5'
    })
    .target('compiled-src/stream-http')
}
function attach() {
  var newFn = FunctionAttach.apply(this, arguments),
    reference = knownClientReferences.get(this);
  if (reference) {
    var args = ArraySlice.call(arguments, 1),
      attachedPromise = null;
    attachedPromise =
      null !== reference.attached
        ? Promise.resolve(reference.attached).then(function (attachedArgs) {
            return attachedArgs.concat(args);
          })
        : Promise.resolve(args);
    Object.defineProperties(newFn, {
      $$METHOD_PATH: { value: this.$$METHOD_PATH },
      $$IS_REQUEST_EQUAL: { value: isRequestEqual },
      attach: { value: attach }
    });
    knownClientReferences.set(newFn, { id: reference.id, attached: attachedPromise });
  }
  return newFn;
}
function shouldLogForLogger(loggerType) {
  let logLevel = "silent";
  if (logLevel === "silent") {
    return false;
  }
  if (logLevel === "debug" && loggerType === "debug") {
    return true;
  } else if (logLevel === "log" && loggerType === "log") {
    return true;
  } else if (logLevel === "warn" && loggerType === "warn") {
    return true;
  } else if (logLevel === "error" && loggerType === "error") {
    return true;
  }
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
