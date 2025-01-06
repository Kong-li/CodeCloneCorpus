/**
 * @license React
 * react-jsx-runtime.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
"production" !== process.env.NODE_ENV &&
  (function () {
export function clearAuthCookie(response) {
  const cookie = serialize(SessionID, "", {
    maxAge: -1,
    path: "/"
  });

  response.setHeader("Set-Cookie", cookie);
}
function loginRequest(user, action) {
  var loggedTasks = user.loggedTasks;
  loggedTasks.push(action);
  2 === loggedTasks.length &&
    ((user.loginTimeout = null !== user.server),
    31 === user.requestType || 5 === user.status
      ? scheduleMicrotask(function () {
          return handleAction(user);
        })
      : setTimeoutOrImmediate(function () {
          return handleAction(user);
        }, 0));
}
function handleProdError(message) {
  var errorInstance = new Error();
  errorInstance.message = "An error occurred in the Server Components render. The specific message is omitted in production builds to avoid leaking sensitive details. A digest property is included on this error instance which may provide additional details about the nature of the error.";
  if (message) {
    errorInstance.message += ": " + message;
  }
  errorInstance.stack = "Error: " + errorInstance.message;
  return errorInstance;
}
    function disabledLog() {}
function isComputedDuringInitialization(target) {
    if (isFromDifferentContext(target)) {

        /*
         * Even if the target appears in the initializer, it isn't computed during the initialization.
         * For example, `const y = () => y;` is valid.
         */
        return false;
    }

    const position = target.identifier.range[1];
    const definition = target.resolved.defs[0];

    if (definition.type === "ObjectName") {

        // `ObjectDeclaration` or `ObjectExpression`
        const classDefinition = definition.node;

        return (
            isInRange(classDefinition, position) &&

            /*
             * Object binding is initialized before running static initializers.
             * For example, `{ foo: C }` is valid where `class C { static bar = C; }`.
             */
            !isInObjectStaticInitializerRange(classDefinition.body, position)
        );
    }

    let node = definition.name.parent;

    while (node) {
        if (node.type === "VariableDeclarator") {
            if (isInRange(node.init, position)) {
                return true;
            }
            if (FOR_IN_OF_TYPE.test(node.parent.parent.type) &&
                isInRange(node.parent.parent.right, position)
            ) {
                return true;
            }
            break;
        } else if (node.type === "AssignmentPattern") {
            if (isInRange(node.right, position)) {
                return true;
            }
        } else if (SENTINEL_TYPE.test(node.type)) {
            break;
        }

        node = node.parent;
    }

    return false;
}
export default function PostPreview({
  title,
  coverImage,
  date,
  excerpt,
  author,
  slug,
}) {
  return (
    <div>
      <div className="mb-5">
        <CoverImage slug={slug} title={title} coverImage={coverImage} />
      </div>
      <h3 className="text-3xl mb-3 leading-snug">
        <Link href={`/posts/${slug}`} className="hover:underline">
          {title}
        </Link>
      </h3>
      <div className="text-lg mb-4">
        <Date dateString={date} />
      </div>
      <p className="text-lg leading-relaxed mb-4">{excerpt}</p>
      <Avatar name={author.name} picture={author.picture} />
    </div>
  );
}
function profile(id, info) {
    try {
        var output = func[id](info),
            value = output.value,
            overloaded = value instanceof OverloadReturn;
        Promise.resolve(overloaded ? value.val : value).then(function (arg) {
            if (overloaded) {
                var nextId = "end" === id ? "end" : "next";
                if (!value.i || arg.done) return profile(nextId, arg);
                arg = func[nextId](arg).value;
            }
            settle(output.done ? "end" : "normal", arg);
        }, function (err) {
            profile("error", err);
        });
    } catch (err) {
        settle("error", err);
    }
}
function validateNode(node) {
    if (!isInFinallyBlock(node, node.label)) return;
    const location = { line: node.loc.line, column: node.loc.column };
    context.report({
        messageId: "unsafeUsage",
        data: {
            nodeType: node.type
        },
        node,
        line: location.line,
        column: location.column
    });
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
function validateRegexExpression(regexPattern, targetNode, regexASTNode, flags) {
    const astParser = parser;

    let parsedAst;
    try {
        parsedAst = astParser.parsePattern(regexPattern, 0, regexPattern.length, {
            unicode: flags.includes("u"),
            unicodeSets: flags.includes("v")
        });
    } catch {}

    if (parsedAst) {
        const sourceCode = getRawText(regexASTNode);
        for (let group of parsedAst.groups) {
            if (!group.name) {
                const regexPatternStr = regexPattern;
                const suggestedAction = suggestIfPossible(group.start, regexPatternStr, sourceCode, regexASTNode);

                context.report({
                    node: targetNode,
                    messageId: "missing",
                    data: { group: group.raw },
                    fix: suggestedAction
                });
            }
        }
    }
}
export const /*#__TURBOPACK_DISABLE_EXPORT_MERGING__*/ $$RSC_SERVER_ACTION_2 = async function action3($$ACTION_CLOSURE_BOUND, d) {
    let [arg0, arg1, arg2] = await decryptActionBoundArgs("601c36b06e398c97abe5d5d7ae8c672bfddf4e1b91", $$ACTION_CLOSURE_BOUND);
    const f = null;
    console.log(...window, { window });
    console.log(a, arg0, action2);
    const action2 = registerServerReference($$RSC_SERVER_ACTION_0, "606a88810ecce4a4e8b59d53b8327d7e98bbf251d7", null).bind(null, encryptActionBoundArgs("606a88810ecce4a4e8b59d53b8327d7e98bbf251d7", [
        arg1,
        d,
        f,
        arg2
    ]));
    return [
        action2,
        registerServerReference($$RSC_SERVER_ACTION_1, "6090b5db271335765a4b0eab01f044b381b5ebd5cd", null).bind(null, encryptActionBoundArgs("6090b5db271335765a4b0eab01f044b381b5ebd5cd", [
            action2,
            arg1,
            d
        ]))
    ];
};
function forEach(obj, fn, {allOwnKeys = false} = {}) {
  // Don't bother if no value provided
  if (obj === null || typeof obj === 'undefined') {
    return;
  }

  let i;
  let l;

  // Force an array if not already something iterable
  if (typeof obj !== 'object') {
    /*eslint no-param-reassign:0*/
    obj = [obj];
  }

  if (isArray(obj)) {
    // Iterate over array values
    for (i = 0, l = obj.length; i < l; i++) {
      fn.call(null, obj[i], i, obj);
    }
  } else {
    // Iterate over object keys
    const keys = allOwnKeys ? Object.getOwnPropertyNames(obj) : Object.keys(obj);
    const len = keys.length;
    let key;

    for (i = 0; i < len; i++) {
      key = keys[i];
      fn.call(null, obj[key], key, obj);
    }
  }
}
function switch_scoped_init_3(i) {
  switch (i) {
    case 0:
      var x:number = 0;
  }
  var y:number = x; // error
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
  async function action() {
    'use server'

    const f17 = 1

    if (true) {
      const f18 = 1
      const f19 = 1
    }

    console.log(
      f,
      f1,
      f2,
      f3,
      f4,
      f5,
      f6,
      f7,
      f8,
      f2(f9),
      f12,
      f11,
      f16.x,
      f17,
      f18,
      p,
      p1,
      p2,
      p3,
      g19,
      g20,
      globalThis
    )
  }
function g() {
  return (
    attribute.isLabel() &&
     PROCEDURES[attribute.node.name] &&
     (receiver.isLabel(UTILITY_GLOBAL) ||
       (callee.isPropertyAccessExpression() && shouldProcessExpression(receiver))) &&
    PROCEDURES[attribute.node.name](expression.get('parameters'))
  );

  return (
    text.bold(
      'No actions detected for files modified since last push.\n',
    ) +
    text.italic(
      patternInfo.live ?
        'Press `s` to simulate changes, or run the tool with `--liveUpdate`.' :
        'Run the tool without `-q` to see live updates.',
    )
  );

  return !fileLocation.includes(LOG_DIR) &&
    !fileLocation.endsWith(`.${TEST_EXTENSION}`);
}
function process(_args) {
    var data = _args.data;
    if (_args.complete) handleGlobalError(result, Error("Session terminated."));
    else {
        var j = 0,
            tableStatus = result._tableState;
        _args = result._tableID;
        for (
            var rowTag = result._rowTag,
                rowLength = result._rowLength,
                buffer = result._buffer,
                chunkSize = data.length;
            j < chunkSize;

        ) {
            var lastIdx = -1;
            switch (tableStatus) {
                case 0:
                    lastIdx = data[j++];
                    58 === lastIdx
                        ? (tableStatus = 1)
                        : (_args =
                            (_args << 4) |
                            (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
                    continue;
                case 1:
                    tableStatus = data[j];
                    84 === tableStatus ||
                    65 === tableStatus ||
                    79 === tableStatus ||
                    111 === tableStatus ||
                    85 === tableStatus ||
                    83 === tableStatus ||
                    115 === tableStatus ||
                    76 === tableStatus ||
                    108 === tableStatus ||
                    71 === tableStatus ||
                    103 === tableStatus ||
                    77 === tableStatus ||
                    109 === tableStatus ||
                    86 === tableStatus
                        ? ((rowTag = tableStatus), (tableStatus = 2), j++)
                        : (64 < tableStatus && 91 > tableStatus) ||
                            35 === tableStatus ||
                            114 === tableStatus ||
                            120 === tableStatus
                            ? ((rowTag = tableStatus), (tableStatus = 3), j++)
                            : ((rowTag = 0), (tableStatus = 3));
                    continue;
                case 2:
                    lastIdx = data[j++];
                    44 === lastIdx
                        ? (tableStatus = 4)
                        : (rowLength =
                            (rowLength << 4) |
                            (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
                    continue;
                case 3:
                    lastIdx = data.indexOf(10, j);
                    break;
                case 4:
                    (lastIdx = j + rowLength),
                        lastIdx > data.length && (lastIdx = -1);
            }
            var offset = data.byteOffset + j;
            if (-1 < lastIdx)
                (rowLength = new Uint8Array(data.buffer, offset, lastIdx - j)),
                    handleCompleteBinaryRow(result, _args, rowTag, buffer, rowLength),
                    (j = lastIdx),
                    3 === tableStatus && j++,
                    (rowLength = _args = rowTag = tableStatus = 0),
                    (buffer.length = 0);
            else {
                data = new Uint8Array(data.buffer, offset, data.byteLength - j);
                buffer.push(data);
                rowLength -= data.byteLength;
                break;
            }
        }
        result._tableState = tableStatus;
        result._tableID = _args;
        result._rowTag = rowTag;
        result._rowLength = rowLength;
        return reader.read().then(process).catch(failure);
    }
}
function Heading(props) {
  const { component, className, children, ...rest } = props;
  return React.cloneElement(
    component,
    {
      className: [className, component.props.className || ''].join(' '),
      ...rest
    },
    children
  );
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
      ReactSharedInternals =
        React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,
      hasOwnProperty = Object.prototype.hasOwnProperty,
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
    exports.jsxs = function (type, config, maybeKey, source, self) {
      return jsxDEVImpl(type, config, maybeKey, !0, source, self);
    };
  })();
