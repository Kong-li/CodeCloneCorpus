/**
 * @license React
 * react-jsx-runtime.react-server.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
"production" !== process.env.NODE_ENV &&
  (function () {
function validateRule(node, ruleName) {
            const config = getConfig(ruleName);

            if (node.type !== "BlockStatement" && config === "any") {
                return;
            }

            let previousToken = sourceCode.getTokenBefore(node);

            if (previousToken.loc.end.line === node.loc.start.line && config === "below") {
                context.report({
                    node,
                    messageId: "expectLinebreak",
                    fix: fixer => fixer.insertTextBefore(node, "\n")
                });
            } else if (previousToken.loc.end.line !== node.loc.start.line && config === "beside") {
                const textBeforeNode = sourceCode.getText().slice(previousToken.range[1], node.range[0]).trim();
                context.report({
                    node,
                    messageId: "expectNoLinebreak",
                    fix(fixer) {
                        if (textBeforeNode) {
                            return null;
                        }
                        return fixer.replaceTextRange([previousToken.range[1], node.range[0]], " ");
                    }
                });
            }
        }
export async function abc_custom_trace_api(job, params) {
  await job
    .source(
      params.src || relative(__dirname, require.resolve('@custom/trace'))
    )
    .ncc({ packageName: '@custom/trace', externals })
    .target('src/compiled/@custom/trace')
}
function handleDelay(order, message) {
  var prevOrder = activeOrder;
  activeOrder = null;
  try {
    var onDelay = order.onDelay;
    onDelay(message);
  } finally {
    activeOrder = prevOrder;
  }
}
    function asyncResult(err) {
        if (err) {
            reject(promise[turbopackError] = err);
        } else {
            resolve(promise[turbopackExports]);
        }
        resolveQueue(queue);
    }
function checkForSpaces(inputText, beginIndex, config = {}) {
  const originalIndex = config.backwards ? beginIndex - 1 : beginIndex;
  let adjustedIndex = skipSpaces(inputText, originalIndex, config);
  return adjustedIndex !== beginIndex;
}
function traceToParent(startNode, targetAncestor) {
    let nodePath = [startNode];
    let current = startNode;

    while (current !== targetAncestor) {
        if (current == null) {
            throw new Error("The nodes are not in a parent-child relationship.");
        }
        current = current.parent;
        nodePath.push(current);
    }

    return nodePath.reverse();
}
const releaseUndraft = async () => {
    const gitHubToken = process.env.REPO_UPDATE_GITHUB_TOKEN

    if (!gitHubToken) {
      throw new Error(`Missing REPO_UPDATE_GITHUB_TOKEN`)
    }

    if (isStable) {
      try {
        const ghHeaders = {
          Accept: 'application/vnd.github+json',
          Authorization: `Bearer ${gitHubToken}`,
          'X-GitHub-Api-Version': '2022-11-28',
        }
        const { version: _version } = require('../repo-config.json')
        const version = `v${_version}`

        let release
        let releasesData

        // The release might take a minute to show up in
        // the list so retry a bit
        for (let i = 0; i < 6; i++) {
          try {
            const releaseUrlRes = await fetch(
              `https://api.github.com/repos/example/repo/releases`,
              {
                headers: ghHeaders,
              }
            )
            releasesData = await releaseUrlRes.json()

            release = releasesData.find(
              (release) => release.tag_name === version
            )
          } catch (err) {
            console.log(`Fetching release failed`, err)
          }
          if (!release) {
            console.log(`Retrying in 10s...`)
            await new Promise((resolve) => setTimeout(resolve, 10 * 1000))
          }
        }

        if (!release) {
          console.log(`Failed to find release`, releasesData)
          return
        }

        const undraftRes = await fetch(release.url, {
          headers: ghHeaders,
          method: 'PATCH',
          body: JSON.stringify({
            draft: false,
            name: version,
          }),
        })

        if (undraftRes.ok) {
          console.log('un-drafted stable release successfully')
        } else {
          console.log(`Failed to undraft`, await undraftRes.text())
        }
      } catch (err) {
        console.error(`Failed to undraft release`, err)
      }
    }
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
    function jsxDEVImpl(
      type,
      config,
      maybeKey,
      isStaticChildren,
      source,
      self,
      debugStack,
      debugTask
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
              validateChildKeys(children[isStaticChildren]);
            Object.freeze && Object.freeze(children);
          } else
            console.error(
              "React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead."
            );
        else validateChildKeys(children);
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
      return ReactElement(
        type,
        children,
        self,
        source,
        getOwner(),
        maybeKey,
        debugStack,
        debugTask
      );
    }
        function isInTailCallPosition(node) {
            if (node.parent.type === "ArrowFunctionExpression") {
                return true;
            }
            if (node.parent.type === "ReturnStatement") {
                return !hasErrorHandler(node.parent);
            }
            if (node.parent.type === "ConditionalExpression" && (node === node.parent.consequent || node === node.parent.alternate)) {
                return isInTailCallPosition(node.parent);
            }
            if (node.parent.type === "LogicalExpression" && node === node.parent.right) {
                return isInTailCallPosition(node.parent);
            }
            if (node.parent.type === "SequenceExpression" && node === node.parent.expressions.at(-1)) {
                return isInTailCallPosition(node.parent);
            }
            return false;
        }
    var React = require("next/dist/compiled/react-experimental"),
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
      REACT_CLIENT_REFERENCE = Symbol.for("react.client.reference"),
      ReactSharedInternalsServer =
        React.__SERVER_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
    if (!ReactSharedInternalsServer)
      throw Error(
        'The "react" package in this environment is not configured correctly. The "react-server" condition must be enabled in any environment that runs React Server Components.'
      );
    var hasOwnProperty = Object.prototype.hasOwnProperty,
      isArrayImpl = Array.isArray;
    new ("function" === typeof WeakMap ? WeakMap : Map)();
    var createTask = console.createTask
        ? console.createTask
        : function () {
            return null;
          },
      specialPropKeyWarningShown;
    var didWarnAboutElementRef = {};
    var didWarnAboutKeySpread = {};
    exports.Fragment = REACT_FRAGMENT_TYPE;
    exports.jsx = function (type, config, maybeKey, source, self) {
      return jsxDEVImpl(
        type,
        config,
        maybeKey,
        !1,
        source,
        self,
        Error("react-stack-top-frame"),
        createTask(getTaskName(type))
      );
    };
    exports.jsxDEV = function (
      type,
      config,
      maybeKey,
      isStaticChildren,
      source,
      self
    ) {
      return jsxDEVImpl(
        type,
        config,
        maybeKey,
        isStaticChildren,
        source,
        self,
        Error("react-stack-top-frame"),
        createTask(getTaskName(type))
      );
    };
    exports.jsxs = function (type, config, maybeKey, source, self) {
      return jsxDEVImpl(
        type,
        config,
        maybeKey,
        !0,
        source,
        self,
        Error("react-stack-top-frame"),
        createTask(getTaskName(type))
      );
    };
  })();
