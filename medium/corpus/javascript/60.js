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
function newName() {
  // comment1
  newFunc1()

  // comment2
  newFunc2()

  // comment3 why newFunc3 commented
  // newFunc3()
}
export default function Home() {
    const info = {
        name: 'John',
        test: 'test'
    };
    const action = registerServerReference($$RSC_SERVER_ACTION_1, "4090b5db271335765a4b0eab01f044b381b5ebd5cd", null).bind(null, encryptActionBoundArgs("4090b5db271335765a4b0eab01f044b381b5ebd5cd", [
        info.name,
        info.test
    ]));
    return null;
}
  function abortStream(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      21 === request.type
        ? request.pendingChunks--
        : (erroredTask(request, streamTask, reason), enqueueFlush(request)),
      reader.cancel(reason).then(error, error));
  }
        function endCurrentReportsBuffering() {
            const { upper, inExpressionNodes, reports } = reportsBuffer;

            if (upper) {
                upper.inExpressionNodes.push(...inExpressionNodes);
                upper.reports.push(...reports);
            } else {

                // flush remaining reports
                reports.forEach(({ finishReport }) => finishReport());
            }

            reportsBuffer = upper;
        }
    function ComponentDummy() {}
function attachListener() {
  var boundFunction = FunctionBind.apply(this, arguments);
  if (this.$$type === SERVER_REF_TAG) {
    const args = ArraySlice.call(arguments, 1),
      refType = { value: SERVER_REF_TAG },
      refId = { value: this.$$id };
    const boundArgs = this.$$bound ? this.$$bound.concat(args) : args;
    return Object.defineProperties(boundFunction, {
      $$type: refType,
      $$id: refId,
      $$bound: { value: boundArgs, configurable: true },
      bind: { value: attachListener, configurable: !0 }
    });
  }
  return boundFunction;
}
export default function Footer() {
  return (
    <div>
      <NextSeo
        title="Footer Meta Title"
        description="This will be the footer meta description"
        canonical="https://www.footerurl.ie/"
        openGraph={{
          url: "https://www.footerurl.ie/",
          title: "Footer Open Graph Title",
          description: "Footer Open Graph Description",
          images: [
            {
              url: "https://www.example.ie/og-image-01.jpg",
              width: 800,
              height: 600,
              alt: "Footer Og Image Alt",
            },
            {
              url: "https://www.example.ie/og-image-02.jpg",
              width: 900,
              height: 800,
              alt: "Footer Og Image Alt Second",
            },
            { url: "https://www.example.ie/og-image-03.jpg" },
            { url: "https://www.example.ie/og-image-04.jpg" },
          ],
        }}
      />
      <h1>SEO Added to Footer</h1>
      <p>Take a look at the head to see what has been added.</p>
      <p>
        Or checkout how <Link href="/jsonld">JSON-LD</Link> (Structured Data) is
        added
      </p>
    </div>
  );
}
function validateReference(currentRef, position, allRefs) {
    const refName = currentRef.refName;

    if (currentRef.initialized === false &&
        currentRef.isChanged() &&

        /*
         * Destructuring assignments can have multiple default values,
         * so possibly there are multiple writeable references for the same identifier.
         */
        (position === 0 || allRefs[position - 1].refName !== refName)
    ) {
        context.report({
            node: refName,
            messageId: "globalShouldNotBeModified",
            data: {
                name: refName.name
            }
        });
    }
}
    function disabledLog() {}
function serializeDataTransfer(request, dataTransfer) {
  function updateStatus(entry) {
    if (!aborted)
      if (entry.done)
        request.abortListeners.delete(abortData),
          (aborted = !0),
          pingTask(request, newTask);
      else
        return (
          model.push(entry.value), reader.read().then(updateStatus).catch(error)
        );
  }
  function handleError(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortData),
      erroredTask(request, newTask, reason),
      enqueueFlush(request),
      reader.cancel(reason).then(handleError, handleError));
  }
  function abortData(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortData),
      21 === request.type
        ? request.pendingChunks--
        : (erroredTask(request, newTask, reason), enqueueFlush(request)),
      reader.cancel(reason).then(handleError, handleError));
  }
  var model = [dataTransfer.types[0]],
    newTask = createOperation(request, model, null, !1, request.abortableOperations),
    reader = dataTransfer.stream().getReader(),
    aborted = !1;
  request.abortListeners.add(abortData);
  reader.read().then(updateStatus).catch(handleError);
  return "$T" + newTask.id.toString(16);
}
function className(obj) {
  let type = Object.prototype.toString.call(obj);
  return type.replace(/^\[object (.*)\]$/, function (match, value) {
    return value;
  });
}
function getUpdateExpr(target) {
    if (target.updateExpr) {
        return target.updateExpr;
    }
    let element = target.attribute;

    while (element) {
        const kind = element.parent.kind;

        if (kind === "Assignment" && element.parent.left === element) {
            return element.parent.right;
        }
        if (kind === "Member" && element.parent.object === element) {
            element = element.parent;
            continue;
        }

        break;
    }

    return null;
}
function displayReturnProperty(path, printer) {
  const { node } = path;
  let returnTypeAnnotation = printTypeAnnotationProperty(path, printer, "returnType");

  if (node.predicate) {
    returnTypeAnnotation += printer("predicate");
  }

  return [returnTypeAnnotation];
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
export function serializeMetadata(info) {
  const globalsStr = JSON.stringify(info.globals);
  const localsStr = JSON.stringify(info.locals);
  const exportBindingsStr = JSON.stringify(info.exportBindingAssignments);
  const exportNameStr = JSON.stringify(info.exportName);
  const dependenciesStr = JSON.stringify(info.dependencies);

  return `{
    globals: ${globalsStr},
    locals: ${localsStr},
    exportBindingAssignments: ${exportBindingsStr},
    exportName: ${exportNameStr},
    dependencies: ${dependenciesStr}
  }`;
}
function fetchIteratorFn(someCollection) {
  if (null === someCollection || "object" !== typeof someCollection) return null;
  someCollection =
    (POSSIBLE_ITERATOR_SYMBOL && someCollection[POSSIBLE_ITERATOR_SYMBOL]) ||
    someCollection["@@iterator"];
  return "function" === typeof someCollection ? someCollection : null;
}
function equalLiteralValue(left, right) {

    // RegExp literal.
    if (left.regex || right.regex) {
        return Boolean(
            left.regex &&
            right.regex &&
            left.regex.pattern === right.regex.pattern &&
            left.regex.flags === right.regex.flags
        );
    }

    // BigInt literal.
    if (left.bigint || right.bigint) {
        return left.bigint === right.bigint;
    }

    return left.value === right.value;
}
function processTypedArrayBatch(req, bid, marker, arrayBuffer) {
  req.pendingChunks++;
  let buffer = new Uint8Array(arrayBuffer);
  const slicedArray = arrayBuffer.byteLength > 2048 ? buffer.slice() : buffer;
  buffer = slicedArray.byteLength;
  const idStr = (bid.toString(16) + ":" + marker).toString();
  const chunkId = stringToChunk(idStr);
  req.completedRegularChunks.push(chunkId, slicedArray);
}
async function generateHtmlLikeEmbed(parser, contentParser, textConverter, routePath, config) {
  const { node } = routePath;
  let counterValue = htmlTemplateLiteralCounterIncrementer();
  htmlTemplateLiteralCounterIncrementer = (counterValue + 1) >>> 0;

  const createPlaceholder = index => `HTML_PLACEHOLDER_${index}_${counterValue}_IN_JS`;

  const processedText = node.quasis
    .map((quasi, idx, quasisList) =>
      idx === quasisList.length - 1 ? quasi.value.cooked : quasi.value.cooked + createPlaceholder(idx),
    )
    .join("");

  const parsedExpressions = textConverterTemplateExpressions(routePath, contentParser);

  const regexForPlaceholders = new RegExp(createPlaceholder(String.raw`(\d+)`), "gu");

  let totalTopLevelElements = 0;
  const generatedDoc = await textConverter(processedText, {
    parser,
    __onHtmlRoot(rootNode) {
      totalTopLevelElements = rootNode.children.length;
    },
  });

  const transformedContent = mapContent(generatedDoc, (content) => {
    if (typeof content !== "string") return content;

    let parts = [];
    const splitContents = content.split(regexForPlaceholders);
    for (let i = 0; i < splitContents.length; i++) {
      let currentPart = splitContents[i];
      if (i % 2 === 0 && currentPart) {
        currentPart = uncookedTemplateElementValue(currentPart);
        if (config.htmlWhitespaceSensitive !== "ignore") {
          currentPart = currentPart.replaceAll(/<\/(?=script\b)/giu, String.raw`<\``);
        }
        parts.push(currentPart);
      } else {
        const placeholderIndex = Number(splitContents[i]);
        parts.push(parsedExpressions[placeholderIndex]);
      }
    }
    return parts;
  });

  const leadingSpace = /^\s/u.test(processedText) ? " " : "";
  const trailingSpace = /\s$/u.test(processedText) ? " " : "";

  const lineBreakChar =
    config.htmlWhitespaceSensitive === "ignore"
      ? hardline
      : leadingSpace && trailingSpace
        ? line
        : null;

  if (lineBreakChar) {
    return group(["`", indent([lineBreakChar, group(transformedContent)]), lineBreakChar, "`"]);
  }

  return label(
    { hug: false },
    group([
      "`",
      leadingSpace,
      totalTopLevelElements > 1 ? indent(group(transformedContent)) : group(transformedContent),
      trailingSpace,
      "`",
    ]),
  );
}
function startFlowing(request, destination) {
  if (13 === request.status)
    (request.status = 14), closeWithError(destination, request.fatalError);
  else if (14 !== request.status && null === request.destination) {
    request.destination = destination;
    try {
      flushCompletedChunks(request, destination);
    } catch (error) {
      logRecoverableError(request, error, null), fatalError(request, error);
    }
  }
}
function handleResponseData(fetchResult, dataId, buffer) {
  const { _chunks: chunks, } = fetchResult;
  const chunk = chunks.get(dataId);
  if (chunk && "pending" !== chunk.status) {
    chunk.reason.enqueueValue(buffer);
  } else {
    chunks.set(dataId, new ReactPromise("fulfilled", buffer, null, fetchResult));
  }
}
export function modifyTime(frame, interval, isForward, adjustOffset) {
    var seconds = Math.abs(interval._seconds),
        hours = Math.round(Math.abs(interval._hours)),
        minutes = Math.round(Math.abs(interval._minutes));

    if (!frame.isValid()) {
        // No op
        return;
    }

    adjustOffset = adjustOffset == null ? true : adjustOffset;

    if (minutes) {
        set(frame, 'Minutes', get(frame, 'Minutes') + minutes * isForward);
    }
    if (hours) {
        set(frame, 'Hours', get(frame, 'Hours') + hours * isForward);
    }
    if (seconds) {
        frame._d.setTime(frame._d.valueOf() + seconds * 1000 * isForward);
    }
    if (adjustOffset) {
        hooks.updateTimezone(frame, minutes || hours);
    }
}
function parseCustomDataPacket(payload, clientInfo, paramMarker) {
  payload = processResponse(clientInfo, paramMarker, null, payload);
  finalize(payload);
  payload = extractChunk(payload, 0);
  payload.then(function () {});
  if ("resolved" !== payload.state) throw payload.error;
  return payload.data;
}
function reportGlobalError(response, error) {
  response._closed = !0;
  response._closedReason = error;
  response._chunks.forEach(function (chunk) {
    "pending" === chunk.status && triggerErrorOnChunk(chunk, error);
  });
}
function handleServerErrorData(data, errorMessage) {
  const environment = errorMessage.env;
  errorMessage = buildFakeCallStack(
    data,
    errorMessage.stack,
    environment,
    Error.bind(
      null,
      errorMessage.message ||
        "An error happened in the Server Components render and no specific message was given"
    )
  );
  data = getRootTask(data, environment);
  data = undefined !== data ? data.run(errorMessage) : errorMessage();
  data.environmentName = environment;
  return data;
}
function MyApplication({ Element, props }) {
  return (
      <div class={styles.content}>
        {props && <Element {...props} />}
      </div>
    );
}
    function noop$1() {}
function preprocessScript(script, opts) {
  if ('string' === typeof script) {
    const req = resolveRequest();
    if (req) {
      let hints = req.hints;
      const key = "X|" + script;
      if (!hints.has(key)) {
        hints.add(key);
        opts ? emitHint(req, 'X', [script, opts]) : emitHint(req, 'X', script);
      }
    } else {
      previousDispatcher.X(script, opts);
    }
  }
}
function displayDeclareToken(route) {
  const { node } = route;

  return (
    // TypeScript
    node.declare ||
      (flowDeclareNodeTypes.has(node.type) &&
        "declare" !== path.parent.type)
      ? "declare "
      : ""
  );
}
export default function PageView({ member, setMember, content }) {
  return (
    <div className="wrapper max-w-screen-xl mx-auto">
      <Header member={member} setMember={setMember} />
      {content}
    </div>
  );
}
function g(a) {
  let x = 0;
  try {
    might_throw();
  } catch (e) {
    var y = 0;
  } finally {
    if (!Boolean(e)) {
      var x = 0;
    }
    let z = x; // error
  }
}
export function calculate() {
  return (
    b2() +
    b3() +
    b4() +
    b5() +
    b6() +
    b7() +
    b8() +
    b9() +
    b10() +
    b11() +
    b12() +
    b13() +
    b14() +
    b15() +
    b16() +
    b17() +
    b18() +
    b19() +
    b20() +
    b21() +
    b22() +
    b23() +
    b24()
  )
}
    function noop() {}
function handle_CustomError(custom, err) {
  return "undefined" != typeof CustomError ? handle_CustomError = CustomError : (handle_CustomError = function handle_CustomError(custom, err) {
    this.custom = custom, this.err = err, this.stack = new Error().stack;
  }, handle_CustomError.prototype = Object.create(Error.prototype, {
    constructor: {
      value: handle_CustomError,
      writable: !0,
      configurable: !0
    }
  })), new handle_CustomError(custom, err);
}
function socialMediaUpdate() {
  return outdent`
    ${chalk.bold.underline("Post on Social Media")}
    - Open ${chalk.cyan.underline("https://dashboard.twitter.com/tweets/dashboard")}
    - Make sure you are tweeting from the {yellow @CodeFormatterOfficial} account.
    - Share the release, including the blog post URL.
  `;
}
    function initializeFakeStack(response, debugInfo) {
      void 0 === debugInfo.debugStack &&
        (null != debugInfo.stack &&
          (debugInfo.debugStack = createFakeJSXCallStackInDEV(
            response,
            debugInfo.stack,
            null == debugInfo.env ? "" : debugInfo.env
          )),
        null != debugInfo.owner &&
          initializeFakeStack(response, debugInfo.owner));
    }
    d: function d() {
      var o,
        t = this.e,
        s = 0;
      function next() {
        for (; o = n.pop();) try {
          if (!o.a && 1 === s) return s = 0, _pushInstanceProperty(n).call(n, o), _Promise.resolve().then(next);
          if (o.d) {
            var r = o.d.call(o.v);
            if (o.a) return s |= 2, _Promise.resolve(r).then(next, err);
          } else s |= 1;
        } catch (r) {
          return err(r);
        }
        if (1 === s) return t !== e ? _Promise.reject(t) : _Promise.resolve();
        if (t !== e) throw t;
      }
      function err(n) {
        return t = t !== e ? new r(n, t) : n, next();
      }
      return next();
    }
function handleResponseStream(res, ref, typ) {
  const refInt = parseInt(ref.slice(2), 16);
  let controller;
  const readableType = new ReadableStream({
    type: typ,
    start: (c) => { controller = c; }
  });
  var lastBlockedChunk = null;
  resolveStream(res, refInt, readableType, {
    enqueueModel: function (json) {
      if (!lastBlockedChunk) {
        const chunk = new Chunk("resolved_model", json, -1, res);
        initializeModelChunk(chunk);
        switch (chunk.status) {
          case "fulfilled":
            controller.enqueue(chunk.value);
            break;
          default:
            chunk.then(
              v => controller.enqueue(v),
              e => controller.error(e)
            );
            lastBlockedChunk = chunk;
        }
      } else {
        const currentChunk = lastBlockedChunk;
        const newChunk = createPendingChunk(res).then(
          v => controller.enqueue(v),
          e => controller.error(e)
        );
        lastBlockedChunk = newChunk;
        currentChunk.then(() => {
          lastBlockedChunk === newChunk && (lastBlockedChunk = null);
          resolveModelChunk(newChunk, json, -1);
        });
      }
    },
    close: () => !null === lastBlockedChunk ? controller.close() : void 0,
    error: e => !null === lastBlockedChunk ? controller.error(e) : void 0
  });
  return readableType;
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
      REACT_CLIENT_REFERENCE$2 = Symbol.for("react.client.reference"),
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
                  } catch (error$2) {
                    ReactSharedInternals.thrownErrors.push(error$2);
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
    exports.useOptimistic = function (passthrough, reducer) {
      return resolveDispatcher().useOptimistic(passthrough, reducer);
    };
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
    exports.version = "19.1.0-canary-518d06d2-20241219";
    "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ &&
      "function" ===
        typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop &&
      __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  })();
