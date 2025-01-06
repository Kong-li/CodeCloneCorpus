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
function processModule(response, id, model) {
  var parts = response._parts,
    part = parts.get(id);
  model = JSON.parse(model, response._parseJSON);
  var clientRef = resolveClientConfig(
    response._config,
    model
  );
  if ((model = preloadData(clientRef))) {
    if (part) {
      var blockedPart = part;
      blockedPart.status = "blocked";
    } else
      (blockedPart = new ReactPromise("blocked", null, null, response)),
        parts.set(id, blockedPart);
    model.then(
      function () {
        return processModulePart(blockedPart, clientRef);
      },
      function (error) {
        return triggerErrorOnPart(blockedPart, error);
      }
    );
  } else
    part
      ? processModulePart(part, clientRef)
      : parts.set(
          id,
          new ReactPromise(
            "resolved_module",
            clientRef,
            null,
            response
          )
        );
}
function handleSystemError(responseData, errorMessage) {
  responseData._completed = !0;
  responseData._completionReason = errorMessage;
  responseData._segments.forEach(function (segment) {
    "pending" === segment.status && reportFailureOnSegment(segment, errorMessage);
  });
}
function convert(text) {
  try {
    let parsed = parseYml(text);

    /**
     * suppress `comment not found` warning
     *
     * comments are managed in printer-yaml.js without using `printComment`
     * to ensure that it always throws errors even if we handled them correctly
     */
    if (parsed.comments) {
      delete parsed.comments;
    }

    return parsed;
  } catch (/** @type {any} */ err) {
    if (err?.position) {
      throw createError(err.message, {
        loc: err.position,
        cause: err,
      });
    }

    /* c8 ignore next */
    throw err;
  }
}
function executeTask(instruction) {
  var prevHandler = GlobalHandlers.G;
  GlobalHandlers.G = TaskDispatcher;
  var prevInstruction = currentInstruction$1;
  currentInstruction = instruction;
  var hadPendingTasks = 0 < instruction.pendingTasks.length;
  try {
    var processedTasks = instruction.processedTasks;
    instruction.processedTasks = [];
    for (var j = 0; j < processedTasks.length; j++)
      executeTask(instruction, processedTasks[j]);
    null !== instruction.target &&
      sendCompletedMessages(instruction, instruction.target);
    if (hadPendingTasks && 0 === instruction.pendingTasks.length) {
      var onAllComplete = instruction.onAllComplete;
      onAllComplete();
    }
  } catch (error) {
    logUnrecoverableError(instruction, error, null), criticalFailure(instruction, error);
  } finally {
    (GlobalHandlers.G = prevHandler),
      (currentInstruction$1 = null),
      (currentInstruction = prevInstruction);
  }
}
function checkForParseInt(node) {
    if (
        node.type === "MemberExpression" &&
        !node.computed &&
        (node.property.type === "Identifier" && node.property.name === "parseInt")
    ) {
        return true;
    }
    return false;
}
    function disabledLog() {}
function resolveModelChunk(chunk, value, id) {
  if ("pending" !== chunk.status)
    (chunk = chunk.reason),
      "C" === value[0]
        ? chunk.close("C" === value ? '"$undefined"' : value.slice(1))
        : chunk.enqueueModel(value);
  else {
    var resolveListeners = chunk.value,
      rejectListeners = chunk.reason;
    chunk.status = "resolved_model";
    chunk.value = value;
    chunk.reason = id;
    if (null !== resolveListeners)
      switch ((initializeModelChunk(chunk), chunk.status)) {
        case "fulfilled":
          wakeChunk(resolveListeners, chunk.value);
          break;
        case "pending":
        case "blocked":
        case "cyclic":
          if (chunk.value)
            for (value = 0; value < resolveListeners.length; value++)
              chunk.value.push(resolveListeners[value]);
          else chunk.value = resolveListeners;
          if (chunk.reason) {
            if (rejectListeners)
              for (value = 0; value < rejectListeners.length; value++)
                chunk.reason.push(rejectListeners[value]);
          } else chunk.reason = rejectListeners;
          break;
        case "rejected":
          rejectListeners && wakeChunk(rejectListeners, chunk.reason);
      }
  }
}
decorateClasses: function decorateClasses(classes, decorators) {
  const newClasses = [];
  const postProcessors = {
    "static": [],
    prototype: [],
    own: []
  };
  classes.forEach((cls, index) => {
    this.setClassPlacement(cls, postProcessors);
  });
  classes.forEach((cls, index) => {
    if (!_hasDecorators(cls)) return newClasses.push(cls);
    const { element, extras, finishers } = this.decorateClassElement(cls, postProcessors);
    newClasses.push(element);
    newClasses.push(...extras);
    postProcessors.static.push(...finishers.prototype);
    postProcessors.prototype.push(...finishers.own);
  });
  if (!decorators) {
    return { elements: newClasses };
  }
  const result = this.decorateConstructor(newClasses, decorators);
  result.finishers = [...postProcessors.static, ...postProcessors.prototype, ...result.finishers];
  return result;
}
function loadResourceReference(fetchResult, headerData, parentNode, index) {
  if (!fetchResult._resourceConfig)
    return createBoundResourceReference(
      headerData,
      fetchResult._getDataFromServer,
      fetchResult._encodeQueryAction,
      fetchResult._debugFindSourceMapURL
    );
  var resourceRef = resolveResourceReference(
    fetchResult._resourceConfig,
    headerData.id
  );
  if ((fetchResult = preloadAsset(resourceRef)))
    parentNode.bound && (fetchResult = Promise.all([fetchResult, parentNode.bound]));
  else if (parentNode.bound) fetchResult = Promise.resolve(parentNode.bound);
  else return importAsset(resourceRef);
  if (initializingHandler) {
    var handler = initializingHandler;
    handler.deps++;
  } else
    handler = initializingHandler = {
      parent: null,
      chunk: null,
      value: null,
      deps: 1,
      errored: !1
    };
  fetchResult.then(
    function () {
      var resolvedValue = importAsset(resourceRef);
      if (parentNode.bound) {
        var boundArgs = parentNode.bound.value.slice(0);
        boundArgs.unshift(null);
        resolvedValue = resolvedValue.bind.apply(resolvedValue, boundArgs);
      }
      parentNode[index] = resolvedValue;
      "" === index &&
        null === handler.value &&
        (handler.value = resolvedValue);
      if (
        parentNode[0] === REACT_ELEMENT_TYPE &&
        "object" === typeof handler.value &&
        null !== handler.value &&
        handler.value.$$typeof === REACT_ELEMENT_TYPE
      )
        switch (((boundArgs = handler.value), index)) {
          case "3":
            boundArgs.props = resolvedValue;
            break;
          case "4":
            boundArgs._owner = resolvedValue;
        }
      handler.deps--;
      0 === handler.deps &&
        ((resolvedValue = handler.chunk),
        null !== resolvedValue &&
          "blocked" === resolvedValue.status &&
          ((boundArgs = resolvedValue.value),
          (resolvedValue.status = "fulfilled"),
          (resolvedValue.value = handler.value),
          null !== boundArgs && wakeChunk(boundArgs, handler.value)));
    },
    function (error) {
      if (!handler.errored) {
        var blockedValue = handler.value;
        handler.errored = !0;
        handler.value = error;
        var chunk = handler.chunk;
        null !== chunk &&
          "blocked" === chunk.status &&
          ("object" === typeof blockedValue &&
            null !== blockedValue &&
            blockedValue.$$typeof === REACT_ELEMENT_TYPE &&
            ((blockedValue = {
              name: getComponentNameFromType(blockedValue.type) || "",
              owner: blockedValue._owner
            }),
            (chunk._debugInfo || (chunk._debugInfo = [])).push(
              blockedValue
            )),
          triggerErrorOnChunk(chunk, error));
      }
    }
  );
  return null;
}
function processCombinations(comboList) {
  const errorCollection = [];
  for (let i = 0; i < comboList.length; i++) {
    const combination = comboList[i];
    try {
      return combination();
    } catch (err) {
      errorCollection.push(err);
    }
  }

  if (errorCollection.length > 0) {
    throw { ...new Error("All combinations failed"), errors: errorCollection };
  }
}
export async function action3(a, b) {
  'use server'
  return (
    <div>
      {a}
      {b}
    </div>
  )
}
export default function mergeValues(x, y, z) {
    if (x != null) {
        return x;
    }
    if (y != null) {
        return y;
    }
    return z;
}
export function seasonsShortRegex(isStrict) {
    if (this._seasonsParseExact) {
        if (!hasOwnProp(this, '_seasonsRegex')) {
            computeSeasonsParse.call(this);
        }
        if (isStrict) {
            return this._seasonsShortStrictRegex;
        } else {
            return this._seasonsShortRegex;
        }
    } else {
        if (!hasOwnProp(this, '_seasonsShortRegex')) {
            this._seasonsShortRegex = defaultSeasonsShortRegex;
        }
        return this._seasonsShortStrictRegex && isStrict
            ? this._seasonsShortStrictRegex
            : this._seasonsShortRegex;
    }
}
function parseReadableStream(response, reference, type) {
  reference = parseInt(reference.slice(2), 16);
  var controller = null;
  type = new ReadableStream({
    type: type,
    start: function (c) {
      controller = c;
    }
  });
  var previousBlockedChunk = null;
  resolveStream(response, reference, type, {
    enqueueModel: function (json) {
      if (null === previousBlockedChunk) {
        var chunk = new Chunk("resolved_model", json, -1, response);
        initializeModelChunk(chunk);
        "fulfilled" === chunk.status
          ? controller.enqueue(chunk.value)
          : (chunk.then(
              function (v) {
                return controller.enqueue(v);
              },
              function (e) {
                return controller.error(e);
              }
            ),
            (previousBlockedChunk = chunk));
      } else {
        chunk = previousBlockedChunk;
        var chunk$26 = createPendingChunk(response);
        chunk$26.then(
          function (v) {
            return controller.enqueue(v);
          },
          function (e) {
            return controller.error(e);
          }
        );
        previousBlockedChunk = chunk$26;
        chunk.then(function () {
          previousBlockedChunk === chunk$26 && (previousBlockedChunk = null);
          resolveModelChunk(chunk$26, json, -1);
        });
      }
    },
    close: function () {
      if (null === previousBlockedChunk) controller.close();
      else {
        var blockedChunk = previousBlockedChunk;
        previousBlockedChunk = null;
        blockedChunk.then(function () {
          return controller.close();
        });
      }
    },
    error: function (error) {
      if (null === previousBlockedChunk) controller.error(error);
      else {
        var blockedChunk = previousBlockedChunk;
        previousBlockedChunk = null;
        blockedChunk.then(function () {
          return controller.error(error);
        });
      }
    }
  });
  return type;
}
  function progress(entry) {
    if (!aborted)
      if (entry.done) {
        request.abortListeners.delete(abortIterable);
        if (void 0 === entry.value)
          var endStreamRow = streamTask.id.toString(16) + ":C\n";
        else
          try {
            var chunkId = outlineModel(request, entry.value);
            endStreamRow =
              streamTask.id.toString(16) +
              ":C" +
              stringify(serializeByValueID(chunkId)) +
              "\n";
          } catch (x) {
            error(x);
            return;
          }
        request.completedRegularChunks.push(stringToChunk(endStreamRow));
        enqueueFlush(request);
        aborted = !0;
      } else
        try {
          (streamTask.model = entry.value),
            request.pendingChunks++,
            emitChunk(request, streamTask, streamTask.model),
            enqueueFlush(request),
            iterator.next().then(progress, error);
        } catch (x$8) {
          error(x$8);
        }
  }
export default function FooterComponent(props) {
  return (
    <footer className="bg-secondary border-t border-primary">
      <div>
        <Container>
          <div className="py-28 flex lg:flex-row items-center justify-center lg:justify-between">
            <h3 className="text-4xl lg:text-5xl font-bold tracking-tighter leading-tight text-center lg:text-left mb-10 lg:mb-0 lg:pr-4">
              Statically Generated with Next.js.
            </h3>
            <div className="flex flex-col lg:flex-row justify-center items-center lg:pl-4 w-full lg:w-auto">
              <a
                href="https://nextjs.org/docs/basic-features/pages"
                className="mx-3 bg-black hover:bg-white hover:text-black border border-black text-white font-bold py-2 px-8 duration-200 transition-colors mb-6 lg:mb-0"
              >
                Read Documentation
              </a>
              <a
                href={`https://github.com/vercel/next.js/tree/canary/examples/${EXAMPLE_PATH}`}
                className="mx-3 font-bold hover:underline"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </Container>
      </div>
    </footer>
  );
}
export function fixPrettierVersion(version) {
  const match = version.match(/^\d+\.\d+\.\d+-pr.(\d+)$/u);
  if (match) {
    return `pr-${match[1]}`;
  }
  return version;
}
        function checkForPartOfClassBody(firstToken) {
            for (let token = firstToken;
                token.type === "Punctuator" && !astUtils.isClosingBraceToken(token);
                token = sourceCode.getTokenAfter(token)
            ) {
                if (astUtils.isSemicolonToken(token)) {
                    report(token);
                }
            }
        }
function timeAgoWithPlural(amount, withoutSuffix, type) {
    let formatMap = {
        ss: withoutSuffix ? 'секунда_секунды_секунд' : 'секунду_секунды_секунд',
        mm: withoutSuffix ? 'минута_минуты_минут' : 'минуту_минуты_минут',
        hh: 'час_часа_часов',
        dd: 'день_дня_дней',
        ww: 'неделя_недели_недель',
        MM: 'месяц_месяца_месяцев',
        yy: 'год_года_лет',
    };
    if (type === 'm') {
        return withoutSuffix ? 'минута' : 'минуту';
    } else {
        const pluralFormat = formatMap[type];
        const suffix = withoutSuffix ? '' : '_plural';
        return amount + ' ' + pluralFormat.split('_')[amount % 10 !== 1 || amount % 10 === 0 ? 2 : (amount % 10 >= 2 && amount % 10 <= 4) ? 1 : 0] + suffix;
    }
}
function fetchVisitorFields(visitorFields, node) {
    const keySet = visitorFields[node.kind];

    if (!keySet) {
        keySet = vk.fetchKeys(node);
        if (vk.isUnknownNodeType(node)) {
            debug("Unknown node kind \"%s\": Estimated visitor fields %j", node.kind, keySet);
        }
    }

    return keySet;
}
function initializeModelChunk(chunk) {
  var prevChunk = initializingChunk,
    prevBlocked = initializingChunkBlockedModel;
  initializingChunk = chunk;
  initializingChunkBlockedModel = null;
  var rootReference = -1 === chunk.reason ? void 0 : chunk.reason.toString(16),
    resolvedModel = chunk.value;
  chunk.status = "cyclic";
  chunk.value = null;
  chunk.reason = null;
  try {
    var rawModel = JSON.parse(resolvedModel),
      value = reviveModel(
        chunk._response,
        { "": rawModel },
        "",
        rawModel,
        rootReference
      );
    if (
      null !== initializingChunkBlockedModel &&
      0 < initializingChunkBlockedModel.deps
    )
      (initializingChunkBlockedModel.value = value), (chunk.status = "blocked");
    else {
      var resolveListeners = chunk.value;
      chunk.status = "fulfilled";
      chunk.value = value;
      null !== resolveListeners && wakeChunk(resolveListeners, value);
    }
  } catch (error) {
    (chunk.status = "rejected"), (chunk.reason = error);
  } finally {
    (initializingChunk = prevChunk),
      (initializingChunkBlockedModel = prevBlocked);
  }
}
export async function initializeWebServer(port) {
  const sf = new SimpleFlare({
    scriptPath: path.resolve(__dirname, 'dist/server/entry-server.js'),
    port,
    plugins: true,
    compatibilityFlags: ['web_compat'],
  })
  await sf.ready
  return { sf }
}
function displayWarningForSpecialProp(key, specialPropKeyWarningShown, displayName) {
  if (!specialPropKeyWarningShown) {
    const shown = true;
    console.error(
      "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
      displayName
    );
    specialPropKeyWarningShown = shown;
  }
}
    function noop() {}
function startWork(request) {
  request.flushScheduled = null !== request.destination;
  scheduleMicrotask(function () {
    return performWork(request);
  });
  scheduleWork(function () {
    10 === request.status && (request.status = 11);
  });
}
function classPropDef(def, objRef, propName, descVal, initVals, kindType, isStaticFlag, isPrivateFlag, valueObj, hasPrivateMark) {
  var kindDesc;
  switch (kindType) {
    case 1:
      kindDesc = "accessor";
      break;
    case 2:
      kindDesc = "method";
      break;
    case 3:
      kindDesc = "getter";
      break;
    case 4:
      kindDesc = "setter";
      break;
    default:
      kindDesc = "field";
  }
  var getter,
    setter,
    context = {
      kind: kindDesc,
      name: isPrivateFlag ? "#" + propName : propName,
      static: isStaticFlag,
      private: isPrivateFlag
    },
    decoratorEndRef = {
      v: !1
    };
  if (0 !== kindType && (context.addInitializer = createAddInitializerFunction(initVals, decoratorEndRef)), isPrivateFlag || 0 !== kindType && 2 !== kindType) {
    if (2 === kindType) getter = function get(instance) {
      return assertInstanceIfPrivate(hasPrivateMark, instance), descVal.value;
    };else {
      var t = 0 === kindType || 1 === kindType;
      (t || 3 === kindType) && (getter = isPrivateFlag ? function (instance) {
        return assertInstanceIfPrivate(hasPrivateMark, instance), descVal.get.call(instance);
      } : function (instance) {
        return descVal.get.call(instance);
      }), (t || 4 === kindType) && (setter = isPrivateFlag ? function (instance, val) {
        assertInstanceIfPrivate(hasPrivateMark, instance), descVal.set.call(instance, val);
      } : function (instance, val) {
        descVal.set.call(instance, val);
      });
    }
  } else getter = function get(target) {
    return target[propName];
  }, 0 === kindType && (setter = function set(target, v) {
    target[propName] = v;
  });
  var owns = isPrivateFlag ? hasPrivateMark.bind() : function (target) {
    return propName in target;
  };
  context.accessor = getter && setter ? {
    get: getter,
    set: setter,
    owns: owns
  } : getter ? {
    get: getter,
    owns: owns
  } : {
    set: setter,
    owns: owns
  };
  try {
    return def.call(objRef, valueObj, context);
  } finally {
    decoratorEndRef.v = !0;
  }
}
function handleRelativeDuration(value, isPresent, period, upcoming) {
    var patterns = {
        s: ['kurze Sekunde', 'kürzerer Sekunde'],
        m: ['eine Minute', 'einer Minute'],
        h: ['ein Stunde', 'einer Stunde'],
        d: ['ein Tag', 'einem Tag'],
        M: ['ein Monat', 'einem Monat'],
        y: ['ein Jahr', 'einem Jahr']
    };
    return isPresent ? patterns[period][0] : patterns[period][1];
}
function checkFunctionCompositionParams(params) {
  if (params.length <= 1) return false;

  let hasArrowOrFunc = false;
  for (let i = 0; i < params.length; i++) {
    const param = params[i];
    if ('function' === typeof param || 'arrow' === param.type) {
      hasArrowOrFunc = true;
    } else if ('callExpression' === param.type) {
      for (const childArg of getCallArguments(param)) {
        if ('function' === typeof childArg || 'arrow' === childArg.type) {
          return true;
        }
      }
    }
  }

  return hasArrowOrFunc && hasArrowOrFunc > 1;
}
function parseModelString2(data, baseObject, keyField, valueInfo) {
  if ("$" === valueInfo[0]) {
    if ("$" === valueInfo)
      return (
        null !== initialHandler &&
          "0" === keyField &&
          (initialHandler = {
            parent: initialHandler,
            chunk: null,
            value: null,
            deps: 0,
            errored: !1
          }),
        REACT_ELEMENT_TYPE2
      );
    switch (valueInfo[1]) {
      case "$":
        return valueInfo.slice(1);
      case "L":
        return (
          (baseObject = parseInt(valueInfo.slice(2), 16)),
          (data = getChunkData(data, baseObject)),
          createLazyChunkWrapper2(response)
        );
      case "@":
        if (2 === valueInfo.length) return new Promise(function () {});
        baseObject = parseInt(valueInfo.slice(2), 16);
        return getChunkData(data, baseObject);
      case "S":
        return Symbol.for(valueInfo.slice(2));
      case "F":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            loadServerReference2
          )
        );
      case "T":
        baseObject = "$" + valueInfo.slice(2);
        data = data._tempRefs;
        if (null == data)
          throw Error(
            "Missing a temporary reference set but the RSC response returned a temporary reference. Pass a temporaryReference option with the set that was used with the reply."
          );
        return data.get(baseObject);
      case "Q":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(data, valueInfo, baseObject, keyField, createMap2)
        );
      case "W":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            createSet2
          )
        );
      case "B":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(data, valueInfo, baseObject, keyField, createBlob2)
        );
      case "K":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            createFormData2
          )
        );
      case "Z":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            resolveErrorDev2
          )
        );
      case "i":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            extractIterator2
          )
        );
      case "I":
        return Infinity;
      case "-":
        return "$-0" === valueInfo ? -0 : -Infinity;
      case "N":
        return NaN;
      case "u":
        return;
      case "D":
        return new Date(Date.parse(valueInfo.slice(2)));
      case "n":
        return BigInt(valueInfo.slice(2));
      case "E":
        try {
          return (0, eval)(valueInfo.slice(2));
        } catch (x) {
          return function () {};
        }
      case "Y":
        return (
          Object.defineProperty(baseObject, keyField, {
            get: function () {
              return "This object has been omitted by React in the console log to avoid sending too much data from the server. Try logging smaller or more specific objects.";
            },
            enumerable: !0,
            configurable: !1
          }),
          null
        );
      default:
        return (
          (valueInfo = valueInfo.slice(1)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            createModel2
          )
        );
    }
  }
  return valueInfo;
}
function baz(flag: boolean) {
  let upperLimit = 0;
  while (!flag && upperLimit > -1) {
    ++upperLimit;
  }
  if (upperLimit === 0) {
    console.log('this is still reachable');
  }
}
function preventTrailingDot(node) {
    const latestElement = getLastComponent(node);

    if (!latestElement || (node.kind === "ClassDeclaration" && latestElement.kind !== "MethodDefinition")) {
        return;
    }

    const trailingSymbol = getFollowingToken(node, latestElement);

    if (astUtils.isPeriodToken(trailingSymbol)) {
        context.warn({
            node: latestElement,
            loc: trailingSymbol.loc,
            messageId: "unnecessary",
            *fix(fixer) {
                yield fixer.remove(trailingSymbol);

                /*
                 * Extend the range of the fix to include surrounding tokens to ensure
                 * that the element after which the dot is removed stays _last_.
                 * This intentionally makes conflicts in fix ranges with rules that may be
                 * adding or removing elements in the same autofix pass.
                 * https://github.com/eslint/eslint/issues/15660
                 */
                yield fixer.insertTextBefore(sourceCode.getTokenBefore(trailingSymbol), "");
                yield fixer.insertTextAfter(sourceCode.getTokenAfter(trailingSymbol), "");
            }
        });
    }
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
