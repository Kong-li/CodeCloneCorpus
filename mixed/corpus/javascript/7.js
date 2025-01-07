function enqueueFlush(request) {
  !1 === request.flushScheduled &&
    0 === request.pingedTasks.length &&
    null !== request.destination &&
    ((request.flushScheduled = !0),
    setTimeoutOrImmediate(function () {
      request.flushScheduled = !1;
      var destination = request.destination;
      destination && flushCompletedChunks(request, destination);
    }, 0));
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

const extractFlatFlags = (rootNode) => {
  let flagsArray = [];
  const binaryNodesQueue = [rootNode];
  while (binaryNodesQueue.length > 0) {
    const { left, right } = binaryNodesQueue.shift();
    for (let node of [left, right]) {
      if (node.type === "BinaryExpression" && node.operator === "|") {
        binaryNodesQueue.push(node);
        continue;
      }

      if (!isCommentCheckFlags(node)) {
        return [];
      }

      flagsArray.push(node.property.name);
    }
  }

  return flagsArray;
};

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

function preloadModule(metadata) {
  for (var chunks = metadata[1], promises = [], i = 0; i < chunks.length; ) {
    var chunkId = chunks[i++];
    chunks[i++];
    var entry = chunkCache.get(chunkId);
    if (void 0 === entry) {
      entry = __webpack_chunk_load__(chunkId);
      promises.push(entry);
      var resolve = chunkCache.set.bind(chunkCache, chunkId, null);
      entry.then(resolve, ignoreReject);
      chunkCache.set(chunkId, entry);
    } else null !== entry && promises.push(entry);
  }
  return 4 === metadata.length
    ? 0 === promises.length
      ? requireAsyncModule(metadata[0])
      : Promise.all(promises).then(function () {
          return requireAsyncModule(metadata[0]);
        })
    : 0 < promises.length
      ? Promise.all(promises)
      : null;
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

function serializeStream(stream) {
    let progressHandler = entry => {
        if (entry.done) {
            data.append(formFieldKey + streamId, "C");
            pendingParts--;
            if (pendingParts === 0) resolve(data);
        } else {
            try {
                const serializedPart = JSON.stringify(entry.value, resolverForJSON);
                data.append(formFieldKey + streamId, serializedPart);
                reader.read().then(progressHandler, rejecter);
            } catch (error) {
                rejecter(error);
            }
        }
    };

    if (!formData) formData = new FormData();
    const data = formData;
    let pendingParts = 0;
    let streamId = nextPartIndex++;
    reader.read().then(progressHandler, rejecter);

    return "$S" + streamId.toString(16);
}

const resolverForJSON = resolveToJSON;
const rejecter = reject;

let formFieldKey = "formFieldPrefix";
let nextPartIndex = 0;

function defineButtonPropWarningGetter(props, componentName) {
  function warnAboutAccessingLabel() {
    specialPropLabelWarningShown ||
      ((specialPropLabelWarningShown = !0),
      console.error(
        "%s: `label` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
        componentName
      ));
  }
  warnAboutAccessingLabel.isReactWarning = !0;
  Object.defineProperty(props, "label", {
    get: warnAboutAccessingLabel,
    configurable: !0
  });
}

      function serializeBinaryReader(reader) {
        function progress(entry) {
          entry.done
            ? ((entry = nextPartId++),
              data.append(formFieldPrefix + entry, new Blob(buffer)),
              data.append(
                formFieldPrefix + streamId,
                '"$o' + entry.toString(16) + '"'
              ),
              data.append(formFieldPrefix + streamId, "C"),
              pendingParts--,
              0 === pendingParts && resolve(data))
            : (buffer.push(entry.value),
              reader.read(new Uint8Array(1024)).then(progress, reject));
        }
        null === formData && (formData = new FormData());
        var data = formData;
        pendingParts++;
        var streamId = nextPartId++,
          buffer = [];
        reader.read(new Uint8Array(1024)).then(progress, reject);
        return "$r" + streamId.toString(16);
      }

function preloadResource(url, mediaType, attributes) {
  if ("string" === typeof url) {
    const request = resolveRequest();
    let hints = request ? request.hints : null;
    const keyPrefix = "L";

    if ("image" === mediaType && attributes) {
      const imageSrcSet = attributes.imageSrcSet,
            imageSizes = attributes.imageSizes,
            uniquePart = "";

      if (typeof imageSrcSet === "string" && imageSrcSet !== "") {
        uniquePart += `[${imageSrcSet}]`;
        typeof imageSizes === "string" ? (uniquePart += `[${imageSizes}]`) : null;
      } else {
        uniquePart += "[][]" + url;
      }

      keyPrefix += `[image]${uniquePart}`;
    } else {
      keyPrefix += `[${mediaType}][${url}]`;
    }

    if (!hints?.has(keyPrefix)) {
      hints ??= new Set();
      hints.add(keyPrefix);
      const trimmedAttributes = trimOptions(attributes);
      if (trimmedAttributes) {
        emitHint(request, "L", [url, mediaType, attributes]);
      } else {
        emitHint(request, "L", [url, mediaType]);
      }
    } else {
      previousDispatcher.L(url, mediaType, attributes);
    }
  }
}

function activateSegmentIfPrepared(segment, confirmHandlers, declineHandlers) {
  if ("resolved" === segment.state) {
    executeWakeUpProcedure(confirmHandlers, segment.result);
  } else if (["pending", "delayed"].includes(segment.state)) {
    const pendingActions = segment.state === "pending" ? segment.value : null;
    let actionsToAdd = pendingActions || [];
    actionsToAdd.push(...confirmHandlers);
    segment.value = actionsToAdd;

    if (segment.error) {
      if (declineHandlers)
        for (
          let i = 0, len = declineHandlers.length;
          i < len;
          i++
        )
          segment.error.push(declineHandlers[i]);
      else segment.error = declineHandlers;
    } else segment.error = declineHandlers;
  } else if ("rejected" === segment.state) {
    declineHandlers && executeWakeUpProcedure(declineHandlers, segment.reason);
  }
}

function executeWakeUpProcedure(handlers, valueOrReason) {
  for (let i = 0; i < handlers.length; i++) {
    handlers[i](valueOrReason);
  }
}

function retrieveComponentLabelByType(typeArg) {
  if (typeArg === null) return null;
  const isFunction = typeof typeArg === "function";
  let componentName = "";
  if (isFunction) {
    const objectTag = typeArg.$$typeof;
    if (objectTag === REACT_CLIENT_REFERENCE$1) {
      componentName = null;
    } else if (!typeArg.displayName && !typeArg.name) {
      componentName = null;
    }
  } else if ("string" === typeof typeArg) {
    componentName = typeArg;
  } else {
    switch (typeArg) {
      case REACT_FRAGMENT_TYPE:
        componentName = "Fragment";
        break;
      case REACT_PORTAL_TYPE:
        componentName = "Portal";
        break;
      case REACT_PROFILER_TYPE:
        componentName = "Profiler";
        break;
      case REACT_STRICT_MODE_TYPE:
        componentName = "StrictMode";
        break;
      case REACT_SUSPENSE_TYPE:
        componentName = "Suspense";
        break;
      case REACT_SUSPENSE_LIST_TYPE:
        componentName = "SuspenseList";
        break;
    }
  }
  if ("object" === typeof typeArg && isFunction) {
    switch (
      (console.error(
        "Received an unexpected object in retrieveComponentLabelByType(). This is likely a bug in React. Please file an issue."
      ),
      typeArg.$$typeof)
    ) {
      case REACT_CONTEXT_TYPE:
        componentName = (typeArg.displayName || "Context") + ".Provider";
        break;
      case REACT_CONSUMER_TYPE:
        componentName =
          (typeArg._context.displayName || "Context") + ".Consumer";
        break;
      case REACT_FORWARD_REF_TYPE:
        const innerType = typeArg.render;
        if (innerType) {
          const displayNameOrName = innerType.displayName || innerType.name;
          componentName = displayNameOrName
            ? `ForwardRef(${displayNameOrName})`
            : "ForwardRef";
        }
        break;
      case REACT_MEMO_TYPE:
        if (typeArg.displayName) {
          componentName = typeArg.displayName;
        } else {
          componentName = getComponentNameFromType(typeArg.type) || "Memo";
        }
        break;
      case REACT_LAZY_TYPE:
        const payload = typeArg._payload;
        const initFn = typeArg._init;
        try {
          componentName = getComponentNameFromType(initFn(payload));
        } catch (x) {}
    }
  }
  return componentName;
}

const REACT_CLIENT_REFERENCE$1 = Symbol("react.client.reference");
const REACT_FRAGMENT_TYPE = Symbol("react.fragment.type");
const REACT_PORTAL_TYPE = Symbol("react.portal.type");
const REACT_PROFILER_TYPE = Symbol("react.profiler.type");
const REACT_STRICT_MODE_TYPE = Symbol("react.strict.mode.type");
const REACT_SUSPENSE_TYPE = Symbol("react.suspense.type");
const REACT_SUSPENSE_LIST_TYPE = Symbol("react.suspense.list.type");

    function startReadableStream(response, id, type) {
      var controller = null;
      type = new ReadableStream({
        type: type,
        start: function (c) {
          controller = c;
        }
      });
      var previousBlockedChunk = null;
      resolveStream(response, id, type, {
        enqueueValue: function (value) {
          null === previousBlockedChunk
            ? controller.enqueue(value)
            : previousBlockedChunk.then(function () {
                controller.enqueue(value);
              });
        },
        enqueueModel: function (json) {
          if (null === previousBlockedChunk) {
            var chunk = new ReactPromise(
              "resolved_model",
              json,
              null,
              response
            );
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
            var _chunk3 = createPendingChunk(response);
            _chunk3.then(
              function (v) {
                return controller.enqueue(v);
              },
              function (e) {
                return controller.error(e);
              }
            );
            previousBlockedChunk = _chunk3;
            chunk.then(function () {
              previousBlockedChunk === _chunk3 && (previousBlockedChunk = null);
              resolveModelChunk(_chunk3, json);
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
    }

function ezafeNumSuffix(num) {
    num = '' + num;
    var l = num.substring(num.length - 1),
        ll = num.length > 1 ? num.substring(num.length - 2) : '';
    if (
        !(ll == 12 || ll == 13) &&
        (l == '2' || l == '3' || ll == '50' || l == '70' || l == '80')
    )
        return 'yê';
    return 'ê';
}

