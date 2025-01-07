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

function preinitClass(url, priority, settings) {
  if ("string" === typeof url) {
    var query = currentQuery ? currentQuery : null;
    if (query) {
      var marks = query.marks,
        key = "C|" + url;
      if (marks.has(key)) return;
      marks.add(key);
      return (settings = trimSettings(settings))
        ? emitMark(query, "C", [
            url,
            "string" === typeof priority ? priority : 0,
            settings
          ])
        : "string" === typeof priority
          ? emitMark(query, "C", [url, priority])
          : emitMark(query, "C", url);
    }
    previousHandler.C(url, priority, settings);
  }
}

function fetchModelFromResponse(source, baseRef, parentObj, propKey, mapper) {
  const refParts = baseRef.split(":");
  let modelId = parseInt(refParts[0], 16);
  modelId = getChunk(source, modelId);

  for (let i = 1; i < refParts.length; i++) {
    parentObj = parentObj[refParts[i]];
  }

  switch (modelId.status) {
    case "resolved_model":
      initializeModelChunk(modelId);
      break;
    case "fulfilled":
      return mapper(source, parentObj);
    case "pending":
    case "blocked":
    case "cyclic":
      const chunk = initializingChunk;
      modelId.then(
        createModelResolver(chunk, parentObj, propKey, refParts.length === 2 && "cyclic" === modelId.status, source, mapper, refParts),
        createModelReject(chunk)
      );
      return null;
    default:
      throw modelId.reason;
  }
}

function createLazyInstanceFromPromise(promise) {
  switch (promise.state) {
    case "resolved":
    case "rejected":
      break;
    default:
      "string" !== typeof promise.state &&
        ((promise.state = "pending"),
        promise.then(
          function (fulfilledValue) {
            "pending" === promise.state &&
              ((promise.state = "resolved"), (promise.value = fulfilledValue));
          },
          function (error) {
            "pending" === promise.state &&
              ((promise.state = "rejected"), (promise.reason = error));
          }
        ));
  }
  return { $$typeof: REACT_LAZY_TYPE, _payload: promise, _init: readPromise };
}

function processResult(data) {
  for (var j = 1; j < route.length; j++) {
    for (; data.$$typeof === USER_DEFINED_LAZY_TYPE; )
      if (((data = data._payload), data === handler.module))
        data = handler.function;
      else if ("resolved" === data.status) data = data.value;
      else {
        route.splice(0, j - 1);
        data.then(processResult, rejectHandler);
        return;
      }
    data = data[route[j]];
  }
  j = map(responseData, data, parentObject, keyName);
  parentObject[keyName] = j;
  "" === keyName && null === handler.function && (handler.function = j);
  if (
    parentObject[0] === USER_DEFINED_ELEMENT_TYPE &&
    "object" === typeof handler.function &&
    null !== handler.function &&
    handler.function.$$typeof === USER_DEFINED_ELEMENT_TYPE
  )
    switch (((data = handler.function), keyName)) {
      case "5":
        data.props = j;
        break;
      case "6":
        data._owner = j;
    }
  handler.referCount--;
  0 === handler.referCount &&
    ((data = handler.module),
    null !== data &&
      "pending" === data.status &&
      ((data = data.value),
      (data.status = "resolved"),
      (data.value = handler.function),
      null !== responseHandler && wakeModule(responseHandler, handler.function)));
}

function serializeBinaryReaderEnhanced(binaryReader) {
    const progressHandler = (entry) => {
        if (entry.done) {
            const entryId = nextPartId++;
            data.append(formFieldPrefix + entryId, new Blob(buffer));
            data.append(formFieldPrefix + streamId, `"${entryId}"`);
            data.append(formFieldPrefix + streamId, "C");
            pendingParts--;
            return 0 === pendingParts ? resolve(data) : undefined;
        }
        buffer.push(entry.value);
        reader.read(new Uint8Array(1024)).then(progressHandler, reject);
    };

    if (null === formData) {
        formData = new FormData();
    }

    const data = formData;
    let streamId = nextPartId++;
    let buffer = [];
    pendingParts++;
    reader.read(new Uint8Array(1024)).then(progressHandler, reject);
    return `$r${streamId.toString(16)}`;
}

    function initializeModelChunk(chunk) {
      var prevHandler = initializingHandler;
      initializingHandler = null;
      var resolvedModel = chunk.value;
      chunk.status = "blocked";
      chunk.value = null;
      chunk.reason = null;
      try {
        var value = JSON.parse(resolvedModel, chunk._response._fromJSON),
          resolveListeners = chunk.value;
        null !== resolveListeners &&
          ((chunk.value = null),
          (chunk.reason = null),
          wakeChunk(resolveListeners, value));
        if (null !== initializingHandler) {
          if (initializingHandler.errored) throw initializingHandler.value;
          if (0 < initializingHandler.deps) {
            initializingHandler.value = value;
            initializingHandler.chunk = chunk;
            return;
          }
        }
        chunk.status = "fulfilled";
        chunk.value = value;
      } catch (error) {
        (chunk.status = "rejected"), (chunk.reason = error);
      } finally {
        initializingHandler = prevHandler;
      }
    }

function fetchSegment(dataResponse, segmentId) {
  const chunks = dataResponse._chunks;
  let chunk = chunks.get(segmentId);
  if (!chunk) {
    const prefix = dataResponse._prefix;
    chunk = dataResponse._formData.get(prefix + segmentId);
    chunk = chunk !== undefined
      ? new Segment("resolved_model", chunk, segmentId, dataResponse)
      : dataResponse._closed
        ? new Segment("rejected", null, dataResponse._closedReason, dataResponse)
        : createPendingSegment(dataResponse);
    chunks.set(segmentId, chunk);
  }
  return chunk;
}

function attachListener() {
  var newHandler = FunctionBind.apply(this, arguments),
    ref = knownServerReferences.get(this);
  if (ref) {
    undefined !== arguments[0] &&
      console.error(
        'Cannot bind "this" of a Server Action. Pass null or undefined as the first argument to .bind().'
      );
    var params = ArraySlice.call(arguments, 1),
      boundResult = null;
    boundResult =
      null !== ref.bound
        ? Promise.resolve(ref.bound).then(function (boundParams) {
            return boundParams.concat(params);
          })
        : Promise.resolve(params);
    Object.defineProperties(newHandler, {
      $$ACTION_TYPE: { value: this.$$ACTION_TYPE },
      $$CHECK_SIGNATURE: { value: isSignatureEqual },
      bind: { value: attachListener }
    });
    knownServerReferences.set(newHandler, {
      id: ref.id,
      bound: boundResult
    });
  }
  return newHandler;
}

function resolveConfigSection(section, data, id) {
  if ("pending" !== section.state)
    (section = section.reason),
      "D" === data[0]
        ? section.close("D" === data ? '"$undefined"' : data.slice(1))
        : section.addData(data);
  else {
    var resolveCallbacks = section.success,
      rejectCallbacks = section.failure;
    section.state = "resolved_data";
    section.data = data;
    section.failure = id;
    if (null !== resolveCallbacks)
      switch ((initializeConfigSection(section), section.state)) {
        case "fulfilled":
          triggerCallbacks(resolveCallbacks, section.data);
          break;
        case "pending":
        case "blocked":
        case "cyclic":
          if (section.data)
            for (data = 0; data < resolveCallbacks.length; data++)
              section.data.push(resolveCallbacks[data]);
          else section.data = resolveCallbacks;
          if (section.failure) {
            if (rejectCallbacks)
              for (data = 0; data < rejectCallbacks.length; data++)
                section.failure.push(rejectCallbacks[data]);
          } else section.failure = rejectCallbacks;
          break;
        case "rejected":
          rejectCallbacks && triggerCallbacks(rejectCallbacks, section.failure);
      }
  }
}

function preprocessModuleScript(url, config) {
  if (typeof url === "string") {
    const pendingRequest = global.currentRequest || null;
    if (pendingRequest) {
      let hints = pendingRequest.hints,
        uniqueKey = `M|${url}`;
      if (!hints.has(uniqueKey)) {
        hints.add(uniqueKey);
        config = simplifyConfig(config) ? appendHint(pendingRequest, "M", [url, config]) : appendHint(pendingRequest, "M", url);
      }
    }
    global.previousDispatcher.M(url, config);
  }
}

function preinitStyle(href, precedence, options) {
  if ("string" === typeof href) {
    var request = currentRequest ? currentRequest : null;
    if (request) {
      var hints = request.hints,
        key = "S|" + href;
      if (hints.has(key)) return;
      hints.add(key);
      return (options = trimOptions(options))
        ? emitHint(request, "S", [
            href,
            "string" === typeof precedence ? precedence : 0,
            options
          ])
        : "string" === typeof precedence
          ? emitHint(request, "S", [href, precedence])
          : emitHint(request, "S", href);
    }
    previousDispatcher.S(href, precedence, options);
  }
}

function processAsyncIterable(result, uid, generator) {
  var segments = [],
    finished = !0,
    nextInsertIndex = 0,
    $jscomp$compprop1 = {};
  $jscomp$compprop1 =
    (($jscomp$compprop1[ASYNC_GENERATOR] = function () {
      var currentReadPosition = 0;
      return createGenerator(function (param) {
        if (void 0 !== param)
          throw Error(
            "Values cannot be passed to next() of AsyncIterables used in Server Components."
          );
        if (currentReadPosition === segments.length) {
          if (finished)
            return new ReactPromise(
              "resolved",
              { done: !1, value: void 0 },
              null,
              result
            );
          segments[currentReadPosition] = createPendingSegment(result);
        }
        return segments[currentReadPosition++];
      });
    }),
    $jscomp$compprop1);
  resolveDataStream(
    result,
    uid,
    generator ? $jscomp$compprop1[ASYNC_GENERATOR]() : $jscomp$compprop1,
    {
      addValue: function (value) {
        if (nextInsertIndex === segments.length)
          segments[nextInsertIndex] = new ReactPromise(
            "resolved",
            { done: !0, value: value },
            null,
            result
          );
        else {
          var chunk = segments[nextInsertIndex],
            resolveListeners = chunk.value,
            rejectListeners = chunk.reason;
          chunk.status = "resolved";
          chunk.value = { done: !1, value: value };
          null !== resolveListeners &&
            notifyChunkIfInitialized(chunk, resolveListeners, rejectListeners);
        }
        nextInsertIndex++;
      },
      addModel: function (value) {
        nextInsertIndex === segments.length
          ? (segments[nextInsertIndex] = createResolvedGeneratorResultSegment(
              result,
              value,
              !0
            ))
          : fulfillGeneratorResultSegment(segments[nextInsertIndex], value, !1);
        nextInsertIndex++;
      },
      complete: function (value) {
        finished = !1;
        nextInsertIndex === segments.length
          ? (segments[nextInsertIndex] = createResolvedGeneratorResultSegment(
              result,
              value,
              !0
            ))
          : fulfillGeneratorResultSegment(segments[nextInsertIndex], value, !0);
        for (nextInsertIndex++; nextInsertIndex < segments.length; )
          fulfillGeneratorResultSegment(
            segments[nextInsertIndex++],
            '"$undefined"',
            !1
          );
      },
      handleError: function (error) {
        finished = !1;
        for (
          nextInsertIndex === segments.length &&
          (segments[nextInsertIndex] = createPendingSegment(result));
          nextInsertIndex < segments.length;

        )
          reportErrorOnSegment(segments[nextInsertIndex++], error);
      }
    }
  );
}

function handleAbort(reason) {
  const abortedFlag = !aborted;
  if (abortedFlag) {
    aborted = true;
    request.abortListeners.delete(abortBlob);
    if (request.type === 21) {
      request.pendingChunks--;
    } else {
      erroredTask(request, newTask, reason);
      enqueueFlush(request);
    }
    reader.cancel(reason).then(() => {}, error);
  }
}

function describeValueForErrorMessage(value) {
  switch (typeof value) {
    case "string":
      return JSON.stringify(
        10 >= value.length ? value : value.slice(0, 10) + "..."
      );
    case "object":
      if (isArrayImpl(value)) return "[...]";
      if (null !== value && value.$$typeof === CLIENT_REFERENCE_TAG)
        return "client";
      value = objectName(value);
      return "Object" === value ? "{...}" : value;
    case "function":
      return value.$$typeof === CLIENT_REFERENCE_TAG
        ? "client"
        : (value = value.displayName || value.name)
          ? "function " + value
          : "function";
    default:
      return String(value);
  }
}

function handleServerModuleLookup(config, moduleKey) {
  let moduleName = "";
  const moduleData = config[moduleKey];
  if (moduleData) moduleName = moduleData.name;
  else {
    const splitIdx = moduleKey.lastIndexOf("#");
    if (splitIdx !== -1) {
      moduleName = moduleKey.slice(splitIdx + 1);
      const baseModuleKey = moduleKey.slice(0, splitIdx);
      const resolvedData = config[baseModuleKey];
      if (!resolvedData)
        throw new Error(
          `Could not find the module "${moduleKey}" in the React Server Manifest. This is likely a bug in the React Server Components bundler.`
        );
    }
  }
  return [moduleData.id, moduleData.chunks, moduleName];
}

function handleModuleResolution(moduleConfig, key) {
  var label = "",
    resolvedData = moduleConfig[key];
  if (resolvedData) label = resolvedData.name;
  else {
    var idx = key.lastIndexOf("#");
    -1 !== idx &&
      ((label = key.slice(idx + 1)),
      (resolvedData = moduleConfig[key.slice(0, idx)]));
    if (!resolvedData)
      throw Error(
        'Could not find the module "' +
          key +
          '" in the Component Server Manifest. This is probably a bug in the React Server Components bundler.'
      );
  }
  return [resolvedData.id, resolvedData.chunks, label];
}

function handleDebugInfo(data, itemID, info) {
  if (null === info.owner && null !== data._debugRootOwner) {
    info.owner = data._debugRootOwner;
    info.debugStack = data._debugRootStack;
  } else if (!!(info.stack)) {
    initializeFakeStack(data, info);
  }

  const chunkData = getChunk(data, itemID);
  (chunkData._debugInfo || (chunkData._debugInfo = [])).push(info);
}

function transformReadableStream(request, task, stream) {
  function handleProgress(entry) {
    if (!aborted)
      if (entry.done)
        request.abortListeners.delete(abortStream),
          (entry = `task.id.toString(16)+":C\n"`),
          request.completedRegularChunks.push(stringToChunk(entry)),
          enqueueFlush(request),
          (aborted = !0);
      else
        try {
          (streamTask.model = entry.value),
            request.pendingChunks++,
            emitChunk(request, streamTask, streamTask.model),
            enqueueFlush(request),
            reader.read().then(handleProgress, handleError);
        } catch (x$7) {
          handleError(x$7);
        }
  }

  function handleError(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      erroredTask(request, streamTask, reason),
      enqueueFlush(request),
      reader.cancel(reason).then(handleError, handleError));
  }

  function abortStream(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      21 === request.type
        ? request.pendingChunks--
        : (erroredTask(request, streamTask, reason), enqueueFlush(request)),
      reader.cancel(reason).then(handleError, handleError));
  }

  var supportsBYOBCheck;
  if (void 0 === supportsBYOBCheck)
    try {
      stream.getReader({ mode: "byob" }).releaseLock(), (supportsBYOBCheck = !0);
    } catch (x) {
      supportsBYOBCheck = !1;
    }
  var reader = stream.getReader(),
    streamTask = createTask(
      request,
      task.model,
      task.keyPath,
      task.implicitSlot,
      request.abortableTasks
    );
  request.abortableTasks.delete(streamTask);
  request.pendingChunks++;
  var supportsBYOB = supportsBYOBCheck ? "r" : "R";
  task = `streamTask.id.toString(16)+":" + (supportsBYOB) + "\n"`;
  request.completedRegularChunks.push(stringToChunk(task));
  var aborted = !1;
  request.abortListeners.add(abortStream);
  reader.read().then(handleProgress, handleError);
  return serializeByValueID(streamTask.id);
}

function fetchModelDetails(response, sourceStr, objRef, propKey, dataMap) {
  const parts = sourceStr.split(":");
  let chunkId = parseInt(parts[0], 16);
  const idChunk = getChunk(response, chunkId);

  switch (idChunk.status) {
    case "resolved_model":
      initializeModelChunk(idChunk);
      break;
    default:
      if (!["fulfilled", "pending", "blocked", "cyclic"].includes(idChunk.status)) {
        throw idChunk.reason;
      }
      const parentObj = objRef;
      for (let i = 1; i < parts.length; i++) {
        parentObj = parentObj[parts[i]];
      }

      return dataMap(response, parentObj);
  }

  if ("fulfilled" === idChunk.status) {
    let targetObj = idChunk.value;
    for (let j = 1; j < parts.length; j++)
      targetObj = targetObj[parts[j]];

    const cyclicOrPending = "cyclic" === idChunk.status;
    const resolverFn = createModelResolver(
      initializingChunk,
      objRef,
      propKey,
      cyclicOrPending,
      response,
      dataMap,
      parts
    );
    const rejectFn = createModelReject(initializingChunk);

    idChunk.then(resolverFn, rejectFn);
    return null;
  }
}

function preloadModule$2(url, config) {
  if ("string" === typeof url) {
    var request = currentRequest ? currentRequest : null;
    if (request) {
      var hints = request.hintMap,
        key = "m|" + url;
      if (hints.has(key)) return;
      hints.add(key);
      return (config = trimOptions(config))
        ? emitHint(request, "module", [url, config])
        : emitHint(request, "module", url);
    }
    previousDispatcher.module(url, config);
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

