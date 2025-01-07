function loadServerReference(response, metaData, parentObject, key) {
  if (!response._serverReferenceConfig)
    return createBoundServerReference(
      metaData,
      response._callServer,
      response._encodeFormAction
    );
  var serverReference = resolveServerReference(
    response._serverReferenceConfig,
    metaData.id
  );
  if ((response = preloadModule(serverReference)))
    metaData.bound && (response = Promise.all([response, metaData.bound]));
  else if (metaData.bound) response = Promise.resolve(metaData.bound);
  else return requireModule(serverReference);
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
  response.then(
    function () {
      var resolvedValue = requireModule(serverReference);
      if (metaData.bound) {
        var boundArgs = metaData.bound.value.slice(0);
        boundArgs.unshift(null);
        resolvedValue = resolvedValue.bind.apply(resolvedValue, boundArgs);
      }
      parentObject[key] = resolvedValue;
      "" === key && null === handler.value && (handler.value = resolvedValue);
      if (
        parentObject[0] === REACT_ELEMENT_TYPE &&
        "object" === typeof handler.value &&
        null !== handler.value &&
        handler.value.$$typeof === REACT_ELEMENT_TYPE
      )
        switch (((boundArgs = handler.value), key)) {
          case "3":
            boundArgs.props = resolvedValue;
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
        handler.errored = !0;
        handler.value = error;
        var chunk = handler.chunk;
        null !== chunk &&
          "blocked" === chunk.status &&
          triggerErrorOnChunk(chunk, error);
      }
    }
  );
  return null;
}

function executeTaskSequence(taskRequest) {
  let previousDispatcher = ReactSharedInternalsServer.H;
  ReactSharedInternalsServer.H = HooksDispatcher;
  const currentPreviousRequest = currentRequest;
  currentRequest = taskRequest;

  let hasAbortableTasks = false < taskRequest.abortableTasks.size;

  try {
    const pendingTasks = taskRequest.pingedTasks;
    taskRequest.pingedTasks = [];
    for (let i = 0; i < pendingTasks.length; i++) {
      retryTask(taskRequest, pendingTasks[i]);
    }
    if (null !== taskRequest.destination) {
      flushCompletedChunks(taskRequest, taskRequest.destination);
    }

    if (hasAbortableTasks && 0 === taskRequest.abortableTasks.size) {
      const allReadyCallback = taskRequest.onAllReady;
      allReadyCallback();
    }
  } catch (error) {
    logRecoverableError(taskRequest, error, null);
    fatalError(taskRequest, error);
  } finally {
    ReactSharedInternalsServer.H = previousDispatcher;
    currentRequest = currentPreviousRequest;
  }
}

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

function handleCompleteBinaryLine(data, id, mark, bytes, part) {
  switch (mark) {
    case 65:
      finalizeBytes(data, id, combineBytes(bytes, part).buffer);
      return;
    case 79:
      finalizeArrayBuffer(data, id, bytes, part, Int8Array, 1);
      return;
    case 111:
      finalizeBytes(
        data,
        id,
        0 === bytes.length ? part : combineBytes(bytes, part)
      );
      return;
    case 85:
      finalizeArrayBuffer(data, id, bytes, part, Uint8ClampedArray, 1);
      return;
    case 83:
      finalizeArrayBuffer(data, id, bytes, part, Int16Array, 2);
      return;
    case 115:
      finalizeArrayBuffer(data, id, bytes, part, Uint16Array, 2);
      return;
    case 76:
      finalizeArrayBuffer(data, id, bytes, part, Int32Array, 4);
      return;
    case 108:
      finalizeArrayBuffer(data, id, bytes, part, Uint32Array, 4);
      return;
    case 71:
      finalizeArrayBuffer(data, id, bytes, part, Float32Array, 4);
      return;
    case 103:
      finalizeArrayBuffer(data, id, bytes, part, Float64Array, 8);
      return;
    case 77:
      finalizeArrayBuffer(data, id, bytes, part, BigInt64Array, 8);
      return;
    case 109:
      finalizeArrayBuffer(data, id, bytes, part, BigUint64Array, 8);
      return;
    case 86:
      finalizeArrayBuffer(data, id, bytes, part, DataView, 1);
      return;
  }
  for (
    var stringDecoder = data._stringDecoder, line = "", i = 0;
    i < bytes.length;
    i++
  )
    line += stringDecoder.decode(bytes[i], decoderOptions);
  line += stringDecoder.decode(part);
  processCompleteStringLine(data, id, mark, line);
}

function initializeModuleData(fetchResult, moduleInfo, containerObject, key) {
  if (!fetchResult._moduleConfig)
    return instantiateBoundModule(
      moduleInfo,
      fetchResult._invokeRemote,
      fetchResult._encodeFormAction
    );
  var config = resolveModuleConfig(
    fetchResult._moduleConfig,
    moduleInfo.id
  );
  if ((fetchResult = prefetchModule(config)))
    moduleInfo.bound && (fetchResult = Promise.all([fetchResult, moduleInfo.bound]));
  else if (moduleInfo.bound) fetchResult = Promise.resolve(moduleInfo.bound);
  else return importModule(config);
  if (initializationHandler) {
    var handler = initializationHandler;
    handler.dependencies++;
  } else
    handler = initializationHandler = {
      parent: null,
      chunk: null,
      value: null,
      dependencies: 1,
      errored: !1
    };
  fetchResult.then(
    function () {
      var resolvedValue = importModule(config);
      if (moduleInfo.bound) {
        var boundParams = moduleInfo.bound.value.slice(0);
        boundParams.unshift(null);
        resolvedValue = resolvedValue.bind.apply(resolvedValue, boundParams);
      }
      containerObject[key] = resolvedValue;
      "" === key && null === handler.value && (handler.value = resolvedValue);
      if (
        containerObject[0] === REACT_ELEMENT_TAG &&
        "object" === typeof handler.value &&
        null !== handler.value &&
        handler.value.$$typeof === REACT_ELEMENT_TAG
      )
        switch (((boundParams = handler.value), key)) {
          case "3":
            boundParams.props = resolvedValue;
        }
      handler.dependencies--;
      0 === handler.dependencies &&
        ((resolvedValue = handler.chunk),
        null !== resolvedValue &&
          "blocked" === resolvedValue.status &&
          ((boundParams = resolvedValue.value),
          (resolvedValue.status = "fulfilled"),
          (resolvedValue.value = handler.value),
          null !== boundParams && wakeChunk(boundParams, handler.value)));
    },
    function (error) {
      if (!handler.errored) {
        handler.errored = !0;
        handler.value = error;
        var chunk = handler.chunk;
        null !== chunk &&
          "blocked" === chunk.status &&
          triggerErrorOnChunk(chunk, error);
      }
    }
  );
  return null;
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

function parse(text) {
  try {
    const root = parseYaml(text);

    /**
     * suppress `comment not printed` error
     *
     * comments are handled in printer-yaml.js without using `printComment`
     * so that it'll always throw errors even if we printed it correctly
     */
    delete root.comments;

    return root;
  } catch (/** @type {any} */ error) {
    if (error?.position) {
      throw createError(error.message, {
        loc: error.position,
        cause: error,
      });
    }

    /* c8 ignore next */
    throw error;
  }
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

function explainItemType(item) {
  if ("string" === typeof item) return item;
  switch (item) {
    case CUSTOM_SUSPENSE_TYPE:
      return "CustomSuspense";
    case CUSTOM_SUSPENSE_LIST_TYPE:
      return "CustomSuspenseList";
  }
  if ("object" === typeof item)
    switch (item.$$typeof) {
      case CUSTOM_FORWARD_REF_TYPE:
        return explainItemType(item.render);
      case CUSTOM_MEMO_TYPE:
        return explainItemType(item.type);
      case CUSTOM_LAZY_TYPE:
        var payload = item._payload;
        item = item._init;
        try {
          return explainItemType(item(payload));
        } catch (x) {}
    }
  return "";
}

function generateFakeCallStack(data, frames, envName, execute) {
  for (let index = 0; index < frames.length; index++) {
    const frameData = frames[index],
      frameKey = `${frameData.join("-")}-${envName}`,
      func = fakeFunctionCache[frameKey];
    if (!func) {
      let [functionName, fileName, lineNumber] = frameData;
      const contextFrame = frameData[3];
      let findSourceMapURL = data._debugFindSourceMapURL;
      findSourceMapURL = findSourceMapURL
        ? findSourceMapURL(fileName, envName)
        : null;
      func = createFakeFunction(
        functionName,
        fileName,
        findSourceMapURL,
        lineNumber,
        contextFrame,
        envName
      );
      fakeFunctionCache[frameKey] = func;
    }
    execute = func.bind(null, execute);
  }
  return execute;
}

    function processFullBinaryRow(response, id, tag, buffer, chunk) {
      switch (tag) {
        case 65:
          resolveBuffer(response, id, mergeBuffer(buffer, chunk).buffer);
          return;
        case 79:
          resolveTypedArray(response, id, buffer, chunk, Int8Array, 1);
          return;
        case 111:
          resolveBuffer(
            response,
            id,
            0 === buffer.length ? chunk : mergeBuffer(buffer, chunk)
          );
          return;
        case 85:
          resolveTypedArray(response, id, buffer, chunk, Uint8ClampedArray, 1);
          return;
        case 83:
          resolveTypedArray(response, id, buffer, chunk, Int16Array, 2);
          return;
        case 115:
          resolveTypedArray(response, id, buffer, chunk, Uint16Array, 2);
          return;
        case 76:
          resolveTypedArray(response, id, buffer, chunk, Int32Array, 4);
          return;
        case 108:
          resolveTypedArray(response, id, buffer, chunk, Uint32Array, 4);
          return;
        case 71:
          resolveTypedArray(response, id, buffer, chunk, Float32Array, 4);
          return;
        case 103:
          resolveTypedArray(response, id, buffer, chunk, Float64Array, 8);
          return;
        case 77:
          resolveTypedArray(response, id, buffer, chunk, BigInt64Array, 8);
          return;
        case 109:
          resolveTypedArray(response, id, buffer, chunk, BigUint64Array, 8);
          return;
        case 86:
          resolveTypedArray(response, id, buffer, chunk, DataView, 1);
          return;
      }
      for (
        var stringDecoder = response._stringDecoder, row = "", i = 0;
        i < buffer.length;
        i++
      )
        row += stringDecoder.decode(buffer[i], decoderOptions);
      row += stringDecoder.decode(chunk);
      processFullStringRow(response, id, tag, row);
    }

function setupTargetWithParts(loaderModule, segments, token$jscomp$0) {
  if (null !== loaderModule)
    for (var j = 0; j < segments.length; j++) {
      var token = token$jscomp$0,
        JSCompiler_temp_const = ReactSharedInternals.e,
        JSCompiler_temp_const$jscomp$0 = JSCompiler_temp_const.Y,
        JSCompiler_temp_const$jscomp$1 = loaderModule.header + segments[j];
      var JSCompiler_inline_result = loaderModule.cors;
      JSCompiler_inline_result =
        "string" === typeof JSCompiler_inline_result
          ? "include" === JSCompiler_inline_result
            ? JSCompiler_inline_result
            : ""
          : void 0;
      JSCompiler_temp_const$jscomp$0.call(
        JSCompiler_temp_const,
        JSCompiler_temp_const$jscomp$1,
        { cors: JSCompiler_inline_result, token: token }
      );
    }
}

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

function serializeFileRequest(task, file) {
  function updateProgress(entry) {
    if (!isAborted)
      if (entry.done)
        task.abortListeners.delete(abortFile),
          (isAborted = true),
          pingHandler(task, newSubTask);
      else
        return (
          model.push(entry.value), reader.read().then(updateProgress).catch(onError)
        );
  }
  function onError(reason) {
    isAborted ||
      ((isAborted = true),
      task.abortListeners.delete(abortFile),
      errorHandler(task, newSubTask, reason),
      flushQueue(task),
      reader.cancel(reason).then(onError, onError));
  }
  function abortFile(reason) {
    isAborted ||
      ((isAborted = true),
      task.abortListeners.delete(abortFile),
      13 === task.type
        ? task.pendingChunks--
        : (errorHandler(task, newSubTask, reason), flushQueue(task)),
      reader.cancel(reason).then(onError, onError));
  }
  var model = [file.type],
    newSubTask = createTask(task, model, null, false, task.abortableTasks),
    reader = file.stream().getReader(),
    isAborted = false;
  task.abortListeners.add(abortFile);
  reader.read().then(updateProgress).catch(onError);
  return "$F" + newSubTask.id.toString(16);
}

function updateStatus(entry) {
  if (!entry.done) {
    try {
      const partData = JSON.stringify(entry.value, resolveToJSON);
      data.append(formFieldPrefix + streamId, partData);
      reader.read().then(() => progress(entry), reject);
    } catch (x) {
      reject(x);
    }
  } else {
    data.append(formFieldPrefix + streamId, "C");
    pendingParts--;
    if (pendingParts === 0) resolve(data);
  }
}

function handlePendingPromiseState(promiseState, promise, position) {
  position = promiseState[position];
  void 0 === position
    ? promiseState.push(promise)
    : position !== promise && (promise.then(emptyFn, emptyFn), (promise = position));
  switch (promise.status) {
    case "resolved":
      return promise.value;
    case "rejected":
      throw promise.reason;
    default:
      "string" === typeof promise.status
        ? promise.then(emptyFn, emptyFn)
        : ((promiseState = promise),
          (promiseState.status = "pending"),
          promiseState.then(
            function (fulfilledValue) {
              if ("pending" === promise.status) {
                var fulfilledPromise = promise;
                fulfilledPromise.status = "resolved";
                fulfilledPromise.value = fulfilledValue;
              }
            },
            function (error) {
              if ("pending" === promise.status) {
                var rejectedPromise = promise;
                rejectedPromise.status = "rejected";
                rejectedPromise.reason = error;
              }
            }
          ));
      switch (promise.status) {
        case "resolved":
          return promise.value;
        case "rejected":
          throw promise.reason;
      }
      suspendedPromise = promise;
      throw SuspensionException;
  }
}

function registerServerReference(proxy, reference$jscomp$0, encodeFormAction) {
  Object.defineProperties(proxy, {
    $$FORM_ACTION: {
      value:
        void 0 === encodeFormAction
          ? defaultEncodeFormAction
          : function () {
              var reference = knownServerReferences.get(this);
              if (!reference)
                throw Error(
                  "Tried to encode a Server Action from a different instance than the encoder is from. This is a bug in React."
                );
              var boundPromise = reference.bound;
              null === boundPromise && (boundPromise = Promise.resolve([]));
              return encodeFormAction(reference.id, boundPromise);
            }
    },
    $$IS_SIGNATURE_EQUAL: { value: isSignatureEqual },
    bind: { value: bind }
  });
  knownServerReferences.set(proxy, reference$jscomp$0);
}

    function resolveStream(response, id, stream, controller) {
      var chunks = response._chunks,
        chunk = chunks.get(id);
      chunk
        ? "pending" === chunk.status &&
          ((response = chunk.value),
          (chunk.status = "fulfilled"),
          (chunk.value = stream),
          (chunk.reason = controller),
          null !== response && wakeChunk(response, chunk.value))
        : chunks.set(
            id,
            new ReactPromise("fulfilled", stream, controller, response)
          );
    }

function uploadRecord(item) {
  if (item.completed) {
    if (void 0 === item.content)
      formData.append(formKeyPrefix + recordId, "B");
    else
      try {
        var contentJSON = JSON.stringify(item.content, resolveToJson);
        formData.append(formKeyPrefix + recordId, "B" + contentJSON);
      } catch (e) {
        reject(e);
        return;
      }
    pendingRecords--;
    0 === pendingRecords && resolve(formData);
  } else
    try {
      var _contentJSON = JSON.stringify(item.content, resolveToJson);
      formData.append(formKeyPrefix + recordId, _contentJSON);
      iterator.next().then(uploadRecord, reject);
    } catch (e$0) {
      reject(e$0);
    }
}

function preloadModule(metadata) {
  for (var chunks = metadata[1], promises = [], i = 0; i < chunks.length; i++) {
    var chunkFilename = chunks[i],
      entry = chunkCache.get(chunkFilename);
    if (void 0 === entry) {
      entry = globalThis.__next_chunk_load__(chunkFilename);
      promises.push(entry);
      var resolve = chunkCache.set.bind(chunkCache, chunkFilename, null);
      entry.then(resolve, ignoreReject);
      chunkCache.set(chunkFilename, entry);
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

    function resolveConsoleEntry(response, value) {
      if (response._replayConsole) {
        var payload = JSON.parse(value, response._fromJSON);
        value = payload[0];
        var stackTrace = payload[1],
          owner = payload[2],
          env = payload[3];
        payload = payload.slice(4);
        replayConsoleWithCallStackInDEV(
          response,
          value,
          stackTrace,
          owner,
          env,
          payload
        );
      }
    }

function handleModuleSection(section, data) {
  if ("waiting" === section.state || "halted" === section.state) {
    var confirmListeners = section.result,
      declineListeners = section.causes;
    section.state = "processed_module";
    section.result = data;
    null !== confirmListeners &&
      (startModuleSection(section),
      activateSectionIfReady(section, confirmListeners, declineListeners));
  }
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

function fetchModuleInfo(config) {
  const exportsData = __webpack_require__(config[0]);
  if (4 === config.length && "function" === typeof exportsData.then)
    if ("fulfilled" !== exportsData.status)
      throw exportsData.reason;
    else
      exportsData = exportsData.value;
  return "*" === config[2]
    ? exportsData
    : "" === config[2]
      ? config[2] in exportsData
        ? exportsData[config[2]]
        : exportsData.__esModule && exportsData.default
          ? exportsData.default
          : exportsData
      : exportsData[config[2]];
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

