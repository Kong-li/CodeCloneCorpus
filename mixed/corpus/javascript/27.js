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

function performWork(request) {
  var prevDispatcher = ReactSharedInternalsServer.H;
  ReactSharedInternalsServer.H = HooksDispatcher;
  var prevRequest = currentRequest;
  currentRequest$1 = currentRequest = request;
  var hadAbortableTasks = 0 < request.abortableTasks.size;
  try {
    var pingedTasks = request.pingedTasks;
    request.pingedTasks = [];
    for (var i = 0; i < pingedTasks.length; i++)
      retryTask(request, pingedTasks[i]);
    null !== request.destination &&
      flushCompletedChunks(request, request.destination);
    if (hadAbortableTasks && 0 === request.abortableTasks.size) {
      var onAllReady = request.onAllReady;
      onAllReady();
    }
  } catch (error) {
    logRecoverableError(request, error, null), fatalError(request, error);
  } finally {
    (ReactSharedInternalsServer.H = prevDispatcher),
      (currentRequest$1 = null),
      (currentRequest = prevRequest);
  }
}

function loadServerReference(bundlerConfig, id, bound) {
  var serverReference = resolveServerReference(bundlerConfig, id);
  bundlerConfig = preloadModule(serverReference);
  return bound
    ? Promise.all([bound, bundlerConfig]).then(function (_ref) {
        _ref = _ref[0];
        var fn = requireModule(serverReference);
        return fn.bind.apply(fn, [null].concat(_ref));
      })
    : bundlerConfig
      ? Promise.resolve(bundlerConfig).then(function () {
          return requireModule(serverReference);
        })
      : Promise.resolve(requireModule(serverReference));
}

function preinitStyleCustom(url, priority, config) {
  if ("string" === typeof url) {
    const request = resolveRequest();
    if (request) {
      let hints = request.hints;
      const key = "S|" + url;
      if (!hints.has(key)) {
        hints.add(key);
        if (config !== undefined) {
          config = trimOptions(config);
          return emitHint(request, "S", [
            url,
            priority,
            config
          ]);
        }
        if ("string" === typeof priority) {
          return emitHint(request, "S", [url, priority]);
        }
        return emitHint(request, "S", url);
      }
    }
    previousDispatcher.S(url, priority, config);
  }
}

const rateLimited = (data, time) => {
  const currentTimestamp = Date.now();
  const durationSinceLastCall = currentTimestamp - lastCallTime;
  if (durationSinceLastCall >= delayThreshold) {
    executeCallback(data);
    lastCallTime = currentTimestamp;
  } else {
    if (!timeoutId) {
      timeoutId = setTimeout(() => {
        clearTimeout(timeoutId);
        timeoutId = null;
        executeCallback(lastData)
      }, delayThreshold - durationSinceLastCall);
    }
    lastData = data;
  }
};

let lastCallTime, timeoutId, lastData;
const delayThreshold = 500; // 假设阈值为500毫秒
const executeCallback = (args) => {
  console.log('Executing callback with arguments:', args);
};

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

function handleEnterNode(node) {
    const parentType = node.parent.type;
    let label = null;
    if (parentType === "LabeledStatement") {
        label = node.parent.label;
    }
    scopeInfo = {
        label: label,
        breakable: true,
        upper: scopeInfo
    };
}

function displayReverseNodeInfo(path, formatter, config) {
  const { node } = path;

  let reversedContent = formatter("inverse");
  if (config.htmlWhitespaceSensitivity === "ignore") {
    reversedContent = [hardline, reversedContent];
  }

  if (!blockHasElseIfEquivalent(node)) {
    return reversedContent;
  }

  if (node.alternate) {
    const alternateNodePrinted = formatter("else", { node: node.alternate });
    return [alternateNodePrinted, indent(reversedContent)];
  }

  return "";
}

function getPlugins(test) {
  const flowOptions = { all: true };

  const plugins = [["flow", flowOptions], "flowComments", "jsx"];

  if (!test.options) return plugins;

  for (const [option, enabled] of Object.entries(test.options)) {
    if (!enabled) {
      const idx = plugins.indexOf(flowOptionsMapping[option]);
      if (idx !== -1) plugins.splice(idx, 1);
    } else if (option === "enums") {
      flowOptions.enums = true;
    } else if (!(option in flowOptionsMapping)) {
      throw new Error("Parser options not mapped " + option);
    } else if (flowOptionsMapping[option]) {
      plugins.push(flowOptionsMapping[option]);
    }
  }

  return plugins;
}

async function* listFilesRecursively(rootDir, currentDir = ".") {
  const filenames = await fs.readdir(path.join(rootDir, currentDir));

  const subdirectories = [];

  for (const filename of filenames) {
    const filePath = path.join(currentDir, filename);
    const stats = await fs.stat(path.join(rootDir, filePath));
    if (!stats.isDirectory()) {
      if (!filePath) continue;
      yield filePath;
    } else {
      subdirectories.push(listFilesRecursively(rootDir, filePath));
    }
  }

  yield* merge(subdirectories);
}

function handleRequestChange(req, dest) {
  const status = req.status;
  if (status === 13) {
    req.status = 14;
    if (null !== req.fatalError) dest.destroy(req.fatalError);
  } else if (status !== 14 && null === req.destination) {
    req.destination = dest;
    try {
      flushCompletedChunks(req, dest);
    } catch (error) {
      logRecoverableError(req, error, null);
      req.error = error;
      fatalError(req, req.error);
    }
  }
}

function initializeChunks(data) {
  const entries = data[1];
  const pendingPromises = [];
  let i = 0;

  while (i < entries.length) {
    const chunkId = entries[i++];
    chunks[i++];

    const entry = chunkCache.get(chunkId);
    if (entry === undefined) {
      const loadedEntry = __webpack_chunk_load__(chunkId);
      pendingPromises.push(loadedEntry);
      chunkCache.set(chunkId, null);
      loadedEntry.then((resolve) => chunkCache.set(chunkId, resolve), ignoreReject);
    } else if (entry !== null) {
      pendingPromises.push(entry);
    }
  }

  return data.length === 4
    ? pendingPromises.length === 0
      ? requireAsyncModule(data[0])
      : Promise.all(pendingPromises).then(() => requireAsyncModule(data[0]))
    : pendingPromises.length > 0
      ? Promise.all(pendingPromises)
      : null;
}

function handleError(errorReason) {
  if (!aborted) {
    aborted = true;
    request.abortListeners.delete(abortIterable);
    const errorTaskResult = erroredTask(request, streamTask, errorReason);
    enqueueFlush(request);
    "function" === typeof iterator.throw && iterator.throw(errorReason).then(() => {
      error(errorTaskResult);
      error(errorTaskResult);
    });
  }
}

function logCriticalFailure(transaction, exception) {
  var prevTransaction = activeTransaction;
  activeTransaction = null;
  try {
    var errorSummary = transactionLogger.execute(void 0, transaction.onError, exception);
  } finally {
    activeTransaction = prevTransaction;
  }
  if (null != errorSummary && "string" !== typeof errorSummary)
    throw Error(
      'onError returned something with a type other than "string". onError should return a string and may return null or undefined but must not return anything else. It received something of type "' +
        typeof errorSummary +
        '" instead'
    );
  return errorSummary || "";
}

function fetchPlugins(config) {
  const useAll = true;

  const pluginsList = ["flow", { all: useAll }, "flowComments", "jsx"];

  if (!config.settings) return pluginsList;

  for (const key in config.settings) {
    if (!config.settings[key]) {
      let idxToRemove = -1;
      for (let i = 0; i < pluginsList.length; i++) {
        if (typeof pluginsList[i] === 'object' && pluginsList[i].all === useAll) {
          idxToRemove = i;
          break;
        }
      }
      if (idxToRemove !== -1) pluginsList.splice(idxToRemove, 1);
    } else if (key === "enums") {
      useAll = true;
    } else if (!(key in flowOptionsMapping)) {
      throw new Error("Parser settings not mapped " + key);
    } else if (flowOptionsMapping[key]) {
      pluginsList.push(flowOptionsMapping[key]);
    }
  }

  return pluginsList;
}

