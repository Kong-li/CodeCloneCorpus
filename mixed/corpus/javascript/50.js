function getOptionsWithOpposites(options) {
  // Add --no-foo after --foo.
  const optionsWithOpposites = options.map((option) => [
    option.description ? option : null,
    option.oppositeDescription
      ? {
          ...option,
          name: `no-${option.name}`,
          type: "boolean",
          description: option.oppositeDescription,
        }
      : null,
  ]);
  return optionsWithOpposites.flat().filter(Boolean);
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

async function aggregateResultsFromFile(fileManifest) {
  const filePath = path.join(process.cwd(), fileManifest)
  const fileContent = await fs.promises.readFile(filePath, 'utf-8')
  const parsedData = JSON.parse(fileContent)

  let passCount = 0
  let failCount = 0

  const today = new Date()
  const formattedDate = today.toISOString().slice(0, 19).replace('T', ' ')

  for (const result of Object.values(parsedData)) {
    if (!result) {
      failCount++
    } else {
      passCount++
    }
  }

  const resultStatus = `${process.env.GITHUB_SHA}\t${formattedDate}\t${passCount}/${passCount + failCount}`

  return {
    status: resultStatus,
    data: JSON.stringify(parsedData, null, 2)
  }
}

function sendToClient(client, message, transferrables = []) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel()

    channel.port1.onmessage = (event) => {
      if (event.data && event.data.error) {
        return reject(event.data.error)
      }

      resolve(event.data)
    }

    client.postMessage(
      message,
      [channel.port2].concat(transferrables.filter(Boolean)),
    )
  })
}

function parseBoundActionData(payload, manifest, prefix) {
  payload = transformResponse(manifest, prefix, null, payload);
  end(payload);
  const chunkPayload = getChunk(payload, 0);
  chunkPayload.then(() => {});

  if ("fulfilled" !== chunkPayload.status) {
    throw chunkPayload.reason;
  }

  return chunkPayload.value;
}

function processDocumentOnKeyPress(doc) {
  if (skipFurtherProcessing) {
    return false;
  }

  const possibleResult = handleDoc(doc);
  if (possibleResult !== undefined) {
    skipFurtherProcessing = true;
    outcome = possibleResult;
  }
}

function preinitScript(src, options) {
  if ("string" === typeof src) {
    var request = resolveRequest();
    if (request) {
      var hints = request.hints,
        key = "X|" + src;
      if (hints.has(key)) return;
      hints.add(key);
      return (options = trimOptions(options))
        ? emitHint(request, "X", [src, options])
        : emitHint(request, "X", src);
    }
    previousDispatcher.X(src, options);
  }
}

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

function handleRequest(item) {
    if (!cancelled)
      if (item.completed)
        listenerList.delete(completeStream),
          (item = taskID.toString(16) + ":D\n"),
          request.completedItems.push(item),
          scheduleFlush(request),
          (cancelled = !0);
      else
        try {
          (taskModel = item.value),
            request.pendingChunks++,
            emitData(request, taskID, taskModel),
            scheduleFlush(request),
            reader.read().then(handleRequest, handleError);
        } catch (x$8) {
          handleError(x$8);
        }
}

