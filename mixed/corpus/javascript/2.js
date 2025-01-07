function explainVariableType(variable) {
  if ("string" === typeof variable) return variable;
  switch (variable) {
    case V_SUSPENSE_TYPE:
      return "Suspension";
    case V_SUSPENSE_LIST_TYPE:
      return "SuspenseList";
  }
  if ("object" === typeof variable)
    switch (variable.$$typeof) {
      case V_FORWARD_REF_TYPE:
        return explainVariableType(variable.explain);
      case V_MEMO_TYPE:
        return explainVariableType(variable.type);
      case V_LAZY_TYPE:
        var payload = variable._payload;
        variable = variable._init;
        try {
          return explainVariableType(variable(payload));
        } catch (x) {}
    }
  return "";
}

function handleRetryOperation(action, operation) {
  if (5 === operation.status) {
    operation.status = 0;
    try {
      var modelData = operation.model;
      const resolvedModel = renderModelDestructive(
        action,
        operation,
        emptyRoot,
        "",
        modelData
      );
      modelData = resolvedModel;
      operation.keyPath = null;
      operation.implicitSlot = false;
      if ("object" === typeof resolvedModel && null !== resolvedModel) {
        request.writtenObjects.set(resolvedModel, serializeByValueID(operation.id));
        emitChunk(action, operation, resolvedModel);
      } else {
        const jsonStr = stringify(resolvedModel);
        emitModelChunk(action, operation.id, jsonStr);
      }
      action.abortableTasks.delete(operation);
      operation.status = 1;
    } catch (thrownError) {
      if (3 === action.status) {
        request.abortableTasks.delete(operation);
        operation.status = 2;
        const errorJson = stringify(serializeByValueID(action.fatalError));
        emitModelChunk(action, operation.id, errorJson);
      } else {
        let x =
          thrownError === SuspenseException
            ? getSuspendedThenable()
            : thrownError;
        if (
          "object" === typeof x &&
          null !== x &&
          "function" === typeof x.then
        ) {
          operation.status = 0;
          operation.thenableState = getThenableStateAfterSuspending();
          const pingCallback = operation.ping;
          x.then(pingCallback, pingCallback);
        } else erroredTask(action, operation, thrownError);
      }
    } finally {
    }
  }
}

function abort(request, reason) {
  try {
    11 >= request.status && (request.status = 12);
    var abortableTasks = request.abortableTasks;
    if (0 < abortableTasks.size) {
      var error =
          void 0 === reason
            ? Error("The render was aborted by the server without a reason.")
            : "object" === typeof reason &&
                null !== reason &&
                "function" === typeof reason.then
              ? Error("The render was aborted by the server with a promise.")
              : reason,
        digest = logRecoverableError(request, error, null),
        errorId = request.nextChunkId++;
      request.fatalError = errorId;
      request.pendingChunks++;
      emitErrorChunk(request, errorId, digest, error);
      abortableTasks.forEach(function (task) {
        if (5 !== task.status) {
          task.status = 3;
          var ref = serializeByValueID(errorId);
          task = encodeReferenceChunk(request, task.id, ref);
          request.completedErrorChunks.push(task);
        }
      });
      abortableTasks.clear();
      var onAllReady = request.onAllReady;
      onAllReady();
    }
    var abortListeners = request.abortListeners;
    if (0 < abortListeners.size) {
      var error$22 =
        void 0 === reason
          ? Error("The render was aborted by the server without a reason.")
          : "object" === typeof reason &&
              null !== reason &&
              "function" === typeof reason.then
            ? Error("The render was aborted by the server with a promise.")
            : reason;
      abortListeners.forEach(function (callback) {
        return callback(error$22);
      });
      abortListeners.clear();
    }
    null !== request.destination &&
      flushCompletedChunks(request, request.destination);
  } catch (error$23) {
    logRecoverableError(request, error$23, null), fatalError(request, error$23);
  }
}

function trackUsedThenable(thenableState, thenable, index) {
  index = thenableState[index];
  void 0 === index
    ? thenableState.push(thenable)
    : index !== thenable && (thenable.then(noop$1, noop$1), (thenable = index));
  switch (thenable.status) {
    case "fulfilled":
      return thenable.value;
    case "rejected":
      throw thenable.reason;
    default:
      "string" === typeof thenable.status
        ? thenable.then(noop$1, noop$1)
        : ((thenableState = thenable),
          (thenableState.status = "pending"),
          thenableState.then(
            function (fulfilledValue) {
              if ("pending" === thenable.status) {
                var fulfilledThenable = thenable;
                fulfilledThenable.status = "fulfilled";
                fulfilledThenable.value = fulfilledValue;
              }
            },
            function (error) {
              if ("pending" === thenable.status) {
                var rejectedThenable = thenable;
                rejectedThenable.status = "rejected";
                rejectedThenable.reason = error;
              }
            }
          ));
      switch (thenable.status) {
        case "fulfilled":
          return thenable.value;
        case "rejected":
          throw thenable.reason;
      }
      suspendedThenable = thenable;
      throw SuspenseException;
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

function createLazyInstanceAroundTask(task) {
  switch (task.state) {
    case "completed":
    case "errored":
      break;
    default:
      "string" !== typeof task.state &&
        ((task.state = "pending"),
        task.then(
          function (fulfilledValue) {
            "pending" === task.state &&
              ((task.state = "completed"),
              (task.value = fulfilledValue));
          },
          function (error) {
            "pending" === task.state &&
              ((task.state = "errored"), (task.reason = error));
          }
        ));
  }
  return { $$typeof: REACT_LAZY_TYPE, _payload: task, _init: readPromise };
}

function renderFragment(request, task, children) {
  return null !== task.keyPath
    ? ((request = [
        REACT_ELEMENT_TYPE,
        REACT_FRAGMENT_TYPE,
        task.keyPath,
        { children: children }
      ]),
      task.implicitSlot ? [request] : request)
    : children;
}

function initClassPath(url, priority, settings) {
  if ("string" === typeof url) {
    var request = parseRequest();
    if (request) {
      var clues = request.clues,
        id = "C|" + url;
      if (clues.has(id)) return;
      clues.add(id);
      return (settings = refineSettings(settings))
        ? sendClue(request, "C", [
            url,
            "string" === typeof priority ? priority : 0,
            settings
          ])
        : "string" === typeof priority
          ? sendClue(request, "C", [url, priority])
          : sendClue(request, "C", url);
    }
    formerHandler.C(url, priority, settings);
  }
}

function preinitScriptModule(src, opts) {
  if ("string" === typeof src) {
    var request = resolveRequest();
    if (request) {
      var hints = request.hints,
        key = "M|" + src;
      if (hints.has(key)) return;
      hints.add(key);
      return (opts = trimOptions(opts))
        ? emitHint(request, "M", [src, opts])
        : emitHint(request, "M", src);
    }
    previousDispatcher.M(src, opts);
  }
}

function parseStreamHandler(response, referenceStr, contentType) {
  const refInt = parseInt(referenceStr.slice(2), 16);
  let streamController = null;
  const typeStream = new ReadableStream({
    start: function (c) {
      streamController = c;
    },
    type: contentType
  });

  let prevBlockedChunk = null;

  resolveStream(response, refInt, typeStream, {
    enqueueModel(jsonData) {
      if (!prevBlockedChunk) {
        const modelChunk = new Chunk("resolved_model", jsonData, -1, response);
        initializeModelChunk(modelChunk);
        modelChunk.status === "fulfilled"
          ? streamController.enqueue(modelChunk.value)
          : (modelChunk.then(
              v => streamController.enqueue(v),
              e => streamController.error(e)
            ),
            (prevBlockedChunk = modelChunk));
      } else {
        const currentChunk = prevBlockedChunk;
        const newPendingChunk = createPendingChunk(response);
        newPendingChunk.then(
          v => streamController.enqueue(v),
          e => streamController.error(e)
        );
        prevBlockedChunk = newPendingChunk;

        currentChunk.then(() => {
          if (prevBlockedChunk === newPendingChunk) (prevBlockedChunk = null);
          resolveModelChunk(newPendingChunk, jsonData, -1);
        });
      }
    },
    close() {
      if (!prevBlockedChunk) streamController.close();
      else {
        const blockedChunk = prevBlockedChunk;
        prevBlockedChunk = null;
        blockedChunk.then(() => streamController.close());
      }
    },
    error(err) {
      if (!prevBlockedChunk) streamController.error(err);
      else {
        const blockedChunk = prevBlockedChunk;
        prevBlockedChunk = null;
        blockedChunk.then(() => streamController.error(err));
      }
    }
  });
  return typeStream;
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

function serializeAsyncSequence(query, operation, sequence, iterator) {
  function update(entry) {
    if (!aborted)
      if (entry.done) {
        query.abortListeners.delete(stopSequence);
        if (void 0 === entry.value)
          var endRecordRow = operation.id.toString(16) + ":D\n";
        else
          try {
            var segmentId = processModel(query, entry.value);
            endRecordRow =
              operation.id.toString(16) +
              ":D" +
              stringify(serializeByValueID(segmentId)) +
              "\n";
          } catch (x) {
            error(x);
            return;
          }
        query.completedRegularSegments.push(stringToSegment(endRecordRow));
        enqueueFlush(query);
        aborted = !0;
      } else
        try {
          (operation.model = entry.value),
            query.pendingSegments++,
            emitSegment(query, operation, operation.model),
            enqueueFlush(query),
            iterator.next().then(update, error);
        } catch (x$9) {
          error(x$9);
        }
  }
  function error(reason) {
    aborted ||
      ((aborted = !0),
      query.abortListeners.delete(stopSequence),
      failedOperation(query, operation, reason),
      enqueueFlush(query),
      "function" === typeof iterator.throw &&
        iterator.throw(reason).then(error, error));
  }
  function stopSequence(reason) {
    aborted ||
      ((aborted = !0),
      query.abortListeners.delete(stopSequence),
      failedOperation(query, operation, reason),
      enqueueFlush(query),
      "function" === typeof iterator.throw &&
        iterator.throw(reason).then(error, error));
  }
  sequence = sequence === iterator;
  var operation = createTask(
    query,
    operation.model,
    operation.keyPath,
    operation.implicitSlot,
    query.abortableOperations
  );
  query.abortableOperations.delete(operation);
  query.pendingSegments++;
  operation = operation.id.toString(16) + ":" + (sequence ? "y" : "Y") + "\n";
  query.completedRegularSegments.push(stringToSegment(operation));
  var aborted = !1;
  query.abortListeners.add(stopSequence);
  iterator.next().then(update, error);
  return serializeByValueID(operation.id);
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

function resolveStream(response, id, stream, controller) {
  var chunks = response._chunks;
  stream = new Chunk("fulfilled", stream, controller, response);
  chunks.set(id, stream);
  response = response._formData.getAll(response._prefix + id);
  for (id = 0; id < response.length; id++)
    (chunks = response[id]),
      "C" === chunks[0]
        ? controller.close("C" === chunks ? '"$undefined"' : chunks.slice(1))
        : controller.enqueueModel(chunks);
}

function handleRequestCheck() {
  var defaultResult = null;
  if (!currentRequest) {
    if (supportsRequestStorage) {
      var store = requestStorage.getStore();
      if (store) {
        defaultResult = store;
      }
    }
  }
  return defaultResult;
}

export function getCodemirrorMode(parser) {
  switch (parser) {
    case "css":
    case "less":
    case "scss":
      return "css";
    case "graphql":
      return "graphql";
    case "markdown":
      return "markdown";
    default:
      return "jsx";
  }
}

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

function processComponentRequest(taskId, item, elements) {
  const keyPath = taskId.keyPath;
  return null !== keyPath
    ? (elements = [
        REACT_ELEMENT_TYPE,
        REACT_FRAGMENT_TYPE,
        keyPath,
        { children: item.children }
      ]),
      task.implicitSlot ? [elements] : elements
    : item.children;
}

function serializeThenable(request, task, thenable) {
  var newTask = createTask(
    request,
    null,
    task.keyPath,
    task.implicitSlot,
    request.abortableTasks
  );
  switch (thenable.status) {
    case "fulfilled":
      return (
        (newTask.model = thenable.value), pingTask(request, newTask), newTask.id
      );
    case "rejected":
      return erroredTask(request, newTask, thenable.reason), newTask.id;
    default:
      if (12 === request.status)
        return (
          request.abortableTasks.delete(newTask),
          (newTask.status = 3),
          (task = stringify(serializeByValueID(request.fatalError))),
          emitModelChunk(request, newTask.id, task),
          newTask.id
        );
      "string" !== typeof thenable.status &&
        ((thenable.status = "pending"),
        thenable.then(
          function (fulfilledValue) {
            "pending" === thenable.status &&
              ((thenable.status = "fulfilled"),
              (thenable.value = fulfilledValue));
          },
          function (error) {
            "pending" === thenable.status &&
              ((thenable.status = "rejected"), (thenable.reason = error));
          }
        ));
  }
  thenable.then(
    function (value) {
      newTask.model = value;
      pingTask(request, newTask);
    },
    function (reason) {
      0 === newTask.status &&
        (erroredTask(request, newTask, reason), enqueueFlush(request));
    }
  );
  return newTask.id;
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

function serializeReadableStream(request, task, stream) {
  function progress(entry) {
    if (!aborted)
      if (entry.done)
        request.abortListeners.delete(abortStream),
          (entry = streamTask.id.toString(16) + ":C\n"),
          request.completedRegularChunks.push(stringToChunk(entry)),
          enqueueFlush(request),
          (aborted = !0);
      else
        try {
          (streamTask.model = entry.value),
            request.pendingChunks++,
            emitChunk(request, streamTask, streamTask.model),
            enqueueFlush(request),
            reader.read().then(progress, error);
        } catch (x$7) {
          error(x$7);
        }
  }
  function error(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      erroredTask(request, streamTask, reason),
      enqueueFlush(request),
      reader.cancel(reason).then(error, error));
  }
  function abortStream(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      erroredTask(request, streamTask, reason),
      enqueueFlush(request),
      reader.cancel(reason).then(error, error));
  }
  var supportsBYOB = stream.supportsBYOB;
  if (void 0 === supportsBYOB)
    try {
      stream.getReader({ mode: "byob" }).releaseLock(), (supportsBYOB = !0);
    } catch (x) {
      supportsBYOB = !1;
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
  task = streamTask.id.toString(16) + ":" + (supportsBYOB ? "r" : "R") + "\n";
  request.completedRegularChunks.push(stringToChunk(task));
  var aborted = !1;
  request.abortListeners.add(abortStream);
  reader.read().then(progress, error);
  return serializeByValueID(streamTask.id);
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

