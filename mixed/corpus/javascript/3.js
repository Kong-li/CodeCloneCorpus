function handleTaskProcessing(task) {
  const originalDispatcher = ReactSharedInternalsServer.H;
  ReactSharedInternalsServer.H = HooksDispatcher;
  let processedRequest = task.request;
  const previousTasks = task.pingedTasks.slice();
  const initialAbortableCount = previousTasks.filter(t => t.isAbortable).length;

  try {
    for (let i = 0; i < previousTasks.length; i++) {
      retryTask(processedRequest, previousTasks[i]);
    }
    if (processedRequest.destination && processedRequest.pingedTasks.length === 0) {
      flushCompletedChunks(processedRequest, processedRequest.destination);
    }

    if (initialAbortableCount > 0 && task.abortableTasks.size === 0) {
      const onAllReadyCallback = task.onAllReady;
      onAllReadyCallback();
    }
  } catch (error) {
    logRecoverableError(processedRequest, error, null);
    fatalError(processedRequest, error);
  } finally {
    ReactSharedInternalsServer.H = originalDispatcher;
  }
}

function handleServerComponentResult(fetchData, taskDetails, ComponentInfo, outcome) {
  if (
    !outcome ||
    typeof outcome !== "object" ||
    outcome.$$typeof === CLIENT_REFERENCE_TAG$1
  )
    return outcome;
  if (typeof outcome.then === 'function')
    return outcome.status === "fulfilled"
      ? outcome.value
      : createLazyWrapperAroundWakeable(outcome);
  const iteratorFunction = getIteratorFn(outcome);
  return iteratorFunction
    ? (
        fetchData = {},
        (fetchData[Symbol.iterator] = function () {
          return iteratorFunction.call(outcome);
        }),
        fetchData
      )
    : typeof outcome[ASYNC_ITERATOR] !== 'function' ||
        (typeof ReadableStream === 'function' && outcome instanceof ReadableStream)
      ? outcome
      : (
        fetchData = {},
        (fetchData[ASYNC_ITERATOR] = function () {
          return outcome[ASYNC_ITERATOR]();
        }),
        fetchData
      );
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

function resolveServerReference(bundlerConfig, id) {
  var name = "",
    resolvedModuleData = bundlerConfig[id];
  if (resolvedModuleData) name = resolvedModuleData.name;
  else {
    var idx = id.lastIndexOf("#");
    -1 !== idx &&
      ((name = id.slice(idx + 1)),
      (resolvedModuleData = bundlerConfig[id.slice(0, idx)]));
    if (!resolvedModuleData)
      throw Error(
        'Could not find the module "' +
          id +
          '" in the React Server Manifest. This is probably a bug in the React Server Components bundler.'
      );
  }
  return [resolvedModuleData.id, resolvedModuleData.chunks, name];
}

function getChunk(response, id) {
  var chunks = response._chunks,
    chunk = chunks.get(id);
  chunk ||
    ((chunk = response._formData.get(response._prefix + id)),
    (chunk =
      null != chunk
        ? new Chunk("resolved_model", chunk, id, response)
        : response._closed
          ? new Chunk("rejected", null, response._closedReason, response)
          : createPendingChunk(response)),
    chunks.set(id, chunk));
  return chunk;
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

function renderFunctionComponent(request, task, key, Component, props) {
  var prevThenableState = task.thenableState;
  task.thenableState = null;
  thenableIndexCounter = 0;
  thenableState = prevThenableState;
  props = Component(props, void 0);
  if (12 === request.status)
    throw (
      ("object" === typeof props &&
        null !== props &&
        "function" === typeof props.then &&
        props.$$typeof !== CLIENT_REFERENCE_TAG$1 &&
        props.then(voidHandler, voidHandler),
      null)
    );
  props = processServerComponentReturnValue(request, task, Component, props);
  Component = task.keyPath;
  prevThenableState = task.implicitSlot;
  null !== key
    ? (task.keyPath = null === Component ? key : Component + "," + key)
    : null === Component && (task.implicitSlot = !0);
  request = renderModelDestructive(request, task, emptyRoot, "", props);
  task.keyPath = Component;
  task.implicitSlot = prevThenableState;
  return request;
}

function enqueueFlush(request) {
  !1 === request.flushScheduled &&
    0 === request.pingedTasks.length &&
    null !== request.destination &&
    ((request.flushScheduled = !0),
    scheduleWork(function () {
      request.flushScheduled = !1;
      var destination = request.destination;
      destination && flushCompletedChunks(request, destination);
    }));
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

