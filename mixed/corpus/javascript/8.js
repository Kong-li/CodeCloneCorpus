function loadModule(href, opts) {
  if (typeof href === 'string') {
    const req = resolveRequest();
    if (req) {
      const hints = req.hints;
      let key = `m|${href}`;
      !hints.has(key) && hints.add(key);
      return opts ? (opts = trimOptions(opts), emitHint(req, "m", [href, opts])) : emitHint(req, "m", href);
    }
    previousDispatcher.m(href, opts);
  }
}

function flushCompletedChunks(request, destination) {
  currentView = new Uint8Array(2048);
  writtenBytes = 0;
  destinationHasCapacity = !0;
  try {
    for (
      var importsChunks = request.completedImportChunks, i = 0;
      i < importsChunks.length;
      i++
    )
      if (
        (request.pendingChunks--,
        !writeChunkAndReturn(destination, importsChunks[i]))
      ) {
        request.destination = null;
        i++;
        break;
      }
    importsChunks.splice(0, i);
    var hintChunks = request.completedHintChunks;
    for (i = 0; i < hintChunks.length; i++)
      if (!writeChunkAndReturn(destination, hintChunks[i])) {
        request.destination = null;
        i++;
        break;
      }
    hintChunks.splice(0, i);
    var regularChunks = request.completedRegularChunks;
    for (i = 0; i < regularChunks.length; i++)
      if (
        (request.pendingChunks--,
        !writeChunkAndReturn(destination, regularChunks[i]))
      ) {
        request.destination = null;
        i++;
        break;
      }
    regularChunks.splice(0, i);
    var errorChunks = request.completedErrorChunks;
    for (i = 0; i < errorChunks.length; i++)
      if (
        (request.pendingChunks--,
        !writeChunkAndReturn(destination, errorChunks[i]))
      ) {
        request.destination = null;
        i++;
        break;
      }
    errorChunks.splice(0, i);
  } finally {
    (request.flushScheduled = !1),
      currentView &&
        0 < writtenBytes &&
        destination.write(currentView.subarray(0, writtenBytes)),
      (currentView = null),
      (writtenBytes = 0),
      (destinationHasCapacity = !0);
  }
  "function" === typeof destination.flush && destination.flush();
  0 === request.pendingChunks &&
    (cleanupTaintQueue(request),
    (request.status = 14),
    destination.end(),
    (request.destination = null));
}

    function pop(heap) {
      if (0 === heap.length) return null;
      var first = heap[0],
        last = heap.pop();
      if (last !== first) {
        heap[0] = last;
        a: for (
          var index = 0, length = heap.length, halfLength = length >>> 1;
          index < halfLength;

        ) {
          var leftIndex = 2 * (index + 1) - 1,
            left = heap[leftIndex],
            rightIndex = leftIndex + 1,
            right = heap[rightIndex];
          if (0 > compare(left, last))
            rightIndex < length && 0 > compare(right, left)
              ? ((heap[index] = right),
                (heap[rightIndex] = last),
                (index = rightIndex))
              : ((heap[index] = left),
                (heap[leftIndex] = last),
                (index = leftIndex));
          else if (rightIndex < length && 0 > compare(right, last))
            (heap[index] = right),
              (heap[rightIndex] = last),
              (index = rightIndex);
          else break a;
        }
      }
      return first;
    }

function test(useable) {
  if (
    (null !== useable && "object" === typeof useable) ||
    "function" === typeof useable
  ) {
    if ("function" === typeof useable.then) {
      var index = thenableIndexCounter;
      thenableIndexCounter += 1;
      null === thenableState && (thenableState = []);
      return trackUsedThenable(thenableState, useable, index);
    }
    useable.$$typeof === CONTEXT_TYPE && unsupportedContext();
  }
  if (useable.$$typeof === REFERENCE_TAG) {
    if (null != useable.value && useable.value.$$typeof === CONTEXT_TYPE)
      throw Error("Cannot read a Reference Context from a Server Component.");
    throw Error("Cannot test() an already resolved Reference.");
  }
  throw Error("An unsupported type was passed to test(): " + String(useable));
}

function parseModelStringImpl(response, obj, key, value, reference) {
  if ("$" === value[0]) {
    switch (value[1]) {
      case "$":
        return value.slice(1);
      case "@":
        var hexValue = parseInt(value.slice(2), 16);
        obj = hexValue;
        return getChunk(response, obj);
      case "F":
        var modelValue = value.slice(2);
        (value = getOutlinedModel(response, modelValue, obj, key, createModel)),
          loadServerReference$1(
            response,
            value.id,
            value.bound,
            initializingChunk,
            obj,
            key
          );
        return value;
      case "T":
        if (void 0 === reference || void 0 === response._temporaryReferences)
          throw Error(
            "Could not reference an opaque temporary reference. This is likely due to misconfiguring the temporaryReferences options on the server."
          );
        return createTemporaryReference(
          response._temporaryReferences,
          reference
        );
      case "Q":
        var outlinedValue = value.slice(2);
        (value = getOutlinedModel(response, outlinedValue, obj, key, createMap));
        return value;
      case "W":
        var setOutValue = value.slice(2);
        (value = getOutlinedModel(response, setOutValue, obj, key, createSet));
        return value;
      case "K":
        var prefix = response._prefix + value.slice(2),
          formDataObj = new FormData();
        response._formData.forEach(function (entry, entryKey) {
          if (entryKey.startsWith(prefix))
            formDataObj.append(entryKey.slice(prefix.length), entry);
        });
        return formDataObj;
      case "i":
        var iteratorValue = value.slice(2);
        (value = getOutlinedModel(response, iteratorValue, obj, key, extractIterator));
        return value;
      case "I":
        return Infinity;
      case "-":
        return "$-0" === value ? -0 : -Infinity;
      case "N":
        return NaN;
      case "u":
        return undefined;
      case "D":
        return new Date(Date.parse(value.slice(2)));
      case "n":
        return BigInt(value.slice(2));
    }
    switch (value[1]) {
      case "A":
        return parseTypedArray(response, value, ArrayBuffer, 1, obj, key);
      case "O":
        return parseTypedArray(response, value, Int8Array, 1, obj, key);
      case "o":
        return parseTypedArray(response, value, Uint8Array, 1, obj, key);
      case "U":
        return parseTypedArray(response, value, Uint8ClampedArray, 1, obj, key);
      case "S":
        return parseTypedArray(response, value, Int16Array, 2, obj, key);
      case "s":
        return parseTypedArray(response, value, Uint16Array, 2, obj, key);
      case "L":
        return parseTypedArray(response, value, Int32Array, 4, obj, key);
      case "l":
        return parseTypedArray(response, value, Uint32Array, 4, obj, key);
      case "G":
        return parseTypedArray(response, value, Float32Array, 4, obj, key);
      case "g":
        return parseTypedArray(response, value, Float64Array, 8, obj, key);
      case "M":
        return parseTypedArray(response, value, BigInt64Array, 8, obj, key);
      case "m":
        return parseTypedArray(response, value, BigUint64Array, 8, obj, key);
      case "V":
        return parseTypedArray(response, value, DataView, 1, obj, key);
      case "B":
        var hexIndex = parseInt(value.slice(2), 16);
        (obj = hexIndex),
          response._formData.get(response._prefix + obj);
        return obj;
    }
    switch (value[1]) {
      case "R":
        return parseReadableStream(response, value, void 0);
      case "r":
        return parseReadableStream(response, value, "bytes");
      case "X":
        return parseAsyncIterable(response, value, !1);
      case "x":
        return parseAsyncIterable(response, value, !0);
    }
  }
}

function updateCounters(currentTime) {
  for (var interval = peek(intervalQueue); null !== interval; ) {
    if (null === interval.handler) pop(intervalQueue);
    else if (interval.startTime <= currentTime)
      pop(intervalQueue),
        (interval.priority = interval.expirationTime),
        push(eventQueue, interval);
    else break;
    interval = peek(intervalQueue);
  }
}

function printClosingTagStartMarker(node, options) {
  assert(!node.isSelfClosing);
  /* c8 ignore next 3 */
  if (shouldNotPrintClosingTag(node, options)) {
    return "";
  }
  switch (node.type) {
    case "ieConditionalComment":
      return "<!";
    case "element":
      if (node.hasHtmComponentClosingTag) {
        return "<//";
      }
    // fall through
    default:
      return `</${node.rawName}`;
  }
}

  return function _createSuperInternal() {
    var Super = getPrototypeOf(Derived),
      result;
    if (hasNativeReflectConstruct) {
      // NOTE: This doesn't work if this.__proto__.constructor has been modified.
      var NewTarget = getPrototypeOf(this).constructor;
      result = Reflect.construct(Super, arguments, NewTarget);
    } else {
      result = Super.apply(this, arguments);
    }
    return possibleConstructorReturn(this, result);
  };

function handleResponseAsync(response, ref, iter) {
  const refValue = parseInt(ref.slice(2), 16);
  let buffer = [],
    isClosed = false,
    nextWriteIdx = 0,
    props$1 = {};

  props$1[ASYNC_ITERATOR] = function () {
    let readIndex = 0;
    return createIterator((arg) => {
      if (void 0 !== arg)
        throw Error(
          "Values cannot be passed to next() of AsyncIterables passed to Client Components."
        );
      if (readIndex === buffer.length) {
        if (isClosed)
          return new Chunk(
            "fulfilled",
            { done: !0, value: void 0 },
            null,
            response
          );
        buffer[readIndex] = createPendingChunk(response);
      }
      readIndex++;
      return buffer[readIndex - 1];
    });
  };

  const iteratorInstance = iter ? props$1[ASYNC_ITERATOR]() : props$1;
  resolveStream(
    response,
    refValue,
    iteratorInstance,
    {
      enqueueModel: (value) => {
        if (nextWriteIdx === buffer.length)
          buffer[nextWriteIdx] = createResolvedIteratorResultChunk(
            response,
            value,
            false
          );
        else resolveIteratorResultChunk(buffer[nextWriteIdx], value, false);
        nextWriteIdx++;
      },
      close: (value) => {
        isClosed = true;
        if (nextWriteIdx === buffer.length)
          buffer[nextWriteIdx] = createResolvedIteratorResultChunk(
            response,
            value,
            true
          );
        else resolveIteratorResultChunk(buffer[nextWriteIdx], value, true);
        for (; nextWriteIdx < buffer.length; )
          resolveIteratorResultChunk(
            buffer[nextWriteIdx++],
            '"$undefined"',
            true
          );
      },
      error: (error) => {
        isClosed = true;
        if (nextWriteIdx === buffer.length)
          buffer[nextWriteIdx] = createPendingChunk(response);
        for (; nextWriteIdx < buffer.length; )
          triggerErrorOnChunk(buffer[nextWriteIdx++], error);
      }
    }
  );

  return iteratorInstance;
}

function checkNextOpeningTagStartMarker(element) {
  /**
   *     123<p
   *        ^^
   *     >
   */
  const isTrailingSensitive = element.isTrailingSpaceSensitive;
  const hasNoSpaces = !element.hasTrailingSpaces;
  const notTextLikeNext = !isTextLikeNode(element.next);
  const textLikeCurrent = isTextLikeNode(element);

  return (
    element.next &&
    notTextLikeNext &&
    textLikeCurrent &&
    isTrailingSensitive &&
    hasNoSpaces
  );
}

function loadResources(url, resourceType, config) {
  if ("string" === typeof url) {
    const request = resolveRequest();
    if (request) {
      let hints = request.hints,
        keyPrefix = "L";
      if ("image" === resourceType && config) {
        let imageSrcSet = config.imageSrcSet,
          imageSizes = config.imageSizes,
          uniqueKeyPart = "";
        if ("string" !== typeof imageSrcSet || "" === imageSrcSet) {
          imageSrcSet = null;
        }
        if (null !== imageSrcSet) {
          uniqueKeyPart += "[" + imageSrcSet + "]";
          if ("string" === typeof imageSizes && null !== imageSizes) {
            uniqueKeyPart += "[" + imageSizes + "]";
          } else {
            uniqueKeyPart += "[][]" + url;
          }
        } else {
          uniqueKeyPart += "[][]" + url;
        }
        keyPrefix += "[image]" + uniqueKeyPart;
      } else {
        keyPrefix += "[" + resourceType + "]" + url;
      }
      if (!hints.has(keyPrefix)) {
        hints.add(keyPrefix);
        const adjustedConfig = trimOptions(config) ? { ...config } : null;
        emitHint(request, "L", [url, resourceType, adjustedConfig])
          ? console.log("Hint emitted")
          : emitHint(request, "L", [url, resourceType]);
      }
    } else {
      previousDispatcher.L(url, resourceType, config);
    }
  }
}

function createLazyWrapperAroundPromise(promise) {
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

