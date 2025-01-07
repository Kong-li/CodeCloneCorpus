function startFlowing(request, destination) {
  if (13 === request.status)
    (request.status = 14), destination.destroy(request.fatalError);
  else if (14 !== request.status && null === request.destination) {
    request.destination = destination;
    try {
      flushCompletedChunks(request, destination);
    } catch (error) {
      logRecoverableError(request, error, null), fatalError(request, error);
    }
  }
}

function addTaskProcess(task) {
  !1 === task.processScheduled &&
    0 === task.pendingTasks.length &&
    null !== task.target &&
    ((task.processScheduled = !0),
    setTimeout(function () {
      task.processScheduled = !1;
      var target = task.target;
      target && processCompletedTasks(task, target);
    }));
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

    decorateConstructor: function decorateConstructor(elements, decorators) {
      var finishers = [];
      for (var i = decorators.length - 1; i >= 0; i--) {
        var obj = this.fromClassDescriptor(elements);
        var elementsAndFinisher = this.toClassDescriptor((0, decorators[i])(obj) || obj);
        if (elementsAndFinisher.finisher !== undefined) {
          finishers.push(elementsAndFinisher.finisher);
        }
        if (elementsAndFinisher.elements !== undefined) {
          elements = elementsAndFinisher.elements;
          for (var j = 0; j < elements.length - 1; j++) {
            for (var k = j + 1; k < elements.length; k++) {
              if (elements[j].key === elements[k].key && elements[j].placement === elements[k].placement) {
                throw new TypeError("Duplicated element (" + elements[j].key + ")");
              }
            }
          }
        }
      }
      return {
        elements: elements,
        finishers: finishers
      };
    },

function preconnect(href, crossOrigin) {
  if ("string" === typeof href) {
    var request = resolveRequest();
    if (request) {
      var hints = request.hints,
        key = "C|" + (null == crossOrigin ? "null" : crossOrigin) + "|" + href;
      hints.has(key) ||
        (hints.add(key),
        "string" === typeof crossOrigin
          ? emitHint(request, "C", [href, crossOrigin])
          : emitHint(request, "C", href));
    } else previousDispatcher.C(href, crossOrigin);
  }
}

function parseModelString(response, obj, key, value, reference) {
  if ("$" === value[0]) {
    switch (value[1]) {
      case "$":
        return value.slice(1);
      case "@":
        return (obj = parseInt(value.slice(2), 16)), getChunk(response, obj);
      case "F":
        return (
          (value = value.slice(2)),
          (value = getOutlinedModel(response, value, obj, key, createModel)),
          loadServerReference$1(
            response,
            value.id,
            value.bound,
            initializingChunk,
            obj,
            key
          )
        );
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
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, obj, key, createMap)
        );
      case "W":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, obj, key, createSet)
        );
      case "K":
        obj = value.slice(2);
        var formPrefix = response._prefix + obj + "_",
          data = new FormData();
        response._formData.forEach(function (entry, entryKey) {
          entryKey.startsWith(formPrefix) &&
            data.append(entryKey.slice(formPrefix.length), entry);
        });
        return data;
      case "i":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, obj, key, extractIterator)
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
        return (
          (obj = parseInt(value.slice(2), 16)),
          response._formData.get(response._prefix + obj)
        );
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
    value = value.slice(1);
    return getOutlinedModel(response, value, obj, key, createModel);
  }
  return value;
}

function displayItem(query, task, item, key, marker, attributes) {
  if (null !== marker && void 0 !== marker)
    throw Error(
      "Markers cannot be used in Server Components, nor passed to Client Components."
    );
  if (
    "function" === typeof item &&
    item.$$typeof !== USER_REFERENCE_TAG$1 &&
    item.$$typeof !== TEMPORARY_REFERENCE_TAG
  )
    return displayFunctionComponent(query, task, key, item, attributes);
  if (item === ITEM_FRAGMENT_TYPE && null === key)
    return (
      (item = task.implicitSlot),
      null === task.keyPath && (task.implicitSlot = !0),
      (attributes = displayModelDestructive(
        query,
        task,
        emptyRoot,
        "",
        attributes.children
      )),
      (task.implicitSlot = item),
      attributes
    );
  if (
    null != item &&
    "object" === typeof item &&
    item.$$typeof !== USER_REFERENCE_TAG$1
  )
    switch (item.$$typeof) {
      case REACT_LAZY_TYPE:
        var init = item._init;
        item = init(item._payload);
        if (12 === query.status) throw null;
        return displayItem(query, task, item, key, marker, attributes);
      case REACT_FORWARD_REF_TYPE:
        return displayFunctionComponent(query, task, key, item.render, attributes);
      case REACT_MEMO_TYPE:
        return displayItem(query, task, item.type, key, marker, attributes);
    }
  query = key;
  key = task.keyPath;
  null === query
    ? (query = key)
    : null !== key && (query = key + "," + query);
  attributes = [REACT_ELEMENT_TYPE, item, query, attributes];
  task = task.implicitSlot && null !== query ? [attributes] : attributes;
  return task;
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

function processServerReference(bundlerConfig, refId) {
  let moduleName = "",
    resolvedData = bundlerConfig[refId];
  if (!resolvedData) {
    const idx = refId.lastIndexOf("#");
    -1 !== idx &&
      ((moduleName = refId.slice(idx + 1)),
      (resolvedData = bundlerConfig[refId.slice(0, idx)]));
    if (!resolvedData)
      throw Error(
        'Could not find the module "' +
          refId +
          '" in the React Server Manifest. This is probably a bug in the React Server Components bundler.'
      );
  }
  return [resolvedData.id, resolvedData.chunks, moduleName];
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

