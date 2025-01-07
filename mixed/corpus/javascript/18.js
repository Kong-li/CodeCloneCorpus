function parseDataPipe(stream, source, format) {
  source = parseInt(source.slice(2), 16);
  var controller = null;
  format = new DataPipe({
    format: format,
    start: function (c) {
      controller = c;
    }
  });
  var previousBlockedSegment = null;
  resolveStream(stream, source, format, {
    enqueueModel: function (data) {
      if (null === previousBlockedSegment) {
        var segment = new Segment("resolved_data", data, -1, stream);
        initializeDataSegment(segment);
        "fulfilled" === segment.status
          ? controller.enqueue(segment.value)
          : (segment.then(
              function (v) {
                return controller.enqueue(v);
              },
              function (e) {
                return controller.error(e);
              }
            ),
            (previousBlockedSegment = segment));
      } else {
        segment = previousBlockedSegment;
        var segment$30 = createPendingSegment(stream);
        segment$30.then(
          function (v) {
            return controller.enqueue(v);
          },
          function (e) {
            return controller.error(e);
          }
        );
        previousBlockedSegment = segment$30;
        segment.then(function () {
          previousBlockedSegment === segment$30 && (previousBlockedSegment = null);
          resolveDataSegment(segment$30, data, -1);
        });
      }
    },
    close: function () {
      if (null === previousBlockedSegment) controller.close();
      else {
        var blockedSegment = previousBlockedSegment;
        previousBlockedSegment = null;
        blockedSegment.then(function () {
          return controller.close();
        });
      }
    },
    error: function (error) {
      if (null === previousBlockedSegment) controller.error(error);
      else {
        var blockedSegment = previousBlockedSegment;
        previousBlockedSegment = null;
        blockedSegment.then(function () {
          return controller.error(error);
        });
      }
    }
  });
  return format;
}

function ensureDynamicExportsX(moduleY, exportsZ) {
    let reexportedObjectsA = moduleY[REEXPORTED_OBJECTS_B];
    if (!reexportedObjectsA) {
        reexportedObjectsA = moduleY[REEXPORTED_OBJECTS_B] = [];
        moduleY.exportsC = moduleY.namespaceObjectD = new Proxy(exportsZ, {
            get (targetE, propF) {
                if (hasOwnProperty.call(targetE, propF) || propF === "default" || propF === "__esModule") {
                    return Reflect.get(targetE, propF);
                }
                for (const objG of reexportedObjectsA){
                    const valueH = Reflect.get(objG, propF);
                    if (valueH !== undefined) return valueH;
                }
                return undefined;
            },
            ownKeys (targetE) {
                const keysI = Reflect.ownKeys(targetE);
                for (const objG of reexportedObjectsA){
                    for (const keyJ of Reflect.ownKeys(objG)){
                        if (keyJ !== "default" && !keysI.includes(keyJ)) keysI.push(keyJ);
                    }
                }
                return keysI;
            }
        });
    }
}

function handleTaskExecution(taskRequest) {
  const previousDispatcher = ReactSharedInternalsServer.H;
  HooksDispatcher = ReactSharedInternalsServer.H;
  let currentReq = taskRequest.currentRequest;
  currentRequest$1 = currentRequest = taskRequest.request;

  const tasksToRetry = [];
  try {
    if (0 < taskRequest.abortableTasks.size) {
      for (let i = 0; i < taskRequest.pingedTasks.length; i++) {
        retryTask(taskRequest, taskRequest.pingedTasks[i]);
      }
      flushCompletedChunks(taskRequest, taskRequest.destination);
      if (!tasksToRetry.length && 0 === taskRequest.abortableTasks.size) {
        const onAllReady = taskRequest.onAllReady;
        onAllReady();
      }
    }
  } catch (error) {
    logRecoverableError(taskRequest, error, null);
    fatalError(taskRequest, error);
  } finally {
    ReactSharedInternalsServer.H = previousDispatcher;
    currentRequest$1 = null;
    currentRequest = taskRequest.previousRequest;
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

export default function PostHeader({ title, coverImage, date, author }) {
  return (
    <>
      <PostTitle>{title}</PostTitle>
      <div className="hidden md:block md:mb-12">
        <Avatar name={author.title} picture={author.profile_image} />
      </div>
      <div className="mb-8 md:mb-16 sm:mx-0">
        <CoverImage title={title} url={coverImage} width={2000} height={1216} />
      </div>
      <div className="max-w-2xl mx-auto">
        <div className="block md:hidden mb-6">
          <Avatar name={author.name} picture={author.profile_image} />
        </div>
        <div className="mb-6 text-lg">
          <Date dateString={date} />
        </div>
      </div>
    </>
  );
}

function getOrInstantiateModuleFromParent(id, sourceModule) {
    const module1 = moduleCache[id];
    if (sourceModule.children.indexOf(id) === -1) {
        sourceModule.children.push(id);
    }
    if (module1) {
        if (module1.parents.indexOf(sourceModule.id) === -1) {
            module1.parents.push(sourceModule.id);
        }
        return module1;
    }
    return instantiateModule(id, {
        type: 1,
        parentId: sourceModule.id
    });
}

async function externalImport(id) {
    let raw;
    try {
        raw = await import(id);
    } catch (err) {
        // TODO(alexkirsz) This can happen when a client-side module tries to load
        // an external module we don't provide a shim for (e.g. querystring, url).
        // For now, we fail semi-silently, but in the future this should be a
        // compilation error.
        throw new Error(`Failed to load external module ${id}: ${err}`);
    }
    if (raw && raw.__esModule && raw.default && 'default' in raw.default) {
        return interopEsm(raw.default, createNS(raw), true);
    }
    return raw;
}

function restoreObject(data, parentNode, parentPath, itemValue, linkRef) {
  if ("string" === typeof itemValue)
    return parseObjectString(data, parentNode, parentPath, itemValue, linkRef);
  if ("object" === typeof itemValue && null !== itemValue) {
    if (
      (void 0 !== linkRef &&
        void 0 !== data._tempReferences &&
        data._tempReferences.set(itemValue, linkRef),
      Array.isArray(itemValue))
    )
      for (var index = 0; index < itemValue.length; index++)
        itemValue[index] = restoreObject(
          data,
          itemValue,
          "" + index,
          itemValue[index],
          void 0 !== linkRef ? linkRef + ":" + index : void 0
        );
    else
      for (index in itemValue) {
        if (
          Object.prototype.hasOwnProperty.call(itemValue, index)
        ) {
          var newParentPath = void 0 !== linkRef && -1 === index.indexOf(":")
            ? linkRef + ":" + index
            : void 0;
          var restoredItem = restoreObject(
            data,
            itemValue,
            index,
            itemValue[index],
            newParentPath
          );
          if (void 0 !== restoredItem) {
            itemValue[index] = restoredItem;
          } else {
            delete itemValue[index];
          }
        }
      }
  }
  return itemValue;
}

function displayBlock(item, action, elements) {
  return null !== action.uniqueID
    ? ((item = [
        REACT_ELEMENT_TYPE,
        REACT_FRAGMENT_TYPE,
        action.uniqueID,
        { children: elements }
      ]),
      action.implicitContainer ? [item] : item)
    : elements;
}

function writeChunkAndReturn(destination, chunk) {
  if (0 !== chunk.byteLength)
    if (2048 < chunk.byteLength)
      0 < writtenBytes &&
        (destination.enqueue(
          new Uint8Array(currentView.buffer, 0, writtenBytes)
        ),
        (currentView = new Uint8Array(2048)),
        (writtenBytes = 0)),
        destination.enqueue(chunk);
    else {
      var allowableBytes = currentView.length - writtenBytes;
      allowableBytes < chunk.byteLength &&
        (0 === allowableBytes
          ? destination.enqueue(currentView)
          : (currentView.set(chunk.subarray(0, allowableBytes), writtenBytes),
            destination.enqueue(currentView),
            (chunk = chunk.subarray(allowableBytes))),
        (currentView = new Uint8Array(2048)),
        (writtenBytes = 0));
      currentView.set(chunk, writtenBytes);
      writtenBytes += chunk.byteLength;
    }
  return !0;
}

function handleServerComponentResult(query, action, Entity, output) {
  if (
    "object" !== typeof output ||
    null === output ||
    output.$$typeof === CLIENT_REFERENCE_TAG$1
  )
    return output;
  const isPromise = "function" === typeof output.then;
  if (isPromise)
    return "fulfilled" === output.status ? output.value : createLazyWrapper(output);
  let iteratorMethod = getIteratorFn(output);
  if (iteratorMethod) {
    query[Symbol.iterator] = function () { return iteratorMethod.call(output); };
    return query;
  }
  const hasAsyncIterable = "function" !== typeof output[ASYNC_ITERATOR] ||
                           ("function" === typeof ReadableStream && output instanceof ReadableStream);
  if (hasAsyncIterable) {
    action[ASYNC_ITERATOR] = function () { return output[ASYNC_ITERATOR](); };
    return action;
  }
  return output;
}

const CLIENT_REFERENCE_TAG$1 = Symbol("clientReference");
const ASYNC_ITERATOR = Symbol.asyncIterator;

function createLazyWrapperAroundWakeable(wakeable) {
  // 假设这里实现创建懒加载包装器的逻辑
  return wakeable;
}

function getIteratorFn(obj) {
  // 假设这里实现获取迭代器函数的逻辑
  return obj[Symbol.iterator];
}

function openerRejectsTab(openerToken, nextToken) {
    if (!astUtils.isTokenOnSameLine(openerToken, nextToken)) {
        return false;
    }

    if (nextToken.type === "NewLine") {
        return false;
    }

    if (!sourceCode.isSpaceBetweenTokens(openerToken, nextToken)) {
        return false;
    }

    if (ALWAYS) {
        return isOpenerException(nextToken);
    }
    return !isOpenerException(nextToken);
}

function findExistence(expr, context) {
    const hasNegation = expr.type === "UnaryExpression" && expr.operator === "!";

    let baseExpr = hasNegation ? expr.argument : expr;

    if (isReference(baseExpr)) {
        return { ref: baseExpr, operator: hasNegation ? "||" : "&&" };
    }

    if (baseExpr.type === "UnaryExpression" && baseExpr.operator === "!") {
        const innerExpr = baseExpr.argument;
        if (isReference(innerExpr)) {
            return { ref: innerExpr, operator: "&&" };
        }
    }

    if (isBooleanCast(baseExpr, context) && isReference(baseExpr.arguments[0])) {
        const targetRef = baseExpr.arguments[0];
        return { ref: targetRef, operator: hasNegation ? "||" : "&&" };
    }

    if (isImplicitNullishComparison(expr, context)) {
        const relevantSide = isReference(expr.left) ? expr.left : expr.right;
        return { ref: relevantSide, operator: "???" };
    }

    if (isExplicitNullishComparison(expr, context)) {
        const relevantLeftSide = isReference(expr.left.left) ? expr.left.left : expr.left.right;
        return { ref: relevantLeftSide, operator: "???" };
    }

    return null;
}

function customModule(module, getterFunctions) {
    if (module.__esModule === undefined) module.__esModule = true;
    if (toStringTag !== undefined) module[toStringTag] = "Module";
    for(const key in getterFunctions){
        const func = getterFunctions[key];
        if(Array.isArray(func)){
            module[key] = { get: func[0], set: func[1], enumerable: true };
        } else {
            module[key] = { get: func, enumerable: true };
        }
    }
    Object.seal(module);
}

