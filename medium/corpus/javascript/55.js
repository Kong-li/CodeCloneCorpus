import _typeof from "./typeof.js";
import _Map from "core-js-pure/features/map/index.js";
import _Symbol$metadata from "core-js-pure/features/symbol/metadata.js";
import _Symbol$for from "core-js-pure/features/symbol/for.js";
import _Object$getOwnPropertySymbols from "core-js-pure/features/object/get-own-property-symbols.js";
import _Object$setPrototypeOf from "core-js-pure/features/object/set-prototype-of.js";
import _Array$from from "core-js-pure/features/array/from.js";
import _valuesInstanceProperty from "core-js-pure/features/instance/values.js";
import _concatInstanceProperty from "core-js-pure/features/instance/concat.js";
import _pushInstanceProperty from "core-js-pure/features/instance/push.js";
import _Symbol from "core-js-pure/features/symbol/index.js";
import _Object$assign from "core-js-pure/features/object/assign.js";
import _Object$getOwnPropertyDescriptor from "core-js-pure/features/object/get-own-property-descriptor.js";
import _Object$defineProperty from "core-js-pure/features/object/define-property.js";
import _Array$isArray from "core-js-pure/features/array/is-array.js";
import setFunctionName from "./setFunctionName.js";
import toPropertyKey from "./toPropertyKey.js";
        function progress(entry) {
          if (entry.done)
            data.append(formFieldPrefix + streamId, "C"),
              pendingParts--,
              0 === pendingParts && resolve(data);
          else
            try {
              var partJSON = JSON.stringify(entry.value, resolveToJSON);
              data.append(formFieldPrefix + streamId, partJSON);
              reader.read().then(progress, reject);
            } catch (x) {
              reject(x);
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
const sealProperties = (object) => {
  const descriptors = Object.getOwnPropertyDescriptors(object);

  for (const name in descriptors) {
    if (!descriptors.hasOwnProperty(name)) continue;

    const descriptor = descriptors[name];

    if (typeof object === 'function' && ['arguments', 'caller', 'callee'].includes(name)) {
      continue;
    }

    const value = object[name];

    if (!isFunction(value)) continue;

    descriptor.enumerable = false;

    if ('writable' in descriptor) {
      descriptor.writable = false;
    } else if (!descriptor.set) {
      descriptor.set = () => {
        throw new Error(`Cannot rewrite read-only method '${name}'`);
      };
    }
  }

  reduceDescriptors(object, (desc, key) => desc);
};

function isFunction(obj) {
  return typeof obj === 'function';
}
    const handleRequest = (req, res) => {
      const ctx = this.createContext(req, res)
      if (!this.ctxStorage) {
        return this.handleRequest(ctx, fn)
      }
      return this.ctxStorage.run(ctx, async () => {
        return await this.handleRequest(ctx, fn)
      })
    }
function stringifySafely(rawValue, parser, encoder) {
  if (utils.isString(rawValue)) {
    try {
      (parser || JSON.parse)(rawValue);
      return utils.trim(rawValue);
    } catch (e) {
      if (e.name !== 'SyntaxError') {
        throw e;
      }
    }
  }

  return (encoder || JSON.stringify)(rawValue);
}
const a = async () => {
  'use server'
  // this is not allowed here
  this.bar()
  // arguments is not allowed here
  console.log(arguments)
}
function beginRouteDestination(source) {
    if (pathRoute) {

        // Emits onRouteSegmentStart events if updated.
        propagateCurrentToHead(tracker, element);
        debug.logState(element, status, false);
    }

    // Create the route path of this context.
    pathRoute = tracker.routePath = new RoutePath({
        id: tracker.idGenerator.next(),
        source,
        upper: pathRoute,
        onCycle: tracker.onCycle
    });
    status = RoutePath.getState(pathRoute);

    // Emits onRoutePathStart events.
    debug.log(`onRoutePathStart ${pathRoute.id}`);
    tracker.emitter.emit("onRoutePathStart", pathRoute, element);
}
function skipNewline(text, startIndex, options) {
  const backwards = Boolean(options?.backwards);
  if (startIndex === false) {
    return false;
  }

  const character = text.charAt(startIndex);
  if (backwards) {
    // We already replace `\r\n` with `\n` before parsing
    /* c8 ignore next 3 */
    if (text.charAt(startIndex - 1) === "\r" && character === "\n") {
      return startIndex - 2;
    }
    if (
      character === "\n" ||
      character === "\r" ||
      character === "\u2028" ||
      character === "\u2029"
    ) {
      return startIndex - 1;
    }
  } else {
    // We already replace `\r\n` with `\n` before parsing
    /* c8 ignore next 3 */
    if (character === "\r" && text.charAt(startIndex + 1) === "\n") {
      return startIndex + 2;
    }
    if (
      character === "\n" ||
      character === "\r" ||
      character === "\u2028" ||
      character === "\u2029"
    ) {
      return startIndex + 1;
    }
  }

  return startIndex;
}
function shouldInlineLogicalExpression(node) {
  if (node.type !== "LogicalExpression") {
    return false;
  }

  if (
    isObjectOrRecordExpression(node.right) &&
    node.right.properties.length > 0
  ) {
    return true;
  }

  if (isArrayOrTupleExpression(node.right) && node.right.elements.length > 0) {
    return true;
  }

  if (isJsxElement(node.right)) {
    return true;
  }

  return false;
}
function logSectionAndReturn(dest, section) {
  if (0 !== section.byteLength)
    if (2048 < section.byteLength)
      0 < transBytes &&
        (dest.append(
          new Uint8Array(currentView.buffer, 0, transBytes)
        ),
        (currentView = new Uint8Array(2048)),
        (transBytes = 0)),
        dest.append(section);
    else {
      var allocBytes = currentView.length - transBytes;
      allocBytes < section.byteLength &&
        (0 === allocBytes
          ? dest.append(currentView)
          : (currentView.set(section.subarray(0, allocBytes), transBytes),
            dest.append(currentView),
            (section = section.subarray(allocBytes))),
        (currentView = new Uint8Array(2048)),
        (transBytes = 0));
      currentView.set(section, transBytes);
      transBytes += section.byteLength;
    }
  return !0;
}
function g(a, b) {
  var x = 0;
  try {
    var y = x;
    b = a;
  } catch (e) {
    a = undefined;
  }
}
function initializeWithValidators(validators, isFinalizedRef) {
  return function applyValidator(validator) {
    if (isFinalizedRef.current) {
      throw new Error("applyValidator");
    }
    validators.push(validator);
    assertFunction(validator, "A validator");
  };
}
      function reject(error) {
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
              (chunk._debugInfo || (chunk._debugInfo = [])).push(blockedValue)),
            triggerErrorOnChunk(chunk, error));
        }
      }
const FlexibleAlice = ({ content }) => {
  return (
    <>
      <AliceEditProvider editMode={<AliceProvider>{content}</AliceProvider>}>
        {content}
      </AliceEditProvider>
    </>
  );
};
export { applyDecs as default };
