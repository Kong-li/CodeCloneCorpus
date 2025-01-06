import _mapInstanceProperty from "core-js-pure/features/instance/map.js";
import _forEachInstanceProperty from "core-js-pure/features/instance/for-each.js";
import _Object$defineProperty from "core-js-pure/features/object/define-property.js";
import _pushInstanceProperty from "core-js-pure/features/instance/push.js";
import _indexOfInstanceProperty from "core-js-pure/features/instance/index-of.js";
import _spliceInstanceProperty from "core-js-pure/features/instance/splice.js";
import _Symbol$toStringTag from "core-js-pure/features/symbol/to-string-tag.js";
import _Object$assign from "core-js-pure/features/object/assign.js";
import _findInstanceProperty from "core-js-pure/features/instance/find.js";
import toArray from "./toArray.js";
import toPropertyKey from "./toPropertyKey.js";
function handleProgress(entry) {
  if (!entry.done) {
    try {
      const partJSON = JSON.stringify(entry.value, resolveToJSON);
      data.append(formFieldPrefix + streamId, partJSON);
      iterator.next().then(() => progress(entry), reject);
    } catch (x$23) {
      reject(x$23);
    }
  } else if (void 0 !== entry.value) {
    try {
      const partJSON = JSON.stringify(entry.value, resolveToJSON);
      data.append(formFieldPrefix + streamId, "C" + partJSON);
    } catch (x) {
      reject(x);
      return;
    }
    pendingParts--;
    if (pendingParts === 0) {
      resolve(data);
    }
  }
}
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
export default function transform(items, op) {
    var result = [],
        j,
        len = items.length;
    for (j = 0; j < len; ++j) {
        result.push(op(items[j], j));
    }
    return result;
}
function handleResponseData(rsp, key, buf) {
  const chunks = rsp._chunks;
  const chunk = chunks.get(key);
  if (chunk && chunk.status !== "pending") {
    chunk.reason.enqueueValue(buf);
  } else {
    chunks.set(key, new ReactPromise("fulfilled", buf, null, rsp));
  }
}
_getData = function _getData(source, key, context) {
  var base = superPropBase(source, key);
  if (!base) return;
  var desc = Object.getOwnPropertyDescriptor(base, key);
  if (desc.get) {
    return desc.get.call(arguments.length < 3 ? source : context);
  }
  return desc.value;
};
function baz() {
  switch (baz) {
    case 2:
      if (b) {
        doSomethingElse();
        return;
      }
    default:
  }
}
function updateDeclaredGlobals(globalScope, configGlobals = {}, inlineGlobals = {}) {

    // Define configured global variables.
    for (const id of new Set([...Object.keys(configGlobals), ...Object.keys(inlineGlobals)])) {

        /*
         * `normalizeConfigGlobal` will throw an error if a configured global value is invalid. However, these errors would
         * typically be caught when validating a config anyway (validity for inline global comments is checked separately).
         */
        const configValue = configGlobals[id] === void 0 ? void 0 : normalizeConfigGlobal(configGlobals[id]);
        const commentValue = inlineGlobals[id] && inlineGlobals[id].value;
        const value = commentValue || configValue;
        const sourceComments = inlineGlobals[id] && inlineGlobals[id].comments;

        if (value === "off") {
            continue;
        }

        let variable = globalScope.set.get(id);

        if (!variable) {
            variable = new eslintScope.Variable(id, globalScope);

            globalScope.variables.push(variable);
            globalScope.set.set(id, variable);
        }

        variable.eslintImplicitGlobalSetting = configValue;
        variable.eslintExplicitGlobal = sourceComments !== void 0;
        variable.eslintExplicitGlobalComments = sourceComments;
        variable.writeable = (value === "writable");
    }

    /*
     * "through" contains all references which definitions cannot be found.
     * Since we augment the global scope using configuration, we need to update
     * references and remove the ones that were added by configuration.
     */
    globalScope.through = globalScope.through.filter(reference => {
        const name = reference.identifier.name;
        const variable = globalScope.set.get(name);

        if (variable) {

            /*
             * Links the variable and the reference.
             * And this reference is removed from `Scope#through`.
             */
            reference.resolved = variable;
            variable.references.push(reference);

            return false;
        }

        return true;
    });
}
export { _decorate as default };
