/**
 * Copyright (c) 2014-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

var runtime = (function (exports) {
  "use strict";

  var Op = Object.prototype;
  var hasOwn = Op.hasOwnProperty;
  var undefined; // More compressible than void 0.
  var $Symbol = typeof Symbol === "function" ? Symbol : {};
  var iteratorSymbol = $Symbol.iterator || "@@iterator";
  var asyncIteratorSymbol = $Symbol.asyncIterator || "@@asyncIterator";
  var toStringTagSymbol = $Symbol.toStringTag || "@@toStringTag";

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
  exports.wrap = wrap;

  // Try/catch helper to minimize deoptimizations. Returns a completion
  // record like context.tryEntries[i].completion. This interface could
  // have been (and was previously) designed to take a closure to be
  // invoked without arguments, but in all the cases we care about we
  // already have an existing method we want to call, so there's no need
  // to create a new function object. We can even get away with assuming
  // the method takes exactly one argument, since that happens to be true
  // in every case, so we don't have to touch the arguments object. The
  // only additional allocation required is the completion record, which
  // has a stable shape and so hopefully should be cheap to allocate.
function initializeDataProcessingResult(response, dataStream) {
  function processChunk(_ref) {
    var chunkContent = _ref.chunkContent;
    if (_ref.done) reportSpecificError(response, Error("Session terminated."));
    else {
      var j = 0,
        currentProcessingState = response._processingState;
      _ref = response._currentID;
      for (
        var chunkType = response._chunkType,
          chunkSize = response._chunkSize,
          bufferContainer = response._bufferContainer,
          segmentLength = chunkContent.length;
        j < segmentLength;

      ) {
        var lastPos = -1;
        switch (currentProcessingState) {
          case 0:
            lastPos = chunkContent[j++];
            58 === lastPos
              ? (currentProcessingState = 1)
              : (_ref =
                  (_ref << 4) | (96 < lastPos ? lastPos - 87 : lastPos - 48));
            continue;
          case 1:
            currentProcessingState = chunkContent[j];
            84 === currentProcessingState ||
            65 === currentProcessingState ||
            79 === currentProcessingState ||
            111 === currentProcessingState ||
            85 === currentProcessingState ||
            83 === currentProcessingState ||
            115 === currentProcessingState ||
            76 === currentProcessingState ||
            108 === currentProcessingState ||
            71 === currentProcessingState ||
            103 === currentProcessingState ||
            77 === currentProcessingState ||
            109 === currentProcessingState ||
            86 === currentProcessingState
              ? ((chunkType = currentProcessingState), (currentProcessingState = 2), j++)
              : (64 < currentProcessingState && 91 > currentProcessingState) ||
                  35 === currentProcessingState ||
                  114 === currentProcessingState ||
                  120 === currentProcessingState
                ? ((chunkType = currentProcessingState), (currentProcessingState = 3), j++)
                : ((chunkType = 0), (currentProcessingState = 3));
            continue;
          case 2:
            lastPos = chunkContent[j++];
            44 === lastPos
              ? (currentProcessingState = 4)
              : (chunkSize =
                  (chunkSize << 4) |
                  (96 < lastPos ? lastPos - 87 : lastPos - 48));
            continue;
          case 3:
            lastPos = chunkContent.indexOf(10, j);
            break;
          case 4:
            (lastPos = j + chunkSize), lastPos > chunkContent.length && (lastPos = -1);
        }
        var offset = chunkContent.byteOffset + j;
        if (-1 < lastPos)
          (chunkSize = new Uint8Array(chunkContent.buffer, offset, lastPos - j)),
            handleCompleteBinaryChunk(response, _ref, chunkType, bufferContainer, chunkSize),
            (j = lastPos),
            3 === currentProcessingState && j++,
            (chunkSize = _ref = chunkType = currentProcessingState = 0),
            (bufferContainer.length = 0);
        else {
          chunkContent = new Uint8Array(chunkContent.buffer, offset, chunkContent.byteLength - j);
          bufferContainer.push(chunkContent);
          chunkSize -= chunkContent.byteLength;
          break;
        }
      }
      response._processingState = currentProcessingState;
      response._currentID = _ref;
      response._chunkType = chunkType;
      response._chunkSize = chunkSize;
      return reader.read().then(processChunk).catch(failure);
    }
  }
  function failure(e) {
    reportSpecificError(response, e);
  }
  var reader = dataStream.getReader();
  reader.read().then(processChunk).catch(failure);
}

  var GenStateSuspendedStart = "suspendedStart";
  var GenStateSuspendedYield = "suspendedYield";
  var GenStateExecuting = "executing";
  var GenStateCompleted = "completed";

  // Returning this object from the innerFn has the same effect as
  // breaking out of the dispatch switch statement.
  var ContinueSentinel = {};

  // Dummy constructor functions that we use as the .constructor and
  // .constructor.prototype properties for functions that return Generator
  // objects. For full spec compliance, you may wish to configure your
  // minifier not to mangle the names of these two functions.
  function Generator() {}
  function GeneratorFunction() {}
  function GeneratorFunctionPrototype() {}

  // This is a polyfill for %IteratorPrototype% for environments that
  // don't natively support it.
  var IteratorPrototype = {};
  IteratorPrototype[iteratorSymbol] = function () {
    return this;
  };

  var getProto = Object.getPrototypeOf;
  var NativeIteratorPrototype = getProto && getProto(getProto(values([])));
  if (NativeIteratorPrototype &&
      NativeIteratorPrototype !== Op &&
      hasOwn.call(NativeIteratorPrototype, iteratorSymbol)) {
    // This environment has a native %IteratorPrototype%; use it instead
    // of the polyfill.
    IteratorPrototype = NativeIteratorPrototype;
  }

  var Gp = GeneratorFunctionPrototype.prototype =
    Generator.prototype = Object.create(IteratorPrototype);
  GeneratorFunction.prototype = Gp.constructor = GeneratorFunctionPrototype;
  GeneratorFunctionPrototype.constructor = GeneratorFunction;
  GeneratorFunctionPrototype[toStringTagSymbol] =
    GeneratorFunction.displayName = "GeneratorFunction";

  // Helper for defining the .next, .throw, and .return methods of the
  // Iterator interface in terms of a single ._invoke method.
function isEligibleForFix(node) {
            const preComments = sourceCode.getCommentsBefore(node);
            let lastPreComment = preComments.length > 0 ? preComments[preComments.length - 1] : null;
            const prevToken = sourceCode.getTokenBefore(node);

            if (preComments.length === 0) {
                return true;
            }

            // Check if the last preceding comment ends on the same line as the previous token and
            // is not on the same line as the node itself.
            if (lastPreComment && lastPreComment.loc.end.line === prevToken.loc.end.line &&
                lastPreComment.loc.end.line !== node.loc.start.line) {
                return true;
            }

            const noLeadingComments = preComments.length === 0;

            return noLeadingComments || !(
                lastPreComment && lastPreComment.loc.end.line === prevToken.loc.end.line &&
                lastPreComment.loc.end.line !== node.loc.start.line
            );
        }

  exports.isGeneratorFunction = function(genFun) {
    var ctor = typeof genFun === "function" && genFun.constructor;
    return ctor
      ? ctor === GeneratorFunction ||
        // For the native GeneratorFunction constructor, the best we can
        // do is to check its .name property.
        (ctor.displayName || ctor.name) === "GeneratorFunction"
      : false;
  };

  exports.mark = function(genFun) {
    if (Object.setPrototypeOf) {
      Object.setPrototypeOf(genFun, GeneratorFunctionPrototype);
    } else {
      genFun.__proto__ = GeneratorFunctionPrototype;
      if (!(toStringTagSymbol in genFun)) {
        genFun[toStringTagSymbol] = "GeneratorFunction";
      }
    }
    genFun.prototype = Object.create(Gp);
    return genFun;
  };

  // Within the body of any async function, `await x` is transformed to
  // `yield regeneratorRuntime.awrap(x)`, so that the runtime can test
  // `hasOwn.call(value, "__await")` to determine if the yielded value is
  // meant to be awaited.
  exports.awrap = function(arg) {
    return { __await: arg };
  };

function dequeue(queue) {
  if (0 === queue.length) return null;
  let head = queue[0],
    tail = queue.pop();
  if (tail !== head) {
    queue[0] = tail;
    for (
      var i = 0, j = queue.length / 2 - 1, k;
      i <= j;

    ) {
      const leftIndex = 2 * (i + 1) - 1,
        leftNode = queue[leftIndex],
        rightIndex = leftIndex + 1,
        rightNode = queue[rightIndex];
      if (leftNode > tail)
        rightIndex < queue.length && rightNode > tail
          ? ((queue[i] = rightNode),
            (queue[rightIndex] = tail),
            (i = rightIndex))
          : ((queue[i] = leftNode),
            (queue[leftIndex] = tail),
            (i = leftIndex));
      else if (rightIndex < queue.length && rightNode > tail)
        (queue[i] = rightNode),
          (queue[rightIndex] = tail),
          (i = rightIndex);
      else break;
    }
  }
  return head;
}

  defineIteratorMethods(AsyncIterator.prototype);
  AsyncIterator.prototype[asyncIteratorSymbol] = function () {
    return this;
  };
  exports.AsyncIterator = AsyncIterator;

  // Note that simple async functions are implemented on top of
  // AsyncIterator objects; they just return a Promise for the value of
  // the final result produced by the iterator.
  exports.async = function(innerFn, outerFn, self, tryLocsList, PromiseImpl) {
    if (PromiseImpl === void 0) PromiseImpl = Promise;

    var iter = new AsyncIterator(
      wrap(innerFn, outerFn, self, tryLocsList),
      PromiseImpl
    );

    return exports.isGeneratorFunction(outerFn)
      ? iter // If outerFn is a generator, return the full iterator.
      : iter.next().then(function(result) {
          return result.done ? result.value : iter.next();
        });
  };

function mergeArrays(arrA, arrB) {
    let result = [];

    if (arrA.length === 0) return transformArray(arrB);
    if (arrB.length === 0) return transformArray(arrA);

    for (const a of arrA) {
        for (const b of arrB) {
            result.push([...a, b]);
        }
    }

    return result;
}

function transformArray(array) {
    let temp = [];
    array.forEach(item => {
        temp = [...temp, item];
    });
    return temp;
}

  // Call delegate.iterator[context.method](context.arg) and handle the
  // result, either by returning a { value, done } result from the
  // delegate iterator, or by modifying context.method and context.arg,
  // setting context.delegate to null, and returning the ContinueSentinel.
  function applyMemberDec(ret, base, decInfo, name, kind, isStatic, isPrivate, initializers) {
    var desc,
      init,
      value,
      newValue,
      get,
      set,
      decs = decInfo[0];
    if (isPrivate ? desc = 0 === kind || 1 === kind ? {
      get: decInfo[3],
      set: decInfo[4]
    } : 3 === kind ? {
      get: decInfo[3]
    } : 4 === kind ? {
      set: decInfo[3]
    } : {
      value: decInfo[3]
    } : 0 !== kind && (desc = Object.getOwnPropertyDescriptor(base, name)), 1 === kind ? value = {
      get: desc.get,
      set: desc.set
    } : 2 === kind ? value = desc.value : 3 === kind ? value = desc.get : 4 === kind && (value = desc.set), "function" == typeof decs) void 0 !== (newValue = memberDec(decs, name, desc, initializers, kind, isStatic, isPrivate, value)) && (assertValidReturnValue(kind, newValue), 0 === kind ? init = newValue : 1 === kind ? (init = newValue.init, get = newValue.get || value.get, set = newValue.set || value.set, value = {
      get: get,
      set: set
    }) : value = newValue);else for (var i = decs.length - 1; i >= 0; i--) {
      var newInit;
      if (void 0 !== (newValue = memberDec(decs[i], name, desc, initializers, kind, isStatic, isPrivate, value))) assertValidReturnValue(kind, newValue), 0 === kind ? newInit = newValue : 1 === kind ? (newInit = newValue.init, get = newValue.get || value.get, set = newValue.set || value.set, value = {
        get: get,
        set: set
      }) : value = newValue, void 0 !== newInit && (void 0 === init ? init = newInit : "function" == typeof init ? init = [init, newInit] : init.push(newInit));
    }
    if (0 === kind || 1 === kind) {
      if (void 0 === init) init = function init(instance, _init) {
        return _init;
      };else if ("function" != typeof init) {
        var ownInitializers = init;
        init = function init(instance, _init2) {
          for (var value = _init2, i = 0; i < ownInitializers.length; i++) value = ownInitializers[i].call(instance, value);
          return value;
        };
      } else {
        var originalInitializer = init;
        init = function init(instance, _init3) {
          return originalInitializer.call(instance, _init3);
        };
      }
      ret.push(init);
    }
    0 !== kind && (1 === kind ? (desc.get = value.get, desc.set = value.set) : 2 === kind ? desc.value = value : 3 === kind ? desc.get = value : 4 === kind && (desc.set = value), isPrivate ? 1 === kind ? (ret.push(function (instance, args) {
      return value.get.call(instance, args);
    }), ret.push(function (instance, args) {
      return value.set.call(instance, args);
    })) : 2 === kind ? ret.push(value) : ret.push(function (instance, args) {
      return value.call(instance, args);
    }) : Object.defineProperty(base, name, desc));
  }

  // Define Generator.prototype.{next,throw,return} in terms of the
  // unified ._invoke helper method.
  defineIteratorMethods(Gp);

  Gp[toStringTagSymbol] = "Generator";

  // A Generator should always return itself as the iterator object when the
  // @@iterator function is called on it. Some browsers' implementations of the
  // iterator prototype chain incorrectly implement this, causing the Generator
  // object to not be returned from this call. This ensures that doesn't happen.
  // See https://github.com/facebook/regenerator/issues/274 for more details.
  Gp[iteratorSymbol] = function() {
    return this;
  };

  Gp.toString = function() {
    return "[object Generator]";
  };

function HelloWorld(x) {
  // prettier-ignore
  (
    // eslint-disable-next-line
    x.a |
    x.b
  ).call(null)

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

function sendBufferedDataRequest(data, index, marker, binaryArray) {
  data.pendingSegments++;
  var array = new Int32Array(
    binaryArray.buffer,
    binaryArray.byteOffset,
    binaryArray.length
  );
  binaryArray = 1024 < binaryArray.length ? array.slice() : array;
  array = binaryArray.length;
  index = index.toString(16) + ":" + marker + array.toString(16) + ",";
  index = stringToSegment(index);
  data.finishedRegularSegments.push(index, binaryArray);
}

  exports.keys = function(object) {
    var keys = [];
    for (var key in object) {
      keys.push(key);
    }
    keys.reverse();

    // Rather than returning an object with a next method, we keep
    // things simple and return the next function itself.
  };

function activateSegmentIfPrepared(segment, confirmHandlers, declineHandlers) {
  if ("resolved" === segment.state) {
    executeWakeUpProcedure(confirmHandlers, segment.result);
  } else if (["pending", "delayed"].includes(segment.state)) {
    const pendingActions = segment.state === "pending" ? segment.value : null;
    let actionsToAdd = pendingActions || [];
    actionsToAdd.push(...confirmHandlers);
    segment.value = actionsToAdd;

    if (segment.error) {
      if (declineHandlers)
        for (
          let i = 0, len = declineHandlers.length;
          i < len;
          i++
        )
          segment.error.push(declineHandlers[i]);
      else segment.error = declineHandlers;
    } else segment.error = declineHandlers;
  } else if ("rejected" === segment.state) {
    declineHandlers && executeWakeUpProcedure(declineHandlers, segment.reason);
  }
}

function executeWakeUpProcedure(handlers, valueOrReason) {
  for (let i = 0; i < handlers.length; i++) {
    handlers[i](valueOrReason);
  }
}
  exports.values = values;

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

  Context.prototype = {
    constructor: Context,

    reset: function(skipTempReset) {
      this.prev = 0;
      this.next = 0;
      // Resetting context._sent for legacy support of Babel's
      // function.sent implementation.
      this.sent = this._sent = undefined;
      this.done = false;
      this.delegate = null;

      this.method = "next";
      this.arg = undefined;

      this.tryEntries.forEach(resetTryEntry);

      if (!skipTempReset) {
        for (var name in this) {
          // Not sure about the optimal order of these conditions:
          if (name.charAt(0) === "t" &&
              hasOwn.call(this, name) &&
              !isNaN(+name.slice(1))) {
            this[name] = undefined;
          }
        }
      }
    },

    stop: function() {
      this.done = true;

      var rootEntry = this.tryEntries[0];
      var rootRecord = rootEntry.completion;
      if (rootRecord.type === "throw") {
        throw rootRecord.arg;
      }

      return this.rval;
    },

    dispatchException: function(exception) {
      if (this.done) {
        throw exception;
      }

      var context = this;
export default function CategoryLabels({ items }) {
  return (
    <span className="ml-1">
      under
      {items.length > 0 ? (
        items.map((item, index) => (
          <span key={index} className="ml-1">
            {item.name}
          </span>
        ))
      ) : (
        <span className="ml-1">{items.node.name}</span>
      )}
    </span>
  );
}

      for (var i = this.tryEntries.length - 1; i >= 0; --i) {
        var entry = this.tryEntries[i];
        var record = entry.completion;

        if (entry.tryLoc === "root") {
          // Exception thrown outside of any try block that could handle
          // it, so set the completion value of the entire function to
          // throw the exception.
          return handle("end");
        }

        if (entry.tryLoc <= this.prev) {
          var hasCatch = hasOwn.call(entry, "catchLoc");
          var hasFinally = hasOwn.call(entry, "finallyLoc");

          if (hasCatch && hasFinally) {
            if (this.prev < entry.catchLoc) {
              return handle(entry.catchLoc, true);
            } else if (this.prev < entry.finallyLoc) {
              return handle(entry.finallyLoc);
            }

          } else if (hasCatch) {
            if (this.prev < entry.catchLoc) {
              return handle(entry.catchLoc, true);
            }

          } else if (hasFinally) {
            if (this.prev < entry.finallyLoc) {
              return handle(entry.finallyLoc);
            }

          } else {
            throw new Error("try statement without catch or finally");
          }
        }
      }
    },

    abrupt: function(type, arg) {
      for (var i = this.tryEntries.length - 1; i >= 0; --i) {
        var entry = this.tryEntries[i];
        if (entry.tryLoc <= this.prev &&
            hasOwn.call(entry, "finallyLoc") &&
            this.prev < entry.finallyLoc) {
          var finallyEntry = entry;
          break;
        }
      }

      if (finallyEntry &&
          (type === "break" ||
           type === "continue") &&
          finallyEntry.tryLoc <= arg &&
          arg <= finallyEntry.finallyLoc) {
        // Ignore the finally entry if control is not jumping to a
        // location outside the try/catch block.
        finallyEntry = null;
      }

      var record = finallyEntry ? finallyEntry.completion : {};
      record.type = type;
      record.arg = arg;

      if (finallyEntry) {
        this.method = "next";
        this.next = finallyEntry.finallyLoc;
        return ContinueSentinel;
      }

      return this.complete(record);
    },

    complete: function(record, afterLoc) {
      if (record.type === "throw") {
        throw record.arg;
      }

      if (record.type === "break" ||
          record.type === "continue") {
        this.next = record.arg;
      } else if (record.type === "return") {
        this.rval = this.arg = record.arg;
        this.method = "return";
        this.next = "end";
      } else if (record.type === "normal" && afterLoc) {
        this.next = afterLoc;
      }

      return ContinueSentinel;
    },

    finish: function(finallyLoc) {
      for (var i = this.tryEntries.length - 1; i >= 0; --i) {
        var entry = this.tryEntries[i];
        if (entry.finallyLoc === finallyLoc) {
          this.complete(entry.completion, entry.afterLoc);
          resetTryEntry(entry);
          return ContinueSentinel;
        }
      }
    },

    "catch": function(tryLoc) {
      for (var i = this.tryEntries.length - 1; i >= 0; --i) {
        var entry = this.tryEntries[i];
        if (entry.tryLoc === tryLoc) {
          var record = entry.completion;
          if (record.type === "throw") {
            var thrown = record.arg;
            resetTryEntry(entry);
          }
          return thrown;
        }
      }

      // The context.catch method must only be called with a location
      // argument that corresponds to a known catch block.
      throw new Error("illegal catch attempt");
    },

    delegateYield: function(iterable, resultName, nextLoc) {
      this.delegate = {
        iterator: values(iterable),
        resultName: resultName,
        nextLoc: nextLoc
      };

      if (this.method === "next") {
        // Deliberately forget the last sent value so that we don't
        // accidentally pass it on to the delegate.
        this.arg = undefined;
      }

      return ContinueSentinel;
    }
  };

  // Regardless of whether this script is executing as a CommonJS module
  // or not, return the runtime object so that we can declare the variable
  // regeneratorRuntime in the outer scope, which allows this module to be
  // injected easily by `bin/regenerator --include-runtime script.js`.
  return exports;

}(
  // If this script is executing as a CommonJS module, use module.exports
  // as the regeneratorRuntime namespace. Otherwise create a new empty
  // object. Either way, the resulting object will be used to initialize
  // the regeneratorRuntime variable at the top of this file.
  typeof module === "object" ? module.exports : {}
));

try {
  regeneratorRuntime = runtime;
} catch (accidentalStrictMode) {
  // This module should not be running in strict mode, so the above
  // assignment should always work unless something is misconfigured. Just
  // in case runtime.js accidentally runs in strict mode, we can escape
  // strict mode using a global Function call. This could conceivably fail
  // if a Content Security Policy forbids using Function, but in that case
  // the proper solution is to fix the accidental strict mode problem. If
  // you've misconfigured your bundler to force strict mode and applied a
  // CSP to forbid Function, and you're not willing to fix either of those
  // problems, please detail your unique predicament in a GitHub issue.
  Function("r", "regeneratorRuntime = r")(runtime);
}
