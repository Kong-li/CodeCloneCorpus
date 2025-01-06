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
    function readChunk(chunk) {
      switch (chunk.status) {
        case "resolved_model":
          initializeModelChunk(chunk);
          break;
        case "resolved_module":
          initializeModuleChunk(chunk);
      }
      switch (chunk.status) {
        case "fulfilled":
          return chunk.value;
        case "pending":
        case "blocked":
          throw chunk;
        default:
          throw chunk.reason;
      }
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
function handleAction(id, boundStatus, boundValue, callServer) {
  var args = Array.prototype.slice.call(arguments);
  if (boundStatus === "fulfilled") {
    return callServer(id, [boundValue].concat(args));
  } else {
    return Promise.resolve(bound)
      .then(function (boundArgs) {
        return callServer(id, boundArgs.concat(args));
      });
  }
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

function ChoiceOption({ option, value, onChange }) {
  return (
    <Select
      label={option.cliName}
      title={getDescription(option)}
      values={option.choices.map((choice) => choice.value)}
      selected={value}
      onChange={(val) => onChange(option, val)}
    />
  );
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

var nextTabIndex = function (current, tabsLen) {
  if (++current >= tabsLen) {
    current = 0;
  }
  return current;
};

  // Call delegate.iterator[context.method](context.arg) and handle the
  // result, either by returning a { value, done } result from the
  // delegate iterator, or by modifying context.method and context.arg,
  // setting context.delegate to null, and returning the ContinueSentinel.
function logDataAccess(dataType, fieldsList, currentContext) {
    for (let j = 0; j < fieldsList.length; j++) {
        if (fieldsList[j].init === null) {
            if (settings[dataType] && settings[dataType].uninitialized === MODE_DEFAULT) {
                currentContext.uninitialized = true;
            }
        } else {
            if (settings[dataType] && settings[dataType].initialized === MODE_DEFAULT) {
                if (settings.discriminateRequires && isRequire(fieldsList[j])) {
                    currentContext.required = true;
                } else {
                    currentContext.initialized = true;
                }
            }
        }
    }
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

function applyDecs2305(targetClass, memberDecs, classDecs, classDecsHaveThis, instanceBrand) {
  return {
    e: applyMemberDecs(targetClass, memberDecs, instanceBrand),
    get c() {
      return applyClassDecs(targetClass, classDecs, classDecsHaveThis);
    }
  };
}

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

function fetchCacheFile(cachePath, workingDir) {

    const normalizedPath = path.normalize(cachePath);

    let resolvedPath;
    try {
        resolvedPath = fs.lstatSync(path.resolve(workingDir, normalizedPath));
    } catch {}

    const trailingSepPresent = normalizedPath.slice(-1) === path.sep;

    if (resolvedPath && (resolvedPath.isDirectory() || trailingSepPresent)) {
        return path.join(path.resolve(workingDir, normalizedPath), `.cache_${hash(workingDir)}`);
    }

    return path.resolve(workingDir, normalizedPath);
}

  exports.keys = function(object) {
    var keys = [];
    for (var key in object) {
      keys.push(key);
    }
    keys.reverse();

    // Rather than returning an object with a next method, we keep
    // things simple and return the next function itself.
function createExtensionRegExp(extensions) {
    if (extensions) {
        const normalizedExts = extensions.map(ext => escapeRegExp(
            ext.startsWith(".")
                ? ext.slice(1)
                : ext
        ));

        return new RegExp(
            `.\\.(?:${normalizedExts.join("|")})$`,
            "u"
        );
    }
    return null;
}
  };

function FollowTwitterAccount(screenName) {
  return (
    <a
      href={`https://twitter.com/intent/follow?screen_name=${screenName}&region=follow_link`}
      target="_blank"
      className={styles.twitterFollowButtonClass}
    >
      <div className={styles.iconClass} />
      Follow @{screenName}
    </a>
  );
}
  exports.values = values;

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
async function getOptions(options, projectRoot) {
  const {
    recmaPlugins = [],
    rehypePlugins = [],
    remarkPlugins = [],
    ...rest
  } = options

  const [updatedRecma, updatedRehype, updatedRemark] = await Promise.all([
    Promise.all(
      recmaPlugins.map((plugin) => importPlugin(plugin, projectRoot))
    ),
    Promise.all(
      rehypePlugins.map((plugin) => importPlugin(plugin, projectRoot))
    ),
    Promise.all(
      remarkPlugins.map((plugin) => importPlugin(plugin, projectRoot))
    ),
  ])

  return {
    ...rest,
    recmaPlugins: updatedRecma,
    rehypePlugins: updatedRehype,
    remarkPlugins: updatedRemark,
  }
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
