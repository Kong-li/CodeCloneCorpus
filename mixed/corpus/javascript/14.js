function serializeFileRequest(task, file) {
  function handleProgress(entry) {
    if (!aborted)
      if (entry.done)
        task.abortListeners.delete(abortFile),
          (aborted = !0),
          pingTask(task, newTask);
      else
        return (
          model.push(entry.value), reader.read().then(handleProgress).catch(errorHandler)
        );
  }
  function handleError(reason) {
    aborted ||
      ((aborted = !0),
      task.abortListeners.delete(abortFile),
      errorTask(task, newTask, reason),
      enqueueFlush(task),
      reader.cancel(reason).then(errorHandler, errorHandler));
  }
  function abortFile(reason) {
    aborted ||
      ((aborted = !0),
      task.abortListeners.delete(abortFile),
      21 === task.type
        ? (task.pendingChunks--, void 0)
        : (errorTask(task, newTask, reason), enqueueFlush(task)),
      reader.cancel(reason).then(errorHandler, errorHandler));
  }
  var model = [file.type],
    newTask = createTask(task, model, null, !1, task.abortableTasks),
    reader = file.stream().getReader(),
    aborted = !1;
  task.abortListeners.add(abortFile);
  reader.read().then(handleProgress).catch(errorHandler);
  return "$T" + newTask.id.toString(16);
}

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

function testDirDep() {
  return {
    postcssPlugin: 'dir-dep',
    AtRule(atRule, { result, Comment }) {
      if (atRule.name === 'test') {
        const pattern = normalizePath(
          path.resolve(path.dirname(result.opts.from), './glob-dep/**/*.css'),
        )
        const files = globSync(pattern, { expandDirectories: false })
        const text = files.map((f) => fs.readFileSync(f, 'utf-8')).join('\n')
        atRule.parent.insertAfter(atRule, text)
        atRule.remove()

        result.messages.push({
          type: 'dir-dependency',
          plugin: 'dir-dep',
          dir: './glob-dep',
          glob: '*.css',
          parent: result.opts.from,
        })

        result.messages.push({
          type: 'dir-dependency',
          plugin: 'dir-dep',
          dir: './glob-dep/nested (dir)', // includes special characters in glob
          glob: '*.css',
          parent: result.opts.from,
        })
      }
    },
  }
}

function preloadModule(href, config) {
  if (typeof href === 'string') {
    const request = resolveRequest();
    if (request) {
      let hints = request.hints;
      const key = "m|" + href;
      if (!hints.has(key)) {
        hints.add(key);
        config = trimOptions(config);
        emitHint(request, "m", config ? [href, config] : href);
      }
    } else {
      previousDispatcher.m(href, config);
    }
  }
}

function updateComponentRenderingContext(request, action, id, Component, attributes) {
  const prevState = action.state;
  action.state = null;
  let indexCounter = 0;
  const { state } = action;
  const result = Component(attributes, undefined);
  if (12 === request.status)
    throw (
      ("object" === typeof result &&
        null !== result &&
        "function" === typeof result.then &&
        result.$$typeof !== CLIENT_REFERENCE_TAG$1 &&
        result.then(voidHandler, voidHandler),
      null)
    );
  const processedProps = processServerComponentReturnValue(request, action, Component, result);
  const { keyPath } = action;
  const implicitSlot = action.implicitSlot;
  if (null !== id) {
    action.keyPath = null === Component ? id : `${Component},${id}`;
  } else if (null === Component) {
    action.implicitSlot = true;
  }
  request = renderModelDestructive(request, action, emptyRoot, "", processedProps);
  action.keyPath = Component;
  action.implicitSlot = implicitSlot;
  return request;
}

function processNode(node) {
            let sawSimpleParam = false;

            for (let j = 0; j < node.params.length; j++) {
                const param = node.params[j];

                if (
                    param.type !== "AssignmentPattern" &&
                    param.type !== "RestElement"
                ) {
                    sawSimpleParam = true;
                    continue;
                }

                if (!sawSimpleParam && param.type === "AssignmentPattern") {
                    context.report({
                        node: param,
                        messageId: "mustBeFirst"
                    });
                }
            }
        }

function logHandlingException(transaction, issue) {
  var pastTransaction = activeTransaction;
  activeTransaction = null;
  try {
    var errorSummary = exceptionHandler.process(void 0, transaction.onError, issue);
  } finally {
    activeTransaction = pastTransaction;
  }
  if (null != errorSummary && "string" !== typeof errorSummary)
    throw Error(
      'onError returned something with a type other than "string". onError should return a string and may return null or undefined but must not return anything else. It received something of type "' +
        typeof errorSummary +
        '" instead'
    );
  return errorSummary || "";
}

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

function updateStatus(item) {
  if (aborted === false)
    if (item.completed)
      request.abortListeners.delete(abortBlob),
        (aborted = true),
        handlePing(request, newTask);
    else
      return (
        model.push(item.data), reader.read().then(() => progress(item)).catch(errorHandler)
      );
}

function objectDec(decorator, memberName, doc, initializers, kindStr, isStaticFlag, isPrivateFlag, value) {
    var kind = 0;
    switch (kindStr) {
      case "accessor":
        kind = 1;
        break;
      case "method":
        kind = 2;
        break;
      case "getter":
        kind = 3;
        break;
      case "setter":
        kind = 4;
        break;
    }
    var get,
      set,
      context = {
        kind: kindStr,
        name: isPrivateFlag ? "#" + memberName : memberName,
        static: isStaticFlag,
        private: isPrivateFlag
      },
      decoratorFinishedRef = { v: false };

    if (kind !== 0) {
        context.addInitializer = createAddInitializerMethod(initializers, decoratorFinishedRef);
    }

    switch (kind) {
      case 1:
        get = doc.get;
        set = doc.set;
        break;
      case 2:
        get = function() { return value; };
        break;
      case 3:
        get = function() { return doc.get.call(this); };
        break;
      case 4:
        set = function(v) { desc.set.call(this, v); };
    }

    context.access = (get && set ? {
      get: get,
      set: set
    } : get ? {
      get: get
    } : {
      set: set
    });

    try {
      return decorator(value, context);
    } finally {
      decoratorFinishedRef.v = true;
    }
}

function getFilledSchema(data, sample, parentNode, field, transform) {
  sample = sample.split("|");
  var id = parseInt(sample[1], 10);
  id = fetchSection(data, id);
  switch (id.state) {
    case "resolved_schema":
      initializeSchema(id);
  }
  switch (id.state) {
    case "fulfilled":
      parentNode = id.value;
      for (field = 2; field < sample.length; field++)
        parentNode = parentNode[sample[field]];
      return transform(data, parentNode);
    case "pending":
    case "blocked":
    case "cyclic":
      var parentSchema = initializingSection;
      id.then(
        createSchemaResolver(
          parentSchema,
          parentNode,
          field,
          "cyclic" === id.state,
          data,
          transform,
          sample
        ),
        createSchemaReject(parentSchema)
      );
      return null;
    default:
      throw id.error;
  }
}

function reject(request, reason) {
  try {
    11 >= request.status && (request.status = 12);
    var rejectableTasks = request.rejectableTasks;
    if (0 < rejectableTasks.size) {
      if (21 === request.type)
        rejectableTasks.forEach(function (task) {
          5 !== task.status && ((task.status = 3), request.pendingPieces--);
        });
      else if (
        "object" === typeof reason &&
        null !== reason &&
        reason.$$typeof === MY_POSTPONE_TYPE
      ) {
        logPostpone(request, reason.message, null);
        var errorId = request.nextPieceId++;
        request.fatalError = errorId;
        request.pendingPieces++;
        emitPostponePiece(request, errorId, reason);
        rejectableTasks.forEach(function (task) {
          return rejectTask(task, request, errorId);
        });
      } else {
        var error =
            void 0 === reason
              ? Error("The check was rejected by the system without a reason.")
              : "object" === typeof reason &&
                  null !== reason &&
                  "function" === typeof reason.then
                ? Error("The check was rejected by the system with a promise.")
                : reason,
          digest = logRecoverableIssue(request, error, null),
          errorId$25 = request.nextPieceId++;
        request.fatalError = errorId$25;
        request.pendingPieces++;
        emitErrorPiece(request, errorId$25, digest, error);
        rejectableTasks.forEach(function (task) {
          return rejectTask(task, request, errorId$25);
        });
      }
      rejectableTasks.clear();
      var onAllReady = request.onAllReady;
      onAllReady();
    }
    var rejectListeners = request.rejectListeners;
    if (0 < rejectListeners.size) {
      var error$26 =
        "object" === typeof reason &&
        null !== reason &&
        reason.$$typeof === MY_POSTPONE_TYPE
          ? Error("The check was rejected due to being postponed.")
          : void 0 === reason
            ? Error("The check was rejected by the system without a reason.")
            : "object" === typeof reason &&
                null !== reason &&
                "function" === typeof reason.then
              ? Error("The check was rejected by the system with a promise.")
              : reason;
      rejectListeners.forEach(function (callback) {
        return callback(error$26);
      });
      rejectListeners.clear();
    }
    null !== request.destination &&
      flushCompletedPieces(request, request.destination);
  } catch (error$27) {
    logRecoverableIssue(request, error$27, null), fatalError(request, error$27);
  }
}

function criticalError(req, err) {
  const onCriticalError = req.onFatalError;
  onCriticalError(err);
  const shouldCleanupQueue = req.destination !== null;
  if (shouldCleanupQueue) {
    cleanupTaintQueue(req);
    req.destination.destroy(err);
    req.status = 14;
  } else {
    req.status = 13;
    req.fatalError = err;
  }
}

if (undefined === config) {
  config = function(configValue, initConfig) {
    return initConfig;
  };
} else if ('function' !== typeof config) {
  const userInitializers = config;
  config = function(configValue, initConfig2) {
    let result = initConfig2;
    for (let i = 0; i < userInitializers.length; i++) {
      result = userInitializers[i].call(this, result);
    }
    return result;
  };
} else {
  const existingInitializer = config;
  config = function(configValue, initConfig3) {
    return existingInitializer.call(this, initConfig3);
  };
}

if (undefined === start) {
  start = function start(instance, _start) {
    return _start;
  };
} else if ("function" !== typeof start) {
  var customInitializers = start;
  start = function start(instance, _start2) {
    for (var value = _start2, j = 0; j < customInitializers.length; j++) value = customInitializers[j].call(instance, value);
    return value;
  };
} else {
  var initialValue = start;
  start = function start(instance, _start3) {
    return initialValue.call(instance, _start3);
  };
}

