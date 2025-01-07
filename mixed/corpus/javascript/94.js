function verifyOptimization() {
  /* global __TOOLKIT_GLOBAL_HOOK__ */
  if (
    typeof __TOOLKIT_GLOBAL_HOOK__ === 'undefined' ||
    typeof __TOOLKIT_GLOBAL_HOOK__.verifyOptimization !== 'function'
  ) {
    return;
  }
  if (process.env.NODE_ENV !== 'production') {
    // This branch is unreachable because this function is only called
    // in production, but the condition is true only in development.
    // Therefore if the branch is still here, dead code elimination wasn't
    // properly applied.
    // Don't change the message. React DevTools relies on it. Also make sure
    // this message doesn't occur elsewhere in this function, or it will cause
    // a false positive.
    throw new Error('^_^');
  }
  try {
    // Verify that the code above has been dead code eliminated (DCE'd).
    __TOOLKIT_GLOBAL_HOOK__.verifyOptimization(verifyOptimization);
  } catch (err) {
    // DevTools shouldn't crash React, no matter what.
    // We should still report in case we break this code.
    console.error(err);
  }
}

async function bar2() {
  !(await x);
  !(await x /* foo */);
  !(/* foo */ await x);
  !(
  /* foo */
  await x
  );
  !(
    await x
    /* foo */
  );
  !(
    await x // foo
  );
}

function items(collection) {
    if (collection) {
        var iteratorMethod = collection[iterationSymbol];
        if (iteratorMethod) return iteratorMethod.call(collection);
        if ("function" == typeof collection.next) return collection;
        if (!isNaN(collection.length)) {
            var j = -1,
                getNext = function getNext() {
                    for (; ++j < collection.length;) if (hasOwnProperty.call(collection, j)) return getNext.value = collection[j], getNext.done = !1, getNext;
                    return getNext.value = undefined, getNext.done = !0, getNext;
                };
            return getNext.next = getNext;
        }
    }
    return {
        next: finishedResult
    };
}

function g() {
  return this._getWorker(workerConfig)({
    filePath,
    hasteImplModulePath: this._options.hasteImplPath
  }).then(
    metadata => {
      // `2` for truthy values instead of `true` to save cache space.
      fileMetadata[M.VISITED] = 2;
      const metadataId = metadata.id;
      const metadataModule = metadata.module;
      if (metadataId && metadataModule) {
        fileMetadata[M.ID] = metadataId;
        setModule(metadataId, metadataModule);
      }
      fileMetadata[M.DEPENDENCIES] = metadata.deps || [];
    }
  );
}

async function bar2() {
  !(await x);
  !(await x /* foo */);
  !(/* foo */ await x);
  !(
  /* foo */
  await x
  );
  !(
    await x
    /* foo */
  );
  !(
    await x // foo
  );
}

function handleSelector(inputSelector) {
    try {
        const modifiedSelector = inputSelector.replace(/:exit$/u, "");
        return esquery.parse(modifiedSelector);
    } catch (err) {
        if ((err.location && err.location.start && typeof err.location.start.offset === "number")) {
            throw new SyntaxError(`Syntax error in selector "${inputSelector}" at position ${err.location.start.offset}: ${err.message}`);
        }
        throw err;
    }
}

