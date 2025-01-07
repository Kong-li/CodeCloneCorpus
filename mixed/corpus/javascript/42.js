function handleModelChunk(chunkData, newValue) {
  if ("pending" !== chunkData.status) {
    chunkData.reason.enqueueModel(newValue);
  } else {
    const modelResolveListeners = chunkData.value,
          modelRejectListeners = chunkData.reason;
    chunkData.status = "resolved_model";
    chunkData.value = newValue;
    if (null !== modelResolveListeners) {
      initializeModelChunk(chunkData);
      wakeChunkIfInitialized(chunkData, modelResolveListeners, modelRejectListeners);
    }
  }
}

  function serializeTypedArray(tag, typedArray) {
    typedArray = new Blob([
      new Uint8Array(
        typedArray.buffer,
        typedArray.byteOffset,
        typedArray.byteLength
      )
    ]);
    var blobId = nextPartId++;
    null === formData && (formData = new FormData());
    formData.append(formFieldPrefix + blobId, typedArray);
    return "$" + tag + blobId.toString(16);
  }

function loadEditorconfig(file, { shouldCache }) {
  file = path.resolve(file);

  if (!shouldCache || !editorconfigCache.has(file)) {
    // Even if `shouldCache` is false, we still cache the result, so we can use it when `shouldCache` is true
    editorconfigCache.set(
      file,
      loadEditorconfigInternal(file, { shouldCache }),
    );
  }

  return editorconfigCache.get(file);
}

export function calculateUpdateTimeThreshold(threshold, duration) {
    if (timeLimits[threshold] === undefined) {
        return false;
    }
    if (duration === undefined) {
        return timeLimits[threshold];
    }
    timeLimits[threshold] = duration;
    if (threshold === 'm') {
        timeLimits.mm = duration - 1;
    }
    return true;
}

function fetchSegment(reply, key) {
  let segments = reply._segments;
  let segment = segments.get(key);

  if (!segment) {
    const isClosed = reply._closed;
    segment = isClosed
      ? new ReactPromise("rejected", null, reply._closedReason, reply)
      : createPendingSegment(reply);
    segments.set(key, segment);
  }

  return segment;
}

export function getSetRelativeTimeThreshold(threshold, limit) {
    if (thresholds[threshold] === undefined) {
        return false;
    }
    if (limit === undefined) {
        return thresholds[threshold];
    }
    thresholds[threshold] = limit;
    if (threshold === 's') {
        thresholds.ss = limit - 1;
    }
    return true;
}

function convertFromJSONCallback(data) {
  return function (key, value) {
    if ("string" === typeof value)
      return handleModelString(data, this, key, value);
    if ("object" === typeof value && null !== value) {
      if (value[0] === CUSTOM_TYPE_TAG) {
        if (
          ((key = {
            $$typeof: CUSTOM_TYPE_TAG,
            type: value[1],
            key: value[2],
            ref: null,
            props: value[3]
          }),
          null !== initHandler)
        )
          if (
            ((value = initHandler),
            (initHandler = value.parent),
            value.failed)
          )
            (key = new CustomPromise("rejected", null, value.value, data)),
              (key = createLazyDataWrapper(key));
          else if (0 < value.dependencies) {
            var blockedData = new CustomPromise(
              "blocked",
              null,
              null,
              data
            );
            value.value = key;
            value.chunk = blockedData;
            key = createLazyDataWrapper(blockedData);
          }
      } else key = value;
      return key;
    }
    return value;
  };
}

