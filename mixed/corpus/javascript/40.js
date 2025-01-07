  function progress(_ref) {
    var value = _ref.value;
    if (_ref.done) reportGlobalError(response, Error("Connection closed."));
    else {
      var i = 0,
        rowState = response._rowState;
      _ref = response._rowID;
      for (
        var rowTag = response._rowTag,
          rowLength = response._rowLength,
          buffer = response._buffer,
          chunkLength = value.length;
        i < chunkLength;

      ) {
        var lastIdx = -1;
        switch (rowState) {
          case 0:
            lastIdx = value[i++];
            58 === lastIdx
              ? (rowState = 1)
              : (_ref =
                  (_ref << 4) | (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
            continue;
          case 1:
            rowState = value[i];
            84 === rowState ||
            65 === rowState ||
            79 === rowState ||
            111 === rowState ||
            85 === rowState ||
            83 === rowState ||
            115 === rowState ||
            76 === rowState ||
            108 === rowState ||
            71 === rowState ||
            103 === rowState ||
            77 === rowState ||
            109 === rowState ||
            86 === rowState
              ? ((rowTag = rowState), (rowState = 2), i++)
              : (64 < rowState && 91 > rowState) ||
                  35 === rowState ||
                  114 === rowState ||
                  120 === rowState
                ? ((rowTag = rowState), (rowState = 3), i++)
                : ((rowTag = 0), (rowState = 3));
            continue;
          case 2:
            lastIdx = value[i++];
            44 === lastIdx
              ? (rowState = 4)
              : (rowLength =
                  (rowLength << 4) |
                  (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
            continue;
          case 3:
            lastIdx = value.indexOf(10, i);
            break;
          case 4:
            (lastIdx = i + rowLength), lastIdx > value.length && (lastIdx = -1);
        }
        var offset = value.byteOffset + i;
        if (-1 < lastIdx)
          (rowLength = new Uint8Array(value.buffer, offset, lastIdx - i)),
            processFullBinaryRow(response, _ref, rowTag, buffer, rowLength),
            (i = lastIdx),
            3 === rowState && i++,
            (rowLength = _ref = rowTag = rowState = 0),
            (buffer.length = 0);
        else {
          value = new Uint8Array(value.buffer, offset, value.byteLength - i);
          buffer.push(value);
          rowLength -= value.byteLength;
          break;
        }
      }
      response._rowState = rowState;
      response._rowID = _ref;
      response._rowTag = rowTag;
      response._rowLength = rowLength;
      return reader.read().then(progress).catch(error);
    }
  }

function handleGlobalError(responseData, errorInfo) {
  const isClosed = !responseData._closed;
  responseData._closed = isClosed;
  responseData._closedReason = errorInfo;

  for (const chunk of responseData._chunks) {
    if ("pending" === chunk.status) {
      triggerErrorOnChunk(chunk, errorInfo);
    }
  }
}

function asyncIterableSerialize(iterable, iterator) {
    var streamId = nextPartId++;
    var data = new FormData();
    pendingParts++;
    if (iterable !== iterator) {
        iterator.next().then(function(entry) {
            if (entry.done) {
                if (void 0 === entry.value)
                    data.append(formFieldPrefix + streamId, "C");
                else
                    try {
                        var partJSON = JSON.stringify(entry.value, resolveToJSON);
                        data.append(formFieldPrefix + streamId, "C" + partJSON);
                    } catch (x) {
                        reject(x);
                        return;
                    }
                pendingParts--;
                if (0 === pendingParts)
                    resolve(data);
            } else {
                try {
                    var partJSON$22 = JSON.stringify(entry.value, resolveToJSON);
                    data.append(formFieldPrefix + streamId, partJSON$22);
                    iterator.next().then(function(entry) { progress(entry); }, reject);
                } catch (x$23) {
                    reject(x$23);
                }
            }
        }, reject);
    } else {
        var entry = iterator.next();
        if (entry.done) {
            if (void 0 === entry.value)
                data.append(formFieldPrefix + streamId, "C");
            else
                try {
                    var partJSON = JSON.stringify(entry.value, resolveToJSON);
                    data.append(formFieldPrefix + streamId, "C" + partJSON);
                } catch (x) {
                    reject(x);
                    return;
                }
            pendingParts--;
            if (0 === pendingParts)
                resolve(data);
        } else {
            try {
                var partJSON$22 = JSON.stringify(entry.value, resolveToJSON);
                data.append(formFieldPrefix + streamId, partJSON$22);
                iterator.next().then(function(entry) { progress(entry); }, reject);
            } catch (x$23) {
                reject(x$23);
            }
        }
    }
    return "$" + (iterable ? "x" : "X") + streamId.toString(16);
}

function progress(entry) {
    if (entry.done) {
        if (void 0 === entry.value)
            data.append(formFieldPrefix + streamId, "C");
        else
            try {
                var partJSON = JSON.stringify(entry.value, resolveToJSON);
                data.append(formFieldPrefix + streamId, "C" + partJSON);
            } catch (x) {
                reject(x);
                return;
            }
        pendingParts--;
        if (0 === pendingParts)
            resolve(data);
    } else
        try {
            var partJSON$22 = JSON.stringify(entry.value, resolveToJSON);
            data.append(formFieldPrefix + streamId, partJSON$22);
            iterator.next().then(progress, reject);
        } catch (x$23) {
            reject(x$23);
        }
}

  function serializeBinaryReader(reader) {
    function progress(entry) {
      entry.done
        ? ((entry = nextPartId++),
          data.append(formFieldPrefix + entry, new Blob(buffer)),
          data.append(
            formFieldPrefix + streamId,
            '"$o' + entry.toString(16) + '"'
          ),
          data.append(formFieldPrefix + streamId, "C"),
          pendingParts--,
          0 === pendingParts && resolve(data))
        : (buffer.push(entry.value),
          reader.read(new Uint8Array(1024)).then(progress, reject));
    }
    null === formData && (formData = new FormData());
    var data = formData;
    pendingParts++;
    var streamId = nextPartId++,
      buffer = [];
    reader.read(new Uint8Array(1024)).then(progress, reject);
    return "$r" + streamId.toString(16);
  }

function updateRecord(item) {
  if (item.completed) {
    if (undefined === item.result)
      records.append(recordFieldPrefix + streamKey, "D");
    else
      try {
        var jsonResult = JSON.stringify(item.result, resolveToJson);
        records.append(recordFieldPrefix + streamKey, "D" + jsonResult);
      } catch (err) {
        fail(err);
        return;
      }
    pendingRecords--;
    0 === pendingRecords && succeed(records);
  } else
    try {
      var jsonResult$24 = JSON.stringify(item.result, resolveToJson);
      records.append(recordFieldPrefix + streamKey, jsonResult$24);
      iterator.next().then(updateRecord, fail);
    } catch (err$25) {
      fail(err$25);
    }
}

export default function Alert({ preview }) {
  return (
    <div
      className={cn("border-b", {
        "bg-accent-7 border-accent-7 text-white": preview,
        "bg-accent-1 border-accent-2": !preview,
      })}
    >
      <Container>
        <div className="py-2 text-center text-sm">
          {preview ? (
            <>
              This is page is a preview.{" "}
              <a
                href="/api/exit-preview"
                className="underline hover:text-cyan duration-200 transition-colors"
              >
                Click here
              </a>{" "}
              to exit preview mode.
            </>
          ) : (
            <>
              The source code for this blog is{" "}
              <a
                href={`https://github.com/vercel/next.js/tree/canary/examples/${EXAMPLE_PATH}`}
                className="underline hover:text-success duration-200 transition-colors"
              >
                available on GitHub
              </a>
              .
            </>
          )}
        </div>
      </Container>
    </div>
  );
}

function fetchAsyncModule(id) {
  const promise = globalThis.__next_require__(id);
  if ("function" !== typeof promise.then || "fulfilled" === promise.status)
    return null;
  if (promise.then) {
    promise.then(
      function (value) {
        promise.status = "fulfilled";
        promise.value = value;
      },
      function (reason) {
        promise.status = "rejected";
        promise.reason = reason;
      }
    );
  }
  return promise;
}

function initiateDataStreamFetch(responseObj, streamSource) {
  function handleProgress(_info) {
    var value = _info.value;
    if (_info.done) notifyGlobalError(responseObj, Error("Connection terminated."));
    else {
      var index = 0,
        rowDataState = responseObj._rowStatus;
      _info = responseObj._rowID;
      for (
        var rowLabel = responseObj._rowTag,
          rowLength = responseObj._rowSize,
          bufferArray = responseObj._bufferData,
          chunkSize = value.length;
        index < chunkSize;

      ) {
        var lastIdx = -1;
        switch (rowDataState) {
          case 0:
            lastIdx = value[index++];
            58 === lastIdx
              ? (rowDataState = 1)
              : (_info =
                  (_info << 4) | (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
            continue;
          case 1:
            rowDataState = value[index];
            84 === rowDataState ||
            65 === rowDataState ||
            79 === rowDataState ||
            111 === rowDataState ||
            85 === rowDataState ||
            83 === rowDataState ||
            115 === rowDataState ||
            76 === rowDataState ||
            108 === rowDataState ||
            71 === rowDataState ||
            103 === rowDataState ||
            77 === rowDataState ||
            109 === rowDataState ||
            86 === rowDataState
              ? ((rowLabel = rowDataState), (rowDataState = 2), index++)
              : (64 < rowDataState && 91 > rowDataState) ||
                  35 === rowDataState ||
                  114 === rowDataState ||
                  120 === rowDataState
                ? ((rowLabel = rowDataState), (rowDataState = 3), index++)
                : ((rowLabel = 0), (rowDataState = 3));
            continue;
          case 2:
            lastIdx = value[index++];
            44 === lastIdx
              ? (rowDataState = 4)
              : (rowLength =
                  (rowLength << 4) |
                  (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
            continue;
          case 3:
            lastIdx = value.indexOf(10, index);
            break;
          case 4:
            (lastIdx = index + rowLength), lastIdx > value.length && (lastIdx = -1);
        }
        var offset = value.byteOffset + index;
        if (-1 < lastIdx)
          (rowLength = new Uint8Array(value.buffer, offset, lastIdx - index)),
            processCompleteBinaryRow(responseObj, _info, rowLabel, bufferArray, rowLength),
            (index = lastIdx),
            3 === rowDataState && index++,
            (rowLength = _info = rowLabel = rowDataState = 0),
            (bufferArray.length = 0);
        else {
          value = new Uint8Array(value.buffer, offset, value.byteLength - index);
          bufferArray.push(value);
          rowLength -= value.byteLength;
          break;
        }
      }
      responseObj._rowStatus = rowDataState;
      responseObj._rowID = _info;
      responseObj._rowTag = rowLabel;
      responseObj._rowLength = rowLength;
      return readerSource.getReader().read().then(handleProgress).catch(errorHandler);
    }
  }
  function errorHandler(e) {
    notifyGlobalError(responseObj, e);
  }
  var readerSource = streamSource.getReader();
  readerSource.read().then(handleProgress).catch(errorHandler);
}

function mergeBuffer(buffer, lastChunk) {
  for (var l = buffer.length, byteLength = lastChunk.length, i = 0; i < l; i++)
    byteLength += buffer[i].byteLength;
  byteLength = new Uint8Array(byteLength);
  for (var i$53 = (i = 0); i$53 < l; i$53++) {
    var chunk = buffer[i$53];
    byteLength.set(chunk, i);
    i += chunk.byteLength;
  }
  byteLength.set(lastChunk, i);
  return byteLength;
}

function setupDestinationUsingModules(moduleLoader, segments, securityToken) {
  if (null !== moduleLoader)
    for (let index = 0; index < segments.length; index++) {
      const nonce = securityToken,
        contextData = ReactDOMSharedInternals.d,
        requestConfig = contextData.X,
        combinedPath = moduleLoader.prefix + segments[index];
      let headerOption = "string" === typeof moduleLoader.crossOrigin
        ? "use-credentials" === moduleLoader.crossOrigin
          ? moduleLoader.crossOrigin
          : ""
        : void 0;
      contextData.call(contextData, combinedPath, { crossOrigin: headerOption, nonce });
    }
}

