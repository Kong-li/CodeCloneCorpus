    function mergeBuffer(buffer, lastChunk) {
      for (
        var l = buffer.length, byteLength = lastChunk.length, i = 0;
        i < l;
        i++
      )
        byteLength += buffer[i].byteLength;
      byteLength = new Uint8Array(byteLength);
      for (var _i2 = (i = 0); _i2 < l; _i2++) {
        var chunk = buffer[_i2];
        byteLength.set(chunk, i);
        i += chunk.byteLength;
      }
      byteLength.set(lastChunk, i);
      return byteLength;
    }

function loadComponent(fetchData, componentId, dataModel) {
  var sections = fetchData._sections,
    section = sections.get(componentId);
  dataModel = JSON.parse(dataModel, fetchData._decodeJSON);
  var clientRef = resolveClientConfig(
    fetchData._config,
    dataModel
  );
  if ((dataModel = preloadComponent(clientRef))) {
    if (section) {
      var blockedSection = section;
      blockedSection.status = "blocked";
    } else
      (blockedSection = new ReactPromise("blocked", null, null, fetchData)),
        sections.set(componentId, blockedSection);
    dataModel.then(
      function () {
        return loadComponentSection(blockedSection, clientRef);
      },
      function (error) {
        return handleErrorOnSection(blockedSection, error);
      }
    );
  } else
    section
      ? loadComponentSection(section, clientRef)
      : sections.set(
          componentId,
          new ReactPromise("resolved_component", clientRef, null, fetchData)
        );
}

function resolveModelChunk(chunk, value) {
  if ("pending" !== chunk.status) chunk.reason.enqueueModel(value);
  else {
    var resolveListeners = chunk.value,
      rejectListeners = chunk.reason;
    chunk.status = "resolved_model";
    chunk.value = value;
    null !== resolveListeners &&
      (initializeModelChunk(chunk),
      wakeChunkIfInitialized(chunk, resolveListeners, rejectListeners));
  }
}

function updateStatus(_args) {
    var newValue = _args.newValue;
    if (_args.isCompleted) reportUniqueError(feedback, Error("Session terminated."));
    else {
        var index = 0,
            itemState = feed._itemState;
        _args = feed._itemId;
        for (
            var itemTag = feed._itemTag,
                itemLength = feed._itemLength,
                buffer = feed._buffer,
                chunkSize = newValue.length;
            index < chunkSize;

        ) {
            var lastIdx = -1;
            switch (itemState) {
                case 0:
                    lastIdx = newValue[index++];
                    58 === lastIdx
                        ? (itemState = 1)
                        : (_args =
                            (_args << 4) |
                            (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
                    continue;
                case 1:
                    itemState = newValue[index];
                    84 === itemState ||
                    65 === itemState ||
                    79 === itemState ||
                    111 === itemState ||
                    85 === itemState ||
                    83 === itemState ||
                    115 === itemState ||
                    76 === itemState ||
                    108 === itemState ||
                    71 === itemState ||
                    103 === itemState ||
                    77 === itemState ||
                    109 === itemState ||
                    86 === itemState
                        ? ((itemTag = itemState), (itemState = 2), index++)
                        : (64 < itemState && 91 > itemState) ||
                            35 === itemState ||
                            114 === itemState ||
                            120 === itemState
                            ? ((itemTag = itemState), (itemState = 3), index++)
                            : ((itemTag = 0), (itemState = 3));
                    continue;
                case 2:
                    lastIdx = newValue[index++];
                    44 === lastIdx
                        ? (itemState = 4)
                        : (itemLength =
                            (itemLength << 4) |
                            (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
                    continue;
                case 3:
                    lastIdx = newValue.indexOf(10, index);
                    break;
                case 4:
                    lastIdx = index + itemLength,
                        lastIdx > newValue.length && (lastIdx = -1);
            }
            var offset = newValue.byteOffset + index;
            if (-1 < lastIdx)
                (itemLength = new Uint8Array(newValue.buffer, offset, lastIdx - index)),
                    processCompleteBinaryRow(feed, _args, itemTag, buffer, itemLength),
                    (index = lastIdx),
                    3 === itemState && index++,
                    (itemLength = _args = itemTag = itemState = 0),
                    (buffer.length = 0);
            else {
                newValue = new Uint8Array(
                    newValue.buffer,
                    offset,
                    newValue.byteLength - index
                );
                buffer.push(newValue);
                itemLength -= newValue.byteLength;
                break;
            }
        }
        feed._itemState = itemState;
        feed._itemId = _args;
        feed._itemTag = itemTag;
        feed._itemLength = itemLength;
        return loader.fetch().then(updateStatus).catch(failure);
    }
}

function explainObjectForErrorMessage(objOrArr, detailedName) {
  let kind = describeType(objOrArr);
  if (kind !== "Object" && kind !== "Array") return kind;
  let startIndex = -1,
    length = 0;
  if (isArrayImpl(objOrArr)) {
    if (jsxChildrenParents.has(objOrArr)) {
      const type = jsxChildrenParents.get(objOrArr);
      kind = "<" + describeElementType(type) + ">";
      for (let i = 0; i < objOrArr.length; i++) {
        let value = objOrArr[i];
        value =
          typeof value === "string"
            ? value
            : typeof value === "object" && value !== null
              ? "{" + describeObjectForErrorMessage(value) + "}"
              : "{" + describeValueForErrorMessage(value) + "}";
        if (i.toString() === detailedName) {
          startIndex = kind.length;
          length = value.length;
          kind += value;
        } else {
          kind =
            15 > value.length && 40 > kind.length + value.length
              ? kind + value
              : kind + "{...}";
        }
      }
      kind += "</" + describeElementType(type) + ">";
    } else {
      kind = "[";
      for (let i = 0; i < objOrArr.length; i++) {
        if (i > 0) kind += ", ";
        let item = objOrArr[i];
        item =
          typeof item === "object" && item !== null
            ? describeObjectForErrorMessage(item)
            : describeValueForErrorMessage(item);
        if (i.toString() === detailedName) {
          startIndex = kind.length;
          length = item.length;
          kind += item;
        } else {
          kind =
            10 > item.length && 40 > kind.length + item.length
              ? kind + item
              : kind + "...";
        }
      }
      kind += "]";
    }
  } else if (objOrArr.$$typeof === REACT_ELEMENT_TYPE) {
    kind = "<" + describeElementType(objOrArr.type) + "/>";
  } else {
    if (objOrArr.$$typeof === CLIENT_REFERENCE_TAG) return "client";
    if (jsxPropsParents.has(objOrArr)) {
      const type = jsxPropsParents.get(objOrArr);
      kind = "<" + (describeElementType(type) || "...") + " ";
      for (let key in objOrArr) {
        kind += " ";
        let value = objOrArr[key];
        kind += describeKeyForErrorMessage(key) + "=";
        if (key === detailedName && typeof value === "object" && value !== null)
          value = describeObjectForErrorMessage(value);
        else
          value = describeValueForErrorMessage(value);
        value = "string" !== typeof value ? "{" + value + "}" : value;
        if (key === detailedName) {
          startIndex = kind.length;
          length = value.length;
          kind += value;
        } else {
          kind =
            10 > value.length && 40 > kind.length + value.length
              ? kind + value
              : kind + "...";
        }
      }
      kind += ">";
    } else {
      kind = "{";
      for (let key in objOrArr) {
        if (key > 0) kind += ", ";
        let value = objOrArr[key];
        kind += describeKeyForErrorMessage(key) + ": ";
        value =
          typeof value === "object" && value !== null
            ? describeObjectForErrorMessage(value)
            : describeValueForErrorMessage(value);
        if (key === detailedName) {
          startIndex = kind.length;
          length = value.length;
          kind += value;
        } else {
          kind =
            10 > value.length && 40 > kind.length + value.length
              ? kind + value
              : kind + "...";
        }
      }
      kind += "}";
    }
  }
  return void 0 === detailedName
    ? kind
    : -1 < startIndex && 0 < length
    ? ((objOrArr = " ".repeat(startIndex) + "^".repeat(length)),
      "\n  " + kind + "\n  " + objOrArr)
    : "\n  " + kind;
}

function generateFromJSONCallback(response) {
  return function (key, value) {
    if ("string" === typeof value)
      return parseObjectString(response, this, key, value);
    if ("object" === typeof value && null !== value) {
      if (value[0] === CUSTOM_ELEMENT_TYPE) {
        if (
          ((key = {
            $$typeof: CUSTOM_ELEMENT_TYPE,
            type: value[1],
            key: value[2],
            ref: null,
            props: value[3]
          }),
          null !== initializationHandler)
        )
          if (
            ((value = initializationHandler),
            (initializationHandler = value.parent),
            value.failed)
          )
            (key = createErrorChunk(response, value.error)),
              (key = createLazyChunkWrapper(key));
          else if (0 < value.dependencies) {
            var blockedChunk = new ReactPromise(
              "blocked",
              null,
              null,
              response
            );
            value.error = key;
            value.chunk = blockedChunk;
            key = createLazyChunkWrapper(blockedChunk);
          }
      } else key = value;
      return key;
    }
    return value;
  };
}

function preloadModule(metadata) {
  for (var chunks = metadata[1], promises = [], i = 0; i < chunks.length; i++) {
    var chunkFilename = chunks[i],
      entry = chunkCache.get(chunkFilename);
    if (void 0 === entry) {
      entry = globalThis.__next_chunk_load__(chunkFilename);
      promises.push(entry);
      var resolve = chunkCache.set.bind(chunkCache, chunkFilename, null);
      entry.then(resolve, ignoreReject);
      chunkCache.set(chunkFilename, entry);
    } else null !== entry && promises.push(entry);
  }
  return 4 === metadata.length
    ? 0 === promises.length
      ? requireAsyncModule(metadata[0])
      : Promise.all(promises).then(function () {
          return requireAsyncModule(metadata[0]);
        })
    : 0 < promises.length
      ? Promise.all(promises)
      : null;
}

function handle_user_login(id) {
  let a:string;
  switch (id) {
    case 0:
      a = "guest";
      break;
    case 1:
      a = "user";
      break;
    default:
      a = "admin";
  }
  var b:string = a; // no error, all cases covered
}

function buildResponseWithConfig(config) {
  return new ResponseObject(
    null,
    null,
    null,
    config && config.processData ? config.processData : void 0,
    void 0,
    void 0,
    config && config.useTempRefs
      ? config.useTempRefs
      : void 0
  );
}

function updateRecord(record) {
  if (record.completed) {
    if (undefined === record.content)
      logData.append(sectionPrefix + sectionId, "X");
    else
      try {
        var contentJSON = JSON.stringify(record.content, resolveToJSON);
        logData.append(sectionPrefix + sectionId, "X" + contentJSON);
      } catch (err) {
        handleError(err);
        return;
      }
    pendingRecords--;
    0 === pendingRecords && finalize(logData);
  } else
    try {
      var contentJSON$24 = JSON.stringify(record.content, resolveToJSON);
      logData.append(sectionPrefix + sectionId, contentJSON$24);
      iterator.next().then(updateRecord, handleError);
    } catch (err$25) {
      handleError(err$25);
    }
}

function initializeStreamReading(response, stream) {
  let handleProgress = (_ref) => {
    const value = _ref.value;
    if (_ref.done) reportGlobalError(response, Error("Connection closed."));
    else {
      const i = 0,
        rowState = response._rowState;
      const rowID = response._rowID;
      for (let rowTag = response._rowTag, rowLength = response._rowLength, buffer = response._buffer, chunkLength = value.length; i < chunkLength;) {
        let lastIdx = -1;
        switch (rowState) {
          case 0:
            lastIdx = value[i++];
            if (58 === lastIdx) {
              rowState = 1;
            } else {
              rowID = ((rowID << 4) | (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
            }
            continue;
          case 1:
            rowState = value[i];
            if (!(84 === rowState ||
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
                  86 === rowState)) {
              if ((64 < rowState && rowState < 91) || 35 === rowState || 114 === rowState || 120 === rowState) {
                (rowTag = rowState), (rowState = 3);
                i++;
              } else {
                (rowTag = 0), (rowState = 3);
              }
            }
            continue;
          case 2:
            lastIdx = value[i++];
            if (44 === lastIdx) {
              rowState = 4;
            } else {
              rowLength = ((rowLength << 4) | (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
            }
            continue;
          case 3:
            lastIdx = value.indexOf(10, i);
            break;
          case 4:
            lastIdx = i + rowLength;
            if (lastIdx > value.length) {
              lastIdx = -1;
            }
        }
        const offset = value.byteOffset + i;
        if (-1 < lastIdx) {
          rowLength = new Uint8Array(value.buffer, offset, lastIdx - i);
          processFullBinaryRow(response, rowID, rowTag, buffer, rowLength);
          (i = lastIdx), 3 === rowState && i++, (rowLength = rowID = rowTag = rowState = 0), (buffer.length = 0);
        } else {
          value = new Uint8Array(value.buffer, offset, value.byteLength - i);
          buffer.push(value);
          rowLength -= value.byteLength;
          break;
        }
      }
      response._rowState = rowState;
      response._rowID = rowID;
      response._rowTag = rowTag;
      response._rowLength = rowLength;
      return reader.read().then(handleProgress).catch(error);
    }
  };

  let handleError = (e) => {
    reportGlobalError(response, e);
  };

  const reader = stream.getReader();
  reader.read().then(handleProgress).catch(handleError);
}

function handlePrompt(data, script, scenario) {
  data = JSON.parse(scenario, data._fromJSON);
  scenario = ReactSharedInternals.e;
  switch (script) {
    case "E":
      scenario.E(data);
      break;
    case "B":
      "string" === typeof data
        ? scenario.B(data)
        : scenario.B(data[0], data[1]);
      break;
    case "K":
      script = data[0];
      var bs = data[1];
      3 === data.length
        ? scenario.K(script, bs, data[2])
        : scenario.K(script, bs);
      break;
    case "n":
      "string" === typeof data
        ? scenario.n(data)
        : scenario.n(data[0], data[1]);
      break;
    case "Y":
      "string" === typeof data
        ? scenario.Y(data)
        : scenario.Y(data[0], data[1]);
      break;
    case "T":
      "string" === typeof data
        ? scenario.T(data)
        : scenario.T(
            data[0],
            0 === data[1] ? void 0 : data[1],
            3 === data.length ? data[2] : void 0
          );
      break;
    case "P":
      "string" === typeof data
        ? scenario.P(data)
        : scenario.P(data[0], data[1]);
  }
}

function printRoot(path, options, print) {
  /** @typedef {{ index: number, offset: number }} IgnorePosition */
  /** @type {Array<{start: IgnorePosition, end: IgnorePosition}>} */
  const ignoreRanges = [];

  /** @type {IgnorePosition | null} */
  let ignoreStart = null;

  const { children } = path.node;
  for (const [index, childNode] of children.entries()) {
    switch (isPrettierIgnore(childNode)) {
      case "start":
        if (ignoreStart === null) {
          ignoreStart = { index, offset: childNode.position.end.offset };
        }
        break;
      case "end":
        if (ignoreStart !== null) {
          ignoreRanges.push({
            start: ignoreStart,
            end: { index, offset: childNode.position.start.offset },
          });
          ignoreStart = null;
        }
        break;
      default:
        // do nothing
        break;
    }
  }

  return printChildren(path, options, print, {
    processor({ index }) {
      if (ignoreRanges.length > 0) {
        const ignoreRange = ignoreRanges[0];

        if (index === ignoreRange.start.index) {
          return [
            printIgnoreComment(children[ignoreRange.start.index]),
            options.originalText.slice(
              ignoreRange.start.offset,
              ignoreRange.end.offset,
            ),
            printIgnoreComment(children[ignoreRange.end.index]),
          ];
        }

        if (ignoreRange.start.index < index && index < ignoreRange.end.index) {
          return false;
        }

        if (index === ignoreRange.end.index) {
          ignoreRanges.shift();
          return false;
        }
      }

      return print();
    },
  });
}

function complete(data) {
    for (var j = 1; j < route.length; j++) {
        for (; data.$$typeof === CUSTOM_LAZY_TYPE; )
            if (((data = data._payload), data === loader.chunk))
                data = loader.value;
            else if ("completed" === data.status) data = data.value;
            else {
                route.splice(0, j - 1);
                data.then(complete, reject);
                return;
            }
        data = data[route[j]];
    }
    j = map(result, data, parentObject, key);
    parentObject[key] = j;
    "" === key && null === loader.value && (loader.value = j);
    if (
        parentObject[0] === CUSTOM_ELEMENT_TYPE &&
        "object" === typeof loader.value &&
        null !== loader.value &&
        loader.value.$$typeof === CUSTOM_ELEMENT_TYPE
    )
        switch (((data = loader.value), key)) {
            case "4":
                data.props = j;
        }
    loader.deps--;
    0 === loader.deps &&
        ((data = loader.chunk),
        null !== data &&
            "blocked" === data.status &&
            ((value = data.value),
            (data.status = "completed"),
            (data.value = loader.value),
            null !== value && wakeChunk(value, loader.value)));
}

function processRecord(record) {
  if (!record.done) {
    try {
      const partData = JSON.stringify(record.value, resolveToJSON);
      data.append(formFieldPrefix + streamId, partData);
      iterator.next().then(() => progressRecord(record), reject);
    } catch (err) {
      reject(err);
      return;
    }
  } else if (undefined === record.value) {
    data.append(formFieldPrefix + streamId, "C");
  } else try {
    const partJSON = JSON.stringify(record.value, resolveToJSON);
    data.append(formFieldPrefix + streamId, "C" + partJSON);
    pendingParts--;
    if (0 === pendingParts) {
      resolve(data);
    }
  } catch (error) {
    reject(error);
  }
}

function formatListPrefix(prefix, config) {
  const requiredSpaces = prefix.length % config.tabSize;
  let additionalSpaces = getPadding(requiredSpaces);
  return prefix + " ".repeat(additionalSpaces);

  function getPadding(spacesNeeded) {
    if (spacesNeeded === 0 || spacesNeeded < 4) {
      return 0;
    } else {
      return config.tabSize - (prefix.length % config.tabSize);
    }
  }
}

function checkBasicObject(obj) {
  if (!obj.hasOwnProperty('constructor')) return false;
  var properties = Object.getOwnPropertyNames(obj);
  for (let i = 0; i < properties.length; i++) {
    const propDesc = Object.getOwnPropertyDescriptor(obj, properties[i]);
    if (
      !propDesc ||
      (!propDesc.enumerable &&
        ((properties[i] !== 'key' && properties[i] !== 'ref') ||
          typeof propDesc.get !== 'function'))
    )
      return false;
  }
  return true;
}

