var _typeof = require("./typeof.js")["default"];
function setupModelComponent(segment) {
  let oldResolver = processingHandler;
  processingHandler = null;
  const actualModel = segment.value;
  segment.status = "awaiting";
  segment.value = null;
  segment.reason = null;
  try {
    const parsedData = JSON.parse(actualModel, segment._response._deserialize),
      notifyListeners = segment.value;
    if (null !== notifyListeners) {
      segment.value = null;
      segment.reason = null;
      handleActivation(notifyListeners, parsedData);
    }
    if (null !== processingHandler) {
      if (!processingHandler.errored) throw processingHandler.value;
      if (0 < processingHandler.reqs) {
        processingHandler.value = parsedData;
        processingHandler.component = segment;
        return;
      }
    }
    segment.status = "resolved";
    segment.value = parsedData;
  } catch (err) {
    segment.status = "rejected";
    segment.reason = err;
  } finally {
    processingHandler = oldResolver;
  }
}

function handleActivation(listeners, data) {
  listeners.forEach(listener => listener(data));
}
export default function _generateCustomWidget(widgetType, widgetProps, uniqueKey, childElements) {
  const CUSTOM_WIDGET_TYPE || (CUSTOM_WIDGET_TYPE = "function" == typeof Symbol && Symbol["for"] && Symbol["for"]("custom.widget") || 60104);
  var defaultProps = widgetType && widgetType.defaultProps,
    childrenCount = arguments.length - 3;
  if (widgetProps || 0 === childrenCount || (widgetProps = {
    children: void 0
  }), 1 === childrenCount) widgetProps.children = childElements;else if (childrenCount > 1) {
    for (var elementArray = new Array(childrenCount), i = 0; i < childrenCount; i++) elementArray[i] = arguments[i + 3];
    widgetProps.children = elementArray;
  }
  if (widgetProps && defaultProps) for (var propName in defaultProps) void 0 === widgetProps[propName] && (widgetProps[propName] = defaultProps[propName]);else widgetProps || (widgetProps = defaultProps || {});
  return {
    $$typeof: CUSTOM_WIDGET_TYPE,
    type: widgetType,
    key: void 0 === uniqueKey ? null : "" + uniqueKey,
    ref: null,
    props: widgetProps,
    _owner: null
  };
}
function objectName(object) {
  return Object.prototype.toString
    .call(object)
    .replace(/^\[object (.*)\]$/, function (m, p0) {
      return p0;
    });
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
async function collectExamplesResult(manifestFile) {
  const file = path.join(process.cwd(), manifestFile)
  const contents = await fs.readFile(file, 'utf-8')
  const results = JSON.parse(contents)

  let failingCount = 0
  let passingCount = 0

  const currentDate = new Date()
  const isoString = currentDate.toISOString()
  const timestamp = isoString.slice(0, 19).replace('T', ' ')

  for (const isPassing of Object.values(results)) {
    if (isPassing) {
      passingCount += 1
    } else {
      failingCount += 1
    }
  }
  const status = `${process.env.GITHUB_SHA}\t${timestamp}\t${passingCount}/${
    passingCount + failingCount
  }`

  return {
    status,
    // Uses JSON.stringify to create minified JSON, otherwise whitespace is preserved.
    data: JSON.stringify(results),
  }
}
export default function _wrapCustomSuper(Class) {
  var _cache = typeof Map === "function" ? new Map() : undefined;
  _wrapCustomSuper = function _wrapCustomSuper(Class) {
    if (Class === null || !isNativeFunction(Class)) return Class;
    if (typeof Class !== "function") {
      throw new TypeError("Super expression must either be null or a function");
    }
    if (typeof _cache !== "undefined") {
      if (_cache.has(Class)) return _cache.get(Class);
      _cache.set(Class, Wrapper);
    }
    function Wrapper() {
      return construct(Class, arguments, getPrototypeOf(this).constructor);
    }
    Wrapper.prototype = Object.create(Class.prototype, {
      constructor: {
        value: Wrapper,
        enumerable: false,
        writable: true,
        configurable: true
      }
    });
    return setPrototypeOf(Wrapper, Class);
  };
  return _wrapCustomSuper(Class);
}
function customFormEncoder(prefix) {
  var ref = existingServerReferences.get(this);
  if (!ref)
    throw Error(
      "Attempted to encode a Server Action from a different context than the encoder is from. This indicates a React issue."
    );
  var info = null;
  if (null !== ref.linked) {
    info = cachedData.get(ref);
    info ||
      ((info = formEncoding(ref)), cachedData.set(ref, info));
    if ("rejected" === info.status) throw info.reason;
    if ("fulfilled" !== info.status) throw info;
    ref = info.value;
    var prefixedInfo = new FormData();
    ref.forEach(function (value, key) {
      prefixedInfo.append("$OPERATION_" + prefix + ":" + key, value);
    });
    info = prefixedInfo;
    ref = "$OPERATION_REF_" + prefix;
  } else ref = "$OPERATION_ID_" + ref.id;
  return {
    identifier: ref,
    action: "POST",
    encodingType: "multipart/form-data",
    payload: info
  };
}
function parseBoundActionMetadata(data, manifest, prefix) {
  data = getInitialResponse(manifest, prefix, null, data);
  close(data);
  const chunkedData = getChunk(data, 0);
  if (!chunkedData.then) throw new Error("Invalid response");
  try {
    chunkedData.then(() => {});
    if (chunkedData.status !== "fulfilled") {
      throw chunkedData.reason;
    }
  } catch (error) {
    console.error(error);
  }
  return chunkedData.value;
}
export function deprecateDeprecated(feature, message) {
    if (hooks.deprecationCallback != null) {
        hooks.deprecationCallback(feature, message);
    }
    if (!warnings[feature]) {
        console.warn(message);
        warnings[feature] = true;
    }
}
function sendToClient(client, message, transferrables = []) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel()

    channel.port1.onmessage = (event) => {
      if (event.data && event.data.error) {
        return reject(event.data.error)
      }

      resolve(event.data)
    }

    client.postMessage(
      message,
      [channel.port2].concat(transferrables.filter(Boolean)),
    )
  })
}
function checkSCSSDirectiveType(node, opts) {
  const parser = opts.parser;
  const directiveTypes = ["if", "else", "for", "each", "while"];
  const isSCSSParser = parser === "scss";
  const isCSSAtRule = node.type === "css-atrule";
  const nameIsDirective = directiveTypes.includes(node.name);

  return isSCSSParser && isCSSAtRule && nameIsDirective;
}
module.exports = applyDecs, module.exports.__esModule = true, module.exports["default"] = module.exports;
