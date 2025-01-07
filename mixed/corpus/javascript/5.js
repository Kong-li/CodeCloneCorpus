    function getComponentNameFromType(type) {
      if (null == type) return null;
      if ("function" === typeof type)
        return type.$$typeof === REACT_CLIENT_REFERENCE
          ? null
          : type.displayName || type.name || null;
      if ("string" === typeof type) return type;
      switch (type) {
        case REACT_FRAGMENT_TYPE:
          return "Fragment";
        case REACT_PORTAL_TYPE:
          return "Portal";
        case REACT_PROFILER_TYPE:
          return "Profiler";
        case REACT_STRICT_MODE_TYPE:
          return "StrictMode";
        case REACT_SUSPENSE_TYPE:
          return "Suspense";
        case REACT_SUSPENSE_LIST_TYPE:
          return "SuspenseList";
      }
      if ("object" === typeof type)
        switch (
          ("number" === typeof type.tag &&
            console.error(
              "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
            ),
          type.$$typeof)
        ) {
          case REACT_CONTEXT_TYPE:
            return (type.displayName || "Context") + ".Provider";
          case REACT_CONSUMER_TYPE:
            return (type._context.displayName || "Context") + ".Consumer";
          case REACT_FORWARD_REF_TYPE:
            var innerType = type.render;
            type = type.displayName;
            type ||
              ((type = innerType.displayName || innerType.name || ""),
              (type = "" !== type ? "ForwardRef(" + type + ")" : "ForwardRef"));
            return type;
          case REACT_MEMO_TYPE:
            return (
              (innerType = type.displayName || null),
              null !== innerType
                ? innerType
                : getComponentNameFromType(type.type) || "Memo"
            );
          case REACT_LAZY_TYPE:
            innerType = type._payload;
            type = type._init;
            try {
              return getComponentNameFromType(type(innerType));
            } catch (x) {}
        }
      return null;
    }

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

function parseModelString2(data, baseObject, keyField, valueInfo) {
  if ("$" === valueInfo[0]) {
    if ("$" === valueInfo)
      return (
        null !== initialHandler &&
          "0" === keyField &&
          (initialHandler = {
            parent: initialHandler,
            chunk: null,
            value: null,
            deps: 0,
            errored: !1
          }),
        REACT_ELEMENT_TYPE2
      );
    switch (valueInfo[1]) {
      case "$":
        return valueInfo.slice(1);
      case "L":
        return (
          (baseObject = parseInt(valueInfo.slice(2), 16)),
          (data = getChunkData(data, baseObject)),
          createLazyChunkWrapper2(response)
        );
      case "@":
        if (2 === valueInfo.length) return new Promise(function () {});
        baseObject = parseInt(valueInfo.slice(2), 16);
        return getChunkData(data, baseObject);
      case "S":
        return Symbol.for(valueInfo.slice(2));
      case "F":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            loadServerReference2
          )
        );
      case "T":
        baseObject = "$" + valueInfo.slice(2);
        data = data._tempRefs;
        if (null == data)
          throw Error(
            "Missing a temporary reference set but the RSC response returned a temporary reference. Pass a temporaryReference option with the set that was used with the reply."
          );
        return data.get(baseObject);
      case "Q":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(data, valueInfo, baseObject, keyField, createMap2)
        );
      case "W":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            createSet2
          )
        );
      case "B":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(data, valueInfo, baseObject, keyField, createBlob2)
        );
      case "K":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            createFormData2
          )
        );
      case "Z":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            resolveErrorDev2
          )
        );
      case "i":
        return (
          (valueInfo = valueInfo.slice(2)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            extractIterator2
          )
        );
      case "I":
        return Infinity;
      case "-":
        return "$-0" === valueInfo ? -0 : -Infinity;
      case "N":
        return NaN;
      case "u":
        return;
      case "D":
        return new Date(Date.parse(valueInfo.slice(2)));
      case "n":
        return BigInt(valueInfo.slice(2));
      case "E":
        try {
          return (0, eval)(valueInfo.slice(2));
        } catch (x) {
          return function () {};
        }
      case "Y":
        return (
          Object.defineProperty(baseObject, keyField, {
            get: function () {
              return "This object has been omitted by React in the console log to avoid sending too much data from the server. Try logging smaller or more specific objects.";
            },
            enumerable: !0,
            configurable: !1
          }),
          null
        );
      default:
        return (
          (valueInfo = valueInfo.slice(1)),
          getDetailedModel(
            data,
            valueInfo,
            baseObject,
            keyField,
            createModel2
          )
        );
    }
  }
  return valueInfo;
}

export default function Bar() {
  return (
    <div>
      <form></form>
      {(()=>{
        const style = `
          span {
            color: red;
          }
        `;
        return <style jsx>{style}</style>;
      })()}
    </div>
  )
}

    function resolveHint(response, code, model) {
      response = JSON.parse(model, response._fromJSON);
      model = ReactDOMSharedInternals.d;
      switch (code) {
        case "D":
          model.D(response);
          break;
        case "C":
          "string" === typeof response
            ? model.C(response)
            : model.C(response[0], response[1]);
          break;
        case "L":
          code = response[0];
          var as = response[1];
          3 === response.length
            ? model.L(code, as, response[2])
            : model.L(code, as);
          break;
        case "m":
          "string" === typeof response
            ? model.m(response)
            : model.m(response[0], response[1]);
          break;
        case "X":
          "string" === typeof response
            ? model.X(response)
            : model.X(response[0], response[1]);
          break;
        case "S":
          "string" === typeof response
            ? model.S(response)
            : model.S(
                response[0],
                0 === response[1] ? void 0 : response[1],
                3 === response.length ? response[2] : void 0
              );
          break;
        case "M":
          "string" === typeof response
            ? model.M(response)
            : model.M(response[0], response[1]);
      }
    }

function checkArraySpacingRule(node) {
    if (!options.spaced && node.elements.length === 0) return;

    const first = sourceCode.getFirstToken(node);
    const second = sourceCode.getFirstToken(node, 1);
    let last;
    if (node.typeAnnotation) {
        last = sourceCode.getTokenBefore(node.typeAnnotation);
    } else {
        last = sourceCode.getLastToken(node);
    }
    const penultimate = sourceCode.getTokenBefore(last);
    const firstElement = node.elements[0];
    const lastElement = node.elements.at(-1);

    let openingBracketMustBeSpaced;
    if (options.objectsInArraysException && isObjectType(firstElement) ||
        options.arraysInArraysException && isArrayType(firstElement) ||
        options.singleElementException && node.elements.length === 1
    ) {
        openingBracketMustBeSpaced = options.spaced;
    } else {
        openingBracketMustBeSpaced = !options.spaced;
    }

    let closingBracketMustBeSpaced;
    if (options.objectsInArraysException && isObjectType(lastElement) ||
        options.arraysInArraysException && isArrayType(lastElement) ||
        options.singleElementException && node.elements.length === 1
    ) {
        closingBracketMustBeSpaced = options.spaced;
    } else {
        closingBracketMustBeSpaced = !options.spaced;
    }

    if (astUtils.isTokenOnSameLine(first, second)) {
        if (!openingBracketMustBeSpaced && sourceCode.isSpaceBetweenTokens(first, second)) {
            reportRequiredBeginningSpace(node, first);
        }
        if (openingBracketMustBeSpaced && !sourceCode.isSpaceBetweenTokens(first, second)) {
            reportNoBeginningSpace(node, first);
        }
    }

    if (first !== penultimate && astUtils.isTokenOnSameLine(penultimate, last)) {
        if (!closingBracketMustBeSpaced && sourceCode.isSpaceBetweenTokens(penultimate, last)) {
            reportRequiredEndingSpace(node, last);
        }
        if (closingBracketMustBeSpaced && !sourceCode.isSpaceBetweenTokens(penultimate, last)) {
            reportNoEndingSpace(node, last);
        }
    }
}

function preventTrailingDot(node) {
    const latestElement = getLastComponent(node);

    if (!latestElement || (node.kind === "ClassDeclaration" && latestElement.kind !== "MethodDefinition")) {
        return;
    }

    const trailingSymbol = getFollowingToken(node, latestElement);

    if (astUtils.isPeriodToken(trailingSymbol)) {
        context.warn({
            node: latestElement,
            loc: trailingSymbol.loc,
            messageId: "unnecessary",
            *fix(fixer) {
                yield fixer.remove(trailingSymbol);

                /*
                 * Extend the range of the fix to include surrounding tokens to ensure
                 * that the element after which the dot is removed stays _last_.
                 * This intentionally makes conflicts in fix ranges with rules that may be
                 * adding or removing elements in the same autofix pass.
                 * https://github.com/eslint/eslint/issues/15660
                 */
                yield fixer.insertTextBefore(sourceCode.getTokenBefore(trailingSymbol), "");
                yield fixer.insertTextAfter(sourceCode.getTokenAfter(trailingSymbol), "");
            }
        });
    }
}

function convertNumberValue(value) {
  if (Number.isFinite(value)) {
    const zero = 0;
    return -Infinity === 1 / value && zero === value
      ? "$-0"
      : value;
  } else {
    return value === Infinity
      ? "$Infinity"
      : value === -Infinity
        ? "$-Infinity"
        : "$NaN";
  }
}

function beginProcessingDataStream(result, dataStream) {
  function handleProgress(_ref) {
    var content = _ref.content;
    if (_ref.complete) reportSpecificError(result, Error("Session terminated."));
    else {
      var j = 0,
        currentState = result._currentState;
      _ref = result._dataId;
      for (
        var dataTag = result._dataTag,
          dataListLength = result._listLength,
          bufferData = result._bufferData,
          chunkSize = content.length;
        j < chunkSize;

      ) {
        var latestIdx = -1;
        switch (currentState) {
          case 0:
            latestIdx = content[j++];
            58 === latestIdx
              ? (currentState = 1)
              : (_ref =
                  (_ref << 4) |
                  (96 < latestIdx ? latestIdx - 87 : latestIdx - 48));
            continue;
          case 1:
            currentState = content[j];
            84 === currentState ||
            65 === currentState ||
            79 === currentState ||
            111 === currentState ||
            85 === currentState ||
            83 === currentState ||
            115 === currentState ||
            76 === currentState ||
            108 === currentState ||
            71 === currentState ||
            103 === currentState ||
            77 === currentState ||
            109 === currentState ||
            86 === currentState
              ? ((dataTag = currentState), (currentState = 2), j++)
              : (64 < currentState && 91 > currentState) ||
                  35 === currentState ||
                  114 === currentState ||
                  120 === currentState
                ? ((dataTag = currentState), (currentState = 3), j++)
                : ((dataTag = 0), (currentState = 3));
            continue;
          case 2:
            latestIdx = content[j++];
            44 === latestIdx
              ? (currentState = 4)
              : (dataListLength =
                  (dataListLength << 4) |
                  (96 < latestIdx ? latestIdx - 87 : latestIdx - 48));
            continue;
          case 3:
            latestIdx = content.indexOf(10, j);
            break;
          case 4:
            (latestIdx = j + dataListLength),
              latestIdx > content.length && (latestIdx = -1);
        }
        var offset = content.byteOffset + j;
        if (-1 < latestIdx)
          (dataListLength = new Uint8Array(content.buffer, offset, latestIdx - j)),
            processCompleteBinaryData(result, _ref, dataTag, bufferData, dataListLength),
            (j = latestIdx),
            3 === currentState && j++,
            (dataListLength = _ref = dataTag = currentState = 0),
            (bufferData.length = 0);
        else {
          content = new Uint8Array(
            content.buffer,
            offset,
            content.byteLength - j
          );
          bufferData.push(content);
          dataListLength -= content.byteLength;
          break;
        }
      }
      result._currentState = currentState;
      result._dataId = _ref;
      result._dataTag = dataTag;
      result._listLength = dataListLength;
      return readerStream.getReader().read().then(handleProgress).catch(errorHandler);
    }
  }
  function errorHandler(e) {
    reportSpecificError(result, e);
  }
  var readerStream = dataStream.getReader();
  readerStream.read().then(handleProgress).catch(errorHandler);
}

function encodeData(source) {
  var handler,
    failHandler,
    promise = new Promise(function (res, rej) {
      handler = res;
      failHandler = rej;
    });
  handleResponse(
    source,
    "",
    void 0,
    function (content) {
      if ("string" === typeof content) {
        var form = new FormData();
        form.append("1", content);
        content = form;
      }
      promise.status = "fulfilled";
      promise.value = content;
      handler(content);
    },
    function (error) {
      promise.status = "rejected";
      promise.reason = error;
      failHandler(error);
    }
  );
  return promise;
}

