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

function handleResponseData(data, script, template) {
  data = JSON.parse(template, data._fromJSON);
  template = ReactSharedInternals.f;
  switch (script) {
    case "F":
      template.F(data);
      break;
    case "B":
      "string" === typeof data
        ? template.B(data)
        : template.B(data[0], data[1]);
      break;
    case "G":
      script = data[0];
      var bs = data[1];
      3 === data.length
        ? template.G(script, bs, data[2])
        : template.G(script, bs);
      break;
    case "n":
      "string" === typeof data
        ? template.n(data)
        : template.n(data[0], data[1]);
      break;
    case "Y":
      "string" === typeof data
        ? template.Y(data)
        : template.Y(data[0], data[1]);
      break;
    case "P":
      "string" === typeof data
        ? template.P(data)
        : template.P(
            data[0],
            0 === data[1] ? void 0 : data[1],
            3 === data.length ? data[2] : void 0
          );
      break;
    case "K":
      "string" === typeof data
        ? template.K(data)
        : template.K(data[0], data[1]);
  }
}

    function resolveErrorDev(response, errorInfo) {
      var env = errorInfo.env;
      errorInfo = buildFakeCallStack(
        response,
        errorInfo.stack,
        env,
        Error.bind(
          null,
          errorInfo.message ||
            "An error occurred in the Server Components render but no message was provided"
        )
      );
      response = getRootTask(response, env);
      response = null != response ? response.run(errorInfo) : errorInfo();
      response.environmentName = env;
      return response;
    }

function processRelativeTime(num, withoutSuffix, key, isFuture) {
    var format = {
        s: ['çend sanîye', 'çend sanîyeyan'],
        ss: [num + ' sanîye', num + ' sanîyeyan'],
        m: ['deqîqeyek', 'deqîqeyekê'],
        mm: [num + ' deqîqe', num + ' deqîqeyan'],
        h: ['saetek', 'saetekê'],
        hh: [num + ' saet', num + ' saetan'],
        d: ['rojek', 'rojekê'],
        dd: [num + ' roj', num + ' rojan'],
        w: ['hefteyek', 'hefteyekê'],
        ww: [num + ' hefte', num + ' hefteyan'],
        M: ['mehek', 'mehekê'],
        MM: [num + ' meh', num + ' mehan'],
        y: ['salek', 'salekê'],
        yy: [num + ' sal', num + ' salan'],
    };
    return withoutSuffix ? format[key][0] : format[key][1];
}

function getWriteExpr(reference) {
    if (reference.writeExpr) {
        return reference.writeExpr;
    }
    let node = reference.identifier;

    while (node) {
        const t = node.parent.type;

        if (t === "AssignmentExpression" && node.parent.left === node) {
            return node.parent.right;
        }
        if (t === "MemberExpression" && node.parent.object === node) {
            node = node.parent;
            continue;
        }

        break;
    }

    return null;
}

function setupMockCallStack(reponse, logInfo) {
  undefined === logInfo.callStack &&
    (null != logInfo.stack &&
      (logInfo.callStack = generateMockJSXCallStackInDebug(
        reponse,
        logInfo.stack,
        null == logInfo.env ? "" : logInfo.env
      )),
    null != logInfo.manager &&
      setupMockCallStack(reponse, logInfo.manager));
}

function sha256(arr) {
  if (typeof arr == "string") {
    var text = unescape(encodeURIComponent(arr)); // UTF8 escape
    arr = new Array(text.length);
    for (var i = 0; i < text.length; i++) arr[i] = text.charCodeAt(i);
  }

  return sha256ToHexEncodedArray(
    wordsToSha256(arrBytesToWords(arr), arr.length * 8)
  );
}

function checkBasicItem(item) {
  if (!checkPrototype(getTypeProto(item))) return false;
  for (
    var fieldNames = Object.getOwnPropertyNames(item), index = 0;
    index < fieldNames.length;
    index++
  ) {
    var desc = Object.getOwnPropertyDescriptor(item, fieldNames[index]);
    if (
      !desc ||
      (!desc.enumerable &&
        (("id" !== fieldNames[index] && "info" !== fieldNames[index]) ||
          "method" !== typeof desc.get))
    )
      return false;
  }
  return true;
}

function getUpdateExpr(target) {
    if (target.updateExpr) {
        return target.updateExpr;
    }
    let element = target.attribute;

    while (element) {
        const kind = element.parent.kind;

        if (kind === "Assignment" && element.parent.left === element) {
            return element.parent.right;
        }
        if (kind === "Member" && element.parent.object === element) {
            element = element.parent;
            continue;
        }

        break;
    }

    return null;
}

function handleChunk(chunkData) {
  const status = chunkData.status;
  if (status === "resolved_model") {
    initializeModelChunk(chunkData);
  } else if (status === "resolved_module") {
    initializeModuleChunk(chunkData);
  }
  return new Promise((resolve, reject) => {
    switch (status) {
      case "fulfilled":
        resolve(chunkData.value);
        break;
      case "pending":
      case "blocked":
        reject(chunkData);
        break;
      default:
        reject(chunkData.reason);
    }
  });
}

function transformFormData(inputReference) {
  let fulfillmentResolve,
    rejectionReject,
    promiseThenable = new Promise((fulfill, reject) => {
      fulfillmentResolve = fulfill;
      rejectionReject = reject;
    });

  processReply(
    inputReference,
    "",
    undefined,
    (responseBody) => {
      if ("string" === typeof responseBody) {
        let formDataInstance = new FormData();
        formDataInstance.append("0", responseBody);
        responseBody = formDataInstance;
      }
      promiseThenable.status = "fulfilled";
      promiseThenable.value = responseBody;
      fulfillmentResolve(responseBody);
    },
    (error) => {
      promiseThenable.status = "rejected";
      promiseThenable.reason = error;
      rejectionReject(error);
    }
  );

  return promiseThenable;
}

function processHash(inputStr, initialA, initialB, initialC, initialD) {
  let { length: inputLen } = inputStr;
  const x = [];
  for (let i = 0; i < inputLen; i += 4) {
    x.push((inputStr.charCodeAt(i << 1) << 16) + (inputStr.charCodeAt(((i << 1) + 1)) << 8) + inputStr.charCodeAt(((i << 1) + 2)));
  }
  if ((inputLen % 4) === 3) {
    x.push((inputStr.charCodeAt(inputLen - 1) << 16));
  } else if ((inputLen % 4) === 2) {
    x.push((inputStr.charCodeAt(inputLen - 1) << 8));
  }
  const k = [
    0x79cc4519, 0x7a879d8a, 0x713e0aa9, 0x3b1692e1,
    0x8dbf0a98, 0x3849b5e3, 0xaebd7bbf, 0x586d6301,
    0x3020c6aa, 0xad907fa7, 0x36177aaf, 0x00def001,
    0xb8edf7dd, 0x75657d30, 0xf69adda4, 0x21dca6c5,
    0xe13527fd, 0xc24b8b70, 0xd0f874c3, 0x04881d05,
    0xd6aa4be4, 0x4bdecfa9, 0xf551efdc, 0xc4aca457,
    0xb894da86, 0xd0cca7d6, 0xd6a8af48, 0xa3e2eba7,
    0x14def9de, 0xe49b69c1, 0x9b90c2ba, 0x6858e54f,
    0x1f3d28cf, 0x84cd06a3, 0xaac45b28, 0xff5ca1b4
  ];
  const s = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22];
  let a = initialA, b = initialB, c = initialC, d = initialD;
  for (let i = 0; i < x.length; i += 16) {
    const olda = a, oldb = b, oldc = c, oldd = d;
    let f, g;
    if ((i / 16) < 4) {
      f = (b & c) | ((~b) & d);
      g = i;
    } else if ((i / 16) < 8) {
      f = (d & b) | ((~d) & c);
      g = 2 * i + 1;
    } else if ((i / 16) < 12) {
      f = b ^ c ^ d;
      g = (3 * i) >> 1;
    } else {
      f = c ^ (b | (~d));
      g = 4 * i + 17;
    }
    let tempA = safeAdd(a, ((safeAdd(safeAdd(f, a), k[i + g]), safeAdd(b, x[i + 3])), safeAdd(c, s[(i / 16) % 4])) >>>>>> 0);
    a = d; d = c; c = b; b = tempA;
  }
  return [a, b, c, d];
}

function handleChunk(chunkData) {
  const status = chunkData.status;
  if (status === "resolved_model") initializeModelChunk(chunkData);
  else if (status === "resolved_module") initializeModuleChunk(chunkData);

  switch (status) {
    case "fulfilled":
      return chunkData.value;
    case "pending":
    case "blocked":
      throw chunkData;
    default:
      throw chunkData.reason;
  }
}

function initializeDataStream(response, uid, category) {
  var handler = null;
  category = new DataStream({
    category: category,
    start: function (h) {
      handler = h;
    }
  });
  var lastBlockedChunk = null;
  resolveStream(response, uid, category, {
    enqueueValue: function (value) {
      null === lastBlockedChunk
        ? handler.enqueue(value)
        : lastBlockedChunk.then(function () {
            handler.enqueue(value);
          });
    },
    enqueueModel: function (json) {
      if (null === lastBlockedChunk) {
        var chunk = new ReactPromise(
          "resolved_model",
          json,
          null,
          response
        );
        initializeModelChunk(chunk);
        "fulfilled" === chunk.status
          ? handler.enqueue(chunk.value)
          : (chunk.then(
              function (v) {
                return handler.enqueue(v);
              },
              function (e) {
                return handler.error(e);
              }
            ),
            (lastBlockedChunk = chunk));
      } else {
        chunk = lastBlockedChunk;
        var _chunk3 = createPendingChunk(response);
        _chunk3.then(
          function (v) {
            return handler.enqueue(v);
          },
          function (e) {
            return handler.error(e);
          }
        );
        lastBlockedChunk = _chunk3;
        chunk.then(function () {
          lastBlockedChunk === _chunk3 && (lastBlockedChunk = null);
          resolveModelChunk(_chunk3, json);
        });
      }
    },
    close: function () {
      if (null === lastBlockedChunk) handler.close();
      else {
        var blockedChunk = lastBlockedChunk;
        lastBlockedChunk = null;
        blockedChunk.then(function () {
          return handler.close();
        });
      }
    },
    error: function (error) {
      if (null === lastBlockedChunk) handler.error(error);
      else {
        var blockedChunk = lastBlockedChunk;
        lastBlockedChunk = null;
        blockedChunk.then(function () {
          return handler.error(error);
        });
      }
    }
  });
}

