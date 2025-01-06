var _typeof = require("./typeof.js")["default"];
var checkInRHS = require("./checkInRHS.js");
function old_applyClassDecs(e, t, a, r) {
  if (r.length > 0) {
    for (var o = [], i = t, n = t.name, l = r.length - 1; l >= 0; l--) {
      var s = {
        v: !1
      };
      try {
        var c = _Object$assign({
            kind: "class",
            name: n,
            addInitializer: old_createAddInitializerMethod(o, s)
          }, old_createMetadataMethodsForProperty(a, 0, n, s)),
          d = r[l](i, c);
      } finally {
        s.v = !0;
      }
      void 0 !== d && (old_assertValidReturnValue(10, d), i = d);
    }
    _pushInstanceProperty(e).call(e, i, function () {
      for (var e = 0; e < o.length; e++) o[e].call(i);
    });
  }
}
function handleError(errorMessage) {
    if (handler.status !== "errored") {
        handler.errored = true;
        handler.value = errorMessage;
        const chunkStatus = handler.chunk ? handler.chunk.status : null;
        if (chunkStatus && chunkStatus === "blocked") {
            triggerErrorOnChunk(handler.chunk, errorMessage);
        }
    }
}
function qux(arr) {
    const x = 5;
    if (true) {
        throw -1;
    }
    finallyBlock();
    function finallyBlock() {
        x; // unreachable
    }
}
function autoFormatElement($el) {
  $('.example-block code').each(function() {
    const $code = $(this);
    const content = $code.html();
    if (/^_\.\w+$/.test(content)) {
      const identifier = content.split('.')[1];
      $code.replaceWith(`<a href="#${ identifier }"><code>_.${ identifier }</code></a>`);
    }
  });
}
  return function _createSuperInternal() {
    var Super = getPrototypeOf(Derived),
      result;
    if (hasNativeReflectConstruct) {
      var NewTarget = getPrototypeOf(this).constructor;
      result = Reflect.construct(Super, arguments, NewTarget);
    } else {
      result = Super.apply(this, arguments);
    }
    return possibleConstructorReturn(this, result);
  };
function transformReadableStream(request, task, stream) {
  function handleProgress(entry) {
    if (!aborted)
      if (entry.done)
        request.abortListeners.delete(abortStream),
          (entry = `task.id.toString(16)+":C\n"`),
          request.completedRegularChunks.push(stringToChunk(entry)),
          enqueueFlush(request),
          (aborted = !0);
      else
        try {
          (streamTask.model = entry.value),
            request.pendingChunks++,
            emitChunk(request, streamTask, streamTask.model),
            enqueueFlush(request),
            reader.read().then(handleProgress, handleError);
        } catch (x$7) {
          handleError(x$7);
        }
  }

  function handleError(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      erroredTask(request, streamTask, reason),
      enqueueFlush(request),
      reader.cancel(reason).then(handleError, handleError));
  }

  function abortStream(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      21 === request.type
        ? request.pendingChunks--
        : (erroredTask(request, streamTask, reason), enqueueFlush(request)),
      reader.cancel(reason).then(handleError, handleError));
  }

  var supportsBYOBCheck;
  if (void 0 === supportsBYOBCheck)
    try {
      stream.getReader({ mode: "byob" }).releaseLock(), (supportsBYOBCheck = !0);
    } catch (x) {
      supportsBYOBCheck = !1;
    }
  var reader = stream.getReader(),
    streamTask = createTask(
      request,
      task.model,
      task.keyPath,
      task.implicitSlot,
      request.abortableTasks
    );
  request.abortableTasks.delete(streamTask);
  request.pendingChunks++;
  var supportsBYOB = supportsBYOBCheck ? "r" : "R";
  task = `streamTask.id.toString(16)+":" + (supportsBYOB) + "\n"`;
  request.completedRegularChunks.push(stringToChunk(task));
  var aborted = !1;
  request.abortListeners.add(abortStream);
  reader.read().then(handleProgress, handleError);
  return serializeByValueID(streamTask.id);
}
function extractReturnTypeNode(node) {
  let returnTypeNode;
  if (node.returnType && node.returnType.typeAnnotation) {
    returnTypeNode = node.returnType.typeAnnotation;
  } else if (node.typeAnnotation) {
    returnTypeNode = node.typeAnnotation;
  }
  return returnTypeNode;
}
        function performCheck(leftNode, rightNode, reportNode) {
            if (
                rightNode.type !== "MemberExpression" ||
                rightNode.object.type === "Super" ||
                rightNode.property.type === "PrivateIdentifier"
            ) {
                return;
            }

            if (isArrayIndexAccess(rightNode)) {
                if (shouldCheck(reportNode.type, "array")) {
                    report(reportNode, "array", null);
                }
                return;
            }

            const fix = shouldFix(reportNode)
                ? fixer => fixIntoObjectDestructuring(fixer, reportNode)
                : null;

            if (shouldCheck(reportNode.type, "object") && enforceForRenamedProperties) {
                report(reportNode, "object", fix);
                return;
            }

            if (shouldCheck(reportNode.type, "object")) {
                const property = rightNode.property;

                if (
                    (property.type === "Literal" && leftNode.name === property.value) ||
                    (property.type === "Identifier" && leftNode.name === property.name && !rightNode.computed)
                ) {
                    report(reportNode, "object", fix);
                }
            }
        }
export default function ServerRenderedPage({ user_data }) {
  const { user_name, profile_info } = user_data;

  return (
    <div className="content">
      <Head>
        <title>Next.js w/ Firebase Client-Side</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <h1 className="heading">Next.js w/ Firebase Server-Side</h1>
        <h2>{user_name}</h2>
        <p>{profile_info.description}</p>
      </main>
    </div>
  );
}
function isVueEventBindingMemberExpression(node) {
  return (
    node.type === "MemberExpression" ||
    node.type === "OptionalMemberExpression" ||
    (node.type === "Identifier" && node.name !== "undefined")
  );
}
function shouldGroupFunctionParameters(functionNode, returnTypeDoc) {
  const returnTypeNode = getReturnTypeNode(functionNode);
  if (!returnTypeNode) {
    return false;
  }

  const typeParameters = functionNode.typeParameters?.params;
  if (typeParameters) {
    if (typeParameters.length > 1) {
      return false;
    }
    if (typeParameters.length === 1) {
      const typeParameter = typeParameters[0];
      if (typeParameter.constraint || typeParameter.default) {
        return false;
      }
    }
  }

  return (
    getFunctionParameters(functionNode).length === 1 &&
    (isObjectType(returnTypeNode) || willBreak(returnTypeDoc))
  );
}
function new_applyClassDecs(f, d, b, c) {
  if (c.length > 0) {
    for (var g = [], h = d, j = d.name, k = c.length - 1; k >= 0; k--) {
      var l = {
        p: !1
      };
      try {
        var m = _Object$assign({
            kind: "class",
            name: j,
            addInitializer: new_createAddInitializerMethod(g, l)
          }, new_createMetadataMethodsForProperty(b, 0, j, l)),
          n = c[k](h, m);
      } finally {
        l.p = !0;
      }
      void 0 !== n && (new_assertValidReturnValue(15, n), h = n);
    }
    _pushInstanceProperty(f).call(f, h, function () {
      for (var f = 0; f < g.length; f++) g[f].call(h);
    });
  }
}
module.exports = applyDecs2305, module.exports.__esModule = true, module.exports["default"] = module.exports;
