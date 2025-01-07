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

function printTernaryOld(path, options, print) {
  const { node } = path;
  const isConditionalExpression = node.type === "ConditionalExpression";
  const consequentNodePropertyName = isConditionalExpression
    ? "consequent"
    : "trueType";
  const alternateNodePropertyName = isConditionalExpression
    ? "alternate"
    : "falseType";
  const testNodePropertyNames = isConditionalExpression
    ? ["test"]
    : ["checkType", "extendsType"];
  const consequentNode = node[consequentNodePropertyName];
  const alternateNode = node[alternateNodePropertyName];
  const parts = [];

  // We print a ConditionalExpression in either "JSX mode" or "normal mode".
  // See `tests/format/jsx/conditional-expression.js` for more info.
  let jsxMode = false;
  const { parent } = path;
  const isParentTest =
    parent.type === node.type &&
    testNodePropertyNames.some((prop) => parent[prop] === node);
  let forceNoIndent = parent.type === node.type && !isParentTest;

  // Find the outermost non-ConditionalExpression parent, and the outermost
  // ConditionalExpression parent. We'll use these to determine if we should
  // print in JSX mode.
  let currentParent;
  let previousParent;
  let i = 0;
  do {
    previousParent = currentParent || node;
    currentParent = path.getParentNode(i);
    i++;
  } while (
    currentParent &&
    currentParent.type === node.type &&
    testNodePropertyNames.every(
      (prop) => currentParent[prop] !== previousParent,
    )
  );
  const firstNonConditionalParent = currentParent || parent;
  const lastConditionalParent = previousParent;

  if (
    isConditionalExpression &&
    (isJsxElement(node[testNodePropertyNames[0]]) ||
      isJsxElement(consequentNode) ||
      isJsxElement(alternateNode) ||
      conditionalExpressionChainContainsJsx(lastConditionalParent))
  ) {
    jsxMode = true;
    forceNoIndent = true;

    // Even though they don't need parens, we wrap (almost) everything in
    // parens when using ?: within JSX, because the parens are analogous to
    // curly braces in an if statement.
    const wrap = (doc) => [
      ifBreak("("),
      indent([softline, doc]),
      softline,
      ifBreak(")"),
    ];

    // The only things we don't wrap are:
    // * Nested conditional expressions in alternates
    // * null
    // * undefined
    const isNil = (node) =>
      node.type === "NullLiteral" ||
      (node.type === "Literal" && node.value === null) ||
      (node.type === "Identifier" && node.name === "undefined");

    parts.push(
      " ? ",
      isNil(consequentNode)
        ? print(consequentNodePropertyName)
        : wrap(print(consequentNodePropertyName)),
      " : ",
      alternateNode.type === node.type || isNil(alternateNode)
        ? print(alternateNodePropertyName)
        : wrap(print(alternateNodePropertyName)),
    );
  } else {
    /*
    This does not mean to indent, but make the doc aligned with the first character after `? ` or `: `,
    so we use `2` instead of `options.tabWidth` here.

    ```js
    test
     ? {
         consequent
       }
     : alternate
    ```

    instead of

    ```js
    test
     ? {
       consequent
     }
     : alternate
    ```
    */
    const printBranch = (nodePropertyName) =>
      options.useTabs
        ? indent(print(nodePropertyName))
        : align(2, print(nodePropertyName));
    // normal mode
    const part = [
      line,
      "? ",
      consequentNode.type === node.type ? ifBreak("", "(") : "",
      printBranch(consequentNodePropertyName),
      consequentNode.type === node.type ? ifBreak("", ")") : "",
      line,
      ": ",
      printBranch(alternateNodePropertyName),
    ];
    parts.push(
      parent.type !== node.type ||
        parent[alternateNodePropertyName] === node ||
        isParentTest
        ? part
        : options.useTabs
          ? dedent(indent(part))
          : align(Math.max(0, options.tabWidth - 2), part),
    );
  }

  // We want a whole chain of ConditionalExpressions to all
  // break if any of them break. That means we should only group around the
  // outer-most ConditionalExpression.
  const shouldBreak = [
    consequentNodePropertyName,
    alternateNodePropertyName,
    ...testNodePropertyNames,
  ].some((property) =>
    hasComment(
      node[property],
      (comment) =>
        isBlockComment(comment) &&
        hasNewlineInRange(
          options.originalText,
          locStart(comment),
          locEnd(comment),
        ),
    ),
  );
  const maybeGroup = (doc) =>
    parent === firstNonConditionalParent
      ? group(doc, { shouldBreak })
      : shouldBreak
        ? [doc, breakParent]
        : doc;

  // Break the closing paren to keep the chain right after it:
  // (a
  //   ? b
  //   : c
  // ).call()
  const breakClosingParen =
    !jsxMode &&
    (isMemberExpression(parent) ||
      (parent.type === "NGPipeExpression" && parent.left === node)) &&
    !parent.computed;

  const shouldExtraIndent = shouldExtraIndentForConditionalExpression(path);

  const result = maybeGroup([
    printTernaryTest(path, options, print),
    forceNoIndent ? parts : indent(parts),
    isConditionalExpression && breakClosingParen && !shouldExtraIndent
      ? softline
      : "",
  ]);

  return isParentTest || shouldExtraIndent
    ? group([indent([softline, result]), softline])
    : result;
}

function requireModule(metadata) {
  var moduleExports = globalThis.__next_require__(metadata[0]);
  if (4 === metadata.length && "function" === typeof moduleExports.then)
    if ("fulfilled" === moduleExports.status)
      moduleExports = moduleExports.value;
    else throw moduleExports.reason;
  return "*" === metadata[2]
    ? moduleExports
    : "" === metadata[2]
      ? moduleExports.__esModule
        ? moduleExports.default
        : moduleExports
      : moduleExports[metadata[2]];
}

function handleFullDataLine(feedback, key, label, line) {
  switch (label) {
    case 45:
      loadComponent(feedback, key, line);
      break;
    case 44:
      key = line[0];
      line = line.slice(1);
      feedback = JSON.parse(line, feedback._decode);
      line = ReactSharedInternals.e;
      switch (key) {
        case "E":
          line.E(feedback);
          break;
        case "B":
          "string" === typeof feedback
            ? line.B(feedback)
            : line.B(feedback[0], feedback[1]);
          break;
        case "K":
          key = feedback[0];
          label = feedback[1];
          3 === feedback.length ? line.K(key, label, feedback[2]) : line.K(key, label);
          break;
        case "n":
          "string" === typeof feedback
            ? line.n(feedback)
            : line.n(feedback[0], feedback[1]);
          break;
        case "Y":
          "string" === typeof feedback
            ? line.Y(feedback)
            : line.Y(feedback[0], feedback[1]);
          break;
        case "P":
          "string" === typeof feedback
            ? line.P(feedback)
            : line.P(
                feedback[0],
                0 === feedback[1] ? void 0 : feedback[1],
                3 === feedback.length ? feedback[2] : void 0
              );
          break;
        case "G":
          "string" === typeof feedback
            ? line.G(feedback)
            : line.G(feedback[0], feedback[1]);
      }
      break;
    case 40:
      label = JSON.parse(line);
      line = resolveErrorProd();
      line.digest = label.digest;
      label = feedback._parts;
      var part = label.get(key);
      part
        ? triggerErrorOnPart(part, line)
        : label.set(key, new ReactPromise("rejected", null, line, feedback));
      break;
    case 56:
      label = feedback._parts;
      (part = label.get(key)) && "pending" !== part.status
        ? part.reason.enqueueValue(line)
        : label.set(key, new ReactPromise("fulfilled", line, null, feedback));
      break;
    case 50:
    case 42:
    case 61:
      throw Error(
        "Failed to read a RSC payload created by a development version of React on the server while using a production version on the client. Always use matching versions on the server and the client."
      );
    case 58:
      startReadableStream(feedback, key, void 0);
      break;
    case 130:
      startReadableStream(feedback, key, "bytes");
      break;
    case 64:
      startAsyncIterable(feedback, key, !1);
      break;
    case 136:
      startAsyncIterable(feedback, key, !0);
      break;
    case 41:
      (feedback = feedback._parts.get(key)) &&
        "fulfilled" === feedback.status &&
        feedback.reason.close("" === line ? '"$undefined"' : line);
      break;
    default:
      (label = feedback._parts),
        (part = label.get(key))
          ? resolveModelPart(part, line)
          : label.set(
              key,
              new ReactPromise("resolved_model", line, null, feedback)
            );
  }
}

function resolveThenable(thenable) {
  switch (thenable.status) {
    case "fulfilled":
      return thenable.value;
    case "rejected":
      throw thenable.reason;
    default:
      switch (
        ("string" === typeof thenable.status
          ? thenable.then(noop$1, noop$1)
          : ((thenable.status = "pending"),
            thenable.then(
              function (fulfilledValue) {
                "pending" === thenable.status &&
                  ((thenable.status = "fulfilled"),
                  (thenable.value = fulfilledValue));
              },
              function (error) {
                "pending" === thenable.status &&
                  ((thenable.status = "rejected"), (thenable.reason = error));
              }
            )),
        thenable.status)
      ) {
        case "fulfilled":
          return thenable.value;
        case "rejected":
          throw thenable.reason;
      }
  }
  throw thenable;
}

  function serializeAsyncIterable(iterable, iterator) {
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
        0 === pendingParts && resolve(data);
      } else
        try {
          var partJSON$22 = JSON.stringify(entry.value, resolveToJSON);
          data.append(formFieldPrefix + streamId, partJSON$22);
          iterator.next().then(progress, reject);
        } catch (x$23) {
          reject(x$23);
        }
    }
    null === formData && (formData = new FormData());
    var data = formData;
    pendingParts++;
    var streamId = nextPartId++;
    iterable = iterable === iterator;
    iterator.next().then(progress, reject);
    return "$" + (iterable ? "x" : "X") + streamId.toString(16);
  }

