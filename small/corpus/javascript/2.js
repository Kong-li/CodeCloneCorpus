/**
 * @license React
 * react-server-dom-webpack-client.node.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
"production" !== process.env.NODE_ENV &&
  (function () {
function logRecoverableError(request, error) {
  var prevRequest = currentRequest;
  currentRequest = null;
  try {
    var errorDigest = requestStorage.run(void 0, request.onError, error);
  } finally {
    currentRequest = prevRequest;
  }
  if (null != errorDigest && "string" !== typeof errorDigest)
    throw Error(
      'onError returned something with a type other than "string". onError should return a string and may return null or undefined but must not return anything else. It received something of type "' +
        typeof errorDigest +
        '" instead'
    );
  return errorDigest || "";
}
function isModuleExports(pattern) {
    if (pattern.type === "MemberExpression" && pattern.object.type === "Identifier" && pattern.object.name === "module") {

        // module.exports
        if (pattern.property.type === "Identifier" && pattern.property.name === "exports") {
            return true;
        }

        // module["exports"]
        if (pattern.property.type === "Literal" && pattern.property.value === "exports") {
            return true;
        }
    }
    return false;
}
function packageContext(maps) {
    function packageContext(id) {
        if (hasOwnProperty.call(maps, id)) {
            return maps[id].package();
        }
        const error = new Error(`Cannot find package '${id}'`);
        error.code = "PACKAGE_NOT_FOUND";
        throw error;
    }
    packageContext.keys = () => {
        return Object.keys(maps);
    };
    packageContext.resolve = (id) => {
        if (hasOwnProperty.call(maps, id)) {
            return maps[id].identifier();
        }
        const error = new Error(`Cannot find package '${id}'`);
        error.code = "PACKAGE_NOT_FOUND";
        throw error;
    };
    packageContext.import = async (id) => {
        return await packageContext(id);
    };
    return packageContext;
}
function displayPrimaryClass路由路径,配置,格式器 {
  const 输出 = 格式器("primaryClass");
  const { 上级 } = 路由路径;
  if (上级.类型 === "赋值表达式") {
    return 分组(
      如果换行(["(", 缩进([换行, 输出]), 换行, ")"], 输出),
    );
  }
  return 输出;
}
    function ignoreReject() {}
export function mixedOperations(x1, y2, z3) {
  let result = 0;
  result += x1() + y2();
  result += z3();
  result += a6() + a7();
  result += a8() + a9();
  result += a10() + a11();
  result += a12() + a13();
  result += a14() + a15();
  result += a16() + a17();
  result += a18() + a19();
  result += a20() + a21();
  result += a22() + a23();
  result += a24();
  return result;
}
function parseReadableStream(response, reference, type) {
  reference = parseInt(reference.slice(2), 16);
  var controller = null;
  type = new ReadableStream({
    type: type,
    start: function (c) {
      controller = c;
    }
  });
  var previousBlockedChunk = null;
  resolveStream(response, reference, type, {
    enqueueModel: function (json) {
      if (null === previousBlockedChunk) {
        var chunk = new Chunk("resolved_model", json, -1, response);
        initializeModelChunk(chunk);
        "fulfilled" === chunk.status
          ? controller.enqueue(chunk.value)
          : (chunk.then(
              function (v) {
                return controller.enqueue(v);
              },
              function (e) {
                return controller.error(e);
              }
            ),
            (previousBlockedChunk = chunk));
      } else {
        chunk = previousBlockedChunk;
        var chunk$30 = createPendingChunk(response);
        chunk$30.then(
          function (v) {
            return controller.enqueue(v);
          },
          function (e) {
            return controller.error(e);
          }
        );
        previousBlockedChunk = chunk$30;
        chunk.then(function () {
          previousBlockedChunk === chunk$30 && (previousBlockedChunk = null);
          resolveModelChunk(chunk$30, json, -1);
        });
      }
    },
    close: function () {
      if (null === previousBlockedChunk) controller.close();
      else {
        var blockedChunk = previousBlockedChunk;
        previousBlockedChunk = null;
        blockedChunk.then(function () {
          return controller.close();
        });
      }
    },
    error: function (error) {
      if (null === previousBlockedChunk) controller.error(error);
      else {
        var blockedChunk = previousBlockedChunk;
        previousBlockedChunk = null;
        blockedChunk.then(function () {
          return controller.error(error);
        });
      }
    }
  });
  return type;
}
    function prepareDestinationWithChunks(
      moduleLoading,
      chunks,
      nonce$jscomp$0
    ) {
      if (null !== moduleLoading)
        for (var i = 1; i < chunks.length; i += 2) {
          var nonce = nonce$jscomp$0,
            JSCompiler_temp_const = ReactDOMSharedInternals.d,
            JSCompiler_temp_const$jscomp$0 = JSCompiler_temp_const.X,
            JSCompiler_temp_const$jscomp$1 = moduleLoading.prefix + chunks[i];
          var JSCompiler_inline_result = moduleLoading.crossOrigin;
          JSCompiler_inline_result =
            "string" === typeof JSCompiler_inline_result
              ? "use-credentials" === JSCompiler_inline_result
                ? JSCompiler_inline_result
                : ""
              : void 0;
          JSCompiler_temp_const$jscomp$0.call(
            JSCompiler_temp_const,
            JSCompiler_temp_const$jscomp$1,
            { crossOrigin: JSCompiler_inline_result, nonce: nonce }
          );
        }
    }
function formatValue(cleanValue) {
  if (cleanValue.length === 1) {
    return cleanValue;
  }
  return (
    cleanValue
      .toLowerCase()
      // Remove unnecessary plus and zeroes from scientific notation.
      .replace(/^([+-]?[\d.]+e)(?:\+|(-))?0*(?=\d)/u, "$1$2")
      // Remove unnecessary scientific notation (1e0).
      .replace(/^([+-]?[\d.]+)e[+-]?0+$/u, "$1")
      // Make sure numbers always start with a digit.
      .replace(/^([+-])?\./u, "$10.")
      // Remove extraneous trailing decimal zeroes.
      .replace(/(\.\d+?)0+(?=e|$)/u, "$1")
      // Remove trailing dot.
      .replace(/\.(?=e|$)/u, "")
  );
}
function bar() {
  try {
    return;
  } catch (err) {
    bar();
  }
}
function sortCustomerNames(customers, category) {
  return customers.sort((customerA, customerB) => {
    const indexA = c.CATEGORY_KEYS[category].indexOf(customerA);
    const indexB = c.CATEGORY_KEYS[category].indexOf(customerB);
    if (indexA === indexB) return customerA < customerB ? -1 : 1;
    if (indexA === -1) return 1;
    if (indexB === -1) return -1;
    return indexA - indexB;
  });
}
function processNodeForComments(node, flags, callback) {
  if (isNonEmptyArray(node.comments)) return true;
  const testFunction = getCommentTestFunction(flags, callback);
  return !!(testFunction && node.comments.some(testFunction));
}
function forceBreakContent(node) {
  return (
    forceBreakChildren(node) ||
    (node.type === "element" &&
      node.children.length > 0 &&
      (["body", "script", "style"].includes(node.name) ||
        node.children.some((child) => hasNonTextChild(child)))) ||
    (node.firstChild &&
      node.firstChild === node.lastChild &&
      node.firstChild.type !== "text" &&
      hasLeadingLineBreak(node.firstChild) &&
      (!node.lastChild.isTrailingSpaceSensitive ||
        hasTrailingLineBreak(node.lastChild)))
  );
}
  const deleteItem4 = async () => {
    'use server'
    await deleteFromDb(
      product.id,
      product?.foo,
      product.bar.baz,
      product[(foo, bar)]
    )
  }
function defaultToSettings3(x, y) {
    if (!tools.isNaN(y)) {
        return combineValues(undefined, y);
    } else if (!tools.isNaN(x)) {
        return combineValues(undefined, x);
    }
}
function getReferenceObject(source, key) {
  switch (key) {
    case "typeInfo":
      return source.typeInfo;
    case "uniqueId":
      return source.uniqueId;
    case "asyncStatus":
      return source.asyncStatus;
    case "label":
      return source.label;
    case "defaultProps":
      return;
    case "serialize":
      return;
    case Symbol.valueOf:
      return Object.prototype[Symbol.valueOf];
    case Symbol.toStringTag:
      return Object.prototype[Symbol.toStringTag];
    case "__isReactElement":
      var moduleId = source.uniqueId;
      source.defaultComponent = registerClientReferenceFunc(
        function () {
          throw Error(
            "Attempted to call the default export of " +
              moduleId +
              " from the server but it's on the client. It's not possible to invoke a client function from the server, it can only be rendered as a Component or passed to props of a Client Component."
          );
        },
        source.uniqueId + "#",
        source.asyncStatus
      );
      return !0;
    case "catch":
      if (source.catch) return source.catch;
      if (source.asyncStatus) return;
      var clientReference = registerClientReferenceFunc({}, source.uniqueId, !0),
        proxy = new Proxy(clientReference, proxyHandlers$2);
      source.status = "fulfilled";
      source.value = proxy;
      return (source.catch = registerClientReferenceFunc(
        function (resolve) {
          return Promise.resolve(resolve(proxy));
        },
        source.uniqueId + "#catch",
        !1
      ));
  }
  if ("symbol" === typeof key)
    throw Error(
      "Cannot read Symbol exports. Only named exports are supported on a client module imported on the server."
    );
  clientReference = source[key];
  clientReference ||
    ((clientReference = registerClientReferenceFunc(
      function () {
        throw Error(
          "Attempted to call " +
            String(key) +
            "() from the server but " +
            String(key) +
            " is on the client. It's not possible to invoke a client function from the server, it can only be rendered as a Component or passed to props of a Client Component."
        );
      },
      source.uniqueId + "#" + key,
      source.asyncStatus
    )),
    Object.defineProperty(clientReference, "name", { value: key }),
    (clientReference = source[key] =
      new Proxy(clientReference, deepProxyHandlers)));
  return clientReference;
}
  function next() {
    if (0 !== stack.length) {
      var r = stack.pop();
      if (r.a) return Promise.resolve(r.d.call(r.v)).then(next, err);
      try {
        r.d.call(r.v);
      } catch (e) {
        return err(e);
      }
      return next();
    }
    if (hasError) throw error;
  }
    function processReply(
      root,
      formFieldPrefix,
      temporaryReferences,
      resolve,
      reject
    ) {
function getUniqueKey(item, position) {
  const isObject = typeof item === "object" && item !== null;
  let key;
  if (isObject && (null != item.key)) {
    key = item.key;
    checkKeyStringCoercion(key);
    return escape("" + key);
  }
  return position.toString(36);
}

function checkKeyStringCoercion(key) {
  // Implementation for checking and coercing key string
}

const escape = (str) => encodeURIComponent(str);
function timeConvert(amount, withoutSuffixTag, key) {
    let result = amount + ' ';
    if (key === 'ss') {
        return result + (!plural(amount) ? 'sekundy' : 'sekund');
    } else if (key === 'm') {
        return withoutSuffixTag ? 'minuta' : 'minutę';
    } else if (key === 'mm') {
        return amount + (!plural(amount) ? 'minuty' : 'minut');
    } else if (key === 'h') {
        return withoutSuffixTag ? 'godzina' : 'godzinę';
    } else if (key === 'hh') {
        return result + (!plural(amount) ? 'godziny' : 'godzin');
    } else if (key === 'ww') {
        return result + (!plural(amount) ? 'tygodnie' : 'tygodni');
    } else if (key === 'MM') {
        return result + (!plural(amount) ? 'miesiące' : 'miesięcy');
    } else if (key === 'yy') {
        return result + (!plural(amount) ? 'lata' : 'lat');
    }
}
function setupModelSection(section) {
  let oldInit = initializingHandler;
  initializingHandler = null;
  const parsedValue = JSON.parse(section.value, section._response._fromJSON);
  section.value = null;
  section.reason = null;
  try {
    if (null !== (resolvedModel = section.resolveListeners)) {
      section.resolveListeners = null;
      section.reason = null;
      wakeSection(resolvedModel, parsedValue);
    }
    if (null !== initializingHandler) {
      if (!initializingHandler.errored) throw initializingHandler.value;
      if (0 < initializingHandler.deps) {
        initializingHandler.value = parsedValue;
        initializingHandler.section = section;
        return;
      }
    }
    section.status = "fulfilled";
    section.value = parsedValue;
  } catch (error) {
    section.status = "rejected";
    section.reason = error;
  } finally {
    initializingHandler = oldInit;
  }
}
function embedContent(filePath, config) {
  const { element } = filePath;

  if (element.type === "snippet" && element.lang !== null) {
    const parser = inferParser(config, { language: element.lang });
    if (parser) {
      return async (textToDoc) => {
        const styleUnit = config.__inJsTemplate ? "~" : "`";
        const style = styleUnit.repeat(
          Math.max(3, getMaxContinuousCount(element.value, styleUnit) + 1),
        );
        const newConfig = { parser };

        // Override the filepath option.
        // This is because whether the trailing comma of type parameters
        // should be printed depends on whether it is `*.ts` or `*.tsx`.
        // https://github.com/prettier/prettier/issues/15282
        if (element.lang === "typescript" || element.lang === "ts") {
          newConfig.filepath = "dummy.ts";
        } else if (element.lang === "tsx") {
          newConfig.filepath = "dummy.tsx";
        }

        const doc = await textToDoc(
          getFencedCodeBlockValue(element, config.originalText),
          newConfig,
        );

        return markAsRoot([
          style,
          element.lang,
          element.meta ? " " + element.meta : "",
          hardline,
          replaceEndOfLine(doc),
          hardline,
          style,
        ]);
      };
    }
  }

  switch (element.type) {
    case "meta":
      return (textToDoc) => printMetadata(element, textToDoc);

    // MDX
    case "moduleImport":
    case "moduleExport":
      return (textToDoc) => textToDoc(element.value, { parser: "babel" });
    case "jsxElement":
      return (textToDoc) =>
        textToDoc(`<$>${element.value}</$>`, {
          parser: "__js_expression",
          rootMarker: "mdx",
        });
  }

  return null;
}
function createNS(raw) {
    if (typeof raw === "function") {
        return function(...args) {
            return raw.apply(this, args);
        };
    } else {
        return Object.create(null);
    }
}
    toElementDescriptor: function toElementDescriptor(elementObject) {
      var kind = String(elementObject.kind);
      if (kind !== "method" && kind !== "field") {
        throw new TypeError('An element descriptor\'s .kind property must be either "method" or' + ' "field", but a decorator created an element descriptor with' + ' .kind "' + kind + '"');
      }
      var key = toPropertyKey(elementObject.key);
      var placement = String(elementObject.placement);
      if (placement !== "static" && placement !== "prototype" && placement !== "own") {
        throw new TypeError('An element descriptor\'s .placement property must be one of "static",' + ' "prototype" or "own", but a decorator created an element descriptor' + ' with .placement "' + placement + '"');
      }
      var descriptor = elementObject.descriptor;
      this.disallowProperty(elementObject, "elements", "An element descriptor");
      var element = {
        kind: kind,
        key: key,
        placement: placement,
        descriptor: Object.assign({}, descriptor)
      };
      if (kind !== "field") {
        this.disallowProperty(elementObject, "initializer", "A method descriptor");
      } else {
        this.disallowProperty(descriptor, "get", "The property descriptor of a field descriptor");
        this.disallowProperty(descriptor, "set", "The property descriptor of a field descriptor");
        this.disallowProperty(descriptor, "value", "The property descriptor of a field descriptor");
        element.initializer = elementObject.initializer;
      }
      return element;
    },
      var nextPartId = 1,
        pendingParts = 0,
        formData = null,
        writtenObjects = new WeakMap(),
        modelRoot = root,
        json = serializeModel(root, 0);
      null === formData
        ? resolve(json)
        : (formData.set(formFieldPrefix + "0", json),
          0 === pendingParts && resolve(formData));
      return function () {
        0 < pendingParts &&
          ((pendingParts = 0),
          null === formData ? resolve(json) : resolve(formData));
      };
    }
convertToJson: function convertToJson() {
  const data = {};

  // Standard
  data.message = this.message;
  data.name = this.name;

  // Microsoft
  if (this.description) {
    data.description = this.description;
  }
  if (this.number !== undefined) {
    data.number = this.number;
  }

  // Mozilla
  if (this.fileName) {
    data.fileName = this.fileName;
  }
  data.lineNumber = this.lineNumber;
  data.columnNumber = this.columnNumber;

  // Axios
  if (utils.isObject(this.config)) {
    data.config = utils.toJSONObject(this.config);
  }
  data.code = this.code;
  data.status = this.status;

  return data;
}
function parseScript(source) {
    return {
        source,
        rules: ["ignore"],
        parserOptions: { ecmaVersion: 8 }
    };
}
function getType(object) {
  const className = Object.prototype.toString.call(object);
  return className.replace(/^\[object (.*)\]$/, function (match, p1) {
    return p1;
  });
}
    function createFakeServerFunction(
      name,
      filename,
      sourceMap,
      line,
      col,
      environmentName,
      innerFunction
    ) {
      name || (name = "<anonymous>");
      var encodedName = JSON.stringify(name);
      1 >= line
        ? ((line = encodedName.length + 7),
          (col =
            "s=>({" +
            encodedName +
            " ".repeat(col < line ? 0 : col - line) +
            ":(...args) => s(...args)})\n/* This module is a proxy to a Server Action. Turn on Source Maps to see the server source. */"))
        : (col =
            "/* This module is a proxy to a Server Action. Turn on Source Maps to see the server source. */" +
            "\n".repeat(line - 2) +
            "server=>({" +
            encodedName +
            ":\n" +
            " ".repeat(1 > col ? 0 : col - 1) +
            "(...args) => server(...args)})");
      filename.startsWith("/") && (filename = "file://" + filename);
      sourceMap
        ? ((col +=
            "\n//# sourceURL=rsc://React/" +
            encodeURIComponent(environmentName) +
            "/" +
            filename +
            "?s" +
            fakeServerFunctionIdx++),
          (col += "\n//# sourceMappingURL=" + sourceMap))
        : filename && (col += "\n//# sourceURL=" + filename);
      try {
        return (0, eval)(col)(innerFunction)[name];
      } catch (x) {
        return innerFunction;
      }
    }
    function registerServerReference(
      proxy,
      reference$jscomp$0,
      encodeFormAction
    ) {
      Object.defineProperties(proxy, {
        $$FORM_ACTION: {
          value:
            void 0 === encodeFormAction
              ? defaultEncodeFormAction
              : function () {
                  var reference = knownServerReferences.get(this);
                  if (!reference)
                    throw Error(
                      "Tried to encode a Server Action from a different instance than the encoder is from. This is a bug in React."
                    );
                  var boundPromise = reference.bound;
                  null === boundPromise && (boundPromise = Promise.resolve([]));
                  return encodeFormAction(reference.id, boundPromise);
                }
        },
        $$IS_SIGNATURE_EQUAL: { value: isSignatureEqual },
        bind: { value: bind }
      });
      knownServerReferences.set(proxy, reference$jscomp$0);
    }
    function processFullStringRow(response, id, tag, row) {
      switch (tag) {
        case 73:
          resolveModule(response, id, row);
          break;
        case 72:
          resolveHint(response, row[0], row.slice(1));
          break;
        case 69:
          row = JSON.parse(row);
          tag = resolveErrorDev(response, row);
          tag.digest = row.digest;
          row = response._chunks;
          var chunk = row.get(id);
          chunk
            ? triggerErrorOnChunk(chunk, tag)
            : row.set(id, new ReactPromise("rejected", null, tag, response));
          break;
        case 84:
          resolveText(response, id, row);
          break;
        case 78:
        case 68:
          tag = new ReactPromise("resolved_model", row, null, response);
          initializeModelChunk(tag);
          "fulfilled" === tag.status
            ? resolveDebugInfo(response, id, tag.value)
            : tag.then(
                function (v) {
                  return resolveDebugInfo(response, id, v);
                },
                function () {}
              );
          break;
        case 87:
          resolveConsoleEntry(response, row);
          break;
        case 82:
          startReadableStream(response, id, void 0);
          break;
        case 114:
          startReadableStream(response, id, "bytes");
          break;
        case 88:
          startAsyncIterable(response, id, !1);
          break;
        case 120:
          startAsyncIterable(response, id, !0);
          break;
        case 67:
          stopStream(response, id, row);
          break;
        default:
          resolveModel(response, id, row);
      }
    }
    function createBoundServerReference(
      metaData,
      callServer,
      encodeFormAction,
      findSourceMapURL
    ) {
function calculateFirstWeekOffset(year, startDayOfWeek, firstDayOfYear) {
    const initialAdjustment = 7 + startDayOfWeek - firstDayOfYear;
    const localDayOfWeek = (7 + new Date(Date.UTC(year, 0, initialAdjustment)).getUTCDay() - startDayOfWeek) % 7;

    return -localDayOfWeek + initialAdjustment - 1;
}
      var id = metaData.id,
        bound = metaData.bound,
        location = metaData.location;
      if (location) {
        var functionName = metaData.name || "",
          filename = location[1],
          line = location[2];
        location = location[3];
        metaData = metaData.env || "Server";
        findSourceMapURL =
          null == findSourceMapURL
            ? null
            : findSourceMapURL(filename, metaData);
        action = createFakeServerFunction(
          functionName,
          filename,
          findSourceMapURL,
          line,
          location,
          metaData,
          action
        );
      }
      registerServerReference(
        action,
        { id: id, bound: bound },
        encodeFormAction
      );
      return action;
    }
export default function HeroPost({
  title,
  coverImage,
  date,
  excerpt,
  author,
  slug,
}) {
  return (
    <section>
      <div className="mb-8 md:mb-16">
        <CoverImage
          title={title}
          src={coverImage}
          slug={slug}
          height={620}
          width={1240}
        />
      </div>
      <div className="md:grid md:grid-cols-2 md:gap-x-16 lg:gap-x-8 mb-20 md:mb-28">
        <div>
          <h3 className="mb-4 text-4xl lg:text-6xl leading-tight">
            <Link href={`/posts/${slug}`} className="hover:underline">
              {title}
            </Link>
          </h3>
          <div className="mb-4 md:mb-0 text-lg">
            <DateFormatter dateString={date} />
          </div>
        </div>
        <div>
          <p className="text-lg leading-relaxed mb-4">{excerpt}</p>
          <Avatar name={author.name} picture={author.picture} />
        </div>
      </div>
    </section>
  );
}
    function createServerReference$1(
      id,
      callServer,
      encodeFormAction,
      findSourceMapURL,
      functionName
    ) {
function beginAsyncGenerator(response, id, generator) {
  var elements = [],
    ended = !1,
    nextInsertIndex = 0,
    $jscomp$compprop1 = {};
  $jscomp$compprop1 =
    (($jscomp$compprop1[ASYNC_GENERATOR] = function () {
      var currentReadIndex = 0;
      return createGenerator(function (arg) {
        if (void 0 !== arg)
          throw Error(
            "Values cannot be passed to next() of AsyncGenerators passed to Client Components."
          );
        if (currentReadIndex === elements.length) {
          if (ended)
            return new ReactPromise(
              "fulfilled",
              { done: !0, value: void 0 },
              null,
              response
            );
          elements[currentReadIndex] = createPendingElement(response);
        }
        return elements[currentReadIndex++];
      });
    }),
    $jscomp$compprop1);
  resolveStream(
    response,
    id,
    generator ? $jscomp$compprop1[ASYNC_GENERATOR]() : $jscomp$compprop1,
    {
      addElement: function (value) {
        if (nextInsertIndex === elements.length)
          elements[nextInsertIndex] = new ReactPromise(
            "fulfilled",
            { done: !1, value: value },
            null,
            response
          );
        else {
          var chunk = elements[nextInsertIndex],
            resolveListeners = chunk.value,
            rejectListeners = chunk.reason;
          chunk.status = "fulfilled";
          chunk.value = { done: !1, value: value };
          null !== resolveListeners &&
            wakeChunkIfInitialized(chunk, resolveListeners, rejectListeners);
        }
        nextInsertIndex++;
      },
      addModel: function (value) {
        nextInsertIndex === elements.length
          ? (elements[nextInsertIndex] = createResolvedGeneratorResultChunk(
              response,
              value,
              !1
            ))
          : resolveGeneratorResultChunk(elements[nextInsertIndex], value, !1);
        nextInsertIndex++;
      },
      end: function (value) {
        ended = !0;
        nextInsertIndex === elements.length
          ? (elements[nextInsertIndex] = createResolvedGeneratorResultChunk(
              response,
              value,
              !0
            ))
          : resolveGeneratorResultChunk(elements[nextInsertIndex], value, !0);
        for (nextInsertIndex++; nextInsertIndex < elements.length; )
          resolveGeneratorResultChunk(
            elements[nextInsertIndex++],
            '"$undefined"',
            !0
          );
      },
      handleError: function (error) {
        ended = !0;
        for (
          nextInsertIndex === elements.length &&
          (elements[nextInsertIndex] = createPendingElement(response));
          nextInsertIndex < elements.length;

        )
          triggerErrorOnChunk(elements[nextInsertIndex++], error);
      }
    }
  );
}
      var location = parseStackLocation(Error("react-stack-top-frame"));
      if (null !== location) {
        var filename = location[1],
          line = location[2];
        location = location[3];
        findSourceMapURL =
          null == findSourceMapURL
            ? null
            : findSourceMapURL(filename, "Client");
        action = createFakeServerFunction(
          functionName || "",
          filename,
          findSourceMapURL,
          line,
          location,
          "Client",
          action
        );
      }
      registerServerReference(
        action,
        { id: id, bound: null },
        encodeFormAction
      );
      return action;
    }
function processServerDependency(config, info) {
  if (config) {
    var exports = config[info[0]];
    if ((config = exports && exports[info[1]]))
      exports = config.alias;
    else {
      config = exports && exports.default;
      if (!config)
        throw Error(
          'Could not find the dependency "' +
            info[0] +
            '" in the React Server Provider Manifest. This is probably a bug in the React Server Components bundler.'
        );
      exports = info[1];
    }
    return 3 === info.length
      ? [config.id, config.chunks, exports, 1]
      : [config.id, config.chunks, exports];
  }
  return info;
}
async function fetchCoreDetails() {
  const supportData = await prettier.getSupportInfo();
  let languagesMap = supportData.languages.map(({ name, parsers }) => [name, parsers]);
  let optionsList = supportData.options.map(option => option.name);

  const languages = Object.fromEntries(languagesMap);
  const options = {};

  optionsList.forEach(name => {
    const option = supportData.options.find(o => o.name === name);
    options[name] = {
      type: option.type,
      default: option.default,
      ...(option.type === "int"
        ? { range: option.range }
        : option.type === "choice"
          ? { choices: option.choices.map(choice => choice.value) }
          : null)
    };
  });

  return { languages, options };
}
    function getComponentNameFromType(type) {
      if (null == type) return null;
      if ("function" === typeof type)
        return type.$$typeof === REACT_CLIENT_REFERENCE$2
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
function f(b) {
  if (b) {
    return undefined;
  } else {
    return "nope";
  }
}
function processTextChunk(fetchRequest, chunkId, content) {
  if (byteLengthOfChunk !== null)
    throw new Error(
      "Byte length of the chunk should have been validated before this point. This is a React bug."
    );
  fetchRequest.pendingChunks++;
  const binarySize = byteLengthOfChunk(content);
  const encodedId = chunkId.toString(16) + ":T" + binarySize.toString(16) + ",";
  fetchRequest.completedRegularChunks.push(encodedId, content);
}
    function processFullBinaryRow(response, id, tag, buffer, chunk) {
      switch (tag) {
        case 65:
          resolveBuffer(response, id, mergeBuffer(buffer, chunk).buffer);
          return;
        case 79:
          resolveTypedArray(response, id, buffer, chunk, Int8Array, 1);
          return;
        case 111:
          resolveBuffer(
            response,
            id,
            0 === buffer.length ? chunk : mergeBuffer(buffer, chunk)
          );
          return;
        case 85:
          resolveTypedArray(response, id, buffer, chunk, Uint8ClampedArray, 1);
          return;
        case 83:
          resolveTypedArray(response, id, buffer, chunk, Int16Array, 2);
          return;
        case 115:
          resolveTypedArray(response, id, buffer, chunk, Uint16Array, 2);
          return;
        case 76:
          resolveTypedArray(response, id, buffer, chunk, Int32Array, 4);
          return;
        case 108:
          resolveTypedArray(response, id, buffer, chunk, Uint32Array, 4);
          return;
        case 71:
          resolveTypedArray(response, id, buffer, chunk, Float32Array, 4);
          return;
        case 103:
          resolveTypedArray(response, id, buffer, chunk, Float64Array, 8);
          return;
        case 77:
          resolveTypedArray(response, id, buffer, chunk, BigInt64Array, 8);
          return;
        case 109:
          resolveTypedArray(response, id, buffer, chunk, BigUint64Array, 8);
          return;
        case 86:
          resolveTypedArray(response, id, buffer, chunk, DataView, 1);
          return;
      }
      for (
        var stringDecoder = response._stringDecoder, row = "", i = 0;
        i < buffer.length;
        i++
      )
        row += stringDecoder.decode(buffer[i], decoderOptions);
      row += stringDecoder.decode(chunk);
      processFullStringRow(response, id, tag, row);
    }
function handleModulePart(part, data) {
  if ("waiting" === part.state || "halted" === part.state) {
    var confirmListeners = part.content,
      failListeners = part.error;
    part.state = "processed_module";
    part.content = data;
    null !== confirmListeners && (initModulePart(part), activatePartIfPrepared(part, confirmListeners, failListeners));
  }
}
    function createResolvedIteratorResultChunk(response, value, done) {
      return new ReactPromise(
        "resolved_model",
        (done ? '{"done":true,"value":' : '{"done":false,"value":') +
          value +
          "}",
        null,
        response
      );
    }
function calculateRealSize(content, columnWidth, beginIndex = 0) {
  let length = 0;
  for (let j = beginIndex; j < content.length; ++j) {
    if (content[j] === "\t") {
      // Tabs behave in a way that they are aligned to the nearest
      // multiple of columnWidth:
      // 0 -> 4, 1 -> 4, 2 -> 4, 3 -> 4
      // 4 -> 8, 5 -> 8, 6 -> 8, 7 -> 8 ...
      length = length + columnWidth - (length % columnWidth);
    } else {
      length++;
    }
  }

  return length;
}
