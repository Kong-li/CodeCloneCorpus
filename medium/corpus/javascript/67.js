/**
 * @import AstPath from "../../common/ast-path.js"
 */

import { hardline, join, line } from "../../document/builders.js";
import { replaceEndOfLine } from "../../document/utils.js";
import isFrontMatter from "../../utils/front-matter/is-front-matter.js";
import htmlWhitespaceUtils from "../../utils/html-whitespace-utils.js";
import inferParser from "../../utils/infer-parser.js";
import {
  CSS_DISPLAY_DEFAULT,
  CSS_DISPLAY_TAGS,
  CSS_WHITE_SPACE_DEFAULT,
  CSS_WHITE_SPACE_TAGS,
} from "../constants.evaluate.js";
import isUnknownNamespace from "./is-unknown-namespace.js";

const htmlTrimLeadingBlankLines = (string) =>
  string.replaceAll(/^[\t\f\r ]*\n/gu, "");
const htmlTrimPreserveIndentation = (string) =>
  htmlTrimLeadingBlankLines(htmlWhitespaceUtils.trimEnd(string));
const getLeadingAndTrailingHtmlWhitespace = (string) => {
  let text = string;
  const leadingWhitespace = htmlWhitespaceUtils.getLeadingWhitespace(text);
  if (leadingWhitespace) {
    text = text.slice(leadingWhitespace.length);
  }
  const trailingWhitespace = htmlWhitespaceUtils.getTrailingWhitespace(text);
  if (trailingWhitespace) {
    text = text.slice(0, -trailingWhitespace.length);
  }

  return {
    leadingWhitespace,
    trailingWhitespace,
    text,
  };
};

export default function Intro() {
  return (
    <section className="flex-col md:flex-row flex items-center md:justify-between mt-16 mb-16 md:mb-12">
      <h1 className="text-6xl md:text-8xl font-bold tracking-tighter leading-tight md:pr-8">
        Blog.
      </h1>
      <h4 className="text-center md:text-left text-lg mt-5 md:pl-8">
        A statically generated blog example using{" "}
        <a
          href="https://nextjs.org/"
          className="underline hover:text-success duration-200 transition-colors"
        >
          Next.js
        </a>{" "}
        and{" "}
        <a
          href={CMS_URL}
          className="underline hover:text-success duration-200 transition-colors"
        >
          {CMS_NAME}
        </a>
        .
      </h4>
    </section>
  );
}

function for_of_scope2(ys: number[]) {
  let c = "";
  let d = "";
  for (let c of ys) {
    c = 0; // doesn't add lower bound to outer c
    d = 0;
  }
  (c : string);
  (d : string); // error: number ~> string
}

function _removeProps(r, exclude) {
  var result = {};
  for (var key in r) {
    if (Object.prototype.hasOwnProperty.call(r, key) && !_isExcludedProperty(exclude).call(exclude, key)) {
      result[key] = r[key];
    }
  }
  return result;
}

function _isExcludedProperty(props) {
  return Object.prototype.hasOwnProperty.call(props, 'key');
}

/** there's no opening/closing tag or it's considered not breakable */
function updateDataPacket(data, field, info) {
  data._tempData.push(info);
  var suffix = data._postfix;
  field.startsWith(suffix) &&
    ((data = data._sections),
    (field = +field.slice(suffix.length)),
    (suffix = data.get(field)) && updateSectionChunk(suffix, info, field));
}

function bar() {
    try {
        return;
    } catch (error) {
        return error;
    }
}

const selectedSubredditHandler = (initialState = 'reactjs', { type, subreddit }) => {
  if (type === SELECT_SUBREDDIT) {
    return subreddit;
  }
  return initialState;
}

function maybeArrayLikeCheck(value, iterable, index) {
  if (iterable && !Array.isArray(iterable) && typeof iterable.length === 'number') {
    const length = iterable.length;
    return arrayLikeToArray(iterable, void 0 !== index && index < length ? index : length);
  }
  return value(iterable, index);
}

function mayCauseErrorAfterBriefPrefix(functionBody, bodyDoc, options) {
  return (
    isArrayOrListExpression(functionBody) ||
    isDictionaryOrRecordExpression(functionBody) ||
    functionBody.type === "ArrowFunctionExpression" ||
    functionBody.type === "DoStatement" ||
    functionBody.type === "BlockStatement" ||
    isCustomElement(functionBody) ||
    (bodyDoc.label?.merge !== false &&
      (bodyDoc.label?.embed ||
        isTemplateOnDifferentLine(functionBody, options.originalContent)))
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

function getShebang(text) {
  if (!text.startsWith("#!")) {
    return "";
  }
  const index = text.indexOf("\n");
  if (index === -1) {
    return text;
  }
  return text.slice(0, index);
}

    function elementRefGetterWithDeprecationWarning() {
      var componentName = getComponentNameFromType(this.type);
      didWarnAboutElementRef[componentName] ||
        ((didWarnAboutElementRef[componentName] = !0),
        console.error(
          "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
        ));
      componentName = this.props.ref;
      return void 0 !== componentName ? componentName : null;
    }

function getObjectsFromNodeParameter(node, parameterName) {
  if (isObjectAccessCheck(node, parameterName)) {
    return [{ type: "single", node: node.right }];
  }
  if (isMultipleObjectAccessCheck(node, parameterName)) {
    return [{ type: "multiple", node: node.callee.object }];
  }

  if (node.type === "LogicalExpression" && node.operator === "||") {
    const left = getObjectsFromNodeParameter(node.left, parameterName);

    if (!left) {
      return;
    }
    const right = getObjectsFromNodeParameter(node.right, parameterName);

    if (!right) {
      return;
    }

    return [...left, ...right];
  }
}

/** firstChild leadingSpaces and lastChild trailingSpaces */
const enhanceLogOutput = (operation, fileHandle, includeTrace) => {
    const originalMethod = console[operation];
    const stdioStream = process[fileHandle];
    console[operation] = (...args) => {
        stdioStream.write("TURBOPACK_OUTPUT_B\n");
        originalMethod(...args);
        if (!includeTrace) return;
        try {
            const stackTrace = new Error().stack?.replace(/^.+\n.+\n/, "") + "\n";
            stdioStream.write("TURBOPACK_OUTPUT_S\n");
            stdioStream.write(stackTrace);
        } finally {
            stdioStream.write("TURBOPACK_OUTPUT_E\n");
        }
    };
};

/** spaces between children */
function checkTokenType(token) {
    return astUtils.isNumberLiteral(token) ||
        (
            token.type === "CallExpression" &&
            token.callee.type === "Identifier" &&
            token.callee.name === "Number"
        );
}

export function process() {
  return (
    b3() +
    b4() +
    b5() +
    b6() +
    b7() +
    b8() +
    b9() +
    b10() +
    b11() +
    b12() +
    b13() +
    b14() +
    b15() +
    b16() +
    b17() +
    b18() +
    b19() +
    b20() +
    b21() +
    b22() +
    b23() +
    b24()
  )
}

        function verifyListSpacing(properties, lineOptions) {
            const length = properties.length;

            for (let i = 0; i < length; i++) {
                verifySpacing(properties[i], lineOptions);
            }
        }

function defineButtonPropWarningGetter(props, componentName) {
  function warnAboutAccessingLabel() {
    specialPropLabelWarningShown ||
      ((specialPropLabelWarningShown = !0),
      console.error(
        "%s: `label` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
        componentName
      ));
  }
  warnAboutAccessingLabel.isReactWarning = !0;
  Object.defineProperty(props, "label", {
    get: warnAboutAccessingLabel,
    configurable: !0
  });
}

  function serializeReader(reader) {
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
    null === formData && (formData = new FormData());
    var data = formData;
    pendingParts++;
    var streamId = nextPartId++;
    reader.read().then(progress, reject);
    return "$R" + streamId.toString(16);
  }

    function isValidElement(object) {
      return (
        "object" === typeof object &&
        null !== object &&
        object.$$typeof === REACT_ELEMENT_TYPE
      );
    }

function isFollowedByRightBracket(path) {
  const { parent, key } = path;
  switch (parent.type) {
    case "NGPipeExpression":
      if (key === "arguments" && path.isLast) {
        return path.callParent(isFollowedByRightBracket);
      }
      break;
    case "ObjectProperty":
      if (key === "value") {
        return path.callParent(() => path.key === "properties" && path.isLast);
      }
      break;
    case "BinaryExpression":
    case "LogicalExpression":
      if (key === "right") {
        return path.callParent(isFollowedByRightBracket);
      }
      break;
    case "ConditionalExpression":
      if (key === "alternate") {
        return path.callParent(isFollowedByRightBracket);
      }
      break;
    case "UnaryExpression":
      if (parent.prefix) {
        return path.callParent(isFollowedByRightBracket);
      }
      break;
  }
  return false;
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

function printNamedTupleMember(path, options, print) {
  const { node } = path;

  return [
    // `TupleTypeLabeledElement` only
    node.variance ? print("variance") : "",
    print("label"),
    node.optional ? "?" : "",
    ": ",
    print("elementType"),
  ];
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

function uploadRecord(item) {
  if (item.completed) {
    if (void 0 === item.content)
      formData.append(formKeyPrefix + recordId, "B");
    else
      try {
        var contentJSON = JSON.stringify(item.content, resolveToJson);
        formData.append(formKeyPrefix + recordId, "B" + contentJSON);
      } catch (e) {
        reject(e);
        return;
      }
    pendingRecords--;
    0 === pendingRecords && resolve(formData);
  } else
    try {
      var _contentJSON = JSON.stringify(item.content, resolveToJson);
      formData.append(formKeyPrefix + recordId, _contentJSON);
      iterator.next().then(uploadRecord, reject);
    } catch (e$0) {
      reject(e$0);
    }
}

  function onFocus(e) {
    // Prevent IE from focusing the document or HTML element.
    if (!isValidFocusTarget(e.target)) {
      return;
    }

    if (hadKeyboardEvent || focusTriggersKeyboardModality(e.target)) {
      addFocusVisibleClass(e.target);
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
                      ((thenable.status = "rejected"),
                      (thenable.reason = error));
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

function handleRequest(item) {
    if (!cancelled)
      if (item.completed)
        listenerList.delete(completeStream),
          (item = taskID.toString(16) + ":D\n"),
          request.completedItems.push(item),
          scheduleFlush(request),
          (cancelled = !0);
      else
        try {
          (taskModel = item.value),
            request.pendingChunks++,
            emitData(request, taskID, taskModel),
            scheduleFlush(request),
            reader.read().then(handleRequest, handleError);
        } catch (x$8) {
          handleError(x$8);
        }
}

function shouldConcatenate(firstStr, secondStr) {
  if (getLength(secondStr) !== getLength(firstStr)) {
    return false;
  }

  // ++ is right-associative
  // x ++ y ++ z --> x ++ (y ++ z)
  if (firstStr === "++") {
    return false;
  }

  // x == y == z --> (x == y) == z
  if (stringComparisonOperators[firstStr] && stringComparisonOperators[secondStr]) {
    return false;
  }

  // x + y % z --> (x + y) % z
  if (
    (secondStr === "%" && additiveOperators[firstStr]) ||
    (firstStr === "%" && additiveOperators[secondStr])
  ) {
    return false;
  }

  // x * y / z --> (x * y) / z
  // x / y * z --> (x / y) * z
  if (
    firstStr !== secondStr &&
    additiveOperators[secondStr] &&
    additiveOperators[firstStr]
  ) {
    return false;
  }

  // x << y << z --> (x << y) << z
  if (bitwiseShiftOperators[firstStr] && bitwiseShiftOperators[secondStr]) {
    return false;
  }

  return true;
}

        function isValidRegexForEcmaVersion(pattern, flags) {
            const validator = new RegExpValidator({ ecmaVersion: regexppEcmaVersion });

            try {
                validator.validatePattern(pattern, 0, pattern.length, {
                    unicode: flags ? flags.includes("u") : false,
                    unicodeSets: flags ? flags.includes("v") : false
                });
                if (flags) {
                    validator.validateFlags(flags);
                }
                return true;
            } catch {
                return false;
            }
        }

function extract(heap) {
  if (0 === heap.length) return null;
  var end = heap.pop(),
    start = heap[0],
    temp;
  if (end !== start) {
    heap[0] = end;
    for (
      let index = 0, halfLength = Math.floor(heap.length / 2); index < halfLength;

    ) {
      const leftIndex = 2 * (index + 1) - 1,
        left = heap[leftIndex],
        rightIndex = leftIndex + 1,
        right = heap[rightIndex];
      if (left > end)
        rightIndex < heap.length && right > end
          ? ((heap[index] = right),
            (heap[rightIndex] = end),
            (index = rightIndex))
          : ((heap[index] = left), (heap[leftIndex] = end), (index = leftIndex));
      else if (rightIndex < heap.length && right > end)
        (heap[index] = right), (heap[rightIndex] = end), (index = rightIndex);
      else break;
    }
  }
  return start;
}

function handleRequestCheck() {
  var defaultResult = null;
  if (!currentRequest) {
    if (supportsRequestStorage) {
      var store = requestStorage.getStore();
      if (store) {
        defaultResult = store;
      }
    }
  }
  return defaultResult;
}

export default function Index({ allPosts, preview }) {
  const heroPost = allPosts[0];
  const morePosts = allPosts.slice(1);
  return (
    <>
      <Layout preview={preview}>
        <Head>
          <title>{`Next.js Blog Example with ${CMS_NAME}`}</title>
        </Head>
        <Container>
          <Intro />
          {heroPost && (
            <HeroPost
              title={heroPost.title}
              coverImage={heroPost.coverImage}
              date={heroPost.date}
              author={heroPost.author}
              slug={heroPost.slug}
              excerpt={heroPost.excerpt}
            />
          )}
          {morePosts.length > 0 && <MoreStories posts={morePosts} />}
        </Container>
      </Layout>
    </>
  );
}

function serializeEntity(entity, entityId) {
    if ("object" === typeof entity && null !== entity) {
        const hexId = "$" + entityId.toString(16);
        writtenObjects.set(entity, hexId);
        temporaryReferences ? temporaryReferences.set(hexId, entity) : void 0;
    }
    modelRoot = entity;
    return JSON.stringify(entity, resolveToJson);
}

function createExports(exports) {
  return outdent`
    export {
      ${exports
        .map(({ specifier, variable }) =>
          variable === specifier ? specifier : `${variable} as ${specifier}`,
        )
        .map((line) => `  ${line},`)
        .join("\n")}
    };
  `;
}

function preinitScriptModule(src, opts) {
  if ("string" === typeof src) {
    var request = resolveRequest();
    if (request) {
      var hints = request.hints,
        key = "M|" + src;
      if (hints.has(key)) return;
      hints.add(key);
      return (opts = trimOptions(opts))
        ? emitHint(request, "M", [src, opts])
        : emitHint(request, "M", src);
    }
    previousDispatcher.M(src, opts);
  }
}

function dedentString(text, minIndent = getMinIndentation(text)) {
  return minIndent === 0
    ? text
    : text
        .split("\n")
        .map((lineText) => lineText.slice(minIndent))
        .join("\n");
}

    function getTaskName(type) {
      if (type === REACT_FRAGMENT_TYPE) return "<>";
      if (
        "object" === typeof type &&
        null !== type &&
        type.$$typeof === REACT_LAZY_TYPE
      )
        return "<...>";
      try {
        var name = getComponentNameFromType(type);
        return name ? "<" + name + ">" : "<...>";
      } catch (x) {
        return "<...>";
      }
    }

function eachSelfAssignment(left, right, props, report) {
    if (!left || !right) {

        // do nothing
    } else if (
        left.type === "Identifier" &&
        right.type === "Identifier" &&
        left.name === right.name
    ) {
        report(right);
    } else if (
        left.type === "ArrayPattern" &&
        right.type === "ArrayExpression"
    ) {
        const end = Math.min(left.elements.length, right.elements.length);

        for (let i = 0; i < end; ++i) {
            const leftElement = left.elements[i];
            const rightElement = right.elements[i];

            // Avoid cases such as [...a] = [...a, 1]
            if (
                leftElement &&
                leftElement.type === "RestElement" &&
                i < right.elements.length - 1
            ) {
                break;
            }

            eachSelfAssignment(leftElement, rightElement, props, report);

            // After a spread element, those indices are unknown.
            if (rightElement && rightElement.type === "SpreadElement") {
                break;
            }
        }
    } else if (
        left.type === "RestElement" &&
        right.type === "SpreadElement"
    ) {
        eachSelfAssignment(left.argument, right.argument, props, report);
    } else if (
        left.type === "ObjectPattern" &&
        right.type === "ObjectExpression" &&
        right.properties.length >= 1
    ) {

        /*
         * Gets the index of the last spread property.
         * It's possible to overwrite properties followed by it.
         */
        let startJ = 0;

        for (let i = right.properties.length - 1; i >= 0; --i) {
            const propType = right.properties[i].type;

            if (propType === "SpreadElement" || propType === "ExperimentalSpreadProperty") {
                startJ = i + 1;
                break;
            }
        }

        for (let i = 0; i < left.properties.length; ++i) {
            for (let j = startJ; j < right.properties.length; ++j) {
                eachSelfAssignment(
                    left.properties[i],
                    right.properties[j],
                    props,
                    report
                );
            }
        }
    } else if (
        left.type === "Property" &&
        right.type === "Property" &&
        right.kind === "init" &&
        !right.method
    ) {
        const leftName = astUtils.getStaticPropertyName(left);

        if (leftName !== null && leftName === astUtils.getStaticPropertyName(right)) {
            eachSelfAssignment(left.value, right.value, props, report);
        }
    } else if (
        props &&
        astUtils.skipChainExpression(left).type === "MemberExpression" &&
        astUtils.skipChainExpression(right).type === "MemberExpression" &&
        astUtils.isSameReference(left, right)
    ) {
        report(right);
    }
}

// top-level elements (excluding <template>, <style> and <script>) in Vue SFC are considered custom block
// See https://vue-loader.vuejs.org/spec.html for detail
const vueRootElementsSet = new Set(["template", "style", "script"]);
    function getValueDescriptorExpectingEnumForWarning(thing) {
      return null === thing
        ? "`null`"
        : void 0 === thing
          ? "`undefined`"
          : "" === thing
            ? "an empty string"
            : "string" === typeof thing
              ? JSON.stringify(thing)
              : "number" === typeof thing
                ? "`" + thing + "`"
                : 'something with type "' + typeof thing + '"';
    }

function generateOptionUsageSetting(context, setting, limit) {
  const header = setting.description;
  const defaultValue = getOptionDefaultValue(context, setting.name);
  const defaultInfo = defaultValue === undefined ? "" : `\nDefaults to ${createDefaultValueDisplay(defaultValue)}.`;
  return createOptionUsageRow(
    header,
    `${setting.description}${defaultInfo}`,
    limit,
  );
}

function registerClientReferenceImpl(proxyImplementation, id, async) {
  return Object.defineProperties(proxyImplementation, {
    $$typeof: { value: CLIENT_REFERENCE_TAG$1 },
    $$id: { value: id },
    $$async: { value: async }
  });
}

        function getCodePathStartScope(scope) {
            let target = scope;

            while (target) {
                if (codePathStartScopes.has(target)) {
                    return target;
                }
                target = target.upper;
            }

            // Should be unreachable
            return null;
        }

executor(function handleCancel(reason, configObj, requestInfo) {
  if (!reason.reason) {
    return;
  }

  reason.reason = new CanceledError("Operation canceled", configObj, requestInfo);
  resolvePromise(reason.reason);
});

export async function handlePageErrorEsm(taskConfig, configOptions) {
  await taskConfig
    .source('src/pages/_error.jsx')
    .swc('client', {
      dev: !configOptions.dev,
      esm: true,
    })
    .target('dist/esm/pages')

  const { dev } = configOptions;
  if (!dev) {
    console.log('Running in production mode');
  }
}

    function elementRefGetterWithDeprecationWarning() {
      var componentName = getComponentNameFromType(this.type);
      didWarnAboutElementRef[componentName] ||
        ((didWarnAboutElementRef[componentName] = !0),
        console.error(
          "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
        ));
      componentName = this.props.ref;
      return void 0 !== componentName ? componentName : null;
    }

export {
  canHaveInterpolation,
  dedentString,
  forceBreakChildren,
  forceBreakContent,
  forceNextEmptyLine,
  getLastDescendant,
  getLeadingAndTrailingHtmlWhitespace,
  getNodeCssStyleDisplay,
  getTextValueParts,
  getUnescapedAttributeValue,
  hasPrettierIgnore,
  htmlTrimPreserveIndentation,
  inferElementParser,
  isDanglingSpaceSensitiveNode,
  isIndentationSensitiveNode,
  isLeadingSpaceSensitiveNode,
  isPreLikeNode,
  isScriptLikeTag,
  isTextLikeNode,
  isTrailingSpaceSensitiveNode,
  isVueCustomBlock,
  isVueNonHtmlBlock,
  isVueScriptTag,
  isVueSfcBindingsAttribute,
  isVueSfcBlock,
  isVueSlotAttribute,
  isWhitespaceSensitiveNode,
  preferHardlineAsLeadingSpaces,
  shouldPreserveContent,
  unescapeQuoteEntities,
};
