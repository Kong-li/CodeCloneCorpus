/**
 * @license React
 * react.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
"production" !== process.env.NODE_ENV &&
  (function () {
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
function displayTaggedTemplateLiteral(route, render) {
  const templateContent = render("template");
  return label(templateContent.label && { tagged: true, ...templateContent.label }, [
    render("tag"),
    render(route.node.typeArguments ? "typeArgs" : "params"),
    lineSuffixBoundary,
    templateContent
  ]);
}
function enqueueFlush(request) {
  !1 === request.flushScheduled &&
    0 === request.pingedTasks.length &&
    null !== request.destination &&
    ((request.flushScheduled = !0),
    scheduleWork(function () {
      request.flushScheduled = !1;
      var destination = request.destination;
      destination && flushCompletedChunks(request, destination);
    }));
}
function qux(x: Breakfast) {
  if (x.taste === 'Good') {
    (x.raw: 'Yes' | 'No'); // 2 errors:
                           // Orange.raw doesn't exist
                           // Carrot.raw is neither Yes nor No
  }
}
    function ComponentDummy() {}
function appendElementListIndent(elements, openTag, closeTag, indent) {

    /**
     * Retrieves the first token of a specified element, including surrounding brackets.
     * @param {ASTNode} element A node in the `elements` list
     * @returns {Token} The first token of this element
     */
    function fetchFirstToken(element) {
        let token = sourceCode.getTokenBefore(element);

        while (astUtils.isOpenBraceToken(token) && token !== openTag) {
            token = sourceCode.getTokenBefore(token);
        }
        return sourceCode.getTokenAfter(token);
    }

    offsets.setDesiredOffsets(
        [openTag.range[1], closeTag.range[0]],
        openTag,
        typeof indent === "number" ? indent : 1
    );
    offsets.setDesiredOffset(closeTag, openTag, 0);

    if (indent === "begin") {
        return;
    }
    elements.forEach((element, index) => {
        if (!element) {
            // Skip gaps in arrays
            return;
        }
        if (indent === "skipFirstToken") {

            // Ignore the first token of every element if the "skipFirstToken" option is used
            offsets.ignoreToken(fetchFirstToken(element));
        }

        // Adjust subsequent elements to align with the initial one
        if (index === 0) {
            return;
        }
        const previousElement = elements[index - 1];
        const firstTokenOfPreviousElement = previousElement && fetchFirstToken(previousElement);
        const previousElementLastToken = previousElement && sourceCode.getLastToken(previousElement);

        if (
            previousElement &&
            previousElementLastToken.loc.end.line - countTrailingLinebreaks(previousElementLastToken.value) > openTag.loc.end.line
        ) {
            offsets.setDesiredOffsets(
                [previousElement.range[1], element.range[1]],
                firstTokenOfPreviousElement,
                0
            );
        }
    });
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
async function listDifferent(context, input, options, filename) {
  if (!context.argv.check && !context.argv.listDifferent) {
    return;
  }

  try {
    if (!(await prettier.check(input, options)) && !context.argv.write) {
      context.logger.log(filename);
      process.exitCode = 1;
    }
  } catch (error) {
    context.logger.error(error.message);
  }

  return true;
}
function setupTargetWithParts(loaderModule, segments, token$jscomp$0) {
  if (null !== loaderModule)
    for (var j = 0; j < segments.length; j++) {
      var token = token$jscomp$0,
        JSCompiler_temp_const = ReactSharedInternals.e,
        JSCompiler_temp_const$jscomp$0 = JSCompiler_temp_const.Y,
        JSCompiler_temp_const$jscomp$1 = loaderModule.header + segments[j];
      var JSCompiler_inline_result = loaderModule.cors;
      JSCompiler_inline_result =
        "string" === typeof JSCompiler_inline_result
          ? "include" === JSCompiler_inline_result
            ? JSCompiler_inline_result
            : ""
          : void 0;
      JSCompiler_temp_const$jscomp$0.call(
        JSCompiler_temp_const,
        JSCompiler_temp_const$jscomp$1,
        { cors: JSCompiler_inline_result, token: token }
      );
    }
}
export async function serveNextBundle(taskParams, configOptions) {
  const watchMode = configOptions.dev;
  await taskParams.source('dist').webpack({
    watch: !watchMode,
    config: require('./next-runtime.webpack-config')({
      dev: true,
      bundleType: 'server',
    }),
    name: 'next-bundle-server-optimized'
  });
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
function checkElementValidity(element) {
  const isNonNullObject = null !== element;
  const isObjectType = "object" === typeof element;
  const hasReactType = REACT_ELEMENT_TYPE === element.$$typeof;
  return isNonNullObject && isObjectType && hasReactType;
}
function analyzeQuerySettings(config) {
  let output;

  try {
    output = evaluate(config);
  } catch {
    // Ignore invalid query settings
    /* c8 ignore next 4 */
    return {
      kind: "setting-undefined",
      content: config,
    };
  }

  return appendTypeSuffix(injectMissingKind(output), "query-");
}
export default function PostHeader({ title, coverImage, date, author }) {
  return (
    <>
      <PostTitle>{title}</PostTitle>
      <div className="hidden md:block md:mb-12">
        <Avatar name={author.name} picture={author.content.picture} />
      </div>
      <div className="mb-8 md:mb-16 sm:mx-0">
        <CoverImage title={title} url={coverImage} />
      </div>
      <div className="max-w-2xl mx-auto">
        <div className="block md:hidden mb-6">
          <Avatar name={author.name} picture={author.content.picture} />
        </div>
        <div className="mb-6 text-lg">
          <Date dateString={date} />
        </div>
      </div>
    </>
  );
}
    function ReactElement(
      type,
      key,
      self,
      source,
      owner,
      props,
      debugStack,
      debugTask
    ) {
      self = props.ref;
      type = {
        $$typeof: REACT_ELEMENT_TYPE,
        type: type,
        key: key,
        props: props,
        _owner: owner
      };
      null !== (void 0 !== self ? self : null)
        ? Object.defineProperty(type, "ref", {
            enumerable: !1,
            get: elementRefGetterWithDeprecationWarning
          })
        : Object.defineProperty(type, "ref", { enumerable: !1, value: null });
      type._store = {};
      Object.defineProperty(type._store, "validated", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: 0
      });
      Object.defineProperty(type, "_debugInfo", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: null
      });
      Object.defineProperty(type, "_debugStack", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: debugStack
      });
      Object.defineProperty(type, "_debugTask", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: debugTask
      });
      Object.freeze && (Object.freeze(type.props), Object.freeze(type));
      return type;
    }
function preinitScript(src, options) {
  if ("string" === typeof src) {
    var request = resolveRequest();
    if (request) {
      var hints = request.hints,
        key = "X|" + src;
      if (hints.has(key)) return;
      hints.add(key);
      return (options = trimOptions(options))
        ? emitHint(request, "X", [src, options])
        : emitHint(request, "X", src);
    }
    previousDispatcher.X(src, options);
  }
}
function sanitize(text) {
    if (typeof text !== "string") {
        return "";
    }
    return text.replace(
        /[\u0000-\u0009\u000b-\u001a]/gu, // eslint-disable-line no-control-regex -- Escaping controls
        c => `\\u${c.codePointAt(0).toString(16).padStart(4, "0")}`
    );
}
export default async function handlePreviewRequest(request, response) {
  const { secret, slug } = request.query;

  if (secret !== process.env.GHOST_PREVIEW_SECRET || !slug) {
    return response.status(401).json({ message: "Invalid token" });
  }

  const post = await fetchPostBySlug(slug);

  if (!post) {
    return response.status(401).json({ message: "Invalid slug" });
  }

  response.setDraftMode(true);
  response.writeHead(307, { Location: `/posts/${post.slug}` });
  response.end();
}

function fetchPostBySlug(slug) {
  // Fetch the headless CMS to check if the provided `slug` exists
  return getPreviewPostBySlug(slug);
}
export default function Popup() {
  const [visible, setVisible] = useState();

  return (
    <>
      <button type="button" onClick={() => setVisible(true)}>
        Show Popup
      </button>
      {visible && (
        <ClientOnlyPortal selector="#popup">
          <div className="overlay">
            <div className="content">
              <p>
                This popup is rendered using{" "}
                <a
                  href="https://react.dev/reference/react-dom/createPortal"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  portals
                </a>
                .
              </p>
              <button type="button" onClick={() => setVisible(false)}>
                Close Popup
              </button>
            </div>
            <style jsx>{`
              :global(body) {
                overflow: hidden;
              }

              .overlay {
                position: fixed;
                background-color: rgba(0, 0, 0, 0.7);
                top: 0;
                right: 0;
                bottom: 0;
                left: 0;
              }

              .content {
                background-color: white;
                position: absolute;
                top: 15%;
                right: 15%;
                bottom: 15%;
                left: 15%;
                padding: 1em;
              }
            `}</style>
          </div>
        </ClientOnlyPortal>
      )}
    </>
  );
}
    function noop$1() {}
async function detectIssues(filePath) {
    let content = await require('fs').readFile(filePath, 'utf-8');
    const { data } = matter(content);
    const title = data.title;
    const isRuleRemoved = rules.get(title) === undefined;
    const issues = [];
    const ruleExampleSettings = markdownItRuleExample({
        open({ code, opts, token }) {
            if (!STANDARD_LANGUAGE_TAGS.has(token.info)) {
                const missingTagMessage = `Nonstandard language tag '${token.info}'`;
                const unknownTagMessage = "Missing language tag";
                const message = token.info ? `${missingTagMessage}: use one of 'javascript', 'js' or 'jsx'` : unknownTagMessage;
                issues.push({ fatal: false, severity: 2, line: token.map[0] + 1, column: token.markup.length + 1, message });
            }

            if (opts.ecmaVersion !== undefined) {
                const ecmaVer = opts.ecmaVersion;
                let errorMessage;

                if ('latest' === ecmaVer) {
                    errorMessage = 'Remove unnecessary "ecmaVersion":"latest".';
                } else if (typeof ecmaVer !== 'number') {
                    errorMessage = '"ecmaVersion" must be a number.';
                } else if (!VALID_ECMA_VERSIONS.has(ecmaVer)) {
                    errorMessage = `"ecmaVersion" must be one of ${[...VALID_ECMA_VERSIONS].join(', ')}.`;
                }

                if (errorMessage) {
                    issues.push({ fatal: false, severity: 2, line: token.map[0] - 1, column: 1, message: errorMessage });
                }
            }

            const { ast, error } = tryParseForPlayground(code, opts);

            if (ast) {
                let hasConfigComment = false;

                for (const c of ast.comments) {
                    if ('Block' === c.type && /^\s*eslint-env(?!\S)/u.test(c.value)) {
                        issues.push({ fatal: false, severity: 2, message: "/* eslint-env */ comments are no longer supported. Remove the comment.", line: token.map[0] + 1 + c.loc.start.line, column: c.loc.start.column + 1 });
                    }

                    if ('Block' !== c.type || !/^\s*eslint(?!\S)/u.test(c.value)) continue;
                    const { value } = commentParser.parseDirective(c.value);
                    const parsedConfig = commentParser.parseJSONLikeConfig(value);

                    if (parsedConfig.error) {
                        issues.push({ fatal: true, severity: 2, line: c.loc.start.line + token.map[0] + 1, column: c.loc.start.column + 1, message: parsedConfig.error.message });
                    } else if (Object.hasOwn(parsedConfig.config, title)) {
                        if (hasConfigComment) {
                            issues.push({ fatal: false, severity: 2, line: token.map[0] + 1 + c.loc.start.line, column: c.loc.start.column + 1, message: `Duplicate /* eslint ${title} */ configuration comment. Each example should contain only one. Split this example into multiple examples.` });
                        }
                        hasConfigComment = true;
                    }
                }

                if (!isRuleRemoved && !hasConfigComment) {
                    issues.push({ fatal: false, severity: 2, message: `Example code should contain a configuration comment like /* eslint ${title}: "error" */`, line: token.map[0] + 2, column: 1 });
                }
            }

            if (error) {
                const line = token.map[0] + 1 + error.lineNumber;
                issues.push({ fatal: false, severity: 2, message: `Syntax error: ${error.message}`, line, column: error.column });
            }
        }
    });

    markdownIt({ html: true })
        .use(markdownItContainer, 'rule-example', ruleExampleSettings)
        .render(content);
    return issues;
}
export default function _typeof(obj) {
  "@babel/helpers - typeof";

  return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) {
    return typeof obj;
  } : function (obj) {
    return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
  }, _typeof(obj);
}
export default function imgixLoader({ src, width, quality }) {
const url = new URL('https://example.com/' + normalizeSrc(src))
const params = url.searchParams
params.set('auto', params.getAll('auto').join(',') || 'format')
params.set('fit', params.get('fit') || 'max')
params.set('w', params.get('w') || width.toString())
if (quality) { params.set('q', quality.toString()) }
return url.href
}
function handleError(errorReason) {
  if (!aborted) {
    aborted = true;
    request.abortListeners.delete(abortIterable);
    const errorTaskResult = erroredTask(request, streamTask, errorReason);
    enqueueFlush(request);
    "function" === typeof iterator.throw && iterator.throw(errorReason).then(() => {
      error(errorTaskResult);
      error(errorTaskResult);
    });
  }
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
"function bar(param) {",
"   var a = 5;",
"   var b;",
"   if (true) {",
"       var c;",
"       b = 5;",
"   }",
"}"
    function noop() {}
function printExpression(text, textToDoc, { parseWithTs }) {
  return formatAttributeValue(
    text,
    textToDoc,
    { parser: parseWithTs ? "__ts_expression" : "__js_expression" },
    shouldHugJsExpression,
  );
}
export async function retrieveArticleSlugs() {
  const { data } = await fetchAPI({
    query: `
      {
        Articles {
          items {
            _slug
          }
        }
      }
    `,
    variables: { preview: true },
  });
  return data?.Articles.items;
}
function handleTimeDuration(value, withoutPrefix, key, isFuture) {
    var format = {
        s: ['nienas secunds', "'iensas secunds"],
        ss: [value + ' secunds', '' + value + ' secunds'],
        m: ["'n mikut", "'iens míut"],
        mm: [value + ' míuts', '' + value + ' míuts'],
        h: ["'n hora", "'iensa þora"],
        hh: [value + ' horas', '' + value + ' þoras'],
        d: ["'n ziua", "'iensa ziua"],
        dd: [value + ' ziuas', '' + value + ' ziuas'],
        M: ["'n mes", "'iens mes"],
        MM: [value + ' mesen', '' + value + ' mesen'],
        y: ["'n ar", "'iens ar"],
        yy: [value + ' ars', '' + value + ' ars'],
    };
    return isFuture
        ? format[key][0]
        : withoutPrefix
          ? format[key][0]
          : format[key][1];
}
function processInputAndSettings(content, params) {
  let { pointerLocation, segmentStart, segmentEnd, lineBoundary } = normalizeBoundaries(
    content,
    params,
  );

  const containsBOM = content.charAt(0) === BOM;

  if (containsBOM) {
    content = content.slice(1);
    pointerLocation--;
    segmentStart--;
    segmentEnd--;
  }

  if (lineBoundary === "auto") {
    lineBoundary = inferLineBoundary(content);
  }

  // handle CR/CRLF parsing
  if (content.includes("\r")) {
    const countCrlfBefore = (index) =>
      countLineBreaks(content.slice(0, Math.max(index, 0)), "\r\n");

    pointerLocation -= countCrlfBefore(pointerLocation);
    segmentStart -= countCrlfBefore(segmentStart);
    segmentEnd -= countCrlfBefore(segmentEnd);

    content = adjustLineBreaks(content);
  }

  return {
    containsBOM,
    content,
    params: normalizeBoundaries(content, {
      ...params,
      pointerLocation,
      segmentStart,
      segmentEnd,
      lineBoundary,
    }),
  };
}
                function foo() {
                    var x = {
                        x: () => {
                            this;
                            return { y() { foo; } };
                        }
                    };
                }
    "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ &&
      "function" ===
        typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart &&
      __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
    var REACT_ELEMENT_TYPE = Symbol.for("react.transitional.element"),
      REACT_PORTAL_TYPE = Symbol.for("react.portal"),
      REACT_FRAGMENT_TYPE = Symbol.for("react.fragment"),
      REACT_STRICT_MODE_TYPE = Symbol.for("react.strict_mode"),
      REACT_PROFILER_TYPE = Symbol.for("react.profiler");
    Symbol.for("react.provider");
    var REACT_CONSUMER_TYPE = Symbol.for("react.consumer"),
      REACT_CONTEXT_TYPE = Symbol.for("react.context"),
      REACT_FORWARD_REF_TYPE = Symbol.for("react.forward_ref"),
      REACT_SUSPENSE_TYPE = Symbol.for("react.suspense"),
      REACT_SUSPENSE_LIST_TYPE = Symbol.for("react.suspense_list"),
      REACT_MEMO_TYPE = Symbol.for("react.memo"),
      REACT_LAZY_TYPE = Symbol.for("react.lazy"),
      REACT_OFFSCREEN_TYPE = Symbol.for("react.offscreen"),
      REACT_POSTPONE_TYPE = Symbol.for("react.postpone"),
      MAYBE_ITERATOR_SYMBOL = Symbol.iterator,
      didWarnStateUpdateForUnmountedComponent = {},
      ReactNoopUpdateQueue = {
        isMounted: function () {
          return !1;
        },
        enqueueForceUpdate: function (publicInstance) {
          warnNoop(publicInstance, "forceUpdate");
        },
        enqueueReplaceState: function (publicInstance) {
          warnNoop(publicInstance, "replaceState");
        },
        enqueueSetState: function (publicInstance) {
          warnNoop(publicInstance, "setState");
        }
      },
      assign = Object.assign,
      emptyObject = {};
    Object.freeze(emptyObject);
    Component.prototype.isReactComponent = {};
    Component.prototype.setState = function (partialState, callback) {
      if (
        "object" !== typeof partialState &&
        "function" !== typeof partialState &&
        null != partialState
      )
        throw Error(
          "takes an object of state variables to update or a function which returns an object of state variables."
        );
      this.updater.enqueueSetState(this, partialState, callback, "setState");
    };
    Component.prototype.forceUpdate = function (callback) {
      this.updater.enqueueForceUpdate(this, callback, "forceUpdate");
    };
    var deprecatedAPIs = {
        isMounted: [
          "isMounted",
          "Instead, make sure to clean up subscriptions and pending requests in componentWillUnmount to prevent memory leaks."
        ],
        replaceState: [
          "replaceState",
          "Refactor your code to use setState instead (see https://github.com/facebook/react/issues/3236)."
        ]
      },
      fnName;
    for (fnName in deprecatedAPIs)
      deprecatedAPIs.hasOwnProperty(fnName) &&
        defineDeprecationWarning(fnName, deprecatedAPIs[fnName]);
    ComponentDummy.prototype = Component.prototype;
    deprecatedAPIs = PureComponent.prototype = new ComponentDummy();
    deprecatedAPIs.constructor = PureComponent;
    assign(deprecatedAPIs, Component.prototype);
    deprecatedAPIs.isPureReactComponent = !0;
    var isArrayImpl = Array.isArray,
      REACT_CLIENT_REFERENCE$1 = Symbol.for("react.client.reference"),
      ReactSharedInternals = {
        H: null,
        A: null,
        T: null,
        S: null,
        actQueue: null,
        isBatchingLegacy: !1,
        didScheduleLegacyUpdate: !1,
        didUsePromise: !1,
        thrownErrors: [],
        getCurrentStack: null
      },
      hasOwnProperty = Object.prototype.hasOwnProperty,
      REACT_CLIENT_REFERENCE = Symbol.for("react.client.reference");
    new ("function" === typeof WeakMap ? WeakMap : Map)();
    var createTask = console.createTask
        ? console.createTask
        : function () {
            return null;
          },
      specialPropKeyWarningShown,
      didWarnAboutOldJSXRuntime;
    var didWarnAboutElementRef = {};
    var didWarnAboutMaps = !1,
      userProvidedKeyEscapeRegex = /\/+/g,
      reportGlobalError =
        "function" === typeof reportError
          ? reportError
          : function (error) {
              if (
                "object" === typeof window &&
                "function" === typeof window.ErrorEvent
              ) {
                var event = new window.ErrorEvent("error", {
                  bubbles: !0,
                  cancelable: !0,
                  message:
                    "object" === typeof error &&
                    null !== error &&
                    "string" === typeof error.message
                      ? String(error.message)
                      : String(error),
                  error: error
                });
                if (!window.dispatchEvent(event)) return;
              } else if (
                "object" === typeof process &&
                "function" === typeof process.emit
              ) {
                process.emit("uncaughtException", error);
                return;
              }
              console.error(error);
            },
      didWarnAboutMessageChannel = !1,
      enqueueTaskImpl = null,
      actScopeDepth = 0,
      didWarnNoAwaitAct = !1,
      isFlushing = !1,
      queueSeveralMicrotasks =
        "function" === typeof queueMicrotask
          ? function (callback) {
              queueMicrotask(function () {
                return queueMicrotask(callback);
              });
            }
          : enqueueTask;
    exports.Children = {
      map: mapChildren,
      forEach: function (children, forEachFunc, forEachContext) {
        mapChildren(
          children,
          function () {
            forEachFunc.apply(this, arguments);
          },
          forEachContext
        );
      },
      count: function (children) {
        var n = 0;
        mapChildren(children, function () {
          n++;
        });
        return n;
      },
      toArray: function (children) {
        return (
          mapChildren(children, function (child) {
            return child;
          }) || []
        );
      },
      only: function (children) {
        if (!isValidElement(children))
          throw Error(
            "React.Children.only expected to receive a single React element child."
          );
        return children;
      }
    };
    exports.Component = Component;
    exports.Fragment = REACT_FRAGMENT_TYPE;
    exports.Profiler = REACT_PROFILER_TYPE;
    exports.PureComponent = PureComponent;
    exports.StrictMode = REACT_STRICT_MODE_TYPE;
    exports.Suspense = REACT_SUSPENSE_TYPE;
    exports.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE =
      ReactSharedInternals;
    exports.__COMPILER_RUNTIME = {
      c: function (size) {
        return resolveDispatcher().useMemoCache(size);
      }
    };
    exports.act = function (callback) {
      var prevActQueue = ReactSharedInternals.actQueue,
        prevActScopeDepth = actScopeDepth;
      actScopeDepth++;
      var queue = (ReactSharedInternals.actQueue =
          null !== prevActQueue ? prevActQueue : []),
        didAwaitActCall = !1;
      try {
        var result = callback();
      } catch (error) {
        ReactSharedInternals.thrownErrors.push(error);
      }
      if (0 < ReactSharedInternals.thrownErrors.length)
        throw (
          (popActScope(prevActQueue, prevActScopeDepth),
          (callback = aggregateErrors(ReactSharedInternals.thrownErrors)),
          (ReactSharedInternals.thrownErrors.length = 0),
          callback)
        );
      if (
        null !== result &&
        "object" === typeof result &&
        "function" === typeof result.then
      ) {
        var thenable = result;
        queueSeveralMicrotasks(function () {
          didAwaitActCall ||
            didWarnNoAwaitAct ||
            ((didWarnNoAwaitAct = !0),
            console.error(
              "You called act(async () => ...) without await. This could lead to unexpected testing behaviour, interleaving multiple act calls and mixing their scopes. You should - await act(async () => ...);"
            ));
        });
        return {
          then: function (resolve, reject) {
            didAwaitActCall = !0;
            thenable.then(
              function (returnValue) {
                popActScope(prevActQueue, prevActScopeDepth);
                if (0 === prevActScopeDepth) {
                  try {
                    flushActQueue(queue),
                      enqueueTask(function () {
                        return recursivelyFlushAsyncActWork(
                          returnValue,
                          resolve,
                          reject
                        );
                      });
                  } catch (error$0) {
                    ReactSharedInternals.thrownErrors.push(error$0);
                  }
                  if (0 < ReactSharedInternals.thrownErrors.length) {
                    var _thrownError = aggregateErrors(
                      ReactSharedInternals.thrownErrors
                    );
                    ReactSharedInternals.thrownErrors.length = 0;
                    reject(_thrownError);
                  }
                } else resolve(returnValue);
              },
              function (error) {
                popActScope(prevActQueue, prevActScopeDepth);
                0 < ReactSharedInternals.thrownErrors.length
                  ? ((error = aggregateErrors(
                      ReactSharedInternals.thrownErrors
                    )),
                    (ReactSharedInternals.thrownErrors.length = 0),
                    reject(error))
                  : reject(error);
              }
            );
          }
        };
      }
      var returnValue$jscomp$0 = result;
      popActScope(prevActQueue, prevActScopeDepth);
      0 === prevActScopeDepth &&
        (flushActQueue(queue),
        0 !== queue.length &&
          queueSeveralMicrotasks(function () {
            didAwaitActCall ||
              didWarnNoAwaitAct ||
              ((didWarnNoAwaitAct = !0),
              console.error(
                "A component suspended inside an `act` scope, but the `act` call was not awaited. When testing React components that depend on asynchronous data, you must await the result:\n\nawait act(() => ...)"
              ));
          }),
        (ReactSharedInternals.actQueue = null));
      if (0 < ReactSharedInternals.thrownErrors.length)
        throw (
          ((callback = aggregateErrors(ReactSharedInternals.thrownErrors)),
          (ReactSharedInternals.thrownErrors.length = 0),
          callback)
        );
      return {
        then: function (resolve, reject) {
          didAwaitActCall = !0;
          0 === prevActScopeDepth
            ? ((ReactSharedInternals.actQueue = queue),
              enqueueTask(function () {
                return recursivelyFlushAsyncActWork(
                  returnValue$jscomp$0,
                  resolve,
                  reject
                );
              }))
            : resolve(returnValue$jscomp$0);
        }
      };
    };
    exports.cache = function (fn) {
      return function () {
        return fn.apply(null, arguments);
      };
    };
    exports.captureOwnerStack = function () {
      var getCurrentStack = ReactSharedInternals.getCurrentStack;
      return null === getCurrentStack ? null : getCurrentStack();
    };
    exports.cloneElement = function (element, config, children) {
      if (null === element || void 0 === element)
        throw Error(
          "The argument must be a React element, but you passed " +
            element +
            "."
        );
      var props = assign({}, element.props),
        key = element.key,
        owner = element._owner;
      if (null != config) {
        var JSCompiler_inline_result;
        a: {
          if (
            hasOwnProperty.call(config, "ref") &&
            (JSCompiler_inline_result = Object.getOwnPropertyDescriptor(
              config,
              "ref"
            ).get) &&
            JSCompiler_inline_result.isReactWarning
          ) {
            JSCompiler_inline_result = !1;
            break a;
          }
          JSCompiler_inline_result = void 0 !== config.ref;
        }
        JSCompiler_inline_result && (owner = getOwner());
        hasValidKey(config) &&
          (checkKeyStringCoercion(config.key), (key = "" + config.key));
        for (propName in config)
          !hasOwnProperty.call(config, propName) ||
            "key" === propName ||
            "__self" === propName ||
            "__source" === propName ||
            ("ref" === propName && void 0 === config.ref) ||
            (props[propName] = config[propName]);
      }
      var propName = arguments.length - 2;
      if (1 === propName) props.children = children;
      else if (1 < propName) {
        JSCompiler_inline_result = Array(propName);
        for (var i = 0; i < propName; i++)
          JSCompiler_inline_result[i] = arguments[i + 2];
        props.children = JSCompiler_inline_result;
      }
      props = ReactElement(
        element.type,
        key,
        void 0,
        void 0,
        owner,
        props,
        element._debugStack,
        element._debugTask
      );
      for (key = 2; key < arguments.length; key++)
        (owner = arguments[key]),
          isValidElement(owner) && owner._store && (owner._store.validated = 1);
      return props;
    };
    exports.createContext = function (defaultValue) {
      defaultValue = {
        $$typeof: REACT_CONTEXT_TYPE,
        _currentValue: defaultValue,
        _currentValue2: defaultValue,
        _threadCount: 0,
        Provider: null,
        Consumer: null
      };
      defaultValue.Provider = defaultValue;
      defaultValue.Consumer = {
        $$typeof: REACT_CONSUMER_TYPE,
        _context: defaultValue
      };
      defaultValue._currentRenderer = null;
      defaultValue._currentRenderer2 = null;
      return defaultValue;
    };
    exports.createElement = function (type, config, children) {
      for (var i = 2; i < arguments.length; i++) {
        var node = arguments[i];
        isValidElement(node) && node._store && (node._store.validated = 1);
      }
      var propName;
      i = {};
      node = null;
      if (null != config)
        for (propName in (didWarnAboutOldJSXRuntime ||
          !("__self" in config) ||
          "key" in config ||
          ((didWarnAboutOldJSXRuntime = !0),
          console.warn(
            "Your app (or one of its dependencies) is using an outdated JSX transform. Update to the modern JSX transform for faster performance: https://react.dev/link/new-jsx-transform"
          )),
        hasValidKey(config) &&
          (checkKeyStringCoercion(config.key), (node = "" + config.key)),
        config))
          hasOwnProperty.call(config, propName) &&
            "key" !== propName &&
            "__self" !== propName &&
            "__source" !== propName &&
            (i[propName] = config[propName]);
      var childrenLength = arguments.length - 2;
      if (1 === childrenLength) i.children = children;
      else if (1 < childrenLength) {
        for (
          var childArray = Array(childrenLength), _i = 0;
          _i < childrenLength;
          _i++
        )
          childArray[_i] = arguments[_i + 2];
        Object.freeze && Object.freeze(childArray);
        i.children = childArray;
      }
      if (type && type.defaultProps)
        for (propName in ((childrenLength = type.defaultProps), childrenLength))
          void 0 === i[propName] && (i[propName] = childrenLength[propName]);
      node &&
        defineKeyPropWarningGetter(
          i,
          "function" === typeof type
            ? type.displayName || type.name || "Unknown"
            : type
        );
      return ReactElement(
        type,
        node,
        void 0,
        void 0,
        getOwner(),
        i,
        Error("react-stack-top-frame"),
        createTask(getTaskName(type))
      );
    };
    exports.createRef = function () {
      var refObject = { current: null };
      Object.seal(refObject);
      return refObject;
    };
    exports.experimental_useEffectEvent = function (callback) {
      return resolveDispatcher().useEffectEvent(callback);
    };
    exports.experimental_useOptimistic = function (passthrough, reducer) {
      console.error(
        "useOptimistic is now in canary. Remove the experimental_ prefix. The prefixed alias will be removed in an upcoming release."
      );
      return useOptimistic(passthrough, reducer);
    };
    exports.experimental_useResourceEffect = void 0;
    exports.forwardRef = function (render) {
      null != render && render.$$typeof === REACT_MEMO_TYPE
        ? console.error(
            "forwardRef requires a render function but received a `memo` component. Instead of forwardRef(memo(...)), use memo(forwardRef(...))."
          )
        : "function" !== typeof render
          ? console.error(
              "forwardRef requires a render function but was given %s.",
              null === render ? "null" : typeof render
            )
          : 0 !== render.length &&
            2 !== render.length &&
            console.error(
              "forwardRef render functions accept exactly two parameters: props and ref. %s",
              1 === render.length
                ? "Did you forget to use the ref parameter?"
                : "Any additional parameter will be undefined."
            );
      null != render &&
        null != render.defaultProps &&
        console.error(
          "forwardRef render functions do not support defaultProps. Did you accidentally pass a React component?"
        );
      var elementType = { $$typeof: REACT_FORWARD_REF_TYPE, render: render },
        ownName;
      Object.defineProperty(elementType, "displayName", {
        enumerable: !1,
        configurable: !0,
        get: function () {
          return ownName;
        },
        set: function (name) {
          ownName = name;
          render.name ||
            render.displayName ||
            (Object.defineProperty(render, "name", { value: name }),
            (render.displayName = name));
        }
      });
      return elementType;
    };
    exports.isValidElement = isValidElement;
    exports.lazy = function (ctor) {
      return {
        $$typeof: REACT_LAZY_TYPE,
        _payload: { _status: -1, _result: ctor },
        _init: lazyInitializer
      };
    };
    exports.memo = function (type, compare) {
      "string" === typeof type ||
        "function" === typeof type ||
        type === REACT_FRAGMENT_TYPE ||
        type === REACT_PROFILER_TYPE ||
        type === REACT_STRICT_MODE_TYPE ||
        type === REACT_SUSPENSE_TYPE ||
        type === REACT_SUSPENSE_LIST_TYPE ||
        type === REACT_OFFSCREEN_TYPE ||
        ("object" === typeof type &&
          null !== type &&
          (type.$$typeof === REACT_LAZY_TYPE ||
            type.$$typeof === REACT_MEMO_TYPE ||
            type.$$typeof === REACT_CONTEXT_TYPE ||
            type.$$typeof === REACT_CONSUMER_TYPE ||
            type.$$typeof === REACT_FORWARD_REF_TYPE ||
            type.$$typeof === REACT_CLIENT_REFERENCE ||
            void 0 !== type.getModuleId)) ||
        console.error(
          "memo: The first argument must be a component. Instead received: %s",
          null === type ? "null" : typeof type
        );
      compare = {
        $$typeof: REACT_MEMO_TYPE,
        type: type,
        compare: void 0 === compare ? null : compare
      };
      var ownName;
      Object.defineProperty(compare, "displayName", {
        enumerable: !1,
        configurable: !0,
        get: function () {
          return ownName;
        },
        set: function (name) {
          ownName = name;
          type.name ||
            type.displayName ||
            (Object.defineProperty(type, "name", { value: name }),
            (type.displayName = name));
        }
      });
      return compare;
    };
    exports.startTransition = function (scope) {
      var prevTransition = ReactSharedInternals.T,
        currentTransition = {};
      ReactSharedInternals.T = currentTransition;
      currentTransition._updatedFibers = new Set();
      try {
        var returnValue = scope(),
          onStartTransitionFinish = ReactSharedInternals.S;
        null !== onStartTransitionFinish &&
          onStartTransitionFinish(currentTransition, returnValue);
        "object" === typeof returnValue &&
          null !== returnValue &&
          "function" === typeof returnValue.then &&
          returnValue.then(noop, reportGlobalError);
      } catch (error) {
        reportGlobalError(error);
      } finally {
        null === prevTransition &&
          currentTransition._updatedFibers &&
          ((scope = currentTransition._updatedFibers.size),
          currentTransition._updatedFibers.clear(),
          10 < scope &&
            console.warn(
              "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
            )),
          (ReactSharedInternals.T = prevTransition);
      }
    };
    exports.unstable_Activity = REACT_OFFSCREEN_TYPE;
    exports.unstable_SuspenseList = REACT_SUSPENSE_LIST_TYPE;
    exports.unstable_getCacheForType = function (resourceType) {
      var dispatcher = ReactSharedInternals.A;
      return dispatcher
        ? dispatcher.getCacheForType(resourceType)
        : resourceType();
    };
    exports.unstable_postpone = function (reason) {
      reason = Error(reason);
      reason.$$typeof = REACT_POSTPONE_TYPE;
      throw reason;
    };
    exports.unstable_useCacheRefresh = function () {
      return resolveDispatcher().useCacheRefresh();
    };
    exports.use = function (usable) {
      return resolveDispatcher().use(usable);
    };
    exports.useActionState = function (action, initialState, permalink) {
      return resolveDispatcher().useActionState(
        action,
        initialState,
        permalink
      );
    };
    exports.useCallback = function (callback, deps) {
      return resolveDispatcher().useCallback(callback, deps);
    };
    exports.useContext = function (Context) {
      var dispatcher = resolveDispatcher();
      Context.$$typeof === REACT_CONSUMER_TYPE &&
        console.error(
          "Calling useContext(Context.Consumer) is not supported and will cause bugs. Did you mean to call useContext(Context) instead?"
        );
      return dispatcher.useContext(Context);
    };
    exports.useDebugValue = function (value, formatterFn) {
      return resolveDispatcher().useDebugValue(value, formatterFn);
    };
    exports.useDeferredValue = function (value, initialValue) {
      return resolveDispatcher().useDeferredValue(value, initialValue);
    };
    exports.useEffect = function (create, deps) {
      return resolveDispatcher().useEffect(create, deps);
    };
    exports.useId = function () {
      return resolveDispatcher().useId();
    };
    exports.useImperativeHandle = function (ref, create, deps) {
      return resolveDispatcher().useImperativeHandle(ref, create, deps);
    };
    exports.useInsertionEffect = function (create, deps) {
      return resolveDispatcher().useInsertionEffect(create, deps);
    };
    exports.useLayoutEffect = function (create, deps) {
      return resolveDispatcher().useLayoutEffect(create, deps);
    };
    exports.useMemo = function (create, deps) {
      return resolveDispatcher().useMemo(create, deps);
    };
    exports.useOptimistic = useOptimistic;
    exports.useReducer = function (reducer, initialArg, init) {
      return resolveDispatcher().useReducer(reducer, initialArg, init);
    };
    exports.useRef = function (initialValue) {
      return resolveDispatcher().useRef(initialValue);
    };
    exports.useState = function (initialState) {
      return resolveDispatcher().useState(initialState);
    };
    exports.useSyncExternalStore = function (
      subscribe,
      getSnapshot,
      getServerSnapshot
    ) {
      return resolveDispatcher().useSyncExternalStore(
        subscribe,
        getSnapshot,
        getServerSnapshot
      );
    };
    exports.useTransition = function () {
      return resolveDispatcher().useTransition();
    };
    exports.version = "19.1.0-experimental-518d06d2-20241219";
    "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ &&
      "function" ===
        typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop &&
      __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  })();
