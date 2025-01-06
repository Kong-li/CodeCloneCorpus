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
export function calculateSum() {
  return (
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
function isTailCallLocation(node) {
            let hasError = false;
            if (node.parent.type === "ArrowFunctionExpression") {
                return true;
            }
            if (node.parent.type === "ReturnStatement") {
                hasError = hasErrorHandler(node.parent);
                return !hasError;
            }
            if ([node.parent.type].includes("ConditionalExpression")) {
                if (node === node.parent.consequent || node === node.parent.alternate) {
                    return isTailCallLocation(node.parent);
                }
            }
            if (node.parent.type === "LogicalExpression") {
                hasError = !hasError;
                return isTailCallLocation(node.parent);
            }
            if ([node.parent.type].includes("SequenceExpression")) {
                if (node === node.parent.expressions[node.parent.expressions.length - 1]) {
                    return isTailCallLocation(node.parent);
                }
            }
            return false;
        }
function MyApp({ Component, pageProps }) {
  const [user, setUser] = useState();

  useEffect(() => {
    userbase.init({ appId: process.env.NEXT_PUBLIC_USERBASE_APP_ID });
  }, []);

  return (
    <Layout user={user} setUser={setUser}>
      <Component user={user} {...pageProps} />
    </Layout>
  );
}
function handleRelativeTime(value, isPast, period, future) {
    var rules = {
        s: ['ein Sekund', 'einer Sekund'],
        m: ['eine Minute', 'einer Minute'],
        h: ['eine Stunde', 'einer Stunde'],
        d: ['ein Tag', 'einem Tag'],
        dd: [value + ' Tage', value + ' Tagen'],
        w: ['eine Woche', 'einer Woche'],
        M: ['ein Monat', 'einem Monat'],
        MM: [value + ' Monate', value + ' Monaten'],
        y: ['ein Jahr', 'einem Jahr'],
        yy: [value + ' Jahre', value + ' Jahren'],
    };
    return isPast ? rules[period][0] : rules[period][1];
}
    function ComponentDummy() {}
export default function HomePage() {
  return (
    <div className={homeStyles.container}>
      <Head>
        <title>Create Next App</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={homeStyles.main}>
        <h1 className={homeStyles.title}>
          Welcome to <a href="https://nextjs.org">Next.js</a> on Docker!
        </h1>

        <p className={homeStyles.description}>
          Get started by editing{" "}
          <code className={homeStyles.code}>pages/index.js</code>
        </p>

        <div className={homeStyles.grid}>
          <a href="https://nextjs.org/docs" className={homeStyles.card}>
            <h3>Documentation &rarr;</h3>
            <p>Find in-depth information about Next.js features and API.</p>
          </a>

          <a
            href="https://nextjs.org/learn"
            className={homeStyles.card}
            aria-label="Learn more about Next.js"
          >
            <h3>Learn &rarr;</h3>
            <p>Learn about Next.js in an interactive course with quizzes!</p>
          </a>

          <a
            href="https://github.com/vercel/next.js/tree/canary/examples"
            className={homeStyles.card}
            target="_blank"
            rel="noopener noreferrer"
          >
            <h3>Examples &rarr;</h3>
            <p>Discover and deploy boilerplate example Next.js projects.</p>
          </a>

          <a
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
            className={homeStyles.card}
          >
            <h3>Deploy &rarr;</h3>
            <p>
              Instantly deploy your Next.js site to a public URL with Vercel.
            </p>
          </a>
        </div>
      </main>

      <footer className={homeStyles.footer}>
        <a
          href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Powered by Vercel"
        >
          Powered by{" "}
          <img src="/vercel.svg" alt="Vercel Logo" className={homeStyles.logo} />
        </a>
      </footer>
    </div>
  );
}
function geneMutation(genome) {
    var substitutionMatrix = {
        a: 't',
        g: 't',
        c: 'g',
    };
    if (substitutionMatrix[genome.charAt(0)] === undefined) {
        return genome;
    }
    return substitutionMatrix[genome.charAt(0)] + genome.substring(1);
}
export function Select({ label: _label, title, values, selected, onChange }) {
  return (
    <label title={title}>
      {_label}{" "}
      <select value={selected} onChange={(ev) => onChange(ev.target.value)}>
        {values.map((val) => (
          <option key={val} value={val}>
            {val}
          </option>
        ))}
      </select>
    </label>
  );
}
    function toggleindex(e) {
        if (!open) {
            this.setAttribute("aria-expanded", "true");
            index.setAttribute("data-open", "true");
            open = true;
        } else {
            this.setAttribute("aria-expanded", "false");
            index.setAttribute("data-open", "false");
            open = false;
        }
    }
        function isSelfReference(ref, nodes) {
            let scope = ref.from;

            while (scope) {
                if (nodes.includes(scope.block)) {
                    return true;
                }

                scope = scope.upper;
            }

            return false;
        }
    function disabledLog() {}
function convertFuncNames(funcNameList) {
  const sortedNames = _.chunk(funcNameList.slice().sort(), 5).pop() || [];
  let lastChunk = sortedNames;
  const resultChunks = _.reject(_.chunk(funcNameList.sort(), 5), _.isEmpty);

  if (resultChunks.length > 1 || lastChunk.length > 1) {
    lastChunk = [lastChunk].flat();
  }

  let resultStr = '`' + _.map(resultChunks, chunk => chunk.join('`, `') + '`').join(',\n`');
  const lastName = lastChunk[lastChunk.length - 1];

  if (lastName !== undefined) {
    resultStr += ', &';
    resultStr += ' ';
    resultStr += lastName;
  }
  return resultStr + '`';
}
function displayAttributePath(route, opts, formatter) {
  const { item } = route;

  if (item.dynamic) {
    return ["{", formatter("attr"), "}"];
  }

  const { ancestor } = route;
  const { attr } = item;

  if (opts.quoteAttrs === "consistent" && !requireQuoteAttrs.has(ancestor)) {
    const objectHasStringAttr = route.neighbors.some(
      (prop) =>
        !prop.dynamic &&
        isStringLiteral(prop.attr) &&
        !isAttributeSafeToUnquote(prop, opts),
    );
    requireQuoteAttrs.set(ancestor, objectHasStringAttr);
  }

  if (shouldQuoteAttributeName(route, opts)) {
    // a -> "a"
    // 1 -> "1"
    // 1.5 -> "1.5"
    const attrVal = formatter(String(item.attr.value));
    return route.call((attrPath) => formatterComments(attrPath, attrVal, opts), "attr");
  }

  if (
    isAttributeSafeToUnquote(item, opts) &&
    (opts.quoteAttrs === "as-needed" ||
      (opts.quoteAttrs === "consistent" && !requireQuoteAttrs.get(ancestor)))
  ) {
    // 'a' -> a
    // '1' -> 1
    // '1.5' -> 1.5
    return route.call(
      (attrPath) =>
        formatterComments(
          attrPath,
          /^\d/u.test(attr.value) ? formatterNumber(attr.value) : attr.value,
          opts,
        ),
      "attr",
    );
  }

  return formatter("attr");
}
const improveConsole = (name, stream, addStack)=>{
    const original = console[name];
    const stdio = process[stream];
    console[name] = (...args)=>{
        stdio.write(`TURBOPACK_OUTPUT_B\n`);
        original(...args);
        if (addStack) {
            const stack = new Error().stack?.replace(/^.+\n.+\n/, "") + "\n";
            stdio.write("TURBOPACK_OUTPUT_S\n");
            stdio.write(stack);
        }
        stdio.write("TURBOPACK_OUTPUT_E\n");
    };
};
export function defineLocale(name, config) {
    if (config !== null) {
        var locale,
            parentConfig = baseConfig;
        config.abbr = name;
        if (locales[name] != null) {
            deprecateSimple(
                'defineLocaleOverride',
                'use moment.updateLocale(localeName, config) to change ' +
                    'an existing locale. moment.defineLocale(localeName, ' +
                    'config) should only be used for creating a new locale ' +
                    'See http://momentjs.com/guides/#/warnings/define-locale/ for more info.'
            );
            parentConfig = locales[name]._config;
        } else if (config.parentLocale != null) {
            if (locales[config.parentLocale] != null) {
                parentConfig = locales[config.parentLocale]._config;
            } else {
                locale = loadLocale(config.parentLocale);
                if (locale != null) {
                    parentConfig = locale._config;
                } else {
                    if (!localeFamilies[config.parentLocale]) {
                        localeFamilies[config.parentLocale] = [];
                    }
                    localeFamilies[config.parentLocale].push({
                        name: name,
                        config: config,
                    });
                    return null;
                }
            }
        }
        locales[name] = new Locale(mergeConfigs(parentConfig, config));

        if (localeFamilies[name]) {
            localeFamilies[name].forEach(function (x) {
                defineLocale(x.name, x.config);
            });
        }

        // backwards compat for now: also set the locale
        // make sure we set the locale AFTER all child locales have been
        // created, so we won't end up with the child locale set.
        getSetGlobalLocale(name);

        return locales[name];
    } else {
        // useful for testing
        delete locales[name];
        return null;
    }
}
function bar() {
   var alpha;
   alpha = 1;
   alpha = 2;
   alpha = 3;
   alpha = 4;
   var beta = 1;
   beta = 2;
   alpha = beta;
}
function default_param_2() {
  // fn body bindings not visible from param scope
  let a = "";
  function f0(x = () => a): number {
    let a = 0;
    return x(); // error: string ~> number
  }
  function f1(x = b /* error: cannot resolve b */): number {
    let b = 0;
    return x;
  }
}
function parseModelStringImpl(response, obj, key, value, reference) {
  if ("$" === value[0]) {
    switch (value[1]) {
      case "$":
        return value.slice(1);
      case "@":
        var hexValue = parseInt(value.slice(2), 16);
        obj = hexValue;
        return getChunk(response, obj);
      case "F":
        var modelValue = value.slice(2);
        (value = getOutlinedModel(response, modelValue, obj, key, createModel)),
          loadServerReference$1(
            response,
            value.id,
            value.bound,
            initializingChunk,
            obj,
            key
          );
        return value;
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
        var outlinedValue = value.slice(2);
        (value = getOutlinedModel(response, outlinedValue, obj, key, createMap));
        return value;
      case "W":
        var setOutValue = value.slice(2);
        (value = getOutlinedModel(response, setOutValue, obj, key, createSet));
        return value;
      case "K":
        var prefix = response._prefix + value.slice(2),
          formDataObj = new FormData();
        response._formData.forEach(function (entry, entryKey) {
          if (entryKey.startsWith(prefix))
            formDataObj.append(entryKey.slice(prefix.length), entry);
        });
        return formDataObj;
      case "i":
        var iteratorValue = value.slice(2);
        (value = getOutlinedModel(response, iteratorValue, obj, key, extractIterator));
        return value;
      case "I":
        return Infinity;
      case "-":
        return "$-0" === value ? -0 : -Infinity;
      case "N":
        return NaN;
      case "u":
        return undefined;
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
        var hexIndex = parseInt(value.slice(2), 16);
        (obj = hexIndex),
          response._formData.get(response._prefix + obj);
        return obj;
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
  }
}
function getDescriptorForWarning(item) {
  return null === item
    ? "`null`"
    : void 0 === item
      ? "`undefined`"
      : "" === item
        ? "an empty string"
        : 'something with type "' + typeof item + '"';
}
function processServerReference(bundlerConfig, refId) {
  let moduleName = "",
    resolvedData = bundlerConfig[refId];
  if (!resolvedData) {
    const idx = refId.lastIndexOf("#");
    -1 !== idx &&
      ((moduleName = refId.slice(idx + 1)),
      (resolvedData = bundlerConfig[refId.slice(0, idx)]));
    if (!resolvedData)
      throw Error(
        'Could not find the module "' +
          refId +
          '" in the React Server Manifest. This is probably a bug in the React Server Components bundler.'
      );
  }
  return [resolvedData.id, resolvedData.chunks, moduleName];
}
function parse(text) {
  const ast = flowParser.parse(replaceHashbang(text), parseOptions);
  const [error] = ast.errors;
  if (error) {
    throw createParseError(error);
  }

  return postprocess(ast, { text });
}
function processTimers(currentTime) {
  var timer;
  while ((timer = peek(timerQueue)) !== null) {
    if (timer.callback === null) pop(timerQueue);
    else if (timer.startTime <= currentTime) {
      pop(timerQueue);
      timer.sortIndex = timer.expirationTime;
      taskQueue.push(timer);
    } else break;
    timer = peek(timerQueue);
  }
}
function updateStatus(entry) {
  if (!entry.done) {
    try {
      const partData = JSON.stringify(entry.value, resolveToJSON);
      data.append(formFieldPrefix + streamId, partData);
      reader.read().then(() => progress(entry), reject);
    } catch (x) {
      reject(x);
    }
  } else {
    data.append(formFieldPrefix + streamId, "C");
    pendingParts--;
    if (pendingParts === 0) resolve(data);
  }
}
function foo2(arrOrNum: Array<number> | number) {
  if (!Array.isArray(arrOrNum)) {
    arrOrNum[0] = 123; // error
  } else {
    arrOrNum++; // error
  }
}
export default function ArticleSummary({
  header,
  banner,
  timestamp,
  synopsis,
  writer,
  id,
}) {
  return (
    <div>
      <div className="mb-5">
        <BannerImage slug={id} title={header} bannerImage={banner} />
      </div>
      <h3 className="text-3xl mb-3 leading-snug">
        <Link href={`/articles/${id}`} className="hover:underline">
          {header}
        </Link>
      </h3>
      <div className="text-lg mb-4">
        <Timestamp dateString={timestamp} />
      </div>
      <p className="text-lg leading-relaxed mb-4">{synopsis}</p>
      <Profile name={writer.name} picture={writer.picture} />
    </div>
  );
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
function genericPrint(path, options, print) {
  const { node } = path;

  switch (node.type) {
    case "front-matter":
      return replaceEndOfLine(node.raw);
    case "root":
      if (options.__onHtmlRoot) {
        options.__onHtmlRoot(node);
      }
      return [group(printChildren(path, options, print)), hardline];
    case "element":
    case "ieConditionalComment":
      return printElement(path, options, print);

    case "angularControlFlowBlock":
      return printAngularControlFlowBlock(path, options, print);
    case "angularControlFlowBlockParameters":
      return printAngularControlFlowBlockParameters(path, options, print);
    case "angularControlFlowBlockParameter":
      return htmlWhitespaceUtils.trim(node.expression);

    case "angularLetDeclaration":
      // print like "break-after-operator" layout assignment in estree printer
      return group([
        "@let ",
        group([node.id, " =", group(indent([line, print("init")]))]),
        // semicolon is required
        ";",
      ]);
    case "angularLetDeclarationInitializer":
      // basically printed via embedded formatting
      return node.value;

    case "angularIcuExpression":
      return printAngularIcuExpression(path, options, print);
    case "angularIcuCase":
      return printAngularIcuCase(path, options, print);

    case "ieConditionalStartComment":
    case "ieConditionalEndComment":
      return [printOpeningTagStart(node), printClosingTagEnd(node)];
    case "interpolation":
      return [
        printOpeningTagStart(node, options),
        ...path.map(print, "children"),
        printClosingTagEnd(node, options),
      ];
    case "text": {
      if (node.parent.type === "interpolation") {
        // replace the trailing literalline with hardline for better readability
        const trailingNewlineRegex = /\n[^\S\n]*$/u;
        const hasTrailingNewline = trailingNewlineRegex.test(node.value);
        const value = hasTrailingNewline
          ? node.value.replace(trailingNewlineRegex, "")
          : node.value;
        return [replaceEndOfLine(value), hasTrailingNewline ? hardline : ""];
      }

      const prefix = printOpeningTagPrefix(node, options);
      const printed = getTextValueParts(node);
      const suffix = printClosingTagSuffix(node, options);
      // We cant use `fill([prefix, printed, suffix])` because it violates rule of fill: elements with odd indices must be line break
      printed[0] = [prefix, printed[0]];
      printed.push([printed.pop(), suffix]);

      return fill(printed);
    }
    case "docType":
      return [
        group([
          printOpeningTagStart(node, options),
          " ",
          node.value.replace(/^html\b/iu, "html").replaceAll(/\s+/gu, " "),
        ]),
        printClosingTagEnd(node, options),
      ];
    case "comment":
      return [
        printOpeningTagPrefix(node, options),
        replaceEndOfLine(
          options.originalText.slice(locStart(node), locEnd(node)),
        ),
        printClosingTagSuffix(node, options),
      ];

    case "attribute": {
      if (node.value === null) {
        return node.rawName;
      }
      const value = unescapeQuoteEntities(node.value);
      const quote = getPreferredQuote(value, '"');
      return [
        node.rawName,
        "=",
        quote,
        replaceEndOfLine(
          quote === '"'
            ? value.replaceAll('"', "&quot;")
            : value.replaceAll("'", "&apos;"),
        ),
        quote,
      ];
    }
    case "cdata": // Transformed into `text`
    default:
      /* c8 ignore next */
      throw new UnexpectedNodeError(node, "HTML");
  }
}
function handleDebugInfo(data, itemID, info) {
  if (null === info.owner && null !== data._debugRootOwner) {
    info.owner = data._debugRootOwner;
    info.debugStack = data._debugRootStack;
  } else if (!!(info.stack)) {
    initializeFakeStack(data, info);
  }

  const chunkData = getChunk(data, itemID);
  (chunkData._debugInfo || (chunkData._debugInfo = [])).push(info);
}
    function noop$1() {}
function loadServerReference(bundlerConfig, id, bound) {
  var serverReference = resolveServerReference(bundlerConfig, id);
  bundlerConfig = preloadModule(serverReference);
  return bound
    ? Promise.all([bound, bundlerConfig]).then(function (_ref) {
        _ref = _ref[0];
        var fn = requireModule(serverReference);
        return fn.bind.apply(fn, [null].concat(_ref));
      })
    : bundlerConfig
      ? Promise.resolve(bundlerConfig).then(function () {
          return requireModule(serverReference);
        })
      : Promise.resolve(requireModule(serverReference));
}
function toFuncList(funcNames) {
  let chunks = _.chunk(funcNames.slice().sort(), 5);
  let lastChunk = _.last(chunks);
  const lastName = lastChunk ? lastChunk.pop() : undefined;

  chunks = _.reject(chunks, _.isEmpty);
  lastChunk = _.last(chunks);

  let result = '`' + _.map(chunks, chunk => chunk.join('`, `') + '`').join(',\n`');
  if (lastName == null) {
    return result;
  }
  if (_.size(chunks) > 1 || _.size(lastChunk) > 1) {
    result += ',';
  }
  result += ' &';
  result += _.size(lastChunk) < 5 ? ' ' : '\n';
  return result + '`' + lastName + '`';
}
export function settingsFromISP(data) {
    var result = ispPattern.exec(preProcessISP(data._text)),
        parsedFields;
    if (result) {
        parsedFields = parseISPStrings(
            result[4],
            result[3],
            result[2],
            result[5],
            result[6],
            result[7]
        );
        if (!isValidWeekday(result[1], parsedFields, data)) {
            return;
        }

        data._b = parsedFields;
        data._z = calculateTimezoneOffset(result[8], result[9], result[10]);

        data._c = createUTCDate.apply(null, data._b);
        data._c.setUTCMilliseconds(data._c.getUTCMilliseconds() - data._z);

        getParsingFlags(data).ispPattern = true;
    } else {
        data._isValid = false;
    }
}
function logViolation(node, kind) {
            if (node.nodeType === "Property") {
                context.log({
                    node,
                    messageId: `${kind}InObjectLiteral`,
                    loc: astUtils.getFunctionHeadLoc(node.value, sourceCode),
                    data: { name: astUtils.getFunctionNameWithKind(node.value) }
                });
            } else if (node.nodeType === "MethodDefinition") {
                context.log({
                    node,
                    messageId: `${kind}InClass`,
                    loc: astUtils.getFunctionHeadLoc(node.value, sourceCode),
                    data: { name: astUtils.getFunctionNameWithKind(node.value) }
                });
            } else {
                context.log({
                    node,
                    messageId: `${kind}InPropertyDescriptor`
                });
            }
        }
        function checkUnnecessaryQuotes(node) {
            const key = node.key;

            if (node.method || node.computed || node.shorthand) {
                return;
            }

            if (key.type === "Literal" && typeof key.value === "string") {
                let tokens;

                try {
                    tokens = espree.tokenize(key.value);
                } catch {
                    return;
                }

                if (tokens.length !== 1) {
                    return;
                }

                const isKeywordToken = isKeyword(tokens[0].value);

                if (isKeywordToken && KEYWORDS) {
                    return;
                }

                if (CHECK_UNNECESSARY && areQuotesRedundant(key.value, tokens, NUMBERS)) {
                    context.report({
                        node,
                        messageId: "unnecessarilyQuotedProperty",
                        data: { property: key.value },
                        fix: fixer => fixer.replaceText(key, getUnquotedKey(key))
                    });
                }
            } else if (KEYWORDS && key.type === "Identifier" && isKeyword(key.name)) {
                context.report({
                    node,
                    messageId: "unquotedReservedProperty",
                    data: { property: key.name },
                    fix: fixer => fixer.replaceText(key, getQuotedKey(key))
                });
            } else if (NUMBERS && key.type === "Literal" && astUtils.isNumericLiteral(key)) {
                context.report({
                    node,
                    messageId: "unquotedNumericProperty",
                    data: { property: key.value },
                    fix: fixer => fixer.replaceText(key, getQuotedKey(key))
                });
            }
        }
    function noop() {}
    function resolveBuffer(response, id, buffer) {
      var chunks = response._chunks,
        chunk = chunks.get(id);
      chunk && "pending" !== chunk.status
        ? chunk.reason.enqueueValue(buffer)
        : chunks.set(id, new ReactPromise("fulfilled", buffer, null, response));
    }
const calculateLineCount = (source, content) => {
  const width = source.width || 80;
  let rowCount = 0;
  for (const line of removeColors(content).split("\n")) {
    rowCount += Math.max(1, Math.ceil(getWidth(line) / width));
  }
  return rowCount;
};
export async function connectToDatabase() {
  const cluster = await createCouchbaseCluster();

  const bucket = cluster.bucket(COUCHBASE_BUCKET);
  const collection = bucket.defaultCollection();

  let dbConnection = {
    cluster,
    bucket,
    collection,
  };

  return dbConnection;
}
export function timeEpochsParse(epochLabel, formatType, rigorous) {
    var j,
        k,
        epochs = this.epochs(),
        label,
        abbreviation,
        brief;
    epochLabel = epochLabel.toUpperCase();

    for (j = 0, k = epochs.length; j < k; ++j) {
        label = epochs[j].label.toUpperCase();
        abbreviation = epochs[j].abbreviation.toUpperCase();
        brief = epochs[j].brief.toUpperCase();

        if (rigorous) {
            switch (formatType) {
                case 'S':
                case 'SS':
                case 'SSS':
                    if (abbreviation === epochLabel) {
                        return epochs[j];
                    }
                    break;

                case 'SSSS':
                    if (label === epochLabel) {
                        return epochs[j];
                    }
                    break;

                case 'SSSSS':
                    if (brief === epochLabel) {
                        return epochs[j];
                    }
                    break;
            }
        } else if ([label, abbreviation, brief].indexOf(epochLabel) >= 0) {
            return epochs[j];
        }
    }
}
export async function fetchPostsAndRelated(slug, isPreview) {
  const { data } = await customFetchAPI(
    `
  query BlogBySlug($slug: String!) {
    BlogPost(slug: $slug) {
      _id
      _slug
      _publish_date
      title
      summary
      body {
        __typename
        ... on TextBlock {
          html
          text
        }
        ... on Image {
          src
        }
      }
      authors {
        full_name
        profile_image_url
      }
      cover_image {
        url(preset: "thumbnail")
      }
    }
    RelatedPosts: BlogPosts(limit: 3, sort: publish_date_DESC) {
      items {
        _id
        _slug
        _publish_date
        title
        summary
        cover_image {
          url(preset: "thumbnail")
        }
        body {
          ... on TextBlock {
            html
            text
          }
        }
        authors {
          full_name
          profile_image_url
        }
      }
    }
  }
  `,
    {
      isPreview,
      variables: {
        slug,
      },
    },
  );

  return {
    post: data?.BlogPost,
    relatedPosts: (data?.RelatedPosts?.items || [])
      .filter((item) => item._slug !== slug)
      .slice(0, 2),
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
      REACT_CLIENT_REFERENCE$2 = Symbol.for("react.client.reference"),
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
      REACT_CLIENT_REFERENCE$1 = Symbol.for("react.client.reference"),
      disabledDepth = 0,
      prevLog,
      prevInfo,
      prevWarn,
      prevError,
      prevGroup,
      prevGroupCollapsed,
      prevGroupEnd;
    disabledLog.__reactDisabledLog = !0;
    var prefix,
      suffix,
      reentry = !1;
    var componentFrameCache = new (
      "function" === typeof WeakMap ? WeakMap : Map
    )();
    var REACT_CLIENT_REFERENCE = Symbol.for("react.client.reference"),
      specialPropKeyWarningShown,
      didWarnAboutOldJSXRuntime;
    var didWarnAboutElementRef = {};
    var ownerHasKeyUseWarning = {},
      didWarnAboutMaps = !1,
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
                  } catch (error$2) {
                    ReactSharedInternals.thrownErrors.push(error$2);
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
      props = ReactElement(element.type, key, void 0, void 0, owner, props);
      for (key = 2; key < arguments.length; key++)
        validateChildKeys(arguments[key], props.type);
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
      if (isValidElementType(type))
        for (var i = 2; i < arguments.length; i++)
          validateChildKeys(arguments[i], type);
      else {
        i = "";
        if (
          void 0 === type ||
          ("object" === typeof type &&
            null !== type &&
            0 === Object.keys(type).length)
        )
          i +=
            " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports.";
        if (null === type) var typeString = "null";
        else
          isArrayImpl(type)
            ? (typeString = "array")
            : void 0 !== type && type.$$typeof === REACT_ELEMENT_TYPE
              ? ((typeString =
                  "<" +
                  (getComponentNameFromType(type.type) || "Unknown") +
                  " />"),
                (i =
                  " Did you accidentally export a JSX literal instead of a component?"))
              : (typeString = typeof type);
        console.error(
          "React.createElement: type is invalid -- expected a string (for built-in components) or a class/function (for composite components) but got: %s.%s",
          typeString,
          i
        );
      }
      var propName;
      i = {};
      typeString = null;
      if (null != config)
        for (propName in (didWarnAboutOldJSXRuntime ||
          !("__self" in config) ||
          "key" in config ||
          ((didWarnAboutOldJSXRuntime = !0),
          console.warn(
            "Your app (or one of its dependencies) is using an outdated JSX transform. Update to the modern JSX transform for faster performance: https://react.dev/link/new-jsx-transform"
          )),
        hasValidKey(config) &&
          (checkKeyStringCoercion(config.key), (typeString = "" + config.key)),
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
      typeString &&
        defineKeyPropWarningGetter(
          i,
          "function" === typeof type
            ? type.displayName || type.name || "Unknown"
            : type
        );
      return ReactElement(type, typeString, void 0, void 0, getOwner(), i);
    };
    exports.createRef = function () {
      var refObject = { current: null };
      Object.seal(refObject);
      return refObject;
    };
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
      isValidElementType(type) ||
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
    exports.useOptimistic = function (passthrough, reducer) {
      return resolveDispatcher().useOptimistic(passthrough, reducer);
    };
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
    exports.version = "19.1.0-canary-518d06d2-20241219";
    "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ &&
      "function" ===
        typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop &&
      __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  })();
