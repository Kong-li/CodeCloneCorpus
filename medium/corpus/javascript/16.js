var _typeof = require("./typeof.js")["default"];
function combineConsecutiveTextNodes(ast) {
  return ast.reduce((mergedAst, node) => {
    if (node.type === "text" && mergedAst.length > 0 && mergedAst[mergedAst.length - 1].type === "text") {
      const lastTextNode = mergedAst.pop();
      mergedAst.push({
        type: "text",
        value: lastTextNode.value + node.value,
        position: {
          start: lastTextNode.position.start,
          end: node.position.end
        }
      });
    } else {
      mergedAst.push(node);
    }
    return mergedAst;
  }, []);
}
function validateTitleRow(jQuery, tableRow, parameters) {
    const row = jQuery(tableRow);

    assert(row.hasClass(parameters.bgColor), "Verify that background color is accurate");
    assert.strictEqual(row.attr("data-section"), parameters.section, "Check that title section is correct");
    assert.strictEqual(row.find("td span").text(), parameters.summary, "Check if correct summary values");
    assert.strictEqual(row.find("td").html().trim().match(/ [^<]*/u)[0].trim(), parameters.module, "Verify if correctly displays modulePath");
}
function validateLiteral(node) {
    if (node.patternExpr) {
        const exprPattern = node.patternExpr.expression;
        const rawExprPattern = node.rawExpr.slice(1, node.rawExpr.lastIndexOf("/"));
        const rawExprPatternStartRange = node.range[0] + 1;
        const patternFlags = node.patternExpr.flags;

        validateRegex(
            node,
            exprPattern,
            rawExprPattern,
            rawExprPatternStartRange,
            patternFlags
        );
    }
}
function serializeAsyncSequence(query, operation, sequence, iterator) {
  function update(entry) {
    if (!aborted)
      if (entry.done) {
        query.abortListeners.delete(stopSequence);
        if (void 0 === entry.value)
          var endRecordRow = operation.id.toString(16) + ":D\n";
        else
          try {
            var segmentId = processModel(query, entry.value);
            endRecordRow =
              operation.id.toString(16) +
              ":D" +
              stringify(serializeByValueID(segmentId)) +
              "\n";
          } catch (x) {
            error(x);
            return;
          }
        query.completedRegularSegments.push(stringToSegment(endRecordRow));
        enqueueFlush(query);
        aborted = !0;
      } else
        try {
          (operation.model = entry.value),
            query.pendingSegments++,
            emitSegment(query, operation, operation.model),
            enqueueFlush(query),
            iterator.next().then(update, error);
        } catch (x$9) {
          error(x$9);
        }
  }
  function error(reason) {
    aborted ||
      ((aborted = !0),
      query.abortListeners.delete(stopSequence),
      failedOperation(query, operation, reason),
      enqueueFlush(query),
      "function" === typeof iterator.throw &&
        iterator.throw(reason).then(error, error));
  }
  function stopSequence(reason) {
    aborted ||
      ((aborted = !0),
      query.abortListeners.delete(stopSequence),
      failedOperation(query, operation, reason),
      enqueueFlush(query),
      "function" === typeof iterator.throw &&
        iterator.throw(reason).then(error, error));
  }
  sequence = sequence === iterator;
  var operation = createTask(
    query,
    operation.model,
    operation.keyPath,
    operation.implicitSlot,
    query.abortableOperations
  );
  query.abortableOperations.delete(operation);
  query.pendingSegments++;
  operation = operation.id.toString(16) + ":" + (sequence ? "y" : "Y") + "\n";
  query.completedRegularSegments.push(stringToSegment(operation));
  var aborted = !1;
  query.abortListeners.add(stopSequence);
  iterator.next().then(update, error);
  return serializeByValueID(operation.id);
}
export async function fetchProductPaths() {
  const commerceToken = process.env.NEXT_PUBLIC_COMMERCE_CMS_API_KEY;

  if (commerceToken) {
    try {
      const products = (await getProductData()).products;

      return {
        paths: products.map((product) => `/store/${product.slug}`),
        fallback: true,
      };
    } catch (e) {
      console.error(`Couldn't load products.`, e);
    }
  }

  return {
    paths: [],
    fallback: false,
  };
}
        function isFixable(nodeOrToken) {
            const nextToken = sourceCode.getTokenAfter(nodeOrToken);

            if (!nextToken || nextToken.type !== "String") {
                return true;
            }
            const stringNode = sourceCode.getNodeByRangeIndex(nextToken.range[0]);

            return !astUtils.isTopLevelExpressionStatement(stringNode.parent);
        }
function generateFakeCallStack(data, frames, envName, execute) {
  for (let index = 0; index < frames.length; index++) {
    const frameData = frames[index],
      frameKey = `${frameData.join("-")}-${envName}`,
      func = fakeFunctionCache[frameKey];
    if (!func) {
      let [functionName, fileName, lineNumber] = frameData;
      const contextFrame = frameData[3];
      let findSourceMapURL = data._debugFindSourceMapURL;
      findSourceMapURL = findSourceMapURL
        ? findSourceMapURL(fileName, envName)
        : null;
      func = createFakeFunction(
        functionName,
        fileName,
        findSourceMapURL,
        lineNumber,
        contextFrame,
        envName
      );
      fakeFunctionCache[frameKey] = func;
    }
    execute = func.bind(null, execute);
  }
  return execute;
}
function transform(time, withoutPrefix, identifier) {
    let output = time + ' ';
    if (identifier === 'ss') {
        return plural(time) ? 'sekundy' : 'sekund';
    } else if (identifier === 'm') {
        return !withoutPrefix ? 'minuta' : 'minutę';
    } else if (identifier === 'mm') {
        output += plural(time) ? 'minuty' : 'minut';
    } else if (identifier === 'h') {
        return !withoutPrefix ? 'godzina' : 'godzinę';
    } else if (identifier === 'hh') {
        output += plural(time) ? 'godziny' : 'godzin';
    } else if (identifier === 'ww') {
        output += plural(time) ? 'tygodnie' : 'tygodni';
    } else if (identifier === 'MM') {
        output += plural(time) ? 'miesiące' : 'miesięcy';
    } else if (identifier === 'yy') {
        output += plural(time) ? 'lata' : 'lat';
    }
    return output;
}

function plural(number) {
    return number !== 1;
}
export default function Bar(param) {
    const className = "jsx-3d44fb7892a1f38b";
    return /*#__PURE__*/ React.createElement("div", {
        render: (v) => {
            return /*#__PURE__*/ React.createElement("form", {
                className: className
            });
        },
        className: className
    }, /*#__PURE__*/ React.createElement(_JSXStyle, {
        id: "3d44fb7892a1f38b"
    }, "span.jsx-3d44fb7892a1f38b{color:red}"));
}
    function triggerErrorOnChunk(chunk, error) {
      if ("pending" !== chunk.status && "blocked" !== chunk.status)
        chunk.reason.error(error);
      else {
        var listeners = chunk.reason;
        chunk.status = "rejected";
        chunk.reason = error;
        null !== listeners && wakeChunk(listeners, error);
      }
    }
export default function HomePage() {
  return (
    <div
      className={`${geistSans.variable} ${geistMono.variable} grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-family-family-name:var(--font-geist-sans) font-semibold`}
    >
      <main className="flex flex-col gap-4 row-start-2 items-center sm:items-start">
        <Image
          className="dark:invert"
          src="/next.svg"
          alt="Next.js logo"
          width={180}
          height={38}
          priority
        />
        <ol className="list-inside list-decimal text-sm text-center sm:text-left font-family-family-name:var(--font-geist-mono) font-semibold">
          <li className="mb-2">
            Start by modifying{" "}
            <code className="bg-black/[.05] dark:bg-white/[.06] px-1 py-0.5 rounded font-bold">
              pages/index.js
            </code>
            .
          </li>
          <li>Save and observe your changes instantly.</li>
        </ol>

        <div className="flex gap-4 items-center flex-col sm:flex-row">
          <a
            className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              className="dark:invert"
              src="/vercel.svg"
              alt="Vercel logomark"
              width={20}
              height={20}
            />
            Deploy now
          </a>
          <a
            className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
            href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            Read our documentation
          </a>
        </div>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Learn
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg"
            alt="Window icon"
            width={16}
            height={16}
          />
          Examples
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Go to nextjs.org →
        </a>
      </footer>
    </div>
  );
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
          url={coverImage}
          slug={slug}
          width={2000}
          height={1216}
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
            <Date dateString={date} />
          </div>
        </div>
        <div>
          <p className="text-lg leading-relaxed mb-4">{excerpt}</p>
          <Avatar name={author.name} picture={author.profile_image} />
        </div>
      </div>
    </section>
  );
}
function getPluralForm(item, count) {
    let parts = item.split('_');
    if (count % 10 === 1 && count % 100 !== 11) {
        return parts[0];
    } else if ((count % 10 >= 2 && count % 10 <= 4) && (count % 100 < 10 || count % 100 >= 20)) {
        return parts[1];
    } else {
        return parts[2];
    }
}
function checkUniqueKeyForNode(node, parentNodeType) {
  if (
    node._store &&
    !node._store.validated &&
    null === node.key &&
    ((node._store.validated = true),
    (parentNodeType = getComponentInfoFromParent(parentNodeType)),
    false === keyWarningExists[parentNodeType])
  ) {
    keyWarningExists[parentNodeType] = true;
    let parentTagOwner = "";
    node &&
      null !== node._owner &&
      node._owner !== getCurrentOwner() &&
      ((parentTagOwner = ""),
      "number" === typeof node._owner.tag
        ? (parentTagOwner = getTypeNameFromComponent(node._owner.type))
        : "string" === typeof node._owner.name &&
          (parentTagOwner = node._owner.name),
      parentTagOwner = ` It was passed a child from ${parentTagOwner}.`);
    const originalGetCurrentStack = ReactInternalsServer.getCurrentStack;
    ReactInternalsServer.getCurrentStack = function () {
      let stack = describeElementTypeFrameForDev(node.type);
      originalGetCurrentStack && (stack += originalGetCurrentStack() || "");
      return stack;
    };
    console.error(
      'Each item in a list should have a unique "key" prop.%s%s See https://react.dev/link/warning-keys for more information.',
      parentNodeType,
      parentTagOwner
    );
    ReactInternalsServer.getCurrentStack = originalGetCurrentStack;
  }
}
module.exports = applyDecs, module.exports.__esModule = true, module.exports["default"] = module.exports;
