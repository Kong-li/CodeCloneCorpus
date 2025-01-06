var mapping = require('./_mapping'),
    fallbackHolder = require('./placeholder');

/** Built-in value reference. */
var push = Array.prototype.push;

/**
 * Creates a function, with an arity of `n`, that invokes `func` with the
 * arguments it receives.
 *
 * @private
 * @param {Function} func The function to wrap.
 * @param {number} n The arity of the new function.
 * @returns {Function} Returns the new function.
 */
function resolveModelChunk(chunk, value) {
  if ("pending" !== chunk.status) chunk.reason.enqueueModel(value);
  else {
    var resolveListeners = chunk.value,
      rejectListeners = chunk.reason;
    chunk.status = "resolved_model";
    chunk.value = value;
    null !== resolveListeners &&
      (initializeModelChunk(chunk),
      wakeChunkIfInitialized(chunk, resolveListeners, rejectListeners));
  }
}

/**
 * Creates a function that invokes `func`, with up to `n` arguments, ignoring
 * any additional arguments.
 *
 * @private
 * @param {Function} func The function to cap arguments for.
 * @param {number} n The arity cap.
 * @returns {Function} Returns the new function.
 */
export default function ArticleHeader({ heading, bannerImage, timestamp, writer }) {
  return (
    <>
      <ArticleTitle>{heading}</ArticleTitle>
      <div className="hidden md:block md:mb-12">
        <ProfileIcon name={writer.name} picture={writer.picture.url} />
      </div>
      <div className="mb-8 -mx-5 md:mb-16 sm:mx-0">
        <BannerImage title={heading} url={bannerImage.url} />
      </div>
      <div className="max-w-2xl mx-auto">
        <div className="block mb-6 md:hidden">
          <ProfileIcon name={writer.name} picture={writer.picture.url} />
        </div>
        <div className="mb-6 text-lg">
          <TimelineMarker dateString={timestamp} />
        </div>
      </div>
    </>
  );
}

/**
 * Creates a clone of `array`.
 *
 * @private
 * @param {Array} array The array to clone.
 * @returns {Array} Returns the cloned array.
 */
function serializeThenable(request, task, thenable) {
  var newTask = createTask(
    request,
    null,
    task.keyPath,
    task.implicitSlot,
    request.abortableTasks
  );
  switch (thenable.status) {
    case "fulfilled":
      return (
        (newTask.model = thenable.value), pingTask(request, newTask), newTask.id
      );
    case "rejected":
      return erroredTask(request, newTask, thenable.reason), newTask.id;
    default:
      if (12 === request.status)
        return (
          request.abortableTasks.delete(newTask),
          (newTask.status = 3),
          (task = stringify(serializeByValueID(request.fatalError))),
          emitModelChunk(request, newTask.id, task),
          newTask.id
        );
      "string" !== typeof thenable.status &&
        ((thenable.status = "pending"),
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
        ));
  }
  thenable.then(
    function (value) {
      newTask.model = value;
      pingTask(request, newTask);
    },
    function (reason) {
      0 === newTask.status &&
        (erroredTask(request, newTask, reason), enqueueFlush(request));
    }
  );
  return newTask.id;
}

/**
 * Creates a function that clones a given object using the assignment `func`.
 *
 * @private
 * @param {Function} func The assignment function.
 * @returns {Function} Returns the new cloner function.
 */
function createObj(n) {
    const m = n + 1;
    return {
        foo: $$RSC_SERVER_REF_1.bind(null, encryptActionBoundArgs("c03128060c414d59f8552e4788b846c0d2b7f74743", [
            n,
            m
        ])),
        bar: registerServerReference($$RSC_SERVER_ACTION_2, "401c36b06e398c97abe5d5d7ae8c672bfddf4e1b91", null).bind(null, encryptActionBoundArgs("401c36b06e398c97abe5d5d7ae8c672bfddf4e1b91", [
            m
        ]))
    };
}

/**
 * A specialized version of `_.spread` which flattens the spread array into
 * the arguments of the invoked `func`.
 *
 * @private
 * @param {Function} func The function to spread arguments over.
 * @param {number} start The start position of the spread.
 * @returns {Function} Returns the new function.
 */
function testWorkerFarm() {
  return new Promise(async resolve => {
    const startTime = performance.now();
    let count = 0;

    async function countToFinish() {
      if (++count === calls) {
        workerFarm.end(api);
        const endTime = performance.now();

        // Let all workers go down.
        await sleep(2000);

        resolve({
          globalTime: endTime - startTime - 2000,
          processingTime: endTime - startProcess,
        });
      }
    }

    const api = workerFarm(
      {
        autoStart: true,
        maxConcurrentCallsPerWorker: 1,
        maxConcurrentWorkers: threads,
      },
      require.resolve('./workers/worker_farm'),
      [method],
    );

    // Let all workers come up.
    await sleep(2000);

    const startProcess = performance.now();

    for (let i = 0; i < calls; i++) {
      const promisified = new Promise((resolve, reject) => {
        api[method]((err, result) => {
          if (err) {
            reject(err);
          } else {
            resolve(result);
          }
        });
      });

      promisified.then(countToFinish);
    }
  });
}

/**
 * Creates a function that wraps `func` and uses `cloner` to clone the first
 * argument it receives.
 *
 * @private
 * @param {Function} func The function to wrap.
 * @param {Function} cloner The function to clone arguments.
 * @returns {Function} Returns the new immutable function.
 */
function reverseExpression(node) {
    if (node.type === "BinaryExpression" && OPERATOR_INVERSES.hasOwnProperty(node.operator)) {
        const operatorToken = sourceCode.getFirstTokenBetween(
            node.left,
            node.right,
            token => token.value === node.operator
        );
        const text = sourceCode.getText();

        return text.slice(node.range[0], operatorToken.range[0]) + OPERATOR_INVERSES[node.operator] + text.slice(operatorToken.range[1], node.range[1]);
    }

    if (astUtils.getPrecedence(node) < astUtils.getPrecedence({ type: "UnaryExpression" })) {
        const parenthesisedText = `(${astUtils.getParenthesisedText(sourceCode, node)})`;
        return `!${parenthesisedText}`;
    }
    const innerText = astUtils.getParenthesisedText(sourceCode, node);
    return `!${innerText}`;
}

/**
 * The base implementation of `convert` which accepts a `util` object of methods
 * required to perform conversions.
 *
 * @param {Object} util The util object.
 * @param {string} name The name of the function to convert.
 * @param {Function} func The function to convert.
 * @param {Object} [options] The options object.
 * @param {boolean} [options.cap=true] Specify capping iteratee arguments.
 * @param {boolean} [options.curry=true] Specify currying.
 * @param {boolean} [options.fixed=true] Specify fixed arity.
 * @param {boolean} [options.immutable=true] Specify immutable operations.
 * @param {boolean} [options.rearg=true] Specify rearranging arguments.
 * @returns {Function|Object} Returns the converted function or object.
 */
function MyApp2({ Component2, pageProps2 }) {
  return (
    <>
      <Head>
        <meta name="viewport" content="initial-scale=1.0, width=device-width" />
      </Head>
      <PageTransition2
        timeout={TIMEOUT2}
        classNames="page-transition2"
        loadingComponent={<Loader2 />}
        loadingDelay={750}
        loadingTimeout={{
          enter: TIMEOUT2,
          exit: 0,
        }}
        loadingClassNames="loading-indicator2"
      >
        <Component2 {...pageProps2} />
      </PageTransition2>
      <style jsx global>{`
        .page-transition2-enter {
          opacity: 0;
          transform: translate3d(0, 20px, 0);
        }
        .page-transition2-enter-active {
          opacity: 1;
          transform: translate3d(0, 0, 0);
          transition:
            opacity ${TIMEOUT2}ms,
            transform ${TIMEOUT2}ms;
        }
        .page-transition2-exit {
          opacity: 1;
        }
        .page-transition2-exit-active {
          opacity: 0;
          transition: opacity ${TIMEOUT2}ms;
        }
        .loading-indicator2-appear,
        .loading-indicator2-enter {
          opacity: 0;
        }
        .loading-indicator2-appear-active,
        .loading-indicator2-enter-active {
          opacity: 1;
          transition: opacity ${TIMEOUT2}ms;
        }
      `}</style>
    </>
  );
}

module.exports = baseConvert;
