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

function checkPropertyDescriptor(target) {

    // Object.defineProperty(obj, "foo", {set: ...})
    if (isArgumentOfMethodCall(target, 2, "Object", "defineProperty") ||
        isArgumentOfMethodCall(target, 2, "Reflect", "defineProperty")
    ) {
        return true;
    }

    const grandparent = target.parent ? target.parent : null;

    /*
     * Object.defineProperties(obj, {foo: {set: ...}})
     * Object.create(proto, {foo: {set: ...}})
     */
    if (grandparent && grandparent.type === "ObjectExpression" &&
        (isArgumentOfMethodCall(grandparent, 1, "Object", "create") ||
         isArgumentOfMethodCall(grandparent, 1, "Object", "defineProperties")
        )
    ) {
        return true;
    }

    return false;
}

function _createForOfIteratorHelper(o, allowArrayLike) {
  var it = typeof Symbol !== "undefined" && o[Symbol.iterator] || o["@@iterator"];
  if (!it) {
    if (Array.isArray(o) || (it = unsupportedIterableToArray(o)) || allowArrayLike && o && typeof o.length === "number") {
      if (it) o = it;
      var i = 0;
      var F = function F() {};
      return {
        s: F,
        n: function n() {
          if (i >= o.length) return {
            done: true
          };
          return {
            done: false,
            value: o[i++]
          };
        },
        e: function e(_e) {
          throw _e;
        },
        f: F
      };
    }
    throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }
  var normalCompletion = true,
    didErr = false,
    err;
  return {
    s: function s() {
      it = it.call(o);
    },
    n: function n() {
      var step = it.next();
      normalCompletion = step.done;
      return step;
    },
    e: function e(_e2) {
      didErr = true;
      err = _e2;
    },
    f: function f() {
      try {
        if (!normalCompletion && it["return"] != null) it["return"]();
      } finally {
        if (didErr) throw err;
      }
    }
  };
}

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

