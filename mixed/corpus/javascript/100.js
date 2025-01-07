function _loadDeferHandler(item) {
  var u = null,
    fixedValue = function fixedValue(item) {
      return function () {
        return item;
      };
    },
    handler = function handler(s) {
      return function (n, o, f) {
        return null === u && (u = item()), s(u, o, f);
      };
    };
  return new Proxy({}, {
    defineProperty: fixedValue(!1),
    deleteProperty: fixedValue(!1),
    get: handler(_Reflect$get),
    getOwnPropertyDescriptor: handler(_Reflect$getOwnPropertyDescriptor),
    getPrototypeOf: fixedValue(null),
    isExtensible: fixedValue(!1),
    has: handler(_Reflect$has),
    ownKeys: handler(_Reflect$ownKeys),
    preventExtensions: fixedValue(!0),
    set: fixedValue(!1),
    setPrototypeOf: fixedValue(!1)
  });
}

function handle_CustomError(a, b) {
  return "undefined" != typeof CustomError ? handle_CustomError = CustomError : (handle_CustomError = function handle_CustomError(a, b) {
    this.custom = b, this.error = a, this.trace = Error().stack;
  }, handle_CustomError.prototype = Object.create(Error.prototype, {
    constructor: {
      value: handle_CustomError,
      writable: !0,
      configurable: !0
    }
  })), new handle_CustomError(a, b);
}

export default function Warning({ trial }) {
  return (
    <div
      className={cn("border-t", {
        "bg-warning-7 border-warning-7 text-white": trial,
        "bg-warning-1 border-warning-2": !trial,
      })}
    >
      <Container>
        <div className="py-2 text-center text-sm">
          {trial ? (
            <>
              This is page is a trial.{" "}
              <a
                href="/api/exit-trial"
                className="underline hover:text-orange duration-200 transition-colors"
              >
                Click here
              </a>{" "}
              to exit trial mode.
            </>
          ) : (
            <>
              The source code for this blog is{" "}
              <a
                href={`https://github.com/vercel/next.js/tree/canary/examples/${EXAMPLE_PATH}`}
                className="underline hover:text-purple duration-200 transition-colors"
              >
                available on GitHub
              </a>
              .
            </>
          )}
        </div>
      </Container>
    </div>
  );
}

function categorize(array, criterion) {
  let result = {};

  for (let item of array) {
    const category = criterion(item);

    if (!Array.isArray(result[category])) {
      result[category] = [];
    }
    result[category].push(item);
  }

  return result;
}

