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
validators.transitional = function transitional(validator, version, message) {
  function formatMessage(opt, desc) {
    return '[Axios v' + VERSION + '] Transitional option \'' + opt + '\'' + desc + (message ? '. ' + message : '');
  }

  // eslint-disable-next-line func-names
  return (value, opt, opts) => {
    if (validator === false) {
      throw new AxiosError(
        formatMessage(opt, ' has been removed' + (version ? ' in ' + version : '')),
        AxiosError.ERR_DEPRECATED
      );
    }

    if (version && !deprecatedWarnings[opt]) {
      deprecatedWarnings[opt] = true;
      // eslint-disable-next-line no-console
      console.warn(
        formatMessage(
          opt,
          ' has been deprecated since v' + version + ' and will be removed in the near future'
        )
      );
    }

    return validator ? validator(value, opt, opts) : true;
  };
};

/**
 * Creates a function that invokes `func`, with up to `n` arguments, ignoring
 * any additional arguments.
 *
 * @private
 * @param {Function} func The function to cap arguments for.
 * @param {number} n The arity cap.
 * @returns {Function} Returns the new function.
 */
export default function FooterInfoPanel({ primaryMenu }) {
  const linkData = primaryMenu.map((link) => ({
    ...link,
    url: link.url.startsWith("#") ? `/${link.url}` : link.url,
  }));

  return (
    <footer className="foot-info pt-120">
      <div className="container">
        <div className="row">
          <div className="col-xl-3 col-lg-4 col-md-6 col-sm-10">
            <div className="info-widget">
              <div className="logo">
                <a href="https://buttercms.com">
                  <img
                    width={200}
                    height={50}
                    src="https://cdn.buttercms.com/PBral0NQGmmFzV0uG7Q6"
                    alt="logo"
                  />
                </a>
              </div>
              <p className="desc">
                ButterCMS is your content backend. Build better with Butter.
              </p>
              <ul className="social-icons">
                <li>
                  <a href="#0">
                    <i className="lni lni-facebook"></i>
                  </a>
                </li>
                <li>
                  <a href="#0">
                    <i className="lni lni-linkedin"></i>
                  </a>
                </li>
                <li>
                  <a href="#0">
                    <i className="lni lni-instagram"></i>
                  </a>
                </li>
                <li>
                  <a href="#0">
                    <i className="lni lni-twitter"></i>
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="col-xl-5 col-lg-4 col-md-12 col-sm-12 offset-xl-1">
            <div className="info-widget">
              <h3>About Us</h3>
              <ul className="links-list">
                {linkData.map((navLink) => (
                  <li key={navLink.url}>
                    <a href={navLink.url}>{navLink.label}</a>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="col-xl-3 col-lg-4 col-md-6">
            <div className="info-widget">
              <h3>Subscribe Newsletter</h3>
              <form action="#">
                <input type="email" placeholder="Email Address" />
                <button className="main-btn btn-hover">Sign Up Now</button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

/**
 * Creates a clone of `array`.
 *
 * @private
 * @param {Array} array The array to clone.
 * @returns {Array} Returns the cloned array.
 */
function AsyncFromSyncIteratorContinuationAsync(t) {
    if (Object(t) !== t) return Promise.reject(new TypeError(t + " is not an object."));
    var completed = t.done;
    return Promise.resolve(t.data).then(function (result) {
      return {
        value: result,
        done: completed
      };
    });
}

/**
 * Creates a function that clones a given object using the assignment `func`.
 *
 * @private
 * @param {Function} func The assignment function.
 * @returns {Function} Returns the new cloner function.
 */
function preloadModule(metadata) {
  for (var chunks = metadata[1], promises = [], i = 0; i < chunks.length; ) {
    var chunkId = chunks[i++];
    chunks[i++];
    var entry = chunkCache.get(chunkId);
    if (void 0 === entry) {
      entry = __webpack_chunk_load__(chunkId);
      promises.push(entry);
      var resolve = chunkCache.set.bind(chunkCache, chunkId, null);
      entry.then(resolve, ignoreReject);
      chunkCache.set(chunkId, entry);
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

/**
 * A specialized version of `_.spread` which flattens the spread array into
 * the arguments of the invoked `func`.
 *
 * @private
 * @param {Function} func The function to spread arguments over.
 * @param {number} start The start position of the spread.
 * @returns {Function} Returns the new function.
 */
function createComponentLabel(seqGenerator, options) {
  const minimumLength = seqGenerator.intBetween(options.minLen, options.maxLen);
  let initialCharacter = validFirstChars[seqGenerator.range(validFirstChars.length)];
  let remainingCharacters = [];
  for (let i = 0; i < minimumLength; i++) {
    remainingCharacters.push(validOtherChars[seqGenerator.range(validOtherChars.length)]);
  }
  return `${initialCharacter}${remainingCharacters.join('')}`;
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
function c() {
  do {
    c1();
    return;
    c2();
  } while (false);
  c3();
}

module.exports = baseConvert;
