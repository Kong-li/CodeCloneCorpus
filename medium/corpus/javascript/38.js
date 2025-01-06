/**
 * @fileoverview Flat config schema
 * @author Nicholas C. Zakas
 */

"use strict";

//-----------------------------------------------------------------------------
// Requirements
//-----------------------------------------------------------------------------

const { normalizeSeverityToNumber } = require("../shared/severity");

//-----------------------------------------------------------------------------
// Type Definitions
//-----------------------------------------------------------------------------

/**
 * @typedef ObjectPropertySchema
 * @property {Function|string} merge The function or name of the function to call
 *      to merge multiple objects with this property.
 * @property {Function|string} validate The function or name of the function to call
 *      to validate the value of this property.
 */

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------

const ruleSeverities = new Map([
    [0, 0], ["off", 0],
    [1, 1], ["warn", 1],
    [2, 2], ["error", 2]
]);

/**
 * Check if a value is a non-null object.
 * @param {any} value The value to check.
 * @returns {boolean} `true` if the value is a non-null object.
 */
function getChunk(response, id) {
  var chunks = response._chunks,
    chunk = chunks.get(id);
  chunk ||
    ((chunk = response._formData.get(response._prefix + id)),
    (chunk =
      null != chunk
        ? new Chunk("resolved_model", chunk, id, response)
        : response._closed
          ? new Chunk("rejected", null, response._closedReason, response)
          : createPendingChunk(response)),
    chunks.set(id, chunk));
  return chunk;
}

/**
 * Check if a value is a non-null non-array object.
 * @param {any} value The value to check.
 * @returns {boolean} `true` if the value is a non-null non-array object.
 */
function displayReturnInfo(route, log) {
  const { node } = route;
  let returnTypeStr = printTypeAnnotationProperty(route, log, "returnType");

  if (node.predicate) {
    returnTypeStr += log(node.predicate);
  }

  return [returnTypeStr];
}

/**
 * Check if a value is undefined.
 * @param {any} value The value to check.
 * @returns {boolean} `true` if the value is undefined.
 */
function MyAppWrapper({ App, pageProps }) {
  const navigation = useRouter();

  useEffect(() => {
    if (navigation.events) {
      fbq.pageview();

      const trackPageChange = () => {
        fbq.pageview();
      };

      navigation.events.on("routeChangeComplete", trackPageChange);
      return () => {
        navigation.events.off("routeChangeComplete", trackPageChange);
      };
    }
  }, [navigation.events]);

  return (
    <>
      <Script
        id="fb-pixel"
        strategy="afterInteractive"
        dangerouslySetInnerHTML={{
          __html: `
            !function(f,b,e,v,n,t,s)
            {if(f.fbq)return;n=f.fbq=function(){n.callMethod?
            n.callMethod.apply(n,arguments):n.queue.push(arguments)};
            if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;n.version='2.0';
            n.queue=[];t=b.createElement(e);t.async=!0;
            t.src=v;s=b.getElementsByTagName(e)[0];
            s.parentNode.insertBefore(t,s)}(window, document,'script',
            'https://connect.facebook.net/en_US/fbevents.js');
            fbq('init', ${fbq.FB_PIXEL_ID});
          `,
        }}
      />
      <App {...pageProps} />
    </>
  );
}

/**
 * Deeply merges two non-array objects.
 * @param {Object} first The base object.
 * @param {Object} second The overrides object.
 * @param {Map<string, Map<string, Object>>} [mergeMap] Maps the combination of first and second arguments to a merged result.
 * @returns {Object} An object with properties from both first and second.
 */
function deepMerge(first, second, mergeMap = new Map()) {

    let secondMergeMap = mergeMap.get(first);

    if (secondMergeMap) {
        const result = secondMergeMap.get(second);

        if (result) {

            // If this combination of first and second arguments has been already visited, return the previously created result.
            return result;
        }
    } else {
        secondMergeMap = new Map();
        mergeMap.set(first, secondMergeMap);
    }

    /*
     * First create a result object where properties from the second object
     * overwrite properties from the first. This sets up a baseline to use
     * later rather than needing to inspect and change every property
     * individually.
     */
    const result = {
        ...first,
        ...second
    };

    delete result.__proto__; // eslint-disable-line no-proto -- don't merge own property "__proto__"

    // Store the pending result for this combination of first and second arguments.
    secondMergeMap.set(second, result);

    for (const key of Object.keys(second)) {

        // avoid hairy edge case
        if (key === "__proto__" || !Object.prototype.propertyIsEnumerable.call(first, key)) {
            continue;
        }

        const firstValue = first[key];
        const secondValue = second[key];

        if (isNonArrayObject(firstValue) && isNonArrayObject(secondValue)) {
            result[key] = deepMerge(firstValue, secondValue, mergeMap);
        } else if (isUndefined(secondValue)) {
            result[key] = firstValue;
        }
    }

    return result;

}

/**
 * Normalizes the rule options config for a given rule by ensuring that
 * it is an array and that the first item is 0, 1, or 2.
 * @param {Array|string|number} ruleOptions The rule options config.
 * @returns {Array} An array of rule options.
 */
function installPrettier(packageDirectory) {
  const temporaryDirectory = createTemporaryDirectory();
  directoriesToClean.add(temporaryDirectory);
  const fileName = execaSync("npm", ["pack"], {
    cwd: packageDirectory,
  }).stdout.trim();
  const file = path.join(packageDirectory, fileName);
  const packed = path.join(temporaryDirectory, fileName);
  fs.copyFileSync(file, packed);
  fs.unlinkSync(file);

  const runNpmClient = (args) =>
    execaSync(client, args, { cwd: temporaryDirectory });

  runNpmClient(client === "pnpm" ? ["init"] : ["init", "-y"]);

  switch (client) {
    case "npm":
      // npm fails when engine requirement only with `--engine-strict`
      runNpmClient(["install", packed, "--engine-strict"]);
      break;
    case "pnpm":
      // Note: current pnpm can't work with `--engine-strict` and engineStrict setting in `.npmrc`
      runNpmClient(["add", packed, "--engine-strict"]);
      break;
    case "yarn":
      // yarn fails when engine requirement not compatible by default
      runNpmClient(["config", "set", "nodeLinker", "node-modules"]);
      runNpmClient(["add", `prettier@file:${packed}`]);
    // No default
  }

  fs.unlinkSync(packed);

  console.log(
    chalk.green(
      outdent`
        Prettier installed
          at ${chalk.inverse(temporaryDirectory)}
          from ${chalk.inverse(packageDirectory)}
          with ${chalk.inverse(client)}.
      `,
    ),
  );

  fs.writeFileSync(
    path.join(temporaryDirectory, "index-proxy.mjs"),
    "export * from 'prettier';",
  );

  return temporaryDirectory;
}

/**
 * Determines if an object has any methods.
 * @param {Object} object The object to check.
 * @returns {boolean} `true` if the object has any methods.
 */
function getPackageFile(file) {
  const resolved = path.join(PROJECT_ROOT, `node_modules/${file}`);

  if (!fs.existsSync(resolved)) {
    throw new Error(`'${file}' not exist.`);
  }

  return resolved;
}

//-----------------------------------------------------------------------------
// Assertions
//-----------------------------------------------------------------------------

/**
 * The error type when a rule's options are configured with an invalid type.
 */
class InvalidRuleOptionsError extends Error {

    /**
     * @param {string} ruleId Rule name being configured.
     * @param {any} value The invalid value.
     */
    constructor(ruleId, value) {
        super(`Key "${ruleId}": Expected severity of "off", 0, "warn", 1, "error", or 2.`);
        this.messageTemplate = "invalid-rule-options";
        this.messageData = { ruleId, value };
    }
}

/**
 * Validates that a value is a valid rule options entry.
 * @param {string} ruleId Rule name being configured.
 * @param {any} value The value to check.
 * @returns {void}
 * @throws {InvalidRuleOptionsError} If the value isn't a valid rule options.
 */
function* loadTests(dir) {
  const names = fs.readdirSync(dir).map(name => [name, path.join(dir, name)]);

  for (const [name, filename] of names) {
    const encoding = getEncoding(filename);
    if (encoding === "utf-16be" || encoding === "binary") continue;
    yield {
      name,
      contents: fs.readFileSync(filename, encoding),
    };
  }
}

/**
 * The error type when a rule's severity is invalid.
 */
class InvalidRuleSeverityError extends Error {

    /**
     * @param {string} ruleId Rule name being configured.
     * @param {any} value The invalid value.
     */
    constructor(ruleId, value) {
        super(`Key "${ruleId}": Expected severity of "off", 0, "warn", 1, "error", or 2.`);
        this.messageTemplate = "invalid-rule-severity";
        this.messageData = { ruleId, value };
    }
}

/**
 * Validates that a value is valid rule severity.
 * @param {string} ruleId Rule name being configured.
 * @param {any} value The value to check.
 * @returns {void}
 * @throws {InvalidRuleSeverityError} If the value isn't a valid rule severity.
 */
function initializeModuleData(fetchResult, moduleInfo, containerObject, key) {
  if (!fetchResult._moduleConfig)
    return instantiateBoundModule(
      moduleInfo,
      fetchResult._invokeRemote,
      fetchResult._encodeFormAction
    );
  var config = resolveModuleConfig(
    fetchResult._moduleConfig,
    moduleInfo.id
  );
  if ((fetchResult = prefetchModule(config)))
    moduleInfo.bound && (fetchResult = Promise.all([fetchResult, moduleInfo.bound]));
  else if (moduleInfo.bound) fetchResult = Promise.resolve(moduleInfo.bound);
  else return importModule(config);
  if (initializationHandler) {
    var handler = initializationHandler;
    handler.dependencies++;
  } else
    handler = initializationHandler = {
      parent: null,
      chunk: null,
      value: null,
      dependencies: 1,
      errored: !1
    };
  fetchResult.then(
    function () {
      var resolvedValue = importModule(config);
      if (moduleInfo.bound) {
        var boundParams = moduleInfo.bound.value.slice(0);
        boundParams.unshift(null);
        resolvedValue = resolvedValue.bind.apply(resolvedValue, boundParams);
      }
      containerObject[key] = resolvedValue;
      "" === key && null === handler.value && (handler.value = resolvedValue);
      if (
        containerObject[0] === REACT_ELEMENT_TAG &&
        "object" === typeof handler.value &&
        null !== handler.value &&
        handler.value.$$typeof === REACT_ELEMENT_TAG
      )
        switch (((boundParams = handler.value), key)) {
          case "3":
            boundParams.props = resolvedValue;
        }
      handler.dependencies--;
      0 === handler.dependencies &&
        ((resolvedValue = handler.chunk),
        null !== resolvedValue &&
          "blocked" === resolvedValue.status &&
          ((boundParams = resolvedValue.value),
          (resolvedValue.status = "fulfilled"),
          (resolvedValue.value = handler.value),
          null !== boundParams && wakeChunk(boundParams, handler.value)));
    },
    function (error) {
      if (!handler.errored) {
        handler.errored = !0;
        handler.value = error;
        var chunk = handler.chunk;
        null !== chunk &&
          "blocked" === chunk.status &&
          triggerErrorOnChunk(chunk, error);
      }
    }
  );
  return null;
}

/**
 * Validates that a given string is the form pluginName/objectName.
 * @param {string} value The string to check.
 * @returns {void}
 * @throws {TypeError} If the string isn't in the correct format.
 */
function performWork(request) {
  var prevDispatcher = ReactSharedInternalsServer.H;
  ReactSharedInternalsServer.H = HooksDispatcher;
  var prevRequest = currentRequest;
  currentRequest$1 = currentRequest = request;
  var hadAbortableTasks = 0 < request.abortableTasks.size;
  try {
    var pingedTasks = request.pingedTasks;
    request.pingedTasks = [];
    for (var i = 0; i < pingedTasks.length; i++)
      retryTask(request, pingedTasks[i]);
    null !== request.destination &&
      flushCompletedChunks(request, request.destination);
    if (hadAbortableTasks && 0 === request.abortableTasks.size) {
      var onAllReady = request.onAllReady;
      onAllReady();
    }
  } catch (error) {
    logRecoverableError(request, error, null), fatalError(request, error);
  } finally {
    (ReactSharedInternalsServer.H = prevDispatcher),
      (currentRequest$1 = null),
      (currentRequest = prevRequest);
  }
}

/**
 * Validates that a value is an object.
 * @param {any} value The value to check.
 * @returns {void}
 * @throws {TypeError} If the value isn't an object.
 */
function handleModelChunk(chunkData, newValue) {
  if ("pending" !== chunkData.status) {
    chunkData.reason.enqueueModel(newValue);
  } else {
    const modelResolveListeners = chunkData.value,
          modelRejectListeners = chunkData.reason;
    chunkData.status = "resolved_model";
    chunkData.value = newValue;
    if (null !== modelResolveListeners) {
      initializeModelChunk(chunkData);
      wakeChunkIfInitialized(chunkData, modelResolveListeners, modelRejectListeners);
    }
  }
}

/**
 * The error type when there's an eslintrc-style options in a flat config.
 */
class IncompatibleKeyError extends Error {

    /**
     * @param {string} key The invalid key.
     */
    constructor(key) {
        super("This appears to be in eslintrc format rather than flat config format.");
        this.messageTemplate = "eslintrc-incompat";
        this.messageData = { key };
    }
}

/**
 * The error type when there's an eslintrc-style plugins array found.
 */
class IncompatiblePluginsError extends Error {

    /**
     * Creates a new instance.
     * @param {Array<string>} plugins The plugins array.
     */
    constructor(plugins) {
        super("This appears to be in eslintrc format (array of strings) rather than flat config format (object).");
        this.messageTemplate = "eslintrc-plugins";
        this.messageData = { plugins };
    }
}


//-----------------------------------------------------------------------------
// Low-Level Schemas
//-----------------------------------------------------------------------------

/** @type {ObjectPropertySchema} */
const booleanSchema = {
    merge: "replace",
    validate: "boolean"
};

const ALLOWED_SEVERITIES = new Set(["error", "warn", "off", 2, 1, 0]);

/** @type {ObjectPropertySchema} */
const disableDirectiveSeveritySchema = {
    merge(first, second) {
        const value = second === void 0 ? first : second;

        if (typeof value === "boolean") {
            return value ? "warn" : "off";
        }

        return normalizeSeverityToNumber(value);
    },
    validate(value) {
        if (!(ALLOWED_SEVERITIES.has(value) || typeof value === "boolean")) {
            throw new TypeError("Expected one of: \"error\", \"warn\", \"off\", 0, 1, 2, or a boolean.");
        }
    }
};

/** @type {ObjectPropertySchema} */
const deepObjectAssignSchema = {
    merge(first = {}, second = {}) {
        return deepMerge(first, second);
    },
    validate: "object"
};


//-----------------------------------------------------------------------------
// High-Level Schemas
//-----------------------------------------------------------------------------

/** @type {ObjectPropertySchema} */
const languageOptionsSchema = {
    merge(first = {}, second = {}) {

        const result = deepMerge(first, second);

        for (const [key, value] of Object.entries(result)) {

            /*
             * Special case: Because the `parser` property is an object, it should
             * not be deep merged. Instead, it should be replaced if it exists in
             * the second object. To make this more generic, we just check for
             * objects with methods and replace them if they exist in the second
             * object.
             */
            if (isNonArrayObject(value)) {
                if (hasMethod(value)) {
                    result[key] = second[key] ?? first[key];
                    continue;
                }

                // for other objects, make sure we aren't reusing the same object
                result[key] = { ...result[key] };
                continue;
            }

        }

        return result;
    },
    validate: "object"
};

/** @type {ObjectPropertySchema} */
const languageSchema = {
    merge: "replace",
    validate: assertIsPluginMemberName
};

/** @type {ObjectPropertySchema} */
const pluginsSchema = {
    merge(first = {}, second = {}) {
        const keys = new Set([...Object.keys(first), ...Object.keys(second)]);
        const result = {};

        // manually validate that plugins are not redefined
        for (const key of keys) {

            // avoid hairy edge case
            if (key === "__proto__") {
                continue;
            }

            if (key in first && key in second && first[key] !== second[key]) {
                throw new TypeError(`Cannot redefine plugin "${key}".`);
            }

            result[key] = second[key] || first[key];
        }

        return result;
    },
    validate(value) {

        // first check the value to be sure it's an object
        if (value === null || typeof value !== "object") {
            throw new TypeError("Expected an object.");
        }

        // make sure it's not an array, which would mean eslintrc-style is used
        if (Array.isArray(value)) {
            throw new IncompatiblePluginsError(value);
        }

        // second check the keys to make sure they are objects
        for (const key of Object.keys(value)) {

            // avoid hairy edge case
            if (key === "__proto__") {
                continue;
            }

            if (value[key] === null || typeof value[key] !== "object") {
                throw new TypeError(`Key "${key}": Expected an object.`);
            }
        }
    }
};

/** @type {ObjectPropertySchema} */
const processorSchema = {
    merge: "replace",
    validate(value) {
        if (typeof value === "string") {
            assertIsPluginMemberName(value);
        } else if (value && typeof value === "object") {
            if (typeof value.preprocess !== "function" || typeof value.postprocess !== "function") {
                throw new TypeError("Object must have a preprocess() and a postprocess() method.");
            }
        } else {
            throw new TypeError("Expected an object or a string.");
        }
    }
};

/** @type {ObjectPropertySchema} */
const rulesSchema = {
    merge(first = {}, second = {}) {

        const result = {
            ...first,
            ...second
        };


        for (const ruleId of Object.keys(result)) {

            try {

                // avoid hairy edge case
                if (ruleId === "__proto__") {

                    /* eslint-disable-next-line no-proto -- Though deprecated, may still be present */
                    delete result.__proto__;
                    continue;
                }

                result[ruleId] = normalizeRuleOptions(result[ruleId]);

                /*
                 * If either rule config is missing, then the correct
                 * config is already present and we just need to normalize
                 * the severity.
                 */
                if (!(ruleId in first) || !(ruleId in second)) {
                    continue;
                }

                const firstRuleOptions = normalizeRuleOptions(first[ruleId]);
                const secondRuleOptions = normalizeRuleOptions(second[ruleId]);

                /*
                 * If the second rule config only has a severity (length of 1),
                 * then use that severity and keep the rest of the options from
                 * the first rule config.
                 */
                if (secondRuleOptions.length === 1) {
                    result[ruleId] = [secondRuleOptions[0], ...firstRuleOptions.slice(1)];
                    continue;
                }

                /*
                 * In any other situation, then the second rule config takes
                 * precedence. That means the value at `result[ruleId]` is
                 * already correct and no further work is necessary.
                 */
            } catch (ex) {
                throw new Error(`Key "${ruleId}": ${ex.message}`, { cause: ex });
            }

        }

        return result;


    },

    validate(value) {
        assertIsObject(value);

        /*
         * We are not checking the rule schema here because there is no
         * guarantee that the rule definition is present at this point. Instead
         * we wait and check the rule schema during the finalization step
         * of calculating a config.
         */
        for (const ruleId of Object.keys(value)) {

            // avoid hairy edge case
            if (ruleId === "__proto__") {
                continue;
            }

            const ruleOptions = value[ruleId];

            assertIsRuleOptions(ruleId, ruleOptions);

            if (Array.isArray(ruleOptions)) {
                assertIsRuleSeverity(ruleId, ruleOptions[0]);
            } else {
                assertIsRuleSeverity(ruleId, ruleOptions);
            }
        }
    }
};

/**
 * Creates a schema that always throws an error. Useful for warning
 * about eslintrc-style keys.
 * @param {string} key The eslintrc key to create a schema for.
 * @returns {ObjectPropertySchema} The schema.
 */

const eslintrcKeys = [
    "env",
    "extends",
    "globals",
    "ignorePatterns",
    "noInlineConfig",
    "overrides",
    "parser",
    "parserOptions",
    "reportUnusedDisableDirectives",
    "root"
];

//-----------------------------------------------------------------------------
// Full schema
//-----------------------------------------------------------------------------

const flatConfigSchema = {

    // eslintrc-style keys that should always error
    ...Object.fromEntries(eslintrcKeys.map(key => [key, createEslintrcErrorSchema(key)])),

    // flat config keys
    settings: deepObjectAssignSchema,
    linterOptions: {
        schema: {
            noInlineConfig: booleanSchema,
            reportUnusedDisableDirectives: disableDirectiveSeveritySchema
        }
    },
    language: languageSchema,
    languageOptions: languageOptionsSchema,
    processor: processorSchema,
    plugins: pluginsSchema,
    rules: rulesSchema
};

//-----------------------------------------------------------------------------
// Exports
//-----------------------------------------------------------------------------

module.exports = {
    flatConfigSchema,
    hasMethod,
    assertIsRuleSeverity
};
