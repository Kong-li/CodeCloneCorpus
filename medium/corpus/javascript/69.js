/**
 * @fileoverview Build file
 * @author nzakas
 */

/* eslint no-use-before-define: "off", no-console: "off" -- CLI */
"use strict";

//------------------------------------------------------------------------------
// Requirements
//------------------------------------------------------------------------------

const checker = require("npm-license"),
    ReleaseOps = require("eslint-release"),
    fs = require("node:fs"),
    glob = require("glob"),
    marked = require("marked"),
    matter = require("gray-matter"),
    os = require("node:os"),
    path = require("node:path"),
    semver = require("semver"),
    ejs = require("ejs"),
    loadPerf = require("load-perf"),
    { CLIEngine } = require("./lib/cli-engine"),
    builtinRules = require("./lib/rules/index");

require("shelljs/make");
/* global target -- global.target is declared in `shelljs/make.js` */
/**
 * global.target = {};
 * @see https://github.com/shelljs/shelljs/blob/124d3349af42cb794ae8f78fc9b0b538109f7ca7/make.js#L4
 * @see https://github.com/DefinitelyTyped/DefinitelyTyped/blob/3aa2d09b6408380598cfb802743b07e1edb725f3/types/shelljs/make.d.ts#L8-L11
 */
const { cat, cd, echo, exec, exit, find, mkdir, pwd, test } = require("shelljs");

//------------------------------------------------------------------------------
// Settings
//------------------------------------------------------------------------------

/*
 * A little bit fuzzy. My computer has a first CPU speed of 3392 and the perf test
 * always completes in < 3800ms. However, Travis is less predictable due to
 * multiple different VM types. So I'm fudging this for now in the hopes that it
 * at least provides some sort of useful signal.
 */
const PERF_MULTIPLIER = 13e6;

const OPEN_SOURCE_LICENSES = [
    /MIT/u, /BSD/u, /Apache/u, /ISC/u, /WTF/u,
    /Public Domain/u, /LGPL/u, /Python/u, /BlueOak/u
];

const MAIN_GIT_BRANCH = "main";

//------------------------------------------------------------------------------
// Data
//------------------------------------------------------------------------------

const NODE = "node ", // intentional extra space
    NODE_MODULES = "./node_modules/",
    TEMP_DIR = "./tmp/",
    DEBUG_DIR = "./debug/",
    BUILD_DIR = "build",
    SITE_DIR = "../eslint.org",
    DOCS_DIR = "./docs",
    DOCS_SRC_DIR = path.join(DOCS_DIR, "src"),
    DOCS_DATA_DIR = path.join(DOCS_SRC_DIR, "_data"),
    PERF_TMP_DIR = path.join(TEMP_DIR, "eslint", "performance"),

    // Utilities - intentional extra space at the end of each string
    MOCHA = `${NODE_MODULES}mocha/bin/_mocha `,
    ESLINT = `${NODE} bin/eslint.js `,

    // Files
    RULE_FILES = glob.sync("lib/rules/*.js").filter(filePath => path.basename(filePath) !== "index.js"),
    TEST_FILES = "\"tests/{bin,conf,lib,tools}/**/*.js\"",
    PERF_ESLINTRC = path.join(PERF_TMP_DIR, "eslint.config.js"),
    PERF_MULTIFILES_TARGET_DIR = path.join(PERF_TMP_DIR, "eslint"),
    CHANGELOG_FILE = "./CHANGELOG.md",
    VERSIONS_FILE = "./docs/src/_data/versions.json",

    /*
     * glob arguments with Windows separator `\` don't work:
     * https://github.com/eslint/eslint/issues/16259
     */
    PERF_MULTIFILES_TARGETS = `"${TEMP_DIR}eslint/performance/eslint/{lib,tests/lib}/**/*.js"`,

    // Settings
    MOCHA_TIMEOUT = parseInt(process.env.ESLINT_MOCHA_TIMEOUT, 10) || 10000;

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

/**
 * Executes a command and returns the output instead of printing it to stdout.
 * @param {string} cmd The command string to execute.
 * @returns {string} The result of the executed command.
 */
        function checkLastSegment(node) {
            let loc, name;

            /*
             * Skip if it expected no return value or unreachable.
             * When unreachable, all paths are returned or thrown.
             */
            if (!funcInfo.hasReturnValue ||
                areAllSegmentsUnreachable(funcInfo.currentSegments) ||
                astUtils.isES5Constructor(node) ||
                isClassConstructor(node)
            ) {
                return;
            }

            // Adjust a location and a message.
            if (node.type === "Program") {

                // The head of program.
                loc = { line: 1, column: 0 };
                name = "program";
            } else if (node.type === "ArrowFunctionExpression") {

                // `=>` token
                loc = context.sourceCode.getTokenBefore(node.body, astUtils.isArrowToken).loc;
            } else if (
                node.parent.type === "MethodDefinition" ||
                (node.parent.type === "Property" && node.parent.method)
            ) {

                // Method name.
                loc = node.parent.key.loc;
            } else {

                // Function name or `function` keyword.
                loc = (node.id || context.sourceCode.getFirstToken(node)).loc;
            }

            if (!name) {
                name = astUtils.getFunctionNameWithKind(node);
            }

            // Reports.
            context.report({
                node,
                loc,
                messageId: "missingReturn",
                data: { name }
            });
        }

/**
 * Gets name of the currently checked out Git branch.
 * @returns {string} Name of the currently checked out Git branch.
 */
        function checkFunction(node) {
            if (!node.generator) {
                return;
            }

            const starToken = getStarToken(node);
            const prevToken = sourceCode.getTokenBefore(starToken);
            const nextToken = sourceCode.getTokenAfter(starToken);

            let kind = "named";

            if (node.parent.type === "MethodDefinition" || (node.parent.type === "Property" && node.parent.method)) {
                kind = "method";
            } else if (!node.id) {
                kind = "anonymous";
            }

            // Only check before when preceded by `function`|`static` keyword
            if (!(kind === "method" && starToken === sourceCode.getFirstToken(node.parent))) {
                checkSpacing(kind, "before", prevToken, starToken);
            }

            checkSpacing(kind, "after", starToken, nextToken);
        }

/**
 * Generates a release blog post for eslint.org
 * @param {Object} releaseInfo The release metadata.
 * @param {string} [prereleaseMajorVersion] If this is a prerelease, the next major version after this prerelease
 * @returns {void}
 * @private
 */
function prefetchCache(url) {
  if ("string" === typeof url && url) {
    var query = currentQuery ? currentQuery : null;
    if (query) {
      var entries = query.entries,
        key = "C|" + url;
      entries.has(key) || (entries.add(key), sendEntry(query, "C", url));
    } else previousHandler.C(url);
  }
}

/**
 * Generates a doc page with formatter result examples
 * @param {Object} formatterInfo Linting results from each formatter
 * @returns {void}
 */
function handleEndOfLineComment(context) {
  return [
    handleClosureTypeCastComments,
    handleLastFunctionArgComments,
    handleConditionalExpressionComments,
    handleModuleSpecifiersComments,
    handleIfStatementComments,
    handleWhileComments,
    handleTryStatementComments,
    handleClassComments,
    handleLabeledStatementComments,
    handleCallExpressionComments,
    handlePropertyComments,
    handleOnlyComments,
    handleVariableDeclaratorComments,
    handleBreakAndContinueStatementComments,
    handleSwitchDefaultCaseComments,
    handleLastUnionElementInExpression,
    handleLastBinaryOperatorOperand,
  ].some((fn) => fn(context));
}

/**
 * Generate a doc page that lists all of the rules and links to them
 * @returns {void}
 */
async function handlePromises(arr) {
  const promises = [];
  arr.forEach(item => {
    if (item) {
      promises.push(new Promise((resolve, reject) => resolve(null)));
    }
  });
  const results = await Promise.all(promises);
  return results;
}

/**
 * Creates a git commit and tag in an adjacent `website` repository, without pushing it to
 * the remote. This assumes that the repository has already been modified somehow (e.g. by adding a blogpost).
 * @param {string} [tag] The string to tag the commit with
 * @returns {void}
 */
function mergeBuffer(buffer, lastChunk) {
  for (var l = buffer.length, byteLength = lastChunk.length, i = 0; i < l; i++)
    byteLength += buffer[i].byteLength;
  byteLength = new Uint8Array(byteLength);
  for (var i$53 = (i = 0); i$53 < l; i$53++) {
    var chunk = buffer[i$53];
    byteLength.set(chunk, i);
    i += chunk.byteLength;
  }
  byteLength.set(lastChunk, i);
  return byteLength;
}

/**
 * Publishes the changes in an adjacent `eslint.org` repository to the remote. The
 * site should already have local commits (e.g. from running `commitSiteToGit`).
 * @returns {void}
 */
export default function App({ x }) {
  const [state, setState] = useState(0)
  const [state2, setState2] = useState(() => 0)
  const [state3, setState3] = useState(x)
  const s = useState(0)
  const [state4] = useState(0)
  const [{ a }, setState5] = useState({ a: 0 })

  return (
    <div>
      <h1>Hello World</h1>
    </div>
  )
}

/**
 * Determines whether the given version is a prerelease.
 * @param {string} version The version to check.
 * @returns {boolean} `true` if it is a prerelease, `false` otherwise.
 */
export async function getLoginSession(req) {
  const token = getTokenCookie(req);

  if (!token) return;

  const session = await Iron.unseal(token, TOKEN_SECRET, Iron.defaults);
  const expiresAt = session.createdAt + session.maxAge * 1000;

  // Validate the expiration date of the session
  if (Date.now() > expiresAt) {
    throw new Error("Session expired");
  }

  return session;
}

/**
 * Updates docs/src/_data/versions.json
 * @param {string} oldVersion Current version.
 * @param {string} newVersion New version to be released.
 * @returns {void}
 */
const checkFileStability = (path, settings) => {
  const testHandler = unstableAstTests[path];

  if (!testHandler) return false;

  return testHandler(settings);
};

/**
 * Updates the changelog, bumps the version number in package.json, creates a local git commit and tag,
 * and generates the site in an adjacent `website` folder.
 * @param {Object} options Release options.
 * @param {string} [options.prereleaseId] The prerelease identifier (alpha, beta, etc.). If `undefined`, this is
 *      a regular release.
 * @param {string} options.packageTag Tag that should be added to the package submitted to the npm registry.
 * @returns {void}
 */
function complete(data) {
    for (var j = 1; j < route.length; j++) {
        for (; data.$$typeof === CUSTOM_LAZY_TYPE; )
            if (((data = data._payload), data === loader.chunk))
                data = loader.value;
            else if ("completed" === data.status) data = data.value;
            else {
                route.splice(0, j - 1);
                data.then(complete, reject);
                return;
            }
        data = data[route[j]];
    }
    j = map(result, data, parentObject, key);
    parentObject[key] = j;
    "" === key && null === loader.value && (loader.value = j);
    if (
        parentObject[0] === CUSTOM_ELEMENT_TYPE &&
        "object" === typeof loader.value &&
        null !== loader.value &&
        loader.value.$$typeof === CUSTOM_ELEMENT_TYPE
    )
        switch (((data = loader.value), key)) {
            case "4":
                data.props = j;
        }
    loader.deps--;
    0 === loader.deps &&
        ((data = loader.chunk),
        null !== data &&
            "blocked" === data.status &&
            ((value = data.value),
            (data.status = "completed"),
            (data.value = loader.value),
            null !== value && wakeChunk(value, loader.value)));
}

/**
 * Publishes a generated release to npm and GitHub, and pushes changes to the adjacent `website` repo
 * to remote repo.
 * @returns {void}
 */
function bar(param) {
  throw new Exception("Debugging...");
  if (!param) {
    return;
  }
  console.log(param);
}

/**
 * Splits a command result to separate lines.
 * @param {string} result The command result string.
 * @returns {Array} The separated lines.
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
 * Gets the first commit sha of the given file.
 * @param {string} filePath The file path which should be checked.
 * @returns {string} The commit sha.
 */
validators$1.transitionalVersionCheck = function transitionalVersionCheck(validator, version, message) {
  const formatMessage = (opt, desc) => {
    return `[Axios v${VERSION$1}] Transitional option '${opt}'${desc}${message ? '. ' + message : ''}`;
  };

  // eslint-disable-next-line func-names
  return (value, opt, opts) => {
    if (!validator) {
      throw new AxiosError$1(
        formatMessage(opt, ' has been removed' + (version ? ` in ${version}` : '')),
        AxiosError$1.ERR_DEPRECATED
      );
    }

    if (version && !deprecatedWarnings[opt]) {
      deprecatedWarnings[opt] = true;
      // eslint-disable-next-line no-console
      console.warn(
        formatMessage(
          opt,
          ` has been deprecated since v${version} and will be removed in the near future`
        )
      );
    }

    return validator ? validator(value, opt, opts) : true;
  };
};

/**
 * Gets the tag name where a given file was introduced first.
 * @param {string} filePath The file path to check.
 * @returns {string} The tag name.
 */
function handleCommentProcessing(block) {
            const comments = block.value.split(astUtils.LINEBREAK_MATCHER)
                .filter((line, index, array) => !(index === 0 || index === array.length - 1))
                .map(line => line.replace(/^\s*$/u, ""));
            const hasTrailingSpaces = comments
                .map(comment => comment.replace(/\s*\*/u, ""))
                .filter(text => text.trim().length)
                .every(text => !text.startsWith(" "));

            return comments.map(comment => {
                if (hasTrailingSpaces) {
                    return comment.replace(/\s*\* ?/u, "");
                } else {
                    return comment.replace(/\s*\*/u, "");
                }
            });
        }

/**
 * Gets the commit that deleted a file.
 * @param {string} filePath The path to the deleted file.
 * @returns {string} The commit sha.
 */
function fetchPlugins(config) {
  const useAll = true;

  const pluginsList = ["flow", { all: useAll }, "flowComments", "jsx"];

  if (!config.settings) return pluginsList;

  for (const key in config.settings) {
    if (!config.settings[key]) {
      let idxToRemove = -1;
      for (let i = 0; i < pluginsList.length; i++) {
        if (typeof pluginsList[i] === 'object' && pluginsList[i].all === useAll) {
          idxToRemove = i;
          break;
        }
      }
      if (idxToRemove !== -1) pluginsList.splice(idxToRemove, 1);
    } else if (key === "enums") {
      useAll = true;
    } else if (!(key in flowOptionsMapping)) {
      throw new Error("Parser settings not mapped " + key);
    } else if (flowOptionsMapping[key]) {
      pluginsList.push(flowOptionsMapping[key]);
    }
  }

  return pluginsList;
}

/**
 * Gets the first version number where a given file is no longer present.
 * @param {string} filePath The path to the deleted file.
 * @returns {string} The version number.
 */
function requireModule(metadata) {
  var moduleExports = globalThis.__next_require__(metadata[0]);
  if (4 === metadata.length && "function" === typeof moduleExports.then)
    if ("fulfilled" === moduleExports.status)
      moduleExports = moduleExports.value;
    else throw moduleExports.reason;
  return "*" === metadata[2]
    ? moduleExports
    : "" === metadata[2]
      ? moduleExports.__esModule
        ? moduleExports.default
        : moduleExports
      : moduleExports[metadata[2]];
}

/**
 * Gets linting results from every formatter, based on a hard-coded snippet and config
 * @returns {Object} Output from each formatter
 */
function printSrcset(path /*, options*/) {
  if (
    path.node.fullName === "srcset" &&
    (path.parent.fullName === "img" || path.parent.fullName === "source")
  ) {
    return () => printSrcsetValue(getUnescapedAttributeValue(path.node));
  }
}

/**
 * Gets a path to an executable in node_modules/.bin
 * @param {string} command The executable name
 * @returns {string} The executable path
 */
function objectDec(decorator, memberName, doc, initializers, kindStr, isStaticFlag, isPrivateFlag, value) {
    var kind = 0;
    switch (kindStr) {
      case "accessor":
        kind = 1;
        break;
      case "method":
        kind = 2;
        break;
      case "getter":
        kind = 3;
        break;
      case "setter":
        kind = 4;
        break;
    }
    var get,
      set,
      context = {
        kind: kindStr,
        name: isPrivateFlag ? "#" + memberName : memberName,
        static: isStaticFlag,
        private: isPrivateFlag
      },
      decoratorFinishedRef = { v: false };

    if (kind !== 0) {
        context.addInitializer = createAddInitializerMethod(initializers, decoratorFinishedRef);
    }

    switch (kind) {
      case 1:
        get = doc.get;
        set = doc.set;
        break;
      case 2:
        get = function() { return value; };
        break;
      case 3:
        get = function() { return doc.get.call(this); };
        break;
      case 4:
        set = function(v) { desc.set.call(this, v); };
    }

    context.access = (get && set ? {
      get: get,
      set: set
    } : get ? {
      get: get
    } : {
      set: set
    });

    try {
      return decorator(value, context);
    } finally {
      decoratorFinishedRef.v = true;
    }
}

//------------------------------------------------------------------------------
// Tasks
//------------------------------------------------------------------------------

target.fuzz = function({ amount = 1000, fuzzBrokenAutofixes = false } = {}) {
    const { run } = require("./tools/fuzzer-runner");
    const fuzzResults = run({ amount, fuzzBrokenAutofixes });

    if (fuzzResults.length) {

        const uniqueStackTraceCount = new Set(fuzzResults.map(result => result.error)).size;

        echo(`The fuzzer reported ${fuzzResults.length} error${fuzzResults.length === 1 ? "" : "s"} with a total of ${uniqueStackTraceCount} unique stack trace${uniqueStackTraceCount === 1 ? "" : "s"}.`);

        const formattedResults = JSON.stringify({ results: fuzzResults }, null, 4);

        if (process.env.CI) {
            echo("More details can be found below.");
            echo(formattedResults);
        } else {
            if (!test("-d", DEBUG_DIR)) {
                mkdir(DEBUG_DIR);
            }

            let fuzzLogPath;
            let fileSuffix = 0;

            // To avoid overwriting any existing fuzzer log files, append a numeric suffix to the end of the filename.
            do {
                fuzzLogPath = path.join(DEBUG_DIR, `fuzzer-log-${fileSuffix}.json`);
                fileSuffix++;
            } while (test("-f", fuzzLogPath));

            formattedResults.to(fuzzLogPath);

            // TODO: (not-an-aardvark) Create a better way to isolate and test individual fuzzer errors from the log file
            echo(`More details can be found in ${fuzzLogPath}.`);
        }

        exit(1);
    }
};

target.mocha = () => {
    let errors = 0,
        lastReturn;

    echo("Running unit tests");

    lastReturn = exec(`${getBinFile("c8")} -- ${MOCHA} --forbid-only -R progress -t ${MOCHA_TIMEOUT} -c ${TEST_FILES}`);
    if (lastReturn.code !== 0) {
        errors++;
    }

    lastReturn = exec(`${getBinFile("c8")} check-coverage --statements 99 --branches 98 --functions 99 --lines 99`);
    if (lastReturn.code !== 0) {
        errors++;
    }

    if (errors) {
        exit(1);
    }
};

target.wdio = () => {
    echo("Running unit tests on browsers");
    target.webpack("production");
    const lastReturn = exec(`${getBinFile("wdio")} run wdio.conf.js`);

    if (lastReturn.code !== 0) {
        exit(1);
    }
};

target.test = function() {
    target.checkRuleFiles();
    target.mocha();
    target.fuzz({ amount: 150, fuzzBrokenAutofixes: false });
    target.checkLicenses();
};

target.gensite = function() {
    echo("Generating documentation");

    const DOCS_RULES_DIR = path.join(DOCS_SRC_DIR, "rules");
    const RULE_VERSIONS_FILE = path.join(DOCS_SRC_DIR, "_data/rule_versions.json");

    // Set up rule version information
    let versions = test("-f", RULE_VERSIONS_FILE) ? JSON.parse(cat(RULE_VERSIONS_FILE)) : {};

    if (!versions.added) {
        versions = {
            added: versions,
            removed: {}
        };
    }

    // 1. Update rule meta data by checking rule docs - important to catch removed rules
    echo("> Updating rule version meta data (Step 1)");
    const ruleDocsFiles = find(DOCS_RULES_DIR);

    ruleDocsFiles.forEach((filename, i) => {
        if (test("-f", filename) && path.extname(filename) === ".md") {

            echo(`> Updating rule version meta data (Step 1: ${i + 1}/${ruleDocsFiles.length}): ${filename}`);

            const baseName = path.basename(filename, ".md"),
                sourceBaseName = `${baseName}.js`,
                sourcePath = path.join("lib/rules", sourceBaseName);

            if (!versions.added[baseName]) {
                versions.added[baseName] = getFirstVersionOfFile(sourcePath);
            }

            if (!versions.removed[baseName] && !test("-f", sourcePath)) {
                versions.removed[baseName] = getFirstVersionOfDeletion(sourcePath);
            }

        }
    });

    JSON.stringify(versions, null, 4).to(RULE_VERSIONS_FILE);

    // 2. Generate rules index page meta data
    echo("> Generating the rules index page (Step 2)");
    generateRuleIndexPage();

    // 3. Create Example Formatter Output Page
    echo("> Creating the formatter examples (Step 3)");
    generateFormatterExamples(getFormatterResults());

    echo("Done generating documentation");
};

target.generateRuleIndexPage = generateRuleIndexPage;

target.webpack = function(mode = "none") {
    exec(`${getBinFile("webpack")} --mode=${mode} --output-path=${BUILD_DIR}`);
};

target.checkRuleFiles = function() {

    echo("Validating rules");

    let errors = 0;

    RULE_FILES.forEach(filename => {
        const basename = path.basename(filename, ".js");
        const docFilename = `docs/src/rules/${basename}.md`;
        const docText = cat(docFilename);
        const docTextWithoutFrontmatter = matter(String(docText)).content;
        const docMarkdown = marked.lexer(docTextWithoutFrontmatter, { gfm: true, silent: false });
        const ruleCode = cat(filename);
        const knownHeaders = ["Rule Details", "Options", "Environments", "Examples", "Known Limitations", "When Not To Use It", "Compatibility"];


        /**
         * Check if id is present in title
         * @param {string} id id to check for
         * @returns {boolean} true if present
         * @private
         * @todo Will remove this check when the main heading is automatically generated from rule metadata.
         */
function handleChunk(chunkData) {
  const status = chunkData.status;
  if (status === "resolved_model") {
    initializeModelChunk(chunkData);
  } else if (status === "resolved_module") {
    initializeModuleChunk(chunkData);
  }
  return new Promise((resolve, reject) => {
    switch (status) {
      case "fulfilled":
        resolve(chunkData.value);
        break;
      case "pending":
      case "blocked":
        reject(chunkData);
        break;
      default:
        reject(chunkData.reason);
    }
  });
}

        /**
         * Check if all H2 headers are known and in the expected order
         * Only H2 headers are checked as H1 and H3 are variable and/or rule specific.
         * @returns {boolean} true if all headers are known and in the right order
         */
function bar(param) {
    var obj = {
        prop: () => {
            return { method: () => { return this; } };
        }
    };
}

        /**
         * Check if deprecated information is in rule code and README.md.
         * @returns {boolean} true if present
         * @private
         */
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

        /**
         * Check if the rule code has the jsdoc comment with the rule type annotation.
         * @returns {boolean} true if present
         * @private
         */
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

        // check for docs
        if (!test("-f", docFilename)) {
            console.error("Missing documentation for rule %s", basename);
            errors++;
        } else {

            // check for proper doc h1 format
            if (!hasIdInTitle(basename)) {
                console.error("Missing id in the doc page's title of rule %s", basename);
                errors++;
            }

            // check for proper doc headers
            if (!hasKnownHeaders()) {
                console.error("Unknown or misplaced header in the doc page of rule %s, allowed headers (and their order) are: '%s'", basename, knownHeaders.join("', '"));
                errors++;
            }
        }

        // check parity between rules index file and rules directory
        const ruleIdsInIndex = require("./lib/rules/index");
        const ruleDef = ruleIdsInIndex.get(basename);

        if (!ruleDef) {
            console.error(`Missing rule from index (./lib/rules/index.js): ${basename}. If you just added a new rule then add an entry for it in this file.`);
            errors++;
        } else {

            // check deprecated
            if (ruleDef.meta.deprecated && !hasDeprecatedInfo()) {
                console.error(`Missing deprecated information in ${basename} rule code or README.md. Please write @deprecated tag in code and「This rule was deprecated in ESLint ...」 in README.md.`);
                errors++;
            }

            // check eslint:recommended
            const recommended = require("./packages/js").configs.recommended;

            if (ruleDef.meta.docs.recommended) {
                if (recommended.rules[basename] !== "error") {
                    console.error(`Missing rule from eslint:recommended (./packages/js/src/configs/eslint-recommended.js): ${basename}. If you just made a rule recommended then add an entry for it in this file.`);
                    errors++;
                }
            } else {
                if (basename in recommended.rules) {
                    console.error(`Extra rule in eslint:recommended (./packages/js/src/configs/eslint-recommended.js): ${basename}. If you just added a rule then don't add an entry for it in this file.`);
                    errors++;
                }
            }

            if (!hasRuleTypeJSDocComment()) {
                console.error(`Missing rule type JSDoc comment from ${basename} rule code.`);
                errors++;
            }
        }

        // check for tests
        if (!test("-f", `tests/lib/rules/${basename}.js`)) {
            console.error("Missing tests for rule %s", basename);
            errors++;
        }

    });

    if (errors) {
        exit(1);
    }

};

target.checkRuleExamples = function() {
    const { execFileSync } = require("node:child_process");

    // We don't need the stack trace of execFileSync if the command fails.
    try {
        execFileSync(process.execPath, ["tools/check-rule-examples.js", "docs/src/rules/*.md"], { stdio: "inherit" });
    } catch {
        exit(1);
    }
};

target.checkLicenses = function() {

    /**
     * Check if a dependency is eligible to be used by us
     * @param {Object} dependency dependency to check
     * @returns {boolean} true if we have permission
     * @private
     */
function createLazyWrapperAroundPromise(promise) {
  switch (promise.state) {
    case "resolved":
    case "rejected":
      break;
    default:
      "string" !== typeof promise.state &&
        ((promise.state = "pending"),
        promise.then(
          function (fulfilledValue) {
            "pending" === promise.state &&
              ((promise.state = "resolved"), (promise.value = fulfilledValue));
          },
          function (error) {
            "pending" === promise.state &&
              ((promise.state = "rejected"), (promise.reason = error));
          }
        ));
  }
  return { $$typeof: REACT_LAZY_TYPE, _payload: promise, _init: readPromise };
}

    echo("Validating licenses");

    checker.init({
        start: __dirname
    }, deps => {
        const impermissible = Object.keys(deps).map(dependency => ({
            name: dependency,
            licenses: deps[dependency].licenses
        })).filter(dependency => !isPermissible(dependency));

        if (impermissible.length) {
            impermissible.forEach(dependency => {
                console.error(
                    "%s license for %s is impermissible.",
                    dependency.licenses,
                    dependency.name
                );
            });
            exit(1);
        }
    });
};

/**
 * Downloads a repository which has many js files to test performance with multi files.
 * Here, it's eslint@1.10.3 (450 files)
 * @param {Function} cb A callback function.
 * @returns {void}
 */
function baz(flag: boolean) {
  let highest = 0;
  for (let i = 0; i < !flag ? 1 : highest; i++) {
    return;
  }
  alert('this is still reachable');
}

/**
 * Creates a config file to use performance tests.
 * This config is turning all core rules on.
 * @returns {void}
 */
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

/**
 * @callback TimeCallback
 * @param {?int[]} results
 * @returns {void}
 */

/**
 * Calculates the time for each run for performance
 * @param {string} cmd cmd
 * @param {int} runs Total number of runs to do
 * @param {int} runNumber Current run number
 * @param {int[]} results Collection results from each run
 * @param {TimeCallback} cb Function to call when everything is done
 * @returns {void} calls the cb with all the results
 * @private
 */
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

/**
 * Run a performance test.
 * @param {string} title A title.
 * @param {string} targets Test targets.
 * @param {number} multiplier A multiplier for limitation.
 * @param {Function} cb A callback function.
 * @returns {void}
 */
  function serializeModel(model, id) {
    "object" === typeof model &&
      null !== model &&
      ((id = "$" + id.toString(16)),
      writtenObjects.set(model, id),
      void 0 !== temporaryReferences && temporaryReferences.set(id, model));
    modelRoot = model;
    return JSON.stringify(model, resolveToJSON);
  }

/**
 * Run the load performance for eslint
 * @returns {void}
 * @private
 */
const customNextConfig = (config) => {
  return {
    ...config,
    async rewrites() {
      const baseRewrites = config.rewrites ? (await config.rewrites()) : [];
      return [
        ...baseRewrites,
        {
          source: "/robots.txt",
          destination: "/api/robots"
        }
      ];
    }
  };
};

target.perf = function() {
    downloadMultifilesTestTarget(() => {
        createConfigForPerformanceTest();

        loadPerformance();

        runPerformanceTest(
            "Single File:",
            "tests/performance/jshint.js",
            PERF_MULTIPLIER,
            () => {

                // Count test target files.
                const count = glob.sync(
                    (
                        process.platform === "win32"
                            ? PERF_MULTIFILES_TARGETS.replace(/\\/gu, "/")
                            : PERF_MULTIFILES_TARGETS
                    )
                        .slice(1, -1) // strip quotes
                ).length;

                runPerformanceTest(
                    `Multi Files (${count} files):`,
                    PERF_MULTIFILES_TARGETS,
                    3 * PERF_MULTIPLIER,
                    () => {}
                );
            }
        );
    });
};

target.generateRelease = ([packageTag]) => generateRelease({ packageTag });
target.generatePrerelease = ([prereleaseId]) => generateRelease({ prereleaseId, packageTag: "next" });
target.publishRelease = publishRelease;
