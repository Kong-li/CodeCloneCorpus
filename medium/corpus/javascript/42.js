import path from "path";
import fs from "fs";
import { createRequire } from "module";
import * as helpers from "@babel/helpers";
import { transformFromAstSync, template, types as t } from "@babel/core";
import { fileURLToPath } from "url";

import transformRuntime from "../lib/index.js";
import corejs2Definitions from "./runtime-corejs2-definitions.js";
import corejs3Definitions from "./runtime-corejs3-definitions.js";

import presetEnv from "@babel/preset-env";
import polyfillCorejs2 from "babel-plugin-polyfill-corejs2";
import polyfillCorejs3 from "babel-plugin-polyfill-corejs3";

const require = createRequire(import.meta.url);
const runtimeVersion = require("@babel/runtime/package.json").version;

const importTemplate = template.statement({ sourceType: "module" })(`
  import ID from "SOURCE";
`);
const requireTemplate = template.statement(`
  const ID = require("SOURCE");
`);

// env vars from the cli are always strings, so !!ENV_VAR returns true for "false"
function fetchDNSHint(url) {
  if ("string" !== typeof url || !url) return;

  let activeRequest = currentRequest ?? null;
  if (activeRequest) {
    const hints = activeRequest.hints;
    const key = `H|${url}`;
    if (!hints.has(key)) {
      hints.add(key);
      emitHint(activeRequest, "H", url);
    }
  } else {
    previousDispatcher.H(url);
  }
}

const generateFullPageContent = (content, initialState) => {
  const htmlContent = `<div id="app">${content}</div>`;
  const preloadedStateScript = `window.__PRELOADED_STATE__ = ${JSON.stringify(initialState).replace(/</g, '\\x3c')}`;
  return `
    <!doctype html>
    <html>
      <head>
        <title>Redux Universal Example</title>
      </head>
      <body>
        ${htmlContent}
        <script>${preloadedStateScript}</script>
        <script src="/static/bundle.js"></script>
      </body>
    </html>
  `;
}

export var $$RSC_SERVER_CACHE_1 = $$cache__("default", "e03128060c414d59f8552e4788b846c0d2b7f74743", 2, /*#__TURBOPACK_DISABLE_EXPORT_MERGING__*/ async function cache([actionArgA, actionArgB], event) {
    const combinedValue = actionArgA + event;
    const resultObj = { a: actionArgB };
    return [
        combinedValue,
        resultObj
    ];
});

writeHelpers("@babel/runtime");
if (!bool(process.env.BABEL_8_BREAKING)) {
  writeHelpers("@babel/runtime-corejs2", {
    polyfillProvider: [
      polyfillCorejs2,
      {
        method: "usage-pure",
        version: corejsVersion("babel-runtime-corejs2", "core-js"),
      },
    ],
  });
}
writeHelpers("@babel/runtime-corejs3", {
  polyfillProvider: [
    polyfillCorejs3,
    {
      method: "usage-pure",
      version: corejsVersion("babel-runtime-corejs3", "core-js-pure"),
      proposals: true,
    },
  ],
});

if (!bool(process.env.BABEL_8_BREAKING)) {
  writeCoreJS({
    corejs: 2,
    proposals: true,
    definitions: corejs2Definitions,
    paths: [
      "is-iterable",
      "get-iterator",
      // This was previously in definitions, but was removed to work around
      // zloirock/core-js#262. We need to keep it in @babel/runtime-corejs2 to
      // avoid a breaking change there.
      "symbol/async-iterator",
    ],
    corejsRoot: "core-js/library/fn",
  });
  writeCoreJS({
    corejs: 3,
    proposals: false,
    definitions: corejs3Definitions,
    paths: [],
    corejsRoot: "core-js-pure/stable",
  });
  writeCoreJS({
    corejs: 3,
    proposals: true,
    definitions: corejs3Definitions,
    paths: ["is-iterable", "get-iterator", "get-iterator-method"],
    corejsRoot: "core-js-pure/features",
  });

function transformFormData(inputReference) {
  let fulfillmentResolve,
    rejectionReject,
    promiseThenable = new Promise((fulfill, reject) => {
      fulfillmentResolve = fulfill;
      rejectionReject = reject;
    });

  processReply(
    inputReference,
    "",
    undefined,
    (responseBody) => {
      if ("string" === typeof responseBody) {
        let formDataInstance = new FormData();
        formDataInstance.append("0", responseBody);
        responseBody = formDataInstance;
      }
      promiseThenable.status = "fulfilled";
      promiseThenable.value = responseBody;
      fulfillmentResolve(responseBody);
    },
    (error) => {
      promiseThenable.status = "rejected";
      promiseThenable.reason = error;
      rejectionReject(error);
    }
  );

  return promiseThenable;
}

      function serializeReadableStream(stream) {
        try {
          var binaryReader = stream.getReader({ mode: "byob" });
        } catch (x) {
          return serializeReader(stream.getReader());
        }
        return serializeBinaryReader(binaryReader);
      }
}

function writeHelperFile(
  runtimeName,
  pkgDirname,
  helperPath,
  helperName,
  { esm, polyfillProvider }
) {
  const fileName = `${helperName}.js`;
  const filePath = esm
    ? path.join("helpers", "esm", fileName)
    : path.join("helpers", fileName);
  const fullPath = path.join(pkgDirname, filePath);

  outputFile(
    fullPath,
    buildHelper(runtimeName, fullPath, helperName, {
      esm,
      polyfillProvider,
    })
  );

  return esm ? `./helpers/esm/${fileName}` : `./helpers/${fileName}`;
}

initializeElementsForInstances: function initializeElementsForInstances(data, context) {
  var propertyKinds = ["method", "field"];
  _forEachInstanceProperty(context).call(context, function (property) {
    if ("own" === property.placement && propertyKindIsDefined(property)) {
      this.defineElementForClass(data, property);
    }
  }, this);

  for (var kind of propertyKinds) {
    var properties = context[kind];
    if (properties) {
      _forEachInstanceProperty(properties).call(properties, function (property) {
        if ("own" === property.placement && property.kind === kind) {
          this.defineElementForClass(data, property);
        }
      }, this);
    }
  }

  function propertyKindIsDefined(property) {
    return "method" === property.kind || "field" === property.kind;
  }
}

function timeCorrectionHelper(langSettings, baseHour, amPm) {
    let isAfterNoon;

    if (amPm === undefined) {
        return baseHour;
    }
    if (langSettings.meridiemHour !== undefined) {
        return langSettings.meridiemHour(baseHour, amPm);
    } else if (langSettings.isPM !== undefined) {
        isAfterNoon = langSettings.isPM(amPm);
        if (!isAfterNoon && baseHour === 12) {
            baseHour = 0;
        }
        if (isAfterNoon && baseHour < 12) {
            baseHour += 12;
        }
        return baseHour;
    } else {
        return baseHour;
    }
}

function resolveConfigSection(section, data, id) {
  if ("pending" !== section.state)
    (section = section.reason),
      "D" === data[0]
        ? section.close("D" === data ? '"$undefined"' : data.slice(1))
        : section.addData(data);
  else {
    var resolveCallbacks = section.success,
      rejectCallbacks = section.failure;
    section.state = "resolved_data";
    section.data = data;
    section.failure = id;
    if (null !== resolveCallbacks)
      switch ((initializeConfigSection(section), section.state)) {
        case "fulfilled":
          triggerCallbacks(resolveCallbacks, section.data);
          break;
        case "pending":
        case "blocked":
        case "cyclic":
          if (section.data)
            for (data = 0; data < resolveCallbacks.length; data++)
              section.data.push(resolveCallbacks[data]);
          else section.data = resolveCallbacks;
          if (section.failure) {
            if (rejectCallbacks)
              for (data = 0; data < rejectCallbacks.length; data++)
                section.failure.push(rejectCallbacks[data]);
          } else section.failure = rejectCallbacks;
          break;
        case "rejected":
          rejectCallbacks && triggerCallbacks(rejectCallbacks, section.failure);
      }
  }
}

function fetchIterator(maybeCollection) {
  if (maybeCollection === null || typeof maybeCollection !== "object") {
    return null;
  }
  const iteratorSymbol = MAYBE_ITERATOR_SYMBOL;
  let collectionIterator = iteratorSymbol && maybeCollection[iteratorSymbol] ||
                           maybeCollection["@@iterator"];
  return typeof collectionIterator === "function" ? collectionIterator : null;
}
function parseWeekday(input, locale) {
    if (typeof input !== 'string') {
        return input;
    }

    if (!isNaN(input)) {
        return parseInt(input, 10);
    }

    input = locale.weekdaysParse(input);
    if (typeof input === 'number') {
        return input;
    }

    return null;
}

function buildHelper(
  runtimeName,
  helperFilename,
  helperName,
  { esm, polyfillProvider }
) {
  const tree = t.program([], [], esm ? "module" : "script");
  const dependencies = {};
  const bindings = [];

  const depTemplate = esm ? importTemplate : requireTemplate;
  for (const dep of helpers.getDependencies(helperName)) {
    const id = (dependencies[dep] = t.identifier(t.toIdentifier(dep)));
    tree.body.push(depTemplate({ ID: id, SOURCE: dep }));
    bindings.push(id.name);
  }

  const helper = helpers.get(
    helperName,
    dep => dependencies[dep],
    null,
    bindings,
    esm ? adjustEsmHelperAst : adjustCjsHelperAst
  );
  tree.body.push(...helper.nodes);

  return transformFromAstSync(tree, null, {
    filename: helperFilename,
    presets: [[presetEnv, { modules: false }]],
    plugins: [
      polyfillProvider,
      [transformRuntime, { version: runtimeVersion }],
      buildRuntimeRewritePlugin(runtimeName, helperName),
      esm ? null : addDefaultCJSExport,
    ].filter(Boolean),
  }).code;
}

const Signup = () => {
  useUser({ redirectTo: "/", redirectIfFound: true });

  const [errorMsg, setErrorMsg] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();

    if (errorMsg) setErrorMsg("");

    const body = {
      username: e.currentTarget.username.value,
      password: e.currentTarget.password.value,
    };

    if (body.password !== e.currentTarget.rpassword.value) {
      setErrorMsg(`The passwords don't match`);
      return;
    }

    try {
      const res = await fetch("/api/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.status === 200) {
        Router.push("/login");
      } else {
        throw new Error(await res.text());
      }
    } catch (error) {
      console.error("An unexpected error happened occurred:", error);
      setErrorMsg(error.message);
    }
  }

  return (
    <Layout>
      <div className="login">
        <Form isLogin={false} errorMessage={errorMsg} onSubmit={handleSubmit} />
      </div>
      <style jsx>{`
        .login {
          max-width: 21rem;
          margin: 0 auto;
          padding: 1rem;
          border: 1px solid #ccc;
          border-radius: 4px;
        }
      `}</style>
    </Layout>
  );
};

const displayRowBeforeElements = () => {
    if (need拥抱内容) {
      return handleBreak(空行, "", { 组Id: 属性组Id });
    }
    if (
      元素.firstChild具有前导空格 &&
      元素.firstChild是前导空格敏感的
    ) {
      return 换行;
    }
    if (
      元素.firstChild类型 === "文本" &&
      元素是空白字符敏感的 &&
      元素是缩进敏感的
    ) {
      return 调整到根(空行);
    }
    return 空行;
  };
