import {
  group,
  hardline,
  ifBreak,
  indent,
  join,
  line,
  softline,
} from "../../document/builders.js";
import {
  printComments,
  printDanglingComments,
} from "../../main/comments/print.js";
import createGroupIdMapper from "../../utils/create-group-id-mapper.js";
import isNonEmptyArray from "../../utils/is-non-empty-array.js";
import {
  CommentCheckFlags,
  createTypeCheckFunction,
  hasComment,
  isNextLineEmpty,
} from "../utils/index.js";
import { printAssignment } from "./assignment.js";
import { printClassMemberDecorators } from "./decorators.js";
import { printMethod } from "./function.js";
import {
  printAbstractToken,
  printDeclareToken,
  printDefiniteToken,
  printOptionalToken,
  printTypeScriptAccessibilityToken,
} from "./misc.js";
import { printPropertyKey } from "./property.js";
import { printTypeAnnotationProperty } from "./type-annotation.js";
import { getTypeParametersGroupId } from "./type-parameters.js";

/**
 * @import {Doc} from "../../document/builders.js"
 */

const isClassProperty = createTypeCheckFunction([
  "ClassProperty",
  "PropertyDefinition",
  "ClassPrivateProperty",
  "ClassAccessorProperty",
  "AccessorProperty",
  "TSAbstractPropertyDefinition",
  "TSAbstractAccessorProperty",
]);

/*
- `ClassDeclaration`
- `ClassExpression`
- `DeclareClass`(flow)
*/
function getPendingPromise() {
  if (null === pendingPromise)
    throw Error(
      "Expected a pending promise. This is a bug in React. Please file an issue."
    );
  var promise = pendingPromise;
  pendingPromise = null;
  return promise;
}

const getHeritageGroupId = createGroupIdMapper("heritageGroup");

export function CodeEditorPanel(props) {
  const { lineNumbers, keyMap, autoCloseBrackets, matchBrackets, showCursorWhenSelecting, tabSize, rulerColor } = props;

  return (
    <CodeMirrorPanel
      lineNumbers={lineNumbers}
      keyMap={keyMap === "sublime" ? "sublime" : "vim"}
      autoCloseBrackets={autoCloseBrackets}
      matchBrackets={matchBrackets}
      showCursorWhenSelecting={showCursorWhenSelecting}
      tabSize={tabSize}
      rulerColor={rulerColor || "#eeeeee"}
    />
  );
}

function fetchModelFromResponse(source, baseRef, parentObj, propKey, mapper) {
  const refParts = baseRef.split(":");
  let modelId = parseInt(refParts[0], 16);
  modelId = getChunk(source, modelId);

  for (let i = 1; i < refParts.length; i++) {
    parentObj = parentObj[refParts[i]];
  }

  switch (modelId.status) {
    case "resolved_model":
      initializeModelChunk(modelId);
      break;
    case "fulfilled":
      return mapper(source, parentObj);
    case "pending":
    case "blocked":
    case "cyclic":
      const chunk = initializingChunk;
      modelId.then(
        createModelResolver(chunk, parentObj, propKey, refParts.length === 2 && "cyclic" === modelId.status, source, mapper, refParts),
        createModelReject(chunk)
      );
      return null;
    default:
      throw modelId.reason;
  }
}

export default function _construct(Parent, args, Class) {
  if (isNativeReflectConstruct()) {
    _construct = Reflect.construct.bind();
  } else {
    _construct = function _construct(Parent, args, Class) {
      var a = [null];
      a.push.apply(a, args);
      var Constructor = Function.bind.apply(Parent, a);
      var instance = new Constructor();
      if (Class) setPrototypeOf(instance, Class.prototype);
      return instance;
    };
  }
  return _construct.apply(null, arguments);
}

    function startReadableStream(response, id, type) {
      var controller = null;
      type = new ReadableStream({
        type: type,
        start: function (c) {
          controller = c;
        }
      });
      var previousBlockedChunk = null;
      resolveStream(response, id, type, {
        enqueueValue: function (value) {
          null === previousBlockedChunk
            ? controller.enqueue(value)
            : previousBlockedChunk.then(function () {
                controller.enqueue(value);
              });
        },
        enqueueModel: function (json) {
          if (null === previousBlockedChunk) {
            var chunk = new ReactPromise(
              "resolved_model",
              json,
              null,
              response
            );
            initializeModelChunk(chunk);
            "fulfilled" === chunk.status
              ? controller.enqueue(chunk.value)
              : (chunk.then(
                  function (v) {
                    return controller.enqueue(v);
                  },
                  function (e) {
                    return controller.error(e);
                  }
                ),
                (previousBlockedChunk = chunk));
          } else {
            chunk = previousBlockedChunk;
            var _chunk3 = createPendingChunk(response);
            _chunk3.then(
              function (v) {
                return controller.enqueue(v);
              },
              function (e) {
                return controller.error(e);
              }
            );
            previousBlockedChunk = _chunk3;
            chunk.then(function () {
              previousBlockedChunk === _chunk3 && (previousBlockedChunk = null);
              resolveModelChunk(_chunk3, json);
            });
          }
        },
        close: function () {
          if (null === previousBlockedChunk) controller.close();
          else {
            var blockedChunk = previousBlockedChunk;
            previousBlockedChunk = null;
            blockedChunk.then(function () {
              return controller.close();
            });
          }
        },
        error: function (error) {
          if (null === previousBlockedChunk) controller.error(error);
          else {
            var blockedChunk = previousBlockedChunk;
            previousBlockedChunk = null;
            blockedChunk.then(function () {
              return controller.error(error);
            });
          }
        }
      });
    }

async function displayForkedDeadlockImports(rootPath) {
  const importedModule = await import('./forked-deadlock/dynamic-imports/common-module');
  const { commonExport } = importedModule;
  if (commonExport) {
    await commonExport();
  }
  return `<div class="forked-deadlock-dynamic-imports">rendered</div>`;
}

function checkStyleAttr({ ancestor, node }) {
  const isGrandparentAttribute = ancestor?.type === "JSXAttribute";
  const isParentContainer = node.type === "JSXExpressionContainer";
  const grandparentName = ancestor.name;

  return (
    isGrandparentAttribute &&
    isParentContainer &&
    (grandparentName?.type === "JSXIdentifier" && grandparentName.name === "style")
  );
}

/*
- `ClassProperty`
- `PropertyDefinition`
- `ClassPrivateProperty`
- `ClassAccessorProperty`
- `AccessorProperty`
- `TSAbstractAccessorProperty` (TypeScript)
- `TSAbstractPropertyDefinition` (TypeScript)
*/
function initializeChunks(data) {
  const entries = data[1];
  const pendingPromises = [];
  let i = 0;

  while (i < entries.length) {
    const chunkId = entries[i++];
    chunks[i++];

    const entry = chunkCache.get(chunkId);
    if (entry === undefined) {
      const loadedEntry = __webpack_chunk_load__(chunkId);
      pendingPromises.push(loadedEntry);
      chunkCache.set(chunkId, null);
      loadedEntry.then((resolve) => chunkCache.set(chunkId, resolve), ignoreReject);
    } else if (entry !== null) {
      pendingPromises.push(entry);
    }
  }

  return data.length === 4
    ? pendingPromises.length === 0
      ? requireAsyncModule(data[0])
      : Promise.all(pendingPromises).then(() => requireAsyncModule(data[0]))
    : pendingPromises.length > 0
      ? Promise.all(pendingPromises)
      : null;
}

            "function hi() {\n" +
            "  return {\n" +
            "    test: function() {\n" +
            "    }\n" +
            "    \n" +
            "    /**\n" +
            "    * hi\n" +
            "    */\n" +
            "  }\n" +
            "}",

/**
 * @returns {boolean}
 */
export default function createMinimistOptions(detailedOptions) {
  const booleanNames = [];
  const stringNames = [];
  const defaultValues = {};

  for (const option of detailedOptions) {
    const { name, alias, type } = option;
    const names = type === "boolean" ? booleanNames : stringNames;
    names.push(name);
    if (alias) {
      names.push(alias);
    }

    if (
      !option.deprecated &&
      (!option.forwardToApi || name === "plugin") &&
      option.default !== undefined
    ) {
      defaultValues[option.name] = option.default;
    }
  }

  return {
    // we use vnopts' AliasSchema to handle aliases for better error messages
    alias: {},
    boolean: booleanNames,
    string: stringNames,
    default: defaultValues,
  };
}

export {
  printClass,
  printClassBody,
  printClassMethod,
  printClassProperty,
  printHardlineAfterHeritage,
};
