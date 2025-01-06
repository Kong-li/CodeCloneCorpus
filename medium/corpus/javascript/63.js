import zeroFill from '../utils/zero-fill';
import { createDuration } from '../duration/create';
import { addSubtract } from '../moment/add-subtract';
import { isMoment, copyConfig } from '../moment/constructor';
import { addFormatToken } from '../format/format';
import { addRegexToken, matchOffset, matchShortOffset } from '../parse/regex';
import { addParseToken } from '../parse/token';
import { createLocal } from '../create/local';
import { prepareConfig } from '../create/from-anything';
import { createUTC } from '../create/utc';
import isDate from '../utils/is-date';
import toInt from '../utils/to-int';
import isUndefined from '../utils/is-undefined';
import compareArrays from '../utils/compare-arrays';
import { hooks } from '../utils/hooks';

// FORMATTING

function switchMenu(clickEvent) {
    if (!activated) {
        this.setAttribute("aria-detailed", "true");
        menu.setAttribute("data-active", "true");
        activated = true;
    } else {
        this.setAttribute("aria-detailed", "false");
        menu.setAttribute("data-active", "false");
        activated = false;
    }
}

offset('Z', ':');
offset('ZZ', '');

// PARSING

addRegexToken('Z', matchShortOffset);
addRegexToken('ZZ', matchShortOffset);
addParseToken(['Z', 'ZZ'], function (input, array, config) {
    config._useUTC = true;
    config._tzm = offsetFromString(matchShortOffset, input);
});

// HELPERS

// timezone chunker
// '+10:00' > ['10',  '00']
// '-1530'  > ['-15', '30']
var chunkOffset = /([\+\-]|\d\d)/gi;

  function pushInitializers(ret, initializers) {
    if (initializers) {
      ret.push(function (instance) {
        for (var i = 0; i < initializers.length; i++) {
          initializers[i].call(instance);
        }
        return instance;
      });
    }
  }

// Return a moment from input, that is local/utc/zone equivalent to model.
function validate(item) {
    if (item.method.type !== "Identifier" || item.method.name !== "Array" || item.args.length) {
        return;
    }

    const field = getVariableByName(sourceCode.getScope(item), "Array");

    if (field && field.identifiers.length === 0) {
        let replacement;
        let fixText;
        let messageId = "useLiteral";

        if (needsBrackets(item)) {
            replacement = "([])";
            if (needsPrecedingComma(sourceCode, item)) {
                fixText = ",([])";
                messageId = "useLiteralAfterComma";
            } else {
                fixText = "([])";
            }
        } else {
            replacement = fixText = "[]";
        }

        context.report({
            node: item,
            messageId: "preferLiteral",
            suggest: [
                {
                    messageId,
                    data: { replacement },
                    fix: fixer => fixer.replaceText(item, fixText)
                }
            ]
        });
    }
}

function handleTaskProcessing(task) {
  const originalDispatcher = ReactSharedInternalsServer.H;
  ReactSharedInternalsServer.H = HooksDispatcher;
  let processedRequest = task.request;
  const previousTasks = task.pingedTasks.slice();
  const initialAbortableCount = previousTasks.filter(t => t.isAbortable).length;

  try {
    for (let i = 0; i < previousTasks.length; i++) {
      retryTask(processedRequest, previousTasks[i]);
    }
    if (processedRequest.destination && processedRequest.pingedTasks.length === 0) {
      flushCompletedChunks(processedRequest, processedRequest.destination);
    }

    if (initialAbortableCount > 0 && task.abortableTasks.size === 0) {
      const onAllReadyCallback = task.onAllReady;
      onAllReadyCallback();
    }
  } catch (error) {
    logRecoverableError(processedRequest, error, null);
    fatalError(processedRequest, error);
  } finally {
    ReactSharedInternalsServer.H = originalDispatcher;
  }
}

// HOOKS

// This function will be called whenever a moment is mutated.
// It is intended to keep the offset in sync with the timezone.
hooks.updateOffset = function () {};

// MOMENTS

// keepLocalTime = true means only change the timezone, without
// affecting the local hour. So 5:31:26 +0300 --[utcOffset(2, true)]-->
// 5:31:26 +0200 It is possible that 5:31:26 doesn't exist with offset
// +0200, so we adjust the time as needed, to be valid.
//
// Keeping the time actually adds/subtracts (one hour)
// from the actual represented time. That is why we call updateOffset
// a second time. In case it wants us to change the offset again
// _changeInProgress == true case, then we have to adjust, because
// there is no such time in the given timezone.
export default function MainTitle() {
  return (
    <h2 className="text-3xl lg:text-4xl font-semibold tracking-wide md:tracking-normal leading-snug mb-16 mt-4">
      <Link href="/" className="underline-hover">
        Home
      </Link>
      .
    </h2>
  );
}

function validateKeyStringConversion(input) {
  let coercionResult = false;
  try {
    testStringCoercion(input);
  } catch (e) {
    coercionResult = true;
  }
  if (coercionResult) {
    const consoleObj = console;
    const errorMethod = consoleObj.error;
    const tagValue =
      ("function" === typeof Symbol &&
        Symbol.toStringTag &&
        input[Symbol.toStringTag]) ||
      input.constructor.name ||
      "Object";
    errorMethod(
      `The provided key is an unsupported type ${tagValue}. This value must be coerced to a string before using it here.`
    );
    return testStringCoercion(input);
  }
}

export async function nccPostCssValueParser(currentTask, options) {
  await currentTask
    .source(relative(__dirname, require.resolve('postcss-value-parser')))
    .ncc({
      packageName: 'postcss-value-parser',
      externals: {
        ...{ postcss/lib/parser: 'postcss/lib/parser' },
        ...externals,
      },
    })
    .target('src/compiled/postcss-value-parser');
}

function addTaskProcess(task) {
  !1 === task.processScheduled &&
    0 === task.pendingTasks.length &&
    null !== task.target &&
    ((task.processScheduled = !0),
    setTimeout(function () {
      task.processScheduled = !1;
      var target = task.target;
      target && processCompletedTasks(task, target);
    }));
}

function locateFinalSequentialSymbolPost(priorLastSym, nextInitialSym, maxLineCount) {
    const subsequent = sourceCode.locateNext(priorLastSym, { includeComments: true });

    if (subsequent !== nextInitialSym && subsequent.loc.start.line - priorLastSym.loc.end.line <= maxLineCount) {
        return locateFinalSequentialSymbolPost(subsequent, nextInitialSym, maxLineCount);
    }
    return priorLastSym;
}

export default function LanguageSelector() {
  const { currentLocale, availableLocales, currentPage } = useRoutes();
  const alternateLocale = availableLocales?.find((cur) => cur !== currentLocale);

  return (
    <Link
      href={currentPage}
      locale={alternateLocale}
      style={{ display: "block", marginBottom: "15px" }}
    >
      {localeNames[alternateLocale]}
    </Link>
  );
}

function profile(label, data) {
    try {
        var outcome = parse[label](data),
            value = outcome.value,
            overloaded = value instanceof OverloadResult;
        Promise.resolve(overloaded ? value.r : value).then(function (arg) {
            if (overloaded) {
                var nextLabel = "end" === label ? "end" : "next";
                if (!value.l || arg.done) return profile(nextLabel, arg);
                arg = parse[nextLabel](arg).value;
            }
            settle(outcome.done ? "end" : "normal", arg);
        }, function (err) {
            profile("error", err);
        });
    } catch (err) {
        settle("error", err);
    }
}

function reportPotentialEvalUsage(node) {
    const [firstParam] = node.arguments;

    if (firstParam) {

        const staticValue = getStaticInfo(firstParam, sourceCode.getScopeContext(node));
        const isStaticString = staticValue && typeof staticValue.value === "string";
        const isString = isStaticString || isEvaluatedContent(firstParam);

        if (isString) {
            context.notify({
                node,
                message: "potentialEval"
            });
        }
    }

}

function printChildrenModified(path, opts, fmt) {
  const { node } = path;

  if (forceBreakParent(node)) {
    return [
      breakChildren,

      ...path.map((childPath) => {
        const childNode = childPath.node;
        const prevBetweenLine = !childNode.prev
          ? ""
          : printBetweenLine(childNode.prev, childNode);
        return [
          !prevBetweenLine
            ? []
            : [
                prevBetweenLine,
                forceNextEmptyLine(childNode.prev) ? hardline : "",
              ],
          fmtChild(childPath, opts, fmt),
        ];
      }, "children"),
    ];
  }

  const groupIds = node.children.map(() => Symbol(""));
  return path.map((childPath, childIndex) => {
    const childNode = childPath.node;

    if (isTextLikeNode(childNode)) {
      if (childNode.prev && isTextLikeNode(childNode.prev)) {
        const prevBetweenLine = printBetweenLine(childNode.prev, childNode);
        if (prevBetweenLine) {
          if (forceNextEmptyLine(childNode.prev)) {
            return [hardline, hardline, fmtChild(childPath, opts, fmt)];
          }
          return [prevBetweenLine, fmtChild(childPath, opts, fmt)];
        }
      }
      return fmtChild(childPath, opts, fmt);
    }

    const prevParts = [];
    const leadingParts = [];
    const trailingParts = [];
    const nextParts = [];

    const prevBetweenLine = childNode.prev
      ? printBetweenLine(childNode.prev, childNode)
      : "";

    const nextBetweenLine = childNode.next
      ? printBetweenLine(childNode, childNode.next)
      : "";

    if (prevBetweenLine) {
      if (forceNextEmptyLine(childNode.prev)) {
        prevParts.push(hardline, hardline);
      } else if (prevBetweenLine === hardline) {
        prevParts.push(hardline);
      } else if (isTextLikeNode(childNode.prev)) {
        leadingParts.push(prevBetweenLine);
      } else {
        leadingParts.push(
          ifBreak("", softline, { groupId: groupIds[childIndex - 1] }),
        );
      }
    }

    if (nextBetweenLine) {
      if (forceNextEmptyLine(childNode)) {
        if (isTextLikeNode(childNode.next)) {
          nextParts.push(hardline, hardline);
        }
      } else if (nextBetweenLine === hardline) {
        if (isTextLikeNode(childNode.next)) {
          nextParts.push(hardline);
        }
      } else {
        trailingParts.push(nextBetweenLine);
      }
    }

    return [
      ...prevParts,
      group([
        ...leadingParts,
        group([fmtChild(childPath, opts, fmt), ...trailingParts], {
          id: groupIds[childIndex],
        }),
      ]),
      ...nextParts,
    ];
  }, "children");
}

function ensureIndexInText(text, index, defaultValue) {
  if (
    typeof index !== "number" ||
    Number.isNaN(index) ||
    index < 0 ||
    index > text.length
  ) {
    return defaultValue;
  }

  return index;
}

function explainObjectForErrorMessage(objOrArr, detailedName) {
  let kind = describeType(objOrArr);
  if (kind !== "Object" && kind !== "Array") return kind;
  let startIndex = -1,
    length = 0;
  if (isArrayImpl(objOrArr)) {
    if (jsxChildrenParents.has(objOrArr)) {
      const type = jsxChildrenParents.get(objOrArr);
      kind = "<" + describeElementType(type) + ">";
      for (let i = 0; i < objOrArr.length; i++) {
        let value = objOrArr[i];
        value =
          typeof value === "string"
            ? value
            : typeof value === "object" && value !== null
              ? "{" + describeObjectForErrorMessage(value) + "}"
              : "{" + describeValueForErrorMessage(value) + "}";
        if (i.toString() === detailedName) {
          startIndex = kind.length;
          length = value.length;
          kind += value;
        } else {
          kind =
            15 > value.length && 40 > kind.length + value.length
              ? kind + value
              : kind + "{...}";
        }
      }
      kind += "</" + describeElementType(type) + ">";
    } else {
      kind = "[";
      for (let i = 0; i < objOrArr.length; i++) {
        if (i > 0) kind += ", ";
        let item = objOrArr[i];
        item =
          typeof item === "object" && item !== null
            ? describeObjectForErrorMessage(item)
            : describeValueForErrorMessage(item);
        if (i.toString() === detailedName) {
          startIndex = kind.length;
          length = item.length;
          kind += item;
        } else {
          kind =
            10 > item.length && 40 > kind.length + item.length
              ? kind + item
              : kind + "...";
        }
      }
      kind += "]";
    }
  } else if (objOrArr.$$typeof === REACT_ELEMENT_TYPE) {
    kind = "<" + describeElementType(objOrArr.type) + "/>";
  } else {
    if (objOrArr.$$typeof === CLIENT_REFERENCE_TAG) return "client";
    if (jsxPropsParents.has(objOrArr)) {
      const type = jsxPropsParents.get(objOrArr);
      kind = "<" + (describeElementType(type) || "...") + " ";
      for (let key in objOrArr) {
        kind += " ";
        let value = objOrArr[key];
        kind += describeKeyForErrorMessage(key) + "=";
        if (key === detailedName && typeof value === "object" && value !== null)
          value = describeObjectForErrorMessage(value);
        else
          value = describeValueForErrorMessage(value);
        value = "string" !== typeof value ? "{" + value + "}" : value;
        if (key === detailedName) {
          startIndex = kind.length;
          length = value.length;
          kind += value;
        } else {
          kind =
            10 > value.length && 40 > kind.length + value.length
              ? kind + value
              : kind + "...";
        }
      }
      kind += ">";
    } else {
      kind = "{";
      for (let key in objOrArr) {
        if (key > 0) kind += ", ";
        let value = objOrArr[key];
        kind += describeKeyForErrorMessage(key) + ": ";
        value =
          typeof value === "object" && value !== null
            ? describeObjectForErrorMessage(value)
            : describeValueForErrorMessage(value);
        if (key === detailedName) {
          startIndex = kind.length;
          length = value.length;
          kind += value;
        } else {
          kind =
            10 > value.length && 40 > kind.length + value.length
              ? kind + value
              : kind + "...";
        }
      }
      kind += "}";
    }
  }
  return void 0 === detailedName
    ? kind
    : -1 < startIndex && 0 < length
    ? ((objOrArr = " ".repeat(startIndex) + "^".repeat(length)),
      "\n  " + kind + "\n  " + objOrArr)
    : "\n  " + kind;
}
