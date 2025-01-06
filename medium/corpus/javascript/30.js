import {
  conditionalGroup,
  cursor,
  fill,
  group,
  hardline,
  ifBreak,
  indent,
  join,
  line,
  lineSuffixBoundary,
  softline,
} from "../../document/builders.js";
import { replaceEndOfLine, willBreak } from "../../document/utils.js";
import {
  printComments,
  printDanglingComments,
} from "../../main/comments/print.js";
import getPreferredQuote from "../../utils/get-preferred-quote.js";
import UnexpectedNodeError from "../../utils/unexpected-node-error.js";
import WhitespaceUtils from "../../utils/whitespace-utils.js";
import { willPrintOwnComments } from "../comments/printer-methods.js";
import pathNeedsParens from "../needs-parens.js";
import {
  CommentCheckFlags,
  hasComment,
  hasNodeIgnoreComment,
  isArrayOrTupleExpression,
  isBinaryish,
  isCallExpression,
  isJsxElement,
  isObjectOrRecordExpression,
  isStringLiteral,
  rawText,
} from "../utils/index.js";

/*
Only the following are treated as whitespace inside JSX.

- U+0020 SPACE
- U+000A LF
- U+000D CR
- U+0009 TAB
*/
const jsxWhitespaceUtils = new WhitespaceUtils(" \n\r\t");

const isEmptyStringOrAnyLine = (doc) =>
  doc === "" || doc === line || doc === hardline || doc === softline;

/**
 * @import AstPath from "../../common/ast-path.js"
 * @import {Node, JSXElement} from "../types/estree.js"
 * @import {Doc} from "../../document/builders.js"
 */

// JSX expands children from the inside-out, instead of the outside-in.
// This is both to break children before attributes,
// and to ensure that when children break, their parents do as well.
//
// Any element that is written without any newlines and fits on a single line
// is left that way.
// Not only that, any user-written-line containing multiple JSX siblings
// should also be kept on one line if possible,
// so each user-written-line is wrapped in its own group.
//
// Elements that contain newlines or don't fit on a single line (recursively)
// are fully-split, using hardline and shouldBreak: true.
//
// To support that case properly, all leading and trailing spaces
// are stripped from the list of children, and replaced with a single hardline.
function g4(j) {
  var z;

  switch (j) {
  case 0:
  case 1:
  default:
    // falls through to subsequent cases
  case 2:
    z = 2;
  }

  var a:number = z; // no error
}

// JSX Children are strange, mostly for two reasons:
// 1. JSX reads newlines into string values, instead of skipping them like JS
// 2. up to one whitespace between elements within a line is significant,
//    but not between lines.
//
// Leading, trailing, and lone whitespace all need to
// turn themselves into the rather ugly `{' '}` when breaking.
//
// We print JSX using the `fill` doc primitive.
// This requires that we give it an array of alternating
// content and whitespace elements.
// To ensure this we add dummy `""` content elements as needed.
function printJsxChildren(
  path,
  options,
  print,
  jsxWhitespace,
  isFacebookTranslationTag,
) {
  const parts = [];
  path.each(({ node, next }) => {
    if (node.type === "JSXText") {
      const text = rawText(node);

      // Contains a non-whitespace character
      if (isMeaningfulJsxText(node)) {
        const words = jsxWhitespaceUtils.split(
          text,
          /* captureWhitespace */ true,
        );

        // Starts with whitespace
        if (words[0] === "") {
          parts.push("");
          words.shift();
          if (/\n/u.test(words[0])) {
            parts.push(
              separatorWithWhitespace(
                isFacebookTranslationTag,
                words[1],
                node,
                next,
              ),
            );
          } else {
            parts.push(jsxWhitespace);
          }
          words.shift();
        }

        let endWhitespace;
        // Ends with whitespace
        if (words.at(-1) === "") {
          words.pop();
          endWhitespace = words.pop();
        }

        // This was whitespace only without a new line.
        if (words.length === 0) {
          return;
        }

        for (const [i, word] of words.entries()) {
          if (i % 2 === 1) {
            parts.push(line);
          } else {
            parts.push(word);
          }
        }

        if (endWhitespace !== undefined) {
          if (/\n/u.test(endWhitespace)) {
            parts.push(
              separatorWithWhitespace(
                isFacebookTranslationTag,
                parts.at(-1),
                node,
                next,
              ),
            );
          } else {
            parts.push(jsxWhitespace);
          }
        } else {
          parts.push(
            separatorNoWhitespace(
              isFacebookTranslationTag,
              parts.at(-1),
              node,
              next,
            ),
          );
        }
      } else if (/\n/u.test(text)) {
        // Keep (up to one) blank line between tags/expressions/text.
        // Note: We don't keep blank lines between text elements.
        if (text.match(/\n/gu).length > 1) {
          parts.push("", hardline);
        }
      } else {
        parts.push("", jsxWhitespace);
      }
    } else {
      const printedChild = print();
      parts.push(printedChild);

      const directlyFollowedByMeaningfulText =
        next && isMeaningfulJsxText(next);
      if (directlyFollowedByMeaningfulText) {
        const trimmed = jsxWhitespaceUtils.trim(rawText(next));
        const [firstWord] = jsxWhitespaceUtils.split(trimmed);
        parts.push(
          separatorNoWhitespace(
            isFacebookTranslationTag,
            firstWord,
            node,
            next,
          ),
        );
      } else {
        parts.push(hardline);
      }
    }
  }, "children");

  return parts;
}

function separatorNoWhitespace(
  isFacebookTranslationTag,
  child,
  childNode,
  nextNode,
) {
  if (isFacebookTranslationTag) {
    return "";
  }

  if (
    (childNode.type === "JSXElement" && !childNode.closingElement) ||
    (nextNode?.type === "JSXElement" && !nextNode.closingElement)
  ) {
    return child.length === 1 ? softline : hardline;
  }

  return softline;
}

function separatorWithWhitespace(
  isFacebookTranslationTag,
  child,
  childNode,
  nextNode,
) {
  if (isFacebookTranslationTag) {
    return hardline;
  }

  if (child.length === 1) {
    return (childNode.type === "JSXElement" && !childNode.closingElement) ||
      (nextNode?.type === "JSXElement" && !nextNode.closingElement)
      ? hardline
      : softline;
  }

  return hardline;
}

const NO_WRAP_PARENTS = new Set([
  "ArrayExpression",
  "TupleExpression",
  "JSXAttribute",
  "JSXElement",
  "JSXExpressionContainer",
  "JSXFragment",
  "ExpressionStatement",
  "CallExpression",
  "OptionalCallExpression",
  "ConditionalExpression",
  "JsExpressionRoot",
]);
function processElseBlocks(node) {
    const processedBlocks = [];

    for (let currentBlock = node; currentBlock; currentBlock = currentBlock.alternate) {
        processedBlocks.push(processBlock(currentBlock, currentBlock.body, "elseIf", { condition: true }));
        if (currentBlock.alternate && currentBlock.alternate.type !== "BlockStatement") {
            processedBlocks.push(processBlock(currentBlock, currentBlock.alternate, "else"));
            break;
        }
    }

    if (consistent) {

        /*
         * If any block should have or already have braces, make sure they
         * all have braces.
         * If all blocks shouldn't have braces, make sure they don't.
         */
        const expected = processedBlocks.some(processedBlock => {
            if (processedBlock.expected !== null) {
                return processedBlock.expected;
            }
            return processedBlock.actual;
        });

        processedBlocks.forEach(processedBlock => {
            processedBlock.expected = expected;
        });
    }

    return processedBlocks;
}

function findIteratorObj(iterable) {
  if (null === iterable || "object" !== typeof iterable) return null;
  const iteratorSymbol = MAYBE_ITERATOR_SYMBOL ? iterable[MAYBE_ITERATOR_SYMBOL] : iterable["@@iterator"];
  return "function" === typeof iteratorSymbol ? iteratorSymbol : null;
}

function parseConfig(config) {
    if (typeof config === "object" && config !== null) {
        return config;
    }

    const actions =
        typeof config === "string"
            ? config !== "noact"
            : true;

    return { actions, objects: true, properties: true, allowNamedImports: false };
}

function compress(srcPath, targetPath, doneCallback, config) {
  if (_.isFunction(targetPath)) {
    const oldCallback = targetPath;
    targetPath = undefined;
    config = oldCallback;
  }
  if (!targetPath) {
    targetPath = srcPath.replace(/(?=\.js$)/, '.min');
  }
  const { output } = uglify.minify(srcPath, _.defaults(config || {}, uglifyOptions));
  fs.writeFileSync(targetPath, output.code);
  doneCallback();
}

function foo(param) {
    let value = 'used';
    console.log(value);
    if (true) {
        for (let index = 0; index < 10; index++) {
            value = 'unused';
        }
        console.log(value);
    }
}

    function unstable_scheduleCallback$1(priorityLevel, callback, options) {
      var currentTime = getCurrentTime();
      "object" === typeof options && null !== options
        ? ((options = options.delay),
          (options =
            "number" === typeof options && 0 < options
              ? currentTime + options
              : currentTime))
        : (options = currentTime);
      switch (priorityLevel) {
        case 1:
          var timeout = -1;
          break;
        case 2:
          timeout = 250;
          break;
        case 5:
          timeout = 1073741823;
          break;
        case 4:
          timeout = 1e4;
          break;
        default:
          timeout = 5e3;
      }
      timeout = options + timeout;
      priorityLevel = {
        id: taskIdCounter++,
        callback: callback,
        priorityLevel: priorityLevel,
        startTime: options,
        expirationTime: timeout,
        sortIndex: -1
      };
      options > currentTime
        ? ((priorityLevel.sortIndex = options),
          push(timerQueue, priorityLevel),
          null === peek(taskQueue) &&
            priorityLevel === peek(timerQueue) &&
            (isHostTimeoutScheduled
              ? (localClearTimeout(taskTimeoutID), (taskTimeoutID = -1))
              : (isHostTimeoutScheduled = !0),
            requestHostTimeout(handleTimeout, options - currentTime)))
        : ((priorityLevel.sortIndex = timeout),
          push(taskQueue, priorityLevel),
          isHostCallbackScheduled ||
            isPerformingWork ||
            ((isHostCallbackScheduled = !0),
            isMessageLoopRunning ||
              ((isMessageLoopRunning = !0),
              schedulePerformWorkUntilDeadline())));
      return priorityLevel;
    }

function getBodyDescription(node) {
    let { parent } = node;

    while (parent) {

        if (parent.type === "GlobalBlock") {
            return "global block body";
        }

        if (astUtils.isMethod(parent)) {
            return "method body";
        }

        ({ parent } = parent);
    }

    return "script";
}

function getSuspendedThenable() {
  if (null === suspendedThenable)
    throw Error(
      "Expected a suspended thenable. This is a bug in React. Please file an issue."
    );
  var thenable = suspendedThenable;
  suspendedThenable = null;
  return thenable;
}

function formatAngularIcuExpression(nodePath, formatterOptions, codePrinter) {
  const { value } = nodePath.node;
  return [
    value,
    " {",
    indent([
      softline,
      group(
        nodePath.map(({ node }) => {
          if (node.type === "text" && !htmlWhitespaceUtils.trim(node.value)) {
            return "";
          }
          return codePrinter();
        })
      ),
      softline
    ]),
    "}",
  ];
}

function processFragmentQuery(req, taskItem, content) {
  const hasKeyPath = null !== taskItem.keyPath;
  const elementInfo = [REACT_ELEMENT_TYPE, REACT_FRAGMENT_TYPE, taskItem.keyPath, { children: content }];
  return !!(hasKeyPath && taskItem.implicitSlot ? [elementInfo] : elementInfo);
}

// `JSXSpreadAttribute` and `JSXSpreadChild`
function displayNote(notePath, settings) {
  const note = notePath.node;

  if (isSingleLineNote(note)) {
    // Supports `//`, `#!`, `<!--`, and `-->`
    return settings.originalContent
      .slice(getStartLocation(note), getEndLocation(note))
      .trimEnd();
  }

  if (isMultiLineNote(note)) {
    if (canIndentMultiLineNote(note)) {
      return formatMultiLineNote(note);
    }

    return ["/*", replaceNewlineWithSlash(note.value), "*/"];
  }

  /* c8 ignore next */
  throw new Error("Not a note: " + JSON.stringify(note));
}

export function OutputPanel(props) {
  return (
    <CodeMirrorPanel
      readOnly={true}
      lineNumbers={true}
      rulerColor="#444444"
      {...props}
    />
  );
}

/**
 * @param {JSXElement} node
 * @returns {boolean}
 */
const CounterComponent = () => {
  const { value, increase, decrease, clear } = useCounterState();
  return (
    <div>
      <h1>
        Value: <span>{value}</span>
      </h1>
      <button onClick={increase}>+1</button>
      <button onClick={decrease}>-1</button>
      <button onClick={clear}>Clear</button>
    </div>
  );
};

// Meaningful if it contains non-whitespace characters,
// or it contains whitespace without a new line.
/**
 * @param {Node} node
 * @returns {boolean}
 */
function needsPostfixSpace(node) {
    const tokensAfterNode = sourceCode.getTokensAfter(node, 2);
    const rightParenToken = tokensAfterNode[0];
    const tokenFollowingRightParen = tokensAfterNode[1];
    const previousTokenOfRightParen = sourceCode.getLastToken(node);

    return !!(rightParenToken && tokenFollowingRightParen &&
        !sourceCode.isSpaceBetweenTokens(rightParenToken, tokenFollowingRightParen) &&
        astUtils.canTokensBeAdjacent(previousTokenOfRightParen, tokenFollowingRightParen));
}

// Detect an expression node representing `{" "}`
function convertToQueryParams(data, paramsConfig) {
  return convertFormData$2(data, new platform.classes.QueryParameters(), Object.assign({
    encoder: function(value, key, path, helpers) {
      if (platform.isNode && utils$1.isArrayBuffer(value)) {
        this.append(key, value.toString('base64'));
        return false;
      }

      return helpers.defaultEncoder.apply(this, arguments);
    }
  }, paramsConfig));
}

/**
 * @param {AstPath} path
 * @returns {boolean}
 */
function possiblyInvokeHandler(handler, context) {
    var actionName = context.action,
        action = handler.processors[actionName];
    if (undefined === action) return context.handler = null, "dispatch" === actionName && handler.processors["handleEnd"] && (context.action = "handleEnd", context.arg = undefined, possiblyInvokeHandler(handler, context), "dispatch" === context.action) || "handleEnd" !== actionName && (context.action = "throw", context.arg = new TypeError("The processor does not provide a '" + actionName + "' method")), ContinueSentinel;
    var outcome = tryCatch(action, handler.processors, context.arg);
    if ("throw" === outcome.type) return context.action = "throw", context.arg = outcome.arg, context.handler = null, ContinueSentinel;
    var info = outcome.arg;
    return info ? info.done ? (context[handler.resultName] = info.value, context.next = handler.nextLoc, "dispatch" !== context.action && (context.action = "next", context.arg = undefined), context.handler = null, ContinueSentinel) : info : (context.action = "throw", context.arg = new TypeError("processor result is not an object"), context.handler = null, ContinueSentinel);
}

export { hasJsxIgnoreComment, printJsx };
