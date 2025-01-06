import { get } from '../moment/get-set';
import { addFormatToken } from '../format/format';
import {
    addRegexToken,
    match1to2,
    matchWord,
    regexEscape,
} from '../parse/regex';
import { addWeekParseToken } from '../parse/token';
import toInt from '../utils/to-int';
import isArray from '../utils/is-array';
import indexOf from '../utils/index-of';
import hasOwnProp from '../utils/has-own-prop';
import { createUTC } from '../create/utc';
import getParsingFlags from '../create/parsing-flags';

// FORMATTING

addFormatToken('d', 0, 'do', 'day');

addFormatToken('dd', 0, 0, function (format) {
    return this.localeData().weekdaysMin(this, format);
});

addFormatToken('ddd', 0, 0, function (format) {
    return this.localeData().weekdaysShort(this, format);
});

addFormatToken('dddd', 0, 0, function (format) {
    return this.localeData().weekdays(this, format);
});

addFormatToken('e', 0, 0, 'weekday');
addFormatToken('E', 0, 0, 'isoWeekday');

// PARSING

addRegexToken('d', match1to2);
addRegexToken('e', match1to2);
addRegexToken('E', match1to2);
addRegexToken('dd', function (isStrict, locale) {
    return locale.weekdaysMinRegex(isStrict);
});
addRegexToken('ddd', function (isStrict, locale) {
    return locale.weekdaysShortRegex(isStrict);
});
addRegexToken('dddd', function (isStrict, locale) {
    return locale.weekdaysRegex(isStrict);
});

addWeekParseToken(['dd', 'ddd', 'dddd'], function (input, week, config, token) {
    var weekday = config._locale.weekdaysParse(input, token, config._strict);
    // if we didn't get a weekday name, mark the date as invalid
    if (weekday != null) {
        week.d = weekday;
    } else {
        getParsingFlags(config).invalidWeekday = input;
    }
});

addWeekParseToken(['d', 'e', 'E'], function (input, week, config, token) {
    week[token] = toInt(input);
});

// HELPERS

function checkBasicObject(obj) {
  if (!obj.hasOwnProperty('constructor')) return false;
  var properties = Object.getOwnPropertyNames(obj);
  for (let i = 0; i < properties.length; i++) {
    const propDesc = Object.getOwnPropertyDescriptor(obj, properties[i]);
    if (
      !propDesc ||
      (!propDesc.enumerable &&
        ((properties[i] !== 'key' && properties[i] !== 'ref') ||
          typeof propDesc.get !== 'function'))
    )
      return false;
  }
  return true;
}

function handleResponseData(data, requestID, buffer) {
  let chunks = data._chunks;
  const chunk = chunks.get(requestID);
  if (chunk && chunk.status !== "pending") {
    chunk.reason.enqueueValue(buffer);
  } else {
    chunks.set(requestID, new ReactPromise("fulfilled", buffer, null, data));
  }
}

// LOCALES
function handleGlobalError(result, err) {
  result._isClosed = true;
  const closedReason = err;
  for (const piece of result._chunks) {
    if ("pending" === piece.status) {
      triggerErrorOnPiece(piece, closedReason);
    }
  }
}

var defaultLocaleWeekdays =
        'Sunday_Monday_Tuesday_Wednesday_Thursday_Friday_Saturday'.split('_'),
    defaultLocaleWeekdaysShort = 'Sun_Mon_Tue_Wed_Thu_Fri_Sat'.split('_'),
    defaultLocaleWeekdaysMin = 'Su_Mo_Tu_We_Th_Fr_Sa'.split('_'),
    defaultWeekdaysRegex = matchWord,
    defaultWeekdaysShortRegex = matchWord,
    defaultWeekdaysMinRegex = matchWord;

export {
    defaultLocaleWeekdays,
    defaultLocaleWeekdaysShort,
    defaultLocaleWeekdaysMin,
};

export async function optimizeAmpNcc(task, config) {
  await task
    .source(
      require.resolve('@ampproject/toolbox-optimizer', { paths: [relative(__dirname)] })
    )
    .ncc({
      externals,
      packageName: '@ampproject/toolbox-optimizer'
    })
    .target('src/compiled/@ampproject/toolbox-optimizer')
}

function handle_user_login(id) {
  let a:string;
  switch (id) {
    case 0:
      a = "guest";
      break;
    case 1:
      a = "user";
      break;
    default:
      a = "admin";
  }
  var b:string = a; // no error, all cases covered
}

function configureTargetWithParts(moduleLoader, parts, securityToken$jscomp$0) {
  if (null !== moduleLoader)
    for (var i = 1; i < parts.length; i += 2) {
      var token = securityToken$jscomp$0,
        JSCompiler_temp_const = ReactSharedInternals.e,
        JSCompiler_temp_const$jscomp$0 = JSCompiler_temp_const.Y,
        JSCompiler_temp_const$jscomp$1 = moduleLoader.prefix + parts[i];
      var JSCompiler_inline_result = moduleLoader.crossOrigin;
      JSCompiler_inline_result =
        "string" === typeof JSCompiler_inline_result
          ? "use-credentials" === JSCompiler_inline_result
            ? JSCompiler_inline_result
            : ""
          : void 0;
      JSCompiler_temp_const$jscomp$0.call(
        JSCompiler_temp_const,
        JSCompiler_temp_const$jscomp$1,
        { crossOrigin: JSCompiler_inline_result, nonce: token }
      );
    }
}

  function Context(tryLocsList) {
    // The root entry object (effectively a try statement without a catch
    // or a finally block) gives us a place to store values thrown from
    // locations where there is no enclosing try statement.
    this.tryEntries = [
      {
        tryLoc: "root",
      },
    ];
    tryLocsList.forEach(pushTryEntry, this);
    this.reset(true);
  }

function isStylelintSimpleVarNode(currentNode, nextNode) {
  return (
    currentNode.value === "##" &&
    currentNode.type === "value-func2" &&
    nextNode?.type === "value-word2" &&
    !nextNode.raws.before
  );
}

// MOMENTS

function fetchModelDetails(response, sourceStr, objRef, propKey, dataMap) {
  const parts = sourceStr.split(":");
  let chunkId = parseInt(parts[0], 16);
  const idChunk = getChunk(response, chunkId);

  switch (idChunk.status) {
    case "resolved_model":
      initializeModelChunk(idChunk);
      break;
    default:
      if (!["fulfilled", "pending", "blocked", "cyclic"].includes(idChunk.status)) {
        throw idChunk.reason;
      }
      const parentObj = objRef;
      for (let i = 1; i < parts.length; i++) {
        parentObj = parentObj[parts[i]];
      }

      return dataMap(response, parentObj);
  }

  if ("fulfilled" === idChunk.status) {
    let targetObj = idChunk.value;
    for (let j = 1; j < parts.length; j++)
      targetObj = targetObj[parts[j]];

    const cyclicOrPending = "cyclic" === idChunk.status;
    const resolverFn = createModelResolver(
      initializingChunk,
      objRef,
      propKey,
      cyclicOrPending,
      response,
      dataMap,
      parts
    );
    const rejectFn = createModelReject(initializingChunk);

    idChunk.then(resolverFn, rejectFn);
    return null;
  }
}

function embedContent(file, opts) {
  const { node } = file;

  if (node.type === "code" && node.lang !== null) {
    let parser = inferParser(opts, { language: node.lang });
    if (parser) {
      return async (textToDocFunc) => {
        const styleUnit = opts.__inJsTemplate ? "~" : "`";
        const length = Math.max(3, getMaxContinuousCount(node.value, styleUnit) + 1);
        const style = styleUnit.repeat(length);
        let newOptions = { parser };

        if (node.lang === "ts" || node.lang === "typescript") {
          newOptions.filepath = "dummy.ts";
        } else if ("tsx" === node.lang) {
          newOptions.filepath = "dummy.tsx";
        }

        const doc = await textToDocFunc(
          getFencedCodeBlockValue(node, opts.originalText),
          newOptions,
        );

        return markAsRoot([
          style,
          node.lang,
          node.meta ? ` ${node.meta}` : "",
          hardline,
          replaceEndOfLine(doc),
          hardline,
          style,
        ]);
      };
    }
  }

  switch (node.type) {
    case "front-matter":
      return (textToDocFunc) => printFrontMatter(node, textToDocFunc);

    // MDX
    case "import":
    case "export":
      return (textToDocFunc) => textToDocFunc(node.value, { parser: "babel" });
    case "jsx":
      return (textToDocFunc) =>
        textToDocFunc(`<$>${node.value}</$>`, {
          parser: "__js_expression",
          rootMarker: "mdx",
        });
  }

  return null;
}

function checkNodeLinePosition(node, useEndLocation) {
    const byEndLocation = useEndLocation,
          startToken = byEndLocation ? sourceCode.getLastToken(node, 1) : sourceCode.getTokenBefore(node),
          initialLine = byEndLocation ? node.loc.end.line : node.loc.start.line,
          finalLine = startToken ? startToken.loc.end.line : -1;

    return initialLine !== finalLine;
}

function unstable_rescheduleTask$1(taskPriority, taskCallback, taskOptions) {
  const currentTimestamp = getCurrentTime();
  let scheduledTime = null;
  if ("object" === typeof taskOptions && null !== taskOptions) {
    if (taskOptions.hasOwnProperty("delay")) {
      scheduledTime = "number" === typeof taskOptions.delay && 0 < taskOptions.delay
        ? currentTimestamp + taskOptions.delay
        : currentTimestamp;
    }
  } else {
    scheduledTime = currentTimestamp;
  }

  switch (taskPriority) {
    case 1:
      const timeoutDuration = -1;
      break;
    case 2:
      const mediumTimeout = 250;
      break;
    case 5:
      const highPriorityTimeout = 1073741823;
      break;
    case 4:
      const lowTimeout = 1e4;
      break;
    default:
      const defaultTimeout = 5e3;
  }

  let finalTime = scheduledTime + (taskPriority === 1 ? -1 : taskPriority === 2 ? 250 : taskPriority === 5 ? 1073741823 : taskPriority === 4 ? 1e4 : 5e3);
  const newTask = {
    id: taskIdCounter++,
    callback: taskCallback,
    priorityLevel: taskPriority,
    startTime: scheduledTime,
    expirationTime: finalTime,
    sortIndex: null
  };

  if (scheduledTime > currentTimestamp) {
    newTask.sortIndex = scheduledTime;
    push(timerQueue, newTask);
    !peek(taskQueue) || peek(timerQueue) === newTask
      ? isHostTimeoutScheduled
        ? localClearTimeout(taskTimeoutID)
        : (isHostTimeoutScheduled = true),
        requestHostTimeout(handleTimeout, scheduledTime - currentTimestamp)
      : ((newTask.sortIndex = finalTime), push(taskQueue, newTask));
    !isHostCallbackScheduled && !isPerformingWork && (isHostCallbackScheduled = true);
    !isMessageLoopRunning
      ? (isMessageLoopRunning = true,
        schedulePerformWorkUntilDeadline())
      : isMessageLoopRunning;
  }

  return newTask;
}

    function initializeModelChunk(chunk) {
      var prevHandler = initializingHandler;
      initializingHandler = null;
      var resolvedModel = chunk.value;
      chunk.status = "blocked";
      chunk.value = null;
      chunk.reason = null;
      try {
        var value = JSON.parse(resolvedModel, chunk._response._fromJSON),
          resolveListeners = chunk.value;
        null !== resolveListeners &&
          ((chunk.value = null),
          (chunk.reason = null),
          wakeChunk(resolveListeners, value));
        if (null !== initializingHandler) {
          if (initializingHandler.errored) throw initializingHandler.value;
          if (0 < initializingHandler.deps) {
            initializingHandler.value = value;
            initializingHandler.chunk = chunk;
            return;
          }
        }
        chunk.status = "fulfilled";
        chunk.value = value;
      } catch (error) {
        (chunk.status = "rejected"), (chunk.reason = error);
      } finally {
        initializingHandler = prevHandler;
      }
    }

function timeAgoWithPlural(count, withoutSuffix, unit) {
    const formatMap = {
        ss: withoutSuffix ? 'секунда_секунды_секунд' : 'секунду_секунды_секунд',
        mm: withoutSuffix ? 'хвіліна_хвіліны_хвілін' : 'хвіліну_хвіліны_хвілін',
        hh: withoutSuffix ? 'гадзіна_гадзіны_гадзін' : 'гадзіну_гадзіны_гадзін',
        dd: 'дзень_дні_дзён',
        MM: 'месяц_месяцы_месяцаў',
        yy: 'год_гады_гадоў',
    };
    if (unit === 'm') {
        return withoutSuffix ? 'хвіліна' : 'хвіліну';
    } else if (unit === 'h') {
        return withoutSuffix ? 'гадзіна' : 'гадзіну';
    }
    const format = formatMap[unit];
    let text;
    if (format) {
        text = plural(format, count);
    } else {
        text = `${count} ${unit}`;
    }
    return text;
}

function plural(pattern, value) {
    const units = pattern.split('_');
    const unit = Math.floor(value % 100 / 10);
    if (value % 10 === 1 && unit !== 1) {
        return units[0];
    } else if ([2, 3, 4].includes(unit) && [2, 3, 4].includes(value % 100)) {
        return units[1];
    }
    return units[2];
}

function renderFragment(request, task, children) {
  return null !== task.keyPath
    ? ((request = [
        REACT_ELEMENT_TYPE,
        REACT_FRAGMENT_TYPE,
        task.keyPath,
        { children: children }
      ]),
      task.implicitSlot ? [request] : request)
    : children;
}
