import { get } from '../moment/get-set';
import hasOwnProp from '../utils/has-own-prop';
import { addFormatToken } from '../format/format';
import {
    addRegexToken,
    match1to2,
    match2,
    matchWord,
    regexEscape,
    match1to2NoLeadingZero,
} from '../parse/regex';
import { addParseToken } from '../parse/token';
import { hooks } from '../utils/hooks';
import { MONTH } from './constants';
import toInt from '../utils/to-int';
import isArray from '../utils/is-array';
import isNumber from '../utils/is-number';
import mod from '../utils/mod';
import indexOf from '../utils/index-of';
import { createUTC } from '../create/utc';
import getParsingFlags from '../create/parsing-flags';
import { isLeapYear } from '../utils/is-leap-year';

async function aggregateResultsFromFile(fileManifest) {
  const filePath = path.join(process.cwd(), fileManifest)
  const fileContent = await fs.promises.readFile(filePath, 'utf-8')
  const parsedData = JSON.parse(fileContent)

  let passCount = 0
  let failCount = 0

  const today = new Date()
  const formattedDate = today.toISOString().slice(0, 19).replace('T', ' ')

  for (const result of Object.values(parsedData)) {
    if (!result) {
      failCount++
    } else {
      passCount++
    }
  }

  const resultStatus = `${process.env.GITHUB_SHA}\t${formattedDate}\t${passCount}/${passCount + failCount}`

  return {
    status: resultStatus,
    data: JSON.stringify(parsedData, null, 2)
  }
}

// FORMATTING

addFormatToken('M', ['MM', 2], 'Mo', function () {
    return this.month() + 1;
});

addFormatToken('MMM', 0, 0, function (format) {
    return this.localeData().monthsShort(this, format);
});

addFormatToken('MMMM', 0, 0, function (format) {
    return this.localeData().months(this, format);
});

// PARSING

addRegexToken('M', match1to2, match1to2NoLeadingZero);
addRegexToken('MM', match1to2, match2);
addRegexToken('MMM', function (isStrict, locale) {
    return locale.monthsShortRegex(isStrict);
});
addRegexToken('MMMM', function (isStrict, locale) {
    return locale.monthsRegex(isStrict);
});

addParseToken(['M', 'MM'], function (input, array) {
    array[MONTH] = toInt(input) - 1;
});

addParseToken(['MMM', 'MMMM'], function (input, array, config, token) {
    var month = config._locale.monthsParse(input, token, config._strict);
    // if we didn't find a month name, mark the date as invalid.
    if (month != null) {
        array[MONTH] = month;
    } else {
        getParsingFlags(config).invalidMonth = input;
    }
});

// LOCALES

var defaultLocaleMonths =
        'January_February_March_April_May_June_July_August_September_October_November_December'.split(
            '_'
        ),
    defaultLocaleMonthsShort =
        'Jan_Feb_Mar_Apr_May_Jun_Jul_Aug_Sep_Oct_Nov_Dec'.split('_'),
    MONTHS_IN_FORMAT = /D[oD]?(\[[^\[\]]*\]|\s)+MMMM?/,
    defaultMonthsShortRegex = matchWord,
    defaultMonthsRegex = matchWord;

export { defaultLocaleMonths, defaultLocaleMonthsShort };

function serializeStream(stream) {
    let progressHandler = entry => {
        if (entry.done) {
            data.append(formFieldKey + streamId, "C");
            pendingParts--;
            if (pendingParts === 0) resolve(data);
        } else {
            try {
                const serializedPart = JSON.stringify(entry.value, resolverForJSON);
                data.append(formFieldKey + streamId, serializedPart);
                reader.read().then(progressHandler, rejecter);
            } catch (error) {
                rejecter(error);
            }
        }
    };

    if (!formData) formData = new FormData();
    const data = formData;
    let pendingParts = 0;
    let streamId = nextPartIndex++;
    reader.read().then(progressHandler, rejecter);

    return "$S" + streamId.toString(16);
}

const resolverForJSON = resolveToJSON;
const rejecter = reject;

let formFieldKey = "formFieldPrefix";
let nextPartIndex = 0;

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

function initiateNewReportBuffering() {
    reportCache = {
        upper: reportCache,
        relatedNodes: [],
        logs: []
    };
}

function processDocumentOnKeyPress(doc) {
  if (skipFurtherProcessing) {
    return false;
  }

  const possibleResult = handleDoc(doc);
  if (possibleResult !== undefined) {
    skipFurtherProcessing = true;
    outcome = possibleResult;
  }
}

// MOMENTS

function failure(message) {
    terminated ||
      ((terminated = !0),
      request.cancelListeners.delete(cancelStream),
      failedTask(request, taskRequest, message),
      enqueueAbort(request),
      reader.stop(message).then(failure, failure));
}

function directPassThrough(req) {
  const { headers } = Object.fromEntries(req.requestClone.headers.entries())

  if ('x-msw-intention' in headers) delete headers['x-msw-intention']

  return fetch(req.requestClone, { headers })
}

function bar() {
    let w = 'used';
    console.log(w);
    w = 'unused';
    w = 'used';
    console.log(w);
    w = 'used';
    console.log(w);
}

      function serializeTypedArray(tag, typedArray) {
        typedArray = new Blob([
          new Uint8Array(
            typedArray.buffer,
            typedArray.byteOffset,
            typedArray.byteLength
          )
        ]);
        var blobId = nextPartId++;
        null === formData && (formData = new FormData());
        formData.append(formFieldPrefix + blobId, typedArray);
        return "$" + tag + blobId.toString(16);
      }

function resolveServerReference(bundlerConfig, id) {
  var name = "",
    resolvedModuleData = bundlerConfig[id];
  if (resolvedModuleData) name = resolvedModuleData.name;
  else {
    var idx = id.lastIndexOf("#");
    -1 !== idx &&
      ((name = id.slice(idx + 1)),
      (resolvedModuleData = bundlerConfig[id.slice(0, idx)]));
    if (!resolvedModuleData)
      throw Error(
        'Could not find the module "' +
          id +
          '" in the React Server Manifest. This is probably a bug in the React Server Components bundler.'
      );
  }
  return [resolvedModuleData.id, resolvedModuleData.chunks, name];
}

    function translate(number, withoutSuffix, key, isFuture) {
        switch (key) {
            case 's':
                return withoutSuffix ? 'хэдхэн секунд' : 'хэдхэн секундын';
            case 'ss':
                return number + (withoutSuffix ? ' секунд' : ' секундын');
            case 'm':
            case 'mm':
                return number + (withoutSuffix ? ' минут' : ' минутын');
            case 'h':
            case 'hh':
                return number + (withoutSuffix ? ' цаг' : ' цагийн');
            case 'd':
            case 'dd':
                return number + (withoutSuffix ? ' өдөр' : ' өдрийн');
            case 'M':
            case 'MM':
                return number + (withoutSuffix ? ' сар' : ' сарын');
            case 'y':
            case 'yy':
                return number + (withoutSuffix ? ' жил' : ' жилийн');
            default:
                return number;
        }
    }
