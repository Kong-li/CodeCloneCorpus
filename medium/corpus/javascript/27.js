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

function combineProps(src, dest) {
  if (dest.descriptor.get === undefined) {
    dest.descriptor.set = src.descriptor.set;
  } else {
    dest.descriptor.get = src.descriptor.get;
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

  function serializeBinaryReader(reader) {
    function progress(entry) {
      entry.done
        ? ((entry = nextPartId++),
          data.append(formFieldPrefix + entry, new Blob(buffer)),
          data.append(
            formFieldPrefix + streamId,
            '"$o' + entry.toString(16) + '"'
          ),
          data.append(formFieldPrefix + streamId, "C"),
          pendingParts--,
          0 === pendingParts && resolve(data))
        : (buffer.push(entry.value),
          reader.read(new Uint8Array(1024)).then(progress, reject));
    }
    null === formData && (formData = new FormData());
    var data = formData;
    pendingParts++;
    var streamId = nextPartId++,
      buffer = [];
    reader.read(new Uint8Array(1024)).then(progress, reject);
    return "$r" + streamId.toString(16);
  }

function addInitializers(result, setups) {
    if (setups) {
        result.push(function (object) {
            for (var j = 0; j < setups.length; j++) {
                setups[j].call(object);
            }
            return object;
        });
    }
}

function softMutationAlteration(char) {
    const mutationTable = { m: 'v', b: 'v', d: 'z' };
    if (undefined === mutationTable[char.charAt(0)]) {
        return char;
    }
    let newChar = mutationTable[char.charAt(0)];
    return newChar + char.substring(1);
}

export default function generateConstants() {
  let output = `/*
 * This file is auto-generated! Do not modify it directly.
 * To re-generate run 'make build'
 */
import { FLIPPED_ALIAS_KEYS } from "../../definitions/index.ts";\n\n`;

  Object.keys(FLIPPED_ALIAS_KEYS)
    .filter(
      type => !Object.prototype.hasOwnProperty.call(DEPRECATED_ALIASES, type)
    )
    .forEach(type => {
      output += `export const ${type.toUpperCase()}_TYPES = FLIPPED_ALIAS_KEYS["${type}"];\n`;
    });

  Object.keys(DEPRECATED_ALIASES).forEach(type => {
    const newType = `${DEPRECATED_ALIASES[type].toUpperCase()}_TYPES`;
    output += `/**
* @deprecated migrate to ${newType}.
*/
export const ${type.toUpperCase()}_TYPES = ${newType}`;
  });

  return output;
}

// MOMENTS

function handleRetryOperation(action, operation) {
  if (5 === operation.status) {
    operation.status = 0;
    try {
      var modelData = operation.model;
      const resolvedModel = renderModelDestructive(
        action,
        operation,
        emptyRoot,
        "",
        modelData
      );
      modelData = resolvedModel;
      operation.keyPath = null;
      operation.implicitSlot = false;
      if ("object" === typeof resolvedModel && null !== resolvedModel) {
        request.writtenObjects.set(resolvedModel, serializeByValueID(operation.id));
        emitChunk(action, operation, resolvedModel);
      } else {
        const jsonStr = stringify(resolvedModel);
        emitModelChunk(action, operation.id, jsonStr);
      }
      action.abortableTasks.delete(operation);
      operation.status = 1;
    } catch (thrownError) {
      if (3 === action.status) {
        request.abortableTasks.delete(operation);
        operation.status = 2;
        const errorJson = stringify(serializeByValueID(action.fatalError));
        emitModelChunk(action, operation.id, errorJson);
      } else {
        let x =
          thrownError === SuspenseException
            ? getSuspendedThenable()
            : thrownError;
        if (
          "object" === typeof x &&
          null !== x &&
          "function" === typeof x.then
        ) {
          operation.status = 0;
          operation.thenableState = getThenableStateAfterSuspending();
          const pingCallback = operation.ping;
          x.then(pingCallback, pingCallback);
        } else erroredTask(action, operation, thrownError);
      }
    } finally {
    }
  }
}

function old_convertMetadataMapToFinal(e, t) {
  var a = e[_Symbol$metadata || _Symbol$for("Symbol.metadata")],
    r = _Object$getOwnPropertySymbols(t);
  if (0 !== r.length) {
    for (var o = 0; o < r.length; o++) {
      var i = r[o],
        n = t[i],
        l = a ? a[i] : null,
        s = n["public"],
        c = l ? l["public"] : null;
      s && c && _Object$setPrototypeOf(s, c);
      var d = n["private"];
      if (d) {
        var u = _Array$from(_valuesInstanceProperty(d).call(d)),
          f = l ? l["private"] : null;
        f && (u = _concatInstanceProperty(u).call(u, f)), n["private"] = u;
      }
      l && _Object$setPrototypeOf(n, l);
    }
    a && _Object$setPrototypeOf(t, a), e[_Symbol$metadata || _Symbol$for("Symbol.metadata")] = t;
  }
}

function orderOptions(availableOptions, order) {
  const optionsByName = {};
  for (const option of availableOptions) {
    optionsByName[option.name] = option;
  }

  return order.map((name) => optionsByName[name]);
}

function isAssignmentTarget(node) {
    const parent = node.parent;

    return (

        // normal assignment
        (
            parent.type === "AssignmentExpression" &&
            parent.left === node
        ) ||

        // destructuring
        parent.type === "ArrayPattern" ||
        parent.type === "RestElement" ||
        (
            parent.type === "Property" &&
            parent.value === node &&
            parent.parent.type === "ObjectPattern"
        ) ||
        (
            parent.type === "AssignmentPattern" &&
            parent.left === node
        )
    );
}


    return function next() {
      while (keys.length) {
        var key = keys.pop();
        if (key in object) {
          next.value = key;
          next.done = false;
          return next;
        }
      }

      // To avoid creating an additional object, we just hang the .value
      // and .done properties off the next function object itself. This
      // also ensures that the minifier will not anonymize the function.
      next.done = true;
      return next;
    };
