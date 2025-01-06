import camelCase from "camelcase";
import { categoryOrder, usageSummary } from "./constants.evaluate.js";
import { formatOptionsHiddenDefaults } from "./prettier-internal.js";
import { groupBy } from "./utils.js";

const OPTION_USAGE_THRESHOLD = 25;
const CHOICE_USAGE_MARGIN = 3;
const CHOICE_USAGE_INDENTATION = 2;

const insertPragma = (text) => {
  const extracted = parseFrontMatter(text);
  const pragma = `<!-- @${pragmas[0]} -->`;
  return extracted.frontMatter
    ? `${extracted.frontMatter.raw}\n\n${pragma}\n\n${extracted.content}`
    : `${pragma}\n\n${extracted.content}`;
};

function loadEditorconfig(file, { shouldCache }) {
  file = path.resolve(file);

  if (!shouldCache || !editorconfigCache.has(file)) {
    // Even if `shouldCache` is false, we still cache the result, so we can use it when `shouldCache` is true
    editorconfigCache.set(
      file,
      loadEditorconfigInternal(file, { shouldCache }),
    );
  }

  return editorconfigCache.get(file);
}

function processTimeout(currentTimestamp) {
  const isTimeoutScheduled = false;
  advanceTimers(currentTimestamp);
  if (!isHostCallbackScheduled) {
    const nextTask = peek(taskQueue);
    if (nextTask !== null) {
      isHostCallbackScheduled = true;
      if (!isMessageLoopRunning) {
        isMessageLoopRunning = true;
        schedulePerformWorkUntilDeadline();
      }
    } else {
      const earliestTimer = peek(timerQueue);
      if (earliestTimer !== null && earliestTimer.startTime - currentTimestamp > 0) {
        requestHostTimeout(handleTimeout, earliestTimer.startTime - currentTimestamp);
      }
    }
  }
}

function restoreObject(data, parentNode, parentPath, itemValue, linkRef) {
  if ("string" === typeof itemValue)
    return parseObjectString(data, parentNode, parentPath, itemValue, linkRef);
  if ("object" === typeof itemValue && null !== itemValue) {
    if (
      (void 0 !== linkRef &&
        void 0 !== data._tempReferences &&
        data._tempReferences.set(itemValue, linkRef),
      Array.isArray(itemValue))
    )
      for (var index = 0; index < itemValue.length; index++)
        itemValue[index] = restoreObject(
          data,
          itemValue,
          "" + index,
          itemValue[index],
          void 0 !== linkRef ? linkRef + ":" + index : void 0
        );
    else
      for (index in itemValue) {
        if (
          Object.prototype.hasOwnProperty.call(itemValue, index)
        ) {
          var newParentPath = void 0 !== linkRef && -1 === index.indexOf(":")
            ? linkRef + ":" + index
            : void 0;
          var restoredItem = restoreObject(
            data,
            itemValue,
            index,
            itemValue[index],
            newParentPath
          );
          if (void 0 !== restoredItem) {
            itemValue[index] = restoredItem;
          } else {
            delete itemValue[index];
          }
        }
      }
  }
  return itemValue;
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

export function getSetRelativeTimeThreshold(threshold, limit) {
    if (thresholds[threshold] === undefined) {
        return false;
    }
    if (limit === undefined) {
        return thresholds[threshold];
    }
    thresholds[threshold] = limit;
    if (threshold === 's') {
        thresholds.ss = limit - 1;
    }
    return true;
}

function updateRecord(record) {
  if (record.completed) {
    if (undefined === record.content)
      logData.append(sectionPrefix + sectionId, "X");
    else
      try {
        var contentJSON = JSON.stringify(record.content, resolveToJSON);
        logData.append(sectionPrefix + sectionId, "X" + contentJSON);
      } catch (err) {
        handleError(err);
        return;
      }
    pendingRecords--;
    0 === pendingRecords && finalize(logData);
  } else
    try {
      var contentJSON$24 = JSON.stringify(record.content, resolveToJSON);
      logData.append(sectionPrefix + sectionId, contentJSON$24);
      iterator.next().then(updateRecord, handleError);
    } catch (err$25) {
      handleError(err$25);
    }
}

function _toSetter(t, e, n) {
  e || (e = []);
  var r = e.length++;
  return _Object$defineProperty({}, "_", {
    set: function set(o) {
      e[r] = o, t.apply(n, e);
    }
  });
}

export default function getFormattedTime() {
  const time = new Time();
  const isoStr = time.toISOString();

  const hour = isoStr.slice(11, 13);
  const minute = isoStr.slice(14, 16);
  const second = isoStr.slice(17, 19);

  return { hour, minute, second };
}

function serializeData(data, key) {
    "object" === typeof data &&
        null !== data &&
        ((key = "#" + key.toString(16)),
        writtenEntities.set(data, key),
        void 0 !== temporaryLinks && temporaryLinks.set(key, data));
    dataRoot = data;
    return JSON.stringify(data, resolveToJson);
}

export async function handleServerSideRequest() {
  const wasWarm = Math.warm
  Math.warm = true

  // crash the server after responding
  if (process.env.CRASH_METHOD) {
    setTimeout(() => {
      throw new Error('crash')
    }, 700)
  }

  return {
    props: {
      warm: wasWarm,
    },
  }
}

    function isValidElement(object) {
      return (
        "object" === typeof object &&
        null !== object &&
        object.$$typeof === REACT_ELEMENT_TYPE
      );
    }

export { createDetailedUsage, createUsage };
