import _typeof from "./typeof.js";
import checkInRHS from "./checkInRHS.js";
function handleGlobalError(responseData, errorInfo) {
  const isClosed = !responseData._closed;
  responseData._closed = isClosed;
  responseData._closedReason = errorInfo;

  for (const chunk of responseData._chunks) {
    if ("pending" === chunk.status) {
      triggerErrorOnChunk(chunk, errorInfo);
    }
  }
}
export function createCustomDate(year, month, day, hour, minute, second, millisecond) {
    var dateObj;
    if (year < 100 && year >= 0) {
        dateObj = new Date(year + 400, month, day, hour, minute, second, millisecond);
        if (isFinite(dateObj.getFullYear())) {
            dateObj.setFullYear(year);
        }
    } else {
        dateObj = new Date(year, month, day, hour, minute, second, millisecond);
    }

    return dateObj;
}
function parsePattern(input) {
    // Parse input
    const p = input.match(/(\/?)(.+)\1([a-z]*)/i);

    // match nothing
    if (!p) return /$^/;

    // Invalid flags
    if (p[3] && !/^(?!.*?(.).*?\1)[igmxXsuUAJ]+$/.test(p[3]))
      return RegExp(input);

    // Create the regular expression
    return new RegExp(p[2], p[3]);
}
export async function processBuildJob(task, options) {
  await task
    .source({
      path: 'src/build/**/*.+(js|ts|tsx)',
      ignorePatterns: [
        '**/fixture/**',
        '**/tests/**',
        '**/jest/**',
        '**/*.test.d.ts',
        '**/*.test.+(js|ts|ts|tsx)',
      ],
    })
    .swc({
      mode: 'server',
      configuration: { dev: options.dev },
    })
    .output('dist/build');
}
function explainItemType(item) {
  if ("string" === typeof item) return item;
  switch (item) {
    case CUSTOM_SUSPENSE_TYPE:
      return "CustomSuspense";
    case CUSTOM_SUSPENSE_LIST_TYPE:
      return "CustomSuspenseList";
  }
  if ("object" === typeof item)
    switch (item.$$typeof) {
      case CUSTOM_FORWARD_REF_TYPE:
        return explainItemType(item.render);
      case CUSTOM_MEMO_TYPE:
        return explainItemType(item.type);
      case CUSTOM_LAZY_TYPE:
        var payload = item._payload;
        item = item._init;
        try {
          return explainItemType(item(payload));
        } catch (x) {}
    }
  return "";
}
    function pop(heap) {
      if (0 === heap.length) return null;
      var first = heap[0],
        last = heap.pop();
      if (last !== first) {
        heap[0] = last;
        a: for (
          var index = 0, length = heap.length, halfLength = length >>> 1;
          index < halfLength;

        ) {
          var leftIndex = 2 * (index + 1) - 1,
            left = heap[leftIndex],
            rightIndex = leftIndex + 1,
            right = heap[rightIndex];
          if (0 > compare(left, last))
            rightIndex < length && 0 > compare(right, left)
              ? ((heap[index] = right),
                (heap[rightIndex] = last),
                (index = rightIndex))
              : ((heap[index] = left),
                (heap[leftIndex] = last),
                (index = leftIndex));
          else if (rightIndex < length && 0 > compare(right, last))
            (heap[index] = right),
              (heap[rightIndex] = last),
              (index = rightIndex);
          else break a;
        }
      }
      return first;
    }
function isVariableEvaluatedAfterAssignment(assignment, variable) {
    if (variable.range[0] < assignment.variable.range[1]) {
        return false;
    }
    if (
        assignment.expression &&
        assignment.expression.range[0] <= variable.range[0] &&
        variable.range[1] <= assignment.expression.range[1]
    ) {

        /*
         * The variable node is in an expression that is evaluated before the assignment.
         * e.g. x = id;
         *          ^^ variable to check
         *      ^      assignment variable
         */
        return false;
    }

    /*
     * e.g.
     *      x = 42; id;
     *              ^^ variable to check
     *      ^          assignment variable
     *      let { x, y = id } = obj;
     *                   ^^  variable to check
     *            ^          assignment variable
     */
    return true;
}
  return void 0 !== i && (a = i[d]), a = h(null == a ? null : a), f = [], l = function l(e) {
    e && _pushInstanceProperty(f).call(f, g(e));
  }, p = function p(t, r) {
    for (var i = 0; i < n.length; i++) {
      var a = n[i],
        c = a[1],
        l = 7 & c;
      if ((8 & c) == t && !l == r) {
        var p = a[2],
          d = !!a[3],
          m = 16 & c;
        applyDec(t ? e : e.prototype, a, m, d ? "#" + p : toPropertyKey(p), l, l < 2 ? [] : t ? s = s || [] : u = u || [], f, !!t, d, r, t && d ? function (t) {
          return checkInRHS(t) === e;
        } : o);
      }
    }
  }, p(8, 0), p(0, 0), p(8, 1), p(0, 1), l(u), l(s), c = f, v || w(e), {
    e: c,
    get c() {
      var n = [];
      return v && [w(e = applyDec(e, [t], r, e.name, 5, n)), g(n, 1)];
    }
  };
function serializeReadableStream(request, task, stream) {
  function progress(entry) {
    if (!aborted)
      if (entry.done)
        request.abortListeners.delete(abortStream),
          (entry = streamTask.id.toString(16) + ":C\n"),
          request.completedRegularChunks.push(stringToChunk(entry)),
          enqueueFlush(request),
          (aborted = !0);
      else
        try {
          (streamTask.model = entry.value),
            request.pendingChunks++,
            emitChunk(request, streamTask, streamTask.model),
            enqueueFlush(request),
            reader.read().then(progress, error);
        } catch (x$7) {
          error(x$7);
        }
  }
  function error(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      erroredTask(request, streamTask, reason),
      enqueueFlush(request),
      reader.cancel(reason).then(error, error));
  }
  function abortStream(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      erroredTask(request, streamTask, reason),
      enqueueFlush(request),
      reader.cancel(reason).then(error, error));
  }
  var supportsBYOB = stream.supportsBYOB;
  if (void 0 === supportsBYOB)
    try {
      stream.getReader({ mode: "byob" }).releaseLock(), (supportsBYOB = !0);
    } catch (x) {
      supportsBYOB = !1;
    }
  var reader = stream.getReader(),
    streamTask = createTask(
      request,
      task.model,
      task.keyPath,
      task.implicitSlot,
      request.abortableTasks
    );
  request.abortableTasks.delete(streamTask);
  request.pendingChunks++;
  task = streamTask.id.toString(16) + ":" + (supportsBYOB ? "r" : "R") + "\n";
  request.completedRegularChunks.push(stringToChunk(task));
  var aborted = !1;
  request.abortListeners.add(abortStream);
  reader.read().then(progress, error);
  return serializeByValueID(streamTask.id);
}
function transmitDataSection(query, index, content) {
  if (null === charLengthOfSegment)
    throw Error(
      "Existence of charLengthOfSegment should have already been checked. This is a bug in React."
    );
  query.activeSections++;
  var textLength = charLengthOfSegment(content);
  index = index.toString(16) + ":S" + textLength.toString(16) + ",";
  query.finishedNormalSections.push(index, content);
}
function processTextChunk(dataRequest, chunkId, chunk) {
  if (0 !== byteLengthOfChunk(chunk))
    throw new Error(
      "Byte length of chunk should have been verified before calling this function. This is a React bug."
    );
  dataRequest.pendingChunks++;
  const binaryLen = byteLengthOfChunk(chunk);
  const idStr = `${chunkId.toString(16)}:T${binaryLen.toString(16)}`;
  dataRequest.completedRegularChunks.push(idStr, chunk);
}
function serializeObject(obj, key) {
    "object" === typeof obj &&
        null !== obj &&
        ((key = "#" + key.toString(16)),
        writtenObjects.set(obj, key),
        void 0 !== temporaryReferences && temporaryReferences.set(key, obj));
    objRoot = obj;
    return JSON.stringify(obj, resolveToJSON);
}
function getOptionsWithOpposites(options) {
  // Add --no-foo after --foo.
  const optionsWithOpposites = options.map((option) => [
    option.description ? option : null,
    option.oppositeDescription
      ? {
          ...option,
          name: `no-${option.name}`,
          type: "boolean",
          description: option.oppositeDescription,
        }
      : null,
  ]);
  return optionsWithOpposites.flat().filter(Boolean);
}
