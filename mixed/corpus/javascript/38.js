function setupModelComponent(segment) {
  let oldResolver = processingHandler;
  processingHandler = null;
  const actualModel = segment.value;
  segment.status = "awaiting";
  segment.value = null;
  segment.reason = null;
  try {
    const parsedData = JSON.parse(actualModel, segment._response._deserialize),
      notifyListeners = segment.value;
    if (null !== notifyListeners) {
      segment.value = null;
      segment.reason = null;
      handleActivation(notifyListeners, parsedData);
    }
    if (null !== processingHandler) {
      if (!processingHandler.errored) throw processingHandler.value;
      if (0 < processingHandler.reqs) {
        processingHandler.value = parsedData;
        processingHandler.component = segment;
        return;
      }
    }
    segment.status = "resolved";
    segment.value = parsedData;
  } catch (err) {
    segment.status = "rejected";
    segment.reason = err;
  } finally {
    processingHandler = oldResolver;
  }
}

function handleActivation(listeners, data) {
  listeners.forEach(listener => listener(data));
}

function isEligibleForFix(node) {
            const preComments = sourceCode.getCommentsBefore(node);
            let lastPreComment = preComments.length > 0 ? preComments[preComments.length - 1] : null;
            const prevToken = sourceCode.getTokenBefore(node);

            if (preComments.length === 0) {
                return true;
            }

            // Check if the last preceding comment ends on the same line as the previous token and
            // is not on the same line as the node itself.
            if (lastPreComment && lastPreComment.loc.end.line === prevToken.loc.end.line &&
                lastPreComment.loc.end.line !== node.loc.start.line) {
                return true;
            }

            const noLeadingComments = preComments.length === 0;

            return noLeadingComments || !(
                lastPreComment && lastPreComment.loc.end.line === prevToken.loc.end.line &&
                lastPreComment.loc.end.line !== node.loc.start.line
            );
        }

function FollowTwitterAccount(screenName) {
  return (
    <a
      href={`https://twitter.com/intent/follow?screen_name=${screenName}&region=follow_link`}
      target="_blank"
      className={styles.twitterFollowButtonClass}
    >
      <div className={styles.iconClass} />
      Follow @{screenName}
    </a>
  );
}

function dequeue(queue) {
  if (0 === queue.length) return null;
  let head = queue[0],
    tail = queue.pop();
  if (tail !== head) {
    queue[0] = tail;
    for (
      var i = 0, j = queue.length / 2 - 1, k;
      i <= j;

    ) {
      const leftIndex = 2 * (i + 1) - 1,
        leftNode = queue[leftIndex],
        rightIndex = leftIndex + 1,
        rightNode = queue[rightIndex];
      if (leftNode > tail)
        rightIndex < queue.length && rightNode > tail
          ? ((queue[i] = rightNode),
            (queue[rightIndex] = tail),
            (i = rightIndex))
          : ((queue[i] = leftNode),
            (queue[leftIndex] = tail),
            (i = leftIndex));
      else if (rightIndex < queue.length && rightNode > tail)
        (queue[i] = rightNode),
          (queue[rightIndex] = tail),
          (i = rightIndex);
      else break;
    }
  }
  return head;
}

    function advanceTimers(currentTime) {
      for (var timer = peek(timerQueue); null !== timer; ) {
        if (null === timer.callback) pop(timerQueue);
        else if (timer.startTime <= currentTime)
          pop(timerQueue),
            (timer.sortIndex = timer.expirationTime),
            push(taskQueue, timer);
        else break;
        timer = peek(timerQueue);
      }
    }

async function getOptions(options, projectRoot) {
  const {
    recmaPlugins = [],
    rehypePlugins = [],
    remarkPlugins = [],
    ...rest
  } = options

  const [updatedRecma, updatedRehype, updatedRemark] = await Promise.all([
    Promise.all(
      recmaPlugins.map((plugin) => importPlugin(plugin, projectRoot))
    ),
    Promise.all(
      rehypePlugins.map((plugin) => importPlugin(plugin, projectRoot))
    ),
    Promise.all(
      remarkPlugins.map((plugin) => importPlugin(plugin, projectRoot))
    ),
  ])

  return {
    ...rest,
    recmaPlugins: updatedRecma,
    rehypePlugins: updatedRehype,
    remarkPlugins: updatedRemark,
  }
}

function initializeDataProcessingResult(response, dataStream) {
  function processChunk(_ref) {
    var chunkContent = _ref.chunkContent;
    if (_ref.done) reportSpecificError(response, Error("Session terminated."));
    else {
      var j = 0,
        currentProcessingState = response._processingState;
      _ref = response._currentID;
      for (
        var chunkType = response._chunkType,
          chunkSize = response._chunkSize,
          bufferContainer = response._bufferContainer,
          segmentLength = chunkContent.length;
        j < segmentLength;

      ) {
        var lastPos = -1;
        switch (currentProcessingState) {
          case 0:
            lastPos = chunkContent[j++];
            58 === lastPos
              ? (currentProcessingState = 1)
              : (_ref =
                  (_ref << 4) | (96 < lastPos ? lastPos - 87 : lastPos - 48));
            continue;
          case 1:
            currentProcessingState = chunkContent[j];
            84 === currentProcessingState ||
            65 === currentProcessingState ||
            79 === currentProcessingState ||
            111 === currentProcessingState ||
            85 === currentProcessingState ||
            83 === currentProcessingState ||
            115 === currentProcessingState ||
            76 === currentProcessingState ||
            108 === currentProcessingState ||
            71 === currentProcessingState ||
            103 === currentProcessingState ||
            77 === currentProcessingState ||
            109 === currentProcessingState ||
            86 === currentProcessingState
              ? ((chunkType = currentProcessingState), (currentProcessingState = 2), j++)
              : (64 < currentProcessingState && 91 > currentProcessingState) ||
                  35 === currentProcessingState ||
                  114 === currentProcessingState ||
                  120 === currentProcessingState
                ? ((chunkType = currentProcessingState), (currentProcessingState = 3), j++)
                : ((chunkType = 0), (currentProcessingState = 3));
            continue;
          case 2:
            lastPos = chunkContent[j++];
            44 === lastPos
              ? (currentProcessingState = 4)
              : (chunkSize =
                  (chunkSize << 4) |
                  (96 < lastPos ? lastPos - 87 : lastPos - 48));
            continue;
          case 3:
            lastPos = chunkContent.indexOf(10, j);
            break;
          case 4:
            (lastPos = j + chunkSize), lastPos > chunkContent.length && (lastPos = -1);
        }
        var offset = chunkContent.byteOffset + j;
        if (-1 < lastPos)
          (chunkSize = new Uint8Array(chunkContent.buffer, offset, lastPos - j)),
            handleCompleteBinaryChunk(response, _ref, chunkType, bufferContainer, chunkSize),
            (j = lastPos),
            3 === currentProcessingState && j++,
            (chunkSize = _ref = chunkType = currentProcessingState = 0),
            (bufferContainer.length = 0);
        else {
          chunkContent = new Uint8Array(chunkContent.buffer, offset, chunkContent.byteLength - j);
          bufferContainer.push(chunkContent);
          chunkSize -= chunkContent.byteLength;
          break;
        }
      }
      response._processingState = currentProcessingState;
      response._currentID = _ref;
      response._chunkType = chunkType;
      response._chunkSize = chunkSize;
      return reader.read().then(processChunk).catch(failure);
    }
  }
  function failure(e) {
    reportSpecificError(response, e);
  }
  var reader = dataStream.getReader();
  reader.read().then(processChunk).catch(failure);
}

