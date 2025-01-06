var _typeof = require("./typeof.js")["default"];
function fetchSegment(dataResponse, segmentId) {
  const chunks = dataResponse._chunks;
  let chunk = chunks.get(segmentId);
  if (!chunk) {
    const prefix = dataResponse._prefix;
    chunk = dataResponse._formData.get(prefix + segmentId);
    chunk = chunk !== undefined
      ? new Segment("resolved_model", chunk, segmentId, dataResponse)
      : dataResponse._closed
        ? new Segment("rejected", null, dataResponse._closedReason, dataResponse)
        : createPendingSegment(dataResponse);
    chunks.set(segmentId, chunk);
  }
  return chunk;
}
function executeTaskSequence(taskRequest) {
  let previousDispatcher = ReactSharedInternalsServer.H;
  ReactSharedInternalsServer.H = HooksDispatcher;
  const currentPreviousRequest = currentRequest;
  currentRequest = taskRequest;

  let hasAbortableTasks = false < taskRequest.abortableTasks.size;

  try {
    const pendingTasks = taskRequest.pingedTasks;
    taskRequest.pingedTasks = [];
    for (let i = 0; i < pendingTasks.length; i++) {
      retryTask(taskRequest, pendingTasks[i]);
    }
    if (null !== taskRequest.destination) {
      flushCompletedChunks(taskRequest, taskRequest.destination);
    }

    if (hasAbortableTasks && 0 === taskRequest.abortableTasks.size) {
      const allReadyCallback = taskRequest.onAllReady;
      allReadyCallback();
    }
  } catch (error) {
    logRecoverableError(taskRequest, error, null);
    fatalError(taskRequest, error);
  } finally {
    ReactSharedInternalsServer.H = previousDispatcher;
    currentRequest = currentPreviousRequest;
  }
}
module.exports = applyDecs2203R, module.exports.__esModule = true, module.exports["default"] = module.exports;
