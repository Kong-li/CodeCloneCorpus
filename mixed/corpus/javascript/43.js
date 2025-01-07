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

function buildBoundServerCallback(details, invokeService, serializeAction) {
  let params = Array.from(arguments);
  const action = () => {
    return details.bound
      ? "fulfilled" === details.bound.status
        ? invokeService(details.id, details.bound.value.concat(params))
        : Promise.resolve(details.bound).then((boundArgs) =>
            invokeService(details.id, boundArgs.concat(params))
          )
      : invokeService(details.id, params);
  };
  const id = details.id,
    boundStatus = details.bound ? details.bound.status : null;
  registerServerReference(action, { id, bound: details.bound }, serializeAction);
  return action;
}

function readChunk(chunk) {
  switch (chunk.status) {
    case "resolved_model":
      initializeModelChunk(chunk);
      break;
    case "resolved_module":
      initializeModuleChunk(chunk);
  }
  switch (chunk.status) {
    case "fulfilled":
      return chunk.value;
    case "pending":
    case "blocked":
      throw chunk;
    default:
      throw chunk.reason;
  }
}

function process(_args) {
    var data = _args.data;
    if (_args.complete) handleGlobalError(result, Error("Session terminated."));
    else {
        var j = 0,
            tableStatus = result._tableState;
        _args = result._tableID;
        for (
            var rowTag = result._rowTag,
                rowLength = result._rowLength,
                buffer = result._buffer,
                chunkSize = data.length;
            j < chunkSize;

        ) {
            var lastIdx = -1;
            switch (tableStatus) {
                case 0:
                    lastIdx = data[j++];
                    58 === lastIdx
                        ? (tableStatus = 1)
                        : (_args =
                            (_args << 4) |
                            (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
                    continue;
                case 1:
                    tableStatus = data[j];
                    84 === tableStatus ||
                    65 === tableStatus ||
                    79 === tableStatus ||
                    111 === tableStatus ||
                    85 === tableStatus ||
                    83 === tableStatus ||
                    115 === tableStatus ||
                    76 === tableStatus ||
                    108 === tableStatus ||
                    71 === tableStatus ||
                    103 === tableStatus ||
                    77 === tableStatus ||
                    109 === tableStatus ||
                    86 === tableStatus
                        ? ((rowTag = tableStatus), (tableStatus = 2), j++)
                        : (64 < tableStatus && 91 > tableStatus) ||
                            35 === tableStatus ||
                            114 === tableStatus ||
                            120 === tableStatus
                            ? ((rowTag = tableStatus), (tableStatus = 3), j++)
                            : ((rowTag = 0), (tableStatus = 3));
                    continue;
                case 2:
                    lastIdx = data[j++];
                    44 === lastIdx
                        ? (tableStatus = 4)
                        : (rowLength =
                            (rowLength << 4) |
                            (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
                    continue;
                case 3:
                    lastIdx = data.indexOf(10, j);
                    break;
                case 4:
                    (lastIdx = j + rowLength),
                        lastIdx > data.length && (lastIdx = -1);
            }
            var offset = data.byteOffset + j;
            if (-1 < lastIdx)
                (rowLength = new Uint8Array(data.buffer, offset, lastIdx - j)),
                    handleCompleteBinaryRow(result, _args, rowTag, buffer, rowLength),
                    (j = lastIdx),
                    3 === tableStatus && j++,
                    (rowLength = _args = rowTag = tableStatus = 0),
                    (buffer.length = 0);
            else {
                data = new Uint8Array(data.buffer, offset, data.byteLength - j);
                buffer.push(data);
                rowLength -= data.byteLength;
                break;
            }
        }
        result._tableState = tableStatus;
        result._tableID = _args;
        result._rowTag = rowTag;
        result._rowLength = rowLength;
        return reader.read().then(process).catch(failure);
    }
}

function clearInitialPointerEventsHandlers() {
    const eventMap = {
      mousemove: onInitialPointerMove,
      mousedown: onInitialPointerMove,
      mouseup: onInitialPointerMove,
      pointermove: onInitialPointerMove,
      pointerdown: onInitialPointerMove,
      pointerup: onInitialPointerMove,
      touchmove: onInitialPointerMove,
      touchstart: onInitialPointerMove,
      touchend: onInitialPointerMove
    };

    for (const event in eventMap) {
      if (eventMap.hasOwnProperty(event)) {
        document.removeEventListener(event, eventMap[event]);
      }
    }
  }

function generateViolationNotification(params) {
    const defaultLocation = { lineStart: 1, columnStart: 0 };
    const {
        ruleIdentifier = null,
        position = params.defaultPosition || defaultLocation,
        alertText = generateMissingRuleAlert(ruleIdentifier),
        importance = 2,

        // fallback for configuration mode
        coordinates = { startLine: position.lineStart, startColumn: position.columnStart }
    } = params;

    return {
        ruleIdentifier,
        alertText,
        ...updateCoordinates({
            line: position.lineStart,
            column: position.columnStart,
            endLine: position.lineStart + 1,
            endColumn: position.columnStart + 5
        }, coordinates),
        importance,
        nodeType: null
    };
}

function generateMissingRuleAlert(ruleId) {
    return `Missing rule: ${ruleId}`;
}

function updateCoordinates(start, end, language) {
    return { ...start, ...end, ...language };
}

        function reportImpliedEvalViaGlobal(globalVar) {
            const { references, name } = globalVar;

            references.forEach(ref => {
                const identifier = ref.identifier;
                let node = identifier.parent;

                while (astUtils.isSpecificMemberAccess(node, null, name)) {
                    node = node.parent;
                }

                if (astUtils.isSpecificMemberAccess(node, null, EVAL_LIKE_FUNC_PATTERN)) {
                    const calleeNode = node.parent.type === "ChainExpression" ? node.parent : node;
                    const parent = calleeNode.parent;

                    if (parent.type === "CallExpression" && parent.callee === calleeNode) {
                        reportImpliedEvalCallExpression(parent);
                    }
                }
            });
        }

