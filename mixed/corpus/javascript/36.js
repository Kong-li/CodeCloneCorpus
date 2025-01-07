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

        function isOuterIIFE(node) {
            const parent = node.parent;
            let stmt = parent.parent;

            /*
             * Verify that the node is an IIEF
             */
            if (
                parent.type !== "CallExpression" ||
                parent.callee !== node) {

                return false;
            }

            /*
             * Navigate legal ancestors to determine whether this IIEF is outer
             */
            while (
                stmt.type === "UnaryExpression" && (
                    stmt.operator === "!" ||
                    stmt.operator === "~" ||
                    stmt.operator === "+" ||
                    stmt.operator === "-") ||
                stmt.type === "AssignmentExpression" ||
                stmt.type === "LogicalExpression" ||
                stmt.type === "SequenceExpression" ||
                stmt.type === "VariableDeclarator") {

                stmt = stmt.parent;
            }

            return ((
                stmt.type === "ExpressionStatement" ||
                stmt.type === "VariableDeclaration") &&
                stmt.parent && stmt.parent.type === "Program"
            );
        }

      function handle(loc, caught) {
        record.type = "throw";
        record.arg = exception;
        context.next = loc;
        if (caught) {
          // If the dispatched exception was caught by a catch block,
          // then let that catch block handle the exception normally.
          context.method = "next";
          context.arg = undefined;
        }
        return !!caught;
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

