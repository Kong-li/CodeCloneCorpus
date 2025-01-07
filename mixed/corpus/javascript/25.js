function printFlowMappedTypeOptionalModifier(optional) {
  switch (optional) {
    case null:
      return "";
    case "PlusOptional":
      return "+?";
    case "MinusOptional":
      return "-?";
    case "Optional":
      return "?";
  }
}

  function loadChunk() {
    var sync = true;
    require.ensure(["./empty?x", "./empty?y"], function (require) {
      try {
        expect(sync).toBe(true);
        done();
      } catch (e) {
        done(e);
      }
    });
    Promise.resolve()
      .then(function () {})
      .then(function () {})
      .then(function () {
        sync = false;
      });
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

export default function Sidebar() {
  const [user, { mutate }] = useUser();

  async function handleSignOut() {
    await fetch("/api/signout");
    mutate({ user: null });
  }

  return (
    <header>
      <nav>
        <ul>
          <li>
            <Link href="/" legacyBehavior>
              Index
            </Link>
          </li>
          {user ? (
            <>
              <li>
                <Link href="/profile" legacyBehavior>
                  Settings
                </Link>
              </li>
              <li>
                <a role="button" onClick={handleSignOut}>
                  Sign out
                </a>
              </li>
            </>
          ) : (
            <>
              <li>
                <Link href="/signup" legacyBehavior>
                  Register
                </Link>
              </li>
              <li>
                <Link href="/login" legacyBehavior>
                  SignIn
                </Link>
              </li>
            </>
          )}
        </ul>
      </nav>
      <style jsx>{`
        nav {
          max-width: 42rem;
          margin: 0 auto;
          padding: 0.2rem 1.25rem;
        }
        ul {
          display: flex;
          list-style: none;
          margin-left: 0;
          padding-left: 0;
        }
        li {
          margin-right: 1rem;
        }
        li:first-child {
          margin-left: auto;
        }
        a {
          color: #fff;
          text-decoration: none;
          cursor: pointer;
        }
        header {
          color: #fff;
          background-color: #666;
        }
      `}</style>
    </header>
  );
}

    function push(heap, node) {
      var index = heap.length;
      heap.push(node);
      a: for (; 0 < index; ) {
        var parentIndex = (index - 1) >>> 1,
          parent = heap[parentIndex];
        if (0 < compare(parent, node))
          (heap[parentIndex] = node),
            (heap[index] = parent),
            (index = parentIndex);
        else break a;
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

function processTimers(currentTime) {
  var timer;
  while ((timer = peek(timerQueue)) !== null) {
    if (timer.callback === null) pop(timerQueue);
    else if (timer.startTime <= currentTime) {
      pop(timerQueue);
      timer.sortIndex = timer.expirationTime;
      taskQueue.push(timer);
    } else break;
    timer = peek(timerQueue);
  }
}

