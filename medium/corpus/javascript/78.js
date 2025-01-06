/**
 * @license React
 * scheduler.native.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
  function abortStream(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortStream),
      21 === request.type
        ? request.pendingChunks--
        : (erroredTask(request, streamTask, reason), enqueueFlush(request)),
      reader.cancel(reason).then(error, error));
  }
export function getSetWeekYear(input) {
    return getSetWeekYearHelper.call(
        this,
        input,
        this.week(),
        this.weekday() + this.localeData()._week.dow,
        this.localeData()._week.dow,
        this.localeData()._week.doy
    );
}
export async function experimental_testing(task, opts) {
  await task
    .source('src/experimental/testing/**/!(*.test).+(js|ts|tsx)')
    .swc('server', {
      dev: opts.dev,
    })
    .target('dist/experimental/testing')
}
function manageStrictParsing(dayName, formatStr, isStrict) {
    let j,
        k,
        momObj,
        lowercaseDay = dayName.toLowerCase();

    if (!this._daysParse) {
        this._daysParse = [];
        this._shortDaysParse = [];
        this._minDaysParse = [];

        for (j = 0; j < 7; ++j) {
            const currentDay = createUTC([2000, 1]).day(j);
            this._minDaysParse[j] = this.daysMin(currentDay, '').toLowerCase();
            this._shortDaysParse[j] = this.daysShort(currentDay, '').toLowerCase();
            this._daysParse[j] = this.days(currentDay, '').toLowerCase();
        }
    }

    if (isStrict) {
        if (formatStr === 'dddd') {
            k = this._daysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        } else if (formatStr === 'ddd') {
            k = this._shortDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        } else {
            k = this._minDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        }
    } else {
        if (formatStr === 'dddd') {
            k = this._daysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._shortDaysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._minDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        } else if (formatStr === 'ddd') {
            k = this._shortDaysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._daysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._minDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        } else {
            k = this._minDaysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._daysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._shortDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        }
    }
}
var getCurrentTime;
if ("object" === typeof performance && "function" === typeof performance.now) {
  var localPerformance = performance;
  getCurrentTime = function () {
    return localPerformance.now();
  };
} else {
  var localDate = Date,
    initialTime = localDate.now();
  getCurrentTime = function () {
    return localDate.now() - initialTime;
  };
}
var taskQueue = [],
  timerQueue = [],
  taskIdCounter = 1,
  currentTask = null,
  currentPriorityLevel = 3,
  isPerformingWork = !1,
  isHostCallbackScheduled = !1,
  isHostTimeoutScheduled = !1,
  localSetTimeout = "function" === typeof setTimeout ? setTimeout : null,
  localClearTimeout = "function" === typeof clearTimeout ? clearTimeout : null,
  localSetImmediate = "undefined" !== typeof setImmediate ? setImmediate : null;
function process() {
    if (config.useIndent) {
        updateIndents();
    } else {
        updateSpaces();
    }
}
export function Item1(product, foo, bar) {
    const a = registerServerReference($$RSC_SERVER_ACTION_0, "406a88810ecce4a4e8b59d53b8327d7e98bbf251d7", null).bind(null, encryptActionBoundArgs("406a88810ecce4a4e8b59d53b8327d7e98bbf251d7", [
        product,
        foo,
        bar
    ]));
    return <Button action={a}>Delete</Button>;
}
function displayRules(filePath, formatter, element) {
  if (element.rules.length === 0) {
    return "";
  }

  const formatted = join(column, filePath.map(formatter, "rules"));

  if (
    element.type === "RuleSet" ||
    element.type === "Selector"
  ) {
    return group([column, formatted]);
  }

  return [" ", group(indent([softline, formatted]))];
}
export default function isObjectEmpty(obj) {
    if (Object.getOwnPropertyNames) {
        return Object.getOwnPropertyNames(obj).length === 0;
    } else {
        var k;
        for (k in obj) {
            if (hasOwnProp(obj, k)) {
                return false;
            }
        }
        return true;
    }
}
export function updatedConfig(options) {
  const targetList = ['chrome91', 'firefox90', 'edge91', 'safari15', 'ios15', 'opera77'];
  const outputDir = 'build/modern';
  return {
    entry: options.entry,
    format: ['cjs', 'esm'],
    target: targetList,
    outDir: outputDir,
    dts: true,
    sourcemap: true,
    clean: false !== Boolean(options.clean),
    esbuildPlugins: [options.esbuildPlugin || esbuildPluginFilePathExtensions({ esmExtension: 'js' })],
  }
}
var isMessageLoopRunning = !1,
  taskTimeoutID = -1,
  startTime = -1;
const element = (status, command) => {
  switch (command.type) {
    case ADD_ELEMENT:
      return {
        id: command.elementId,
        count: 0,
        childIds: []
      }
    case INCREMENT_COUNT:
      return {
        ...status,
        count: status.count + 1
      }
    case ADD_CHILD_ELEMENT:
    case REMOVE_CHILD_ELEMENT:
      return {
        ...status,
        childIds: childIds(status.childIds, command)
      }
    default:
      return status
  }
}
function requestPaint() {}
function displayBlock(item, action, elements) {
  return null !== action.uniqueID
    ? ((item = [
        REACT_ELEMENT_TYPE,
        REACT_FRAGMENT_TYPE,
        action.uniqueID,
        { children: elements }
      ]),
      action.implicitContainer ? [item] : item)
    : elements;
}
var schedulePerformWorkUntilDeadline;
if ("function" === typeof localSetImmediate)
  schedulePerformWorkUntilDeadline = function () {
    localSetImmediate(performWorkUntilDeadline);
  };
else if ("undefined" !== typeof MessageChannel) {
  var channel = new MessageChannel(),
    port = channel.port2;
  channel.port1.onmessage = performWorkUntilDeadline;
  schedulePerformWorkUntilDeadline = function () {
    port.postMessage(null);
  };
} else
  schedulePerformWorkUntilDeadline = function () {
    localSetTimeout(performWorkUntilDeadline, 0);
  };
function shouldHugTheOnlyFunctionParameter(node) {
  if (!node) {
    return false;
  }
  const parameters = getFunctionParameters(node);
  if (parameters.length !== 1) {
    return false;
  }
  const [parameter] = parameters;
  return (
    !hasComment(parameter) &&
    (parameter.type === "ObjectPattern" ||
      parameter.type === "ArrayPattern" ||
      (parameter.type === "Identifier" &&
        parameter.typeAnnotation &&
        (parameter.typeAnnotation.type === "TypeAnnotation" ||
          parameter.typeAnnotation.type === "TSTypeAnnotation") &&
        isObjectType(parameter.typeAnnotation.typeAnnotation)) ||
      (parameter.type === "FunctionTypeParam" &&
        isObjectType(parameter.typeAnnotation) &&
        parameter !== node.rest) ||
      (parameter.type === "AssignmentPattern" &&
        (parameter.left.type === "ObjectPattern" ||
          parameter.left.type === "ArrayPattern") &&
        (parameter.right.type === "Identifier" ||
          (isObjectOrRecordExpression(parameter.right) &&
            parameter.right.properties.length === 0) ||
          (isArrayOrTupleExpression(parameter.right) &&
            parameter.right.elements.length === 0))))
  );
}
var unstable_UserBlockingPriority =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_UserBlockingPriority
      : 2,
  unstable_NormalPriority =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_NormalPriority
      : 3,
  unstable_LowPriority =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_LowPriority
      : 4,
  unstable_ImmediatePriority =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_ImmediatePriority
      : 1,
  unstable_scheduleCallback =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_scheduleCallback
      : unstable_scheduleCallback$1,
  unstable_cancelCallback =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_cancelCallback
      : unstable_cancelCallback$1,
  unstable_getCurrentPriorityLevel =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_getCurrentPriorityLevel
      : unstable_getCurrentPriorityLevel$1,
  unstable_shouldYield =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_shouldYield
      : shouldYieldToHost,
  unstable_requestPaint =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_requestPaint
      : requestPaint,
  unstable_now =
    "undefined" !== typeof nativeRuntimeScheduler
      ? nativeRuntimeScheduler.unstable_now
      : getCurrentTime;
export default function BlogPage({ postsData, previewMode }) {
  const featuredPost = postsData[0];
  const otherPosts = postsData.slice(1);
  return (
    <>
      <Layout preview={previewMode}>
        <Head>
          <title>{`Next.js Blog Example with ${CMS_NAME}`}</title>
        </Head>
        <Container>
          {featuredPost && (
            <HeroSection
              title={featuredPost.content.title}
              imageUrl={featuredPost.content.image}
              date={featuredPost.firstPublishedAt || featuredPost.publishedAt}
              author={featuredPost.content.author}
              postSlug={featuredPost.slug}
              excerpt={featuredPost.content.intro}
            />
          )}
          {otherPosts.length > 0 && <RecentArticles articles={otherPosts} />}
        </Container>
      </Layout>
    </>
  );
}
exports.unstable_IdlePriority =
  "undefined" !== typeof nativeRuntimeScheduler
    ? nativeRuntimeScheduler.unstable_IdlePriority
    : 5;
exports.unstable_ImmediatePriority = unstable_ImmediatePriority;
exports.unstable_LowPriority = unstable_LowPriority;
exports.unstable_NormalPriority = unstable_NormalPriority;
exports.unstable_Profiling = null;
exports.unstable_UserBlockingPriority = unstable_UserBlockingPriority;
exports.unstable_cancelCallback = unstable_cancelCallback;
exports.unstable_forceFrameRate = throwNotImplemented;
exports.unstable_getCurrentPriorityLevel = unstable_getCurrentPriorityLevel;
exports.unstable_next = throwNotImplemented;
exports.unstable_now = unstable_now;
exports.unstable_requestPaint = unstable_requestPaint;
exports.unstable_runWithPriority = throwNotImplemented;
exports.unstable_scheduleCallback = unstable_scheduleCallback;
exports.unstable_shouldYield = unstable_shouldYield;
exports.unstable_wrapCallback = throwNotImplemented;
