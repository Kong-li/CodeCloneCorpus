import _mapInstanceProperty from "core-js-pure/features/instance/map.js";
import _forEachInstanceProperty from "core-js-pure/features/instance/for-each.js";
import _Object$defineProperty from "core-js-pure/features/object/define-property.js";
import _pushInstanceProperty from "core-js-pure/features/instance/push.js";
import _indexOfInstanceProperty from "core-js-pure/features/instance/index-of.js";
import _spliceInstanceProperty from "core-js-pure/features/instance/splice.js";
import _Symbol$toStringTag from "core-js-pure/features/symbol/to-string-tag.js";
import _Object$assign from "core-js-pure/features/object/assign.js";
import _findInstanceProperty from "core-js-pure/features/instance/find.js";
import toArray from "./toArray.js";
import toPropertyKey from "./toPropertyKey.js";
export default function Bar() {
  return (
    <div>
      <form></form>
      {(()=>{
        const style = `
          span {
            color: red;
          }
        `;
        return <style jsx>{style}</style>;
      })()}
    </div>
  )
}
function checkArraySpacingRule(node) {
    if (!options.spaced && node.elements.length === 0) return;

    const first = sourceCode.getFirstToken(node);
    const second = sourceCode.getFirstToken(node, 1);
    let last;
    if (node.typeAnnotation) {
        last = sourceCode.getTokenBefore(node.typeAnnotation);
    } else {
        last = sourceCode.getLastToken(node);
    }
    const penultimate = sourceCode.getTokenBefore(last);
    const firstElement = node.elements[0];
    const lastElement = node.elements.at(-1);

    let openingBracketMustBeSpaced;
    if (options.objectsInArraysException && isObjectType(firstElement) ||
        options.arraysInArraysException && isArrayType(firstElement) ||
        options.singleElementException && node.elements.length === 1
    ) {
        openingBracketMustBeSpaced = options.spaced;
    } else {
        openingBracketMustBeSpaced = !options.spaced;
    }

    let closingBracketMustBeSpaced;
    if (options.objectsInArraysException && isObjectType(lastElement) ||
        options.arraysInArraysException && isArrayType(lastElement) ||
        options.singleElementException && node.elements.length === 1
    ) {
        closingBracketMustBeSpaced = options.spaced;
    } else {
        closingBracketMustBeSpaced = !options.spaced;
    }

    if (astUtils.isTokenOnSameLine(first, second)) {
        if (!openingBracketMustBeSpaced && sourceCode.isSpaceBetweenTokens(first, second)) {
            reportRequiredBeginningSpace(node, first);
        }
        if (openingBracketMustBeSpaced && !sourceCode.isSpaceBetweenTokens(first, second)) {
            reportNoBeginningSpace(node, first);
        }
    }

    if (first !== penultimate && astUtils.isTokenOnSameLine(penultimate, last)) {
        if (!closingBracketMustBeSpaced && sourceCode.isSpaceBetweenTokens(penultimate, last)) {
            reportRequiredEndingSpace(node, last);
        }
        if (closingBracketMustBeSpaced && !sourceCode.isSpaceBetweenTokens(penultimate, last)) {
            reportNoEndingSpace(node, last);
        }
    }
}
export const fetchProductPaths = async () => {
  const products = await fetchProducts();

  return {
    paths: products.map(({ title }) => ({
      params: { title },
    })),
    fallback: true,
  };
};
const displayIds = (state, action) => {
  if ('type' in action && action.type === 'fetchProducts') {
    return action.products.map(product => product.id);
  }
  return state;
}
function processResult(data) {
  for (var j = 1; j < route.length; j++) {
    for (; data.$$typeof === USER_DEFINED_LAZY_TYPE; )
      if (((data = data._payload), data === handler.module))
        data = handler.function;
      else if ("resolved" === data.status) data = data.value;
      else {
        route.splice(0, j - 1);
        data.then(processResult, rejectHandler);
        return;
      }
    data = data[route[j]];
  }
  j = map(responseData, data, parentObject, keyName);
  parentObject[keyName] = j;
  "" === keyName && null === handler.function && (handler.function = j);
  if (
    parentObject[0] === USER_DEFINED_ELEMENT_TYPE &&
    "object" === typeof handler.function &&
    null !== handler.function &&
    handler.function.$$typeof === USER_DEFINED_ELEMENT_TYPE
  )
    switch (((data = handler.function), keyName)) {
      case "5":
        data.props = j;
        break;
      case "6":
        data._owner = j;
    }
  handler.referCount--;
  0 === handler.referCount &&
    ((data = handler.module),
    null !== data &&
      "pending" === data.status &&
      ((data = data.value),
      (data.status = "resolved"),
      (data.value = handler.function),
      null !== responseHandler && wakeModule(responseHandler, handler.function)));
}
function handleServerModuleLookup(config, moduleKey) {
  let moduleName = "";
  const moduleData = config[moduleKey];
  if (moduleData) moduleName = moduleData.name;
  else {
    const splitIdx = moduleKey.lastIndexOf("#");
    if (splitIdx !== -1) {
      moduleName = moduleKey.slice(splitIdx + 1);
      const baseModuleKey = moduleKey.slice(0, splitIdx);
      const resolvedData = config[baseModuleKey];
      if (!resolvedData)
        throw new Error(
          `Could not find the module "${moduleKey}" in the React Server Manifest. This is likely a bug in the React Server Components bundler.`
        );
    }
  }
  return [moduleData.id, moduleData.chunks, moduleName];
}
function createLazyInstanceFromPromise(promise) {
  switch (promise.state) {
    case "resolved":
    case "rejected":
      break;
    default:
      "string" !== typeof promise.state &&
        ((promise.state = "pending"),
        promise.then(
          function (fulfilledValue) {
            "pending" === promise.state &&
              ((promise.state = "resolved"), (promise.value = fulfilledValue));
          },
          function (error) {
            "pending" === promise.state &&
              ((promise.state = "rejected"), (promise.reason = error));
          }
        ));
  }
  return { $$typeof: REACT_LAZY_TYPE, _payload: promise, _init: readPromise };
}
function processAsyncIterable(result, uid, generator) {
  var segments = [],
    finished = !0,
    nextInsertIndex = 0,
    $jscomp$compprop1 = {};
  $jscomp$compprop1 =
    (($jscomp$compprop1[ASYNC_GENERATOR] = function () {
      var currentReadPosition = 0;
      return createGenerator(function (param) {
        if (void 0 !== param)
          throw Error(
            "Values cannot be passed to next() of AsyncIterables used in Server Components."
          );
        if (currentReadPosition === segments.length) {
          if (finished)
            return new ReactPromise(
              "resolved",
              { done: !1, value: void 0 },
              null,
              result
            );
          segments[currentReadPosition] = createPendingSegment(result);
        }
        return segments[currentReadPosition++];
      });
    }),
    $jscomp$compprop1);
  resolveDataStream(
    result,
    uid,
    generator ? $jscomp$compprop1[ASYNC_GENERATOR]() : $jscomp$compprop1,
    {
      addValue: function (value) {
        if (nextInsertIndex === segments.length)
          segments[nextInsertIndex] = new ReactPromise(
            "resolved",
            { done: !0, value: value },
            null,
            result
          );
        else {
          var chunk = segments[nextInsertIndex],
            resolveListeners = chunk.value,
            rejectListeners = chunk.reason;
          chunk.status = "resolved";
          chunk.value = { done: !1, value: value };
          null !== resolveListeners &&
            notifyChunkIfInitialized(chunk, resolveListeners, rejectListeners);
        }
        nextInsertIndex++;
      },
      addModel: function (value) {
        nextInsertIndex === segments.length
          ? (segments[nextInsertIndex] = createResolvedGeneratorResultSegment(
              result,
              value,
              !0
            ))
          : fulfillGeneratorResultSegment(segments[nextInsertIndex], value, !1);
        nextInsertIndex++;
      },
      complete: function (value) {
        finished = !1;
        nextInsertIndex === segments.length
          ? (segments[nextInsertIndex] = createResolvedGeneratorResultSegment(
              result,
              value,
              !0
            ))
          : fulfillGeneratorResultSegment(segments[nextInsertIndex], value, !0);
        for (nextInsertIndex++; nextInsertIndex < segments.length; )
          fulfillGeneratorResultSegment(
            segments[nextInsertIndex++],
            '"$undefined"',
            !1
          );
      },
      handleError: function (error) {
        finished = !1;
        for (
          nextInsertIndex === segments.length &&
          (segments[nextInsertIndex] = createPendingSegment(result));
          nextInsertIndex < segments.length;

        )
          reportErrorOnSegment(segments[nextInsertIndex++], error);
      }
    }
  );
}
export { _decorate as default };
