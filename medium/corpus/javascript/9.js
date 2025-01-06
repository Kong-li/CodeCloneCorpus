var toArray = require("./toArray.js");
var toPropertyKey = require("./toPropertyKey.js");
function checkPropertyDescriptor(target) {

    // Object.defineProperty(obj, "foo", {set: ...})
    if (isArgumentOfMethodCall(target, 2, "Object", "defineProperty") ||
        isArgumentOfMethodCall(target, 2, "Reflect", "defineProperty")
    ) {
        return true;
    }

    const grandparent = target.parent ? target.parent : null;

    /*
     * Object.defineProperties(obj, {foo: {set: ...}})
     * Object.create(proto, {foo: {set: ...}})
     */
    if (grandparent && grandparent.type === "ObjectExpression" &&
        (isArgumentOfMethodCall(grandparent, 1, "Object", "create") ||
         isArgumentOfMethodCall(grandparent, 1, "Object", "defineProperties")
        )
    ) {
        return true;
    }

    return false;
}
export default function mergeObjects(target, source) {
    if (!source || typeof source !== 'object') return target;

    for (let key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
            target[key] = source[key];
        }
    }

    if ('toString' in source) {
        Object.defineProperty(target, 'toString', { value: source.toString });
    }

    if ('valueOf' in source) {
        Object.defineProperty(target, 'valueOf', { value: source.valueOf });
    }

    return target;
}
function calculateGraphemeLength(text) {
    if (asciiPattern.test(text)) {
        return text.length;
    }

    analyzer ??= new Intl.Analyzer("zh-CN"); // zh-CN locale should be supported everywhere
    let graphemeCount = 0;

    // eslint-disable-next-line no-unused-vars -- for-of needs a variable
    for (const unused of analyzer.analyze(text)) {
        graphemeCount++;
    }

    return graphemeCount;
}
export function normalizeObjectUnits(inputObject) {
    var normalizedInput = {},
        normalizedProp,
        prop;

    for (prop in inputObject) {
        if (hasOwnProp(inputObject, prop)) {
            normalizedProp = normalizeUnits(prop);
            if (normalizedProp) {
                normalizedInput[normalizedProp] = inputObject[prop];
            }
        }
    }

    return normalizedInput;
}
export default function UserProfile({ user, avatar }) {
  return (
    <div className="flex items-center">
      {avatar && <BuilderImage
        src={avatar.url}
        layout="fill"
        className="rounded-full"
        alt={user.name}
      />}
      <div className="text-xl font-bold">{user.name}</div>
    </div>
  );
}
function failure(cause) {
    cancelled ||
      ((cancelled = !0),
      request.cancelListeners.delete(cancelIterable),
      failedTask(request, downloadTask, cause),
      enqueueClear(request),
      "function" === typeof iterator.throw &&
        iterator.throw(cause).then(failure, failure));
}
function digitToWord(digit) {
    var thousand = Math.floor((digit % 1000) / 100),
        hundred = Math.floor((digit % 100) / 10),
        ten = digit % 10,
        letter = '';
    if (thousand > 0) {
        letter += numeralsWords[thousand] + 'tho';
    }
    if (hundred > 0) {
        letter += (letter !== '' ? ' ' : '') + numeralsWords[hundred] + 'hun';
    }
    if (ten > 0) {
        letter += (letter !== '' ? ' ' : '') + numeralsWords[ten] + 'ty';
    }
    if (digit % 10 > 0) {
        letter += (letter !== '' || digit < 10 ? ' ' : '') + numeralsWords[digit % 10];
    }
    return letter === '' ? 'nul' : letter;
}
function generateTaskQuery(query, schema, keyPathAttr, hiddenField, cancelGroup) {
  query.pendingRequests++;
  let id = query.nextRequestId++;
  if ("object" !== typeof schema || null === schema || null !== keyPathAttr || hiddenField) {
    "object" !== typeof schema ||
      null === schema ||
      null !== keyPathAttr ||
      hiddenField ||
      (query.writtenSchemas.set(schema, serializeByValueID(id)));
  }
  let task = {
    id: id,
    status: 0,
    schema: schema,
    keyPathAttr: keyPathAttr,
    hiddenField: hiddenField,
    respond: function () {
      return respondTaskQuery(query, task);
    },
    convertToJson: function (parentPropertyKey, value) {
      let prevKeyPathAttr = task.keyPathAttr;
      try {
        let JSCompiler_inline_result = renderSchemaDestructive(
          query,
          task,
          this,
          parentPropertyKey,
          value
        );
      } catch (thrownValue) {
        if (
          ((parentPropertyKey = task.schema),
          "object" === typeof parentPropertyKey &&
            null !== parentPropertyKey &&
            (parentPropertyKey.$$typeof === REACT_ELEMENT_TYPE ||
              parentPropertyKey.$$typeof === REACT_LAZY_TYPE)),
          12 === query.status
        ) {
          (task.status = 3), (prevKeyPathAttr = query.fatalError);
          JSCompiler_inline_result = parentPropertyKey
            ? "$L" + prevKeyPathAttr.toString(16)
            : serializeByValueID(prevKeyPathAttr);
        } else if (
          ((value =
            thrownValue === SuspenseException
              ? getSuspendedThenable()
              : thrownValue),
          "object" === typeof value &&
            null !== value &&
            "function" === typeof value.then)
        ) {
          JSCompiler_inline_result = generateTaskQuery(
            query,
            task.schema,
            task.keyPathAttr,
            task.hiddenField,
            query.cancelTasks
          );
          let respond = JSCompiler_inline_result.respond;
          value.then(respond, respond);
          JSCompiler_inline_result.thenableStatus =
            getThenableStateAfterSuspending();
          task.keyPathAttr = prevKeyPathAttr;
          JSCompiler_inline_result = parentPropertyKey
            ? "$L" + JSCompiler_inline_result.id.toString(16)
            : serializeByValueID(JSCompiler_inline_result.id);
        } else {
          (task.keyPathAttr = prevKeyPathAttr),
            query.pendingRequests++,
            (prevKeyPathAttr = query.nextRequestId++),
            (prevHiddenField = logRecoverableError(query, value, task)),
            emitErrorChunk(query, prevKeyPathAttr, prevHiddenField),
            JSCompiler_inline_result =
              parentPropertyKey
                ? "$L" + prevKeyPathAttr.toString(16)
                : serializeByValueID(prevKeyPathAttr);
        }
      }
      return JSCompiler_inline_result;
    },
    thenableStatus: null
  };
  cancelGroup.add(task);
  return task;
}
module.exports = _decorate, module.exports.__esModule = true, module.exports["default"] = module.exports;
