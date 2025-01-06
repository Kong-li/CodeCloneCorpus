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
function preloadModule(href, config) {
  if (typeof href === 'string') {
    const request = resolveRequest();
    if (request) {
      let hints = request.hints;
      const key = "m|" + href;
      if (!hints.has(key)) {
        hints.add(key);
        config = trimOptions(config);
        emitHint(request, "m", config ? [href, config] : href);
      }
    } else {
      previousDispatcher.m(href, config);
    }
  }
}
function isReferencedFromOutside(scopeNode) {

    /**
     * Determines if a given variable reference is outside of the specified scope.
     * @param {eslint-scope.Reference} ref A reference to evaluate.
     * @returns {boolean} True if the reference is outside the specified scope.
     */
    function checkRefOutOfScope(ref) {
        const rangeOfScope = scopeNode.range;
        const idRange = ref.identifier.range;

        return idRange[0] < rangeOfScope[0] || idRange[1] > rangeOfScope[1];
    }

    return function(varName) {
        return varName.references.some(checkRefOutOfScope);
    };
}
async function externalImport(id) {
    let raw;
    try {
        raw = await import(id);
    } catch (err) {
        // TODO(alexkirsz) This can happen when a client-side module tries to load
        // an external module we don't provide a shim for (e.g. querystring, url).
        // For now, we fail semi-silently, but in the future this should be a
        // compilation error.
        throw new Error(`Failed to load external module ${id}: ${err}`);
    }
    if (raw && raw.__esModule && raw.default && 'default' in raw.default) {
        return interopEsm(raw.default, createNS(raw), true);
    }
    return raw;
}
function parseModelString(response, obj, key, value, reference) {
  if ("$" === value[0]) {
    switch (value[1]) {
      case "$":
        return value.slice(1);
      case "@":
        return (obj = parseInt(value.slice(2), 16)), getChunk(response, obj);
      case "F":
        return (
          (value = value.slice(2)),
          (value = getOutlinedModel(response, value, obj, key, createModel)),
          loadServerReference$1(
            response,
            value.id,
            value.bound,
            initializingChunk,
            obj,
            key
          )
        );
      case "T":
        if (void 0 === reference || void 0 === response._temporaryReferences)
          throw Error(
            "Could not reference an opaque temporary reference. This is likely due to misconfiguring the temporaryReferences options on the server."
          );
        return createTemporaryReference(
          response._temporaryReferences,
          reference
        );
      case "Q":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, obj, key, createMap)
        );
      case "W":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, obj, key, createSet)
        );
      case "K":
        obj = value.slice(2);
        var formPrefix = response._prefix + obj + "_",
          data = new FormData();
        response._formData.forEach(function (entry, entryKey) {
          entryKey.startsWith(formPrefix) &&
            data.append(entryKey.slice(formPrefix.length), entry);
        });
        return data;
      case "i":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, obj, key, extractIterator)
        );
      case "I":
        return Infinity;
      case "-":
        return "$-0" === value ? -0 : -Infinity;
      case "N":
        return NaN;
      case "u":
        return;
      case "D":
        return new Date(Date.parse(value.slice(2)));
      case "n":
        return BigInt(value.slice(2));
    }
    switch (value[1]) {
      case "A":
        return parseTypedArray(response, value, ArrayBuffer, 1, obj, key);
      case "O":
        return parseTypedArray(response, value, Int8Array, 1, obj, key);
      case "o":
        return parseTypedArray(response, value, Uint8Array, 1, obj, key);
      case "U":
        return parseTypedArray(response, value, Uint8ClampedArray, 1, obj, key);
      case "S":
        return parseTypedArray(response, value, Int16Array, 2, obj, key);
      case "s":
        return parseTypedArray(response, value, Uint16Array, 2, obj, key);
      case "L":
        return parseTypedArray(response, value, Int32Array, 4, obj, key);
      case "l":
        return parseTypedArray(response, value, Uint32Array, 4, obj, key);
      case "G":
        return parseTypedArray(response, value, Float32Array, 4, obj, key);
      case "g":
        return parseTypedArray(response, value, Float64Array, 8, obj, key);
      case "M":
        return parseTypedArray(response, value, BigInt64Array, 8, obj, key);
      case "m":
        return parseTypedArray(response, value, BigUint64Array, 8, obj, key);
      case "V":
        return parseTypedArray(response, value, DataView, 1, obj, key);
      case "B":
        return (
          (obj = parseInt(value.slice(2), 16)),
          response._formData.get(response._prefix + obj)
        );
    }
    switch (value[1]) {
      case "R":
        return parseReadableStream(response, value, void 0);
      case "r":
        return parseReadableStream(response, value, "bytes");
      case "X":
        return parseAsyncIterable(response, value, !1);
      case "x":
        return parseAsyncIterable(response, value, !0);
    }
    value = value.slice(1);
    return getOutlinedModel(response, value, obj, key, createModel);
  }
  return value;
}
function handleTryCatch(func, obj, arg) {
  let result;
  try {
    result = func.call(obj, arg);
  } catch (error) {
    result = error;
  }
  return {
    type: !result ? "throw" : "normal",
    arg: result || error
  };
}
  async function deleteItem2() {
    'use server'
    await deleteFromDb(
      product.id,
      product?.foo,
      product.bar.baz,
      product[(foo, bar)]
    )
  }
function applyPropertyDec(ret, obj, decInfo, propName, dataType, isStatic, isPrivate, inits) {
    var desc,
        initVal,
        val,
        newVal,
        getter,
        setter,
        decs = decInfo[0];
    if (isPrivate ? desc = 2 === dataType || 3 === dataType ? {
            get: decInfo[3],
            set: decInfo[4]
        } : 5 === dataType ? {
            get: decInfo[3]
        } : 6 === dataType ? {
            set: decInfo[3]
        } : {
            value: decInfo[3]
        } : 0 !== dataType && (desc = Object.getOwnPropertyDescriptor(obj, propName)), 1 === dataType ? val = {
            get: desc.get,
            set: desc.set
        } : 2 === dataType ? val = desc.value : 3 === dataType ? val = desc.get : 4 === dataType && (val = desc.set), "function" == typeof decs) void 0 !== (newVal = propertyDec(decs, propName, desc, inits, dataType, isStatic, isPrivate, val)) && (assertValidReturnValue(dataType, newVal), 0 === dataType ? initVal = newVal : 1 === dataType ? (initVal = newVal.init, getter = newVal.get || val.get, setter = newVal.set || val.set, val = {
            get: getter,
            set: setter
        }) : val = newVal);else for (var i = decs.length - 1; i >= 0; i--) {
        var newInitVal;
        if (void 0 !== (newVal = propertyDec(decs[i], propName, desc, inits, dataType, isStatic, isPrivate, val))) assertValidReturnValue(dataType, newVal), 0 === dataType ? newInitVal = newVal : 1 === dataType ? (newInitVal = newVal.init, getter = newVal.get || val.get, setter = newVal.set || val.set, val = {
            get: getter,
            set: setter
        }) : val = newVal, void 0 !== newInitVal && (void 0 === initVal ? initVal = newInitVal : "function" == typeof initVal ? initVal = [initVal, newInitVal] : initVal.push(newInitVal));
    }
    if (0 === dataType || 1 === dataType) {
        if (void 0 === initVal) initVal = function initVal(instance, _initVal) {
            return _initVal;
        };else if ("function" != typeof initVal) {
            var ownInits = initVal;
            initVal = function initVal(instance, _initVal2) {
                for (var val = _initVal2, i = 0; i < ownInits.length; i++) val = ownInits[i].call(instance, val);
                return val;
            };
        } else {
            var originalInit = initVal;
            initVal = function initVal(instance, _initVal3) {
                return originalInit.call(instance, _initVal3);
            };
        }
        ret.push(initVal);
    }
    0 !== dataType && (1 === dataType ? (desc.get = val.get, desc.set = val.set) : 2 === dataType ? desc.value = val : 3 === dataType ? desc.get = val : 4 === dataType && (desc.set = val), isPrivate ? 1 === dataType ? (ret.push(function (instance, args) {
        return val.get.call(instance, args);
    }), ret.push(function (instance, args) {
        return val.set.call(instance, args);
    })) : 2 === dataType ? ret.push(val) : ret.push(function (instance, args) {
        return val.call(instance, args);
    }) : Object.defineProperty(obj, propName, desc));
}
function loadServerReference(response, metaData, parentObject, key) {
  if (!response._serverReferenceConfig)
    return createBoundServerReference(
      metaData,
      response._callServer,
      response._encodeFormAction
    );
  var serverReference = resolveServerReference(
    response._serverReferenceConfig,
    metaData.id
  );
  if ((response = preloadModule(serverReference)))
    metaData.bound && (response = Promise.all([response, metaData.bound]));
  else if (metaData.bound) response = Promise.resolve(metaData.bound);
  else return requireModule(serverReference);
  if (initializingHandler) {
    var handler = initializingHandler;
    handler.deps++;
  } else
    handler = initializingHandler = {
      parent: null,
      chunk: null,
      value: null,
      deps: 1,
      errored: !1
    };
  response.then(
    function () {
      var resolvedValue = requireModule(serverReference);
      if (metaData.bound) {
        var boundArgs = metaData.bound.value.slice(0);
        boundArgs.unshift(null);
        resolvedValue = resolvedValue.bind.apply(resolvedValue, boundArgs);
      }
      parentObject[key] = resolvedValue;
      "" === key && null === handler.value && (handler.value = resolvedValue);
      if (
        parentObject[0] === REACT_ELEMENT_TYPE &&
        "object" === typeof handler.value &&
        null !== handler.value &&
        handler.value.$$typeof === REACT_ELEMENT_TYPE
      )
        switch (((boundArgs = handler.value), key)) {
          case "3":
            boundArgs.props = resolvedValue;
        }
      handler.deps--;
      0 === handler.deps &&
        ((resolvedValue = handler.chunk),
        null !== resolvedValue &&
          "blocked" === resolvedValue.status &&
          ((boundArgs = resolvedValue.value),
          (resolvedValue.status = "fulfilled"),
          (resolvedValue.value = handler.value),
          null !== boundArgs && wakeChunk(boundArgs, handler.value)));
    },
    function (error) {
      if (!handler.errored) {
        handler.errored = !0;
        handler.value = error;
        var chunk = handler.chunk;
        null !== chunk &&
          "blocked" === chunk.status &&
          triggerErrorOnChunk(chunk, error);
      }
    }
  );
  return null;
}
export { _decorate as default };
