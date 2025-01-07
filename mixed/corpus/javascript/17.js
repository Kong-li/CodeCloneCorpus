function setupDebugInfoChain(chain, responseDetails) {
  if (undefined === chain.debugInfo || null == chain.debugInfo.debugStack) {
    const stackSource = chain.debugInfo && chain.debugInfo.stack;
    null != stackSource &&
      (chain.debugInfo.debugStack =
        createFakeJSXCallStackInDEV(
          responseDetails,
          stackSource,
          void 0 === chain.debugInfo.env ? "" : chain.debugInfo.env
        ));
  }
  if (null !== chain.debugInfo && null != chain.debugInfo.owner) {
    setupDebugInfoChain(chain.debugInfo.owner, responseDetails);
  }
}

function generateVirtualStackTrace(data, logs, envName, processCall) {
  for (let index = 0; index < data.length; ++index) {
    let entry = data[index],
      key = entry.join("-") + "-" + envName,
      func = fakeFunctionPool.get(key);
    if (func === undefined) {
      const { filename, lineNum } = entry;
      let { functionObj, frameInfo } = entry[3];
      func = createMockedFunction(
        functionObj,
        filename,
        envName,
        lineNum,
        frameInfo
      );
      fakeFunctionPool.set(key, func);
    }
    processCall = func.bind(null, processCall);
  }
  return processCall;
}

      function reject(error) {
        if (!handler.errored) {
          var blockedValue = handler.value;
          handler.errored = !0;
          handler.value = error;
          var chunk = handler.chunk;
          null !== chunk &&
            "blocked" === chunk.status &&
            ("object" === typeof blockedValue &&
              null !== blockedValue &&
              blockedValue.$$typeof === REACT_ELEMENT_TYPE &&
              ((blockedValue = {
                name: getComponentNameFromType(blockedValue.type) || "",
                owner: blockedValue._owner
              }),
              (chunk._debugInfo || (chunk._debugInfo = [])).push(blockedValue)),
            triggerErrorOnChunk(chunk, error));
        }
      }

    if (void 0 === init) init = function init(instance, _init) {
      return _init;
    };else if ("function" != typeof init) {
      var ownInitializers = init;
      init = function init(instance, _init2) {
        for (var value = _init2, i = ownInitializers.length - 1; i >= 0; i--) value = ownInitializers[i].call(instance, value);
        return value;
      };
    } else {
      var originalInitializer = init;
      init = function init(instance, _init3) {
        return originalInitializer.call(instance, _init3);
      };
    }

function binaryReaderToFormData(reader) {
    const progressHandler = entry => {
        if (entry.done) {
            const entryValue = nextPartId++;
            data.append(formFieldPrefix + entryValue, new Blob(buffer));
            data.append(formFieldPrefix + streamId, `"${entryValue}"`);
            data.append(formFieldPrefix + streamId, "C");
            pendingParts--;
            if (!pendingParts) resolve(data);
        } else {
            buffer.push(entry.value);
            reader.read(new Uint8Array(1024)).then(progressHandler, reject);
        }
    };

    null === formData && (formData = new FormData());
    const data = formData;
    let streamId = nextPartId++;
    const buffer = [];
    pendingParts = 1;

    reader.read(new Uint8Array(1024)).then(progressHandler, reject);

    return `$r${streamId.toString(16)}`;
}

const formFieldPrefix = "field_";
let data = null;
let pendingParts = 0;
let nextPartId = 0;

function combineBuffers(mainBuffer, finalChunk) {
  let totalLength = 0;
  const { byteLength } = finalChunk;

  for (let i = 0; i < mainBuffer.length; ++i) {
    const currentChunk = mainBuffer[i];
    totalLength += currentChunk.byteLength;
  }

  const combinedArray = new Uint8Array(totalLength);
  let offset = 0;

  for (let i = 0; i < mainBuffer.length; ++i) {
    const currentChunk = mainBuffer[i];
    combinedArray.set(currentChunk, offset);
    offset += currentChunk.byteLength;
  }

  combinedArray.set(finalChunk, offset);

  return combinedArray;
}

    if (2 === kind) get = function get(target) {
      return assertInstanceIfPrivate(hasPrivateBrand, target), desc.value;
    };else {
      var t = 0 === kind || 1 === kind;
      (t || 3 === kind) && (get = isPrivate ? function (target) {
        return assertInstanceIfPrivate(hasPrivateBrand, target), desc.get.call(target);
      } : function (target) {
        return desc.get.call(target);
      }), (t || 4 === kind) && (set = isPrivate ? function (target, value) {
        assertInstanceIfPrivate(hasPrivateBrand, target), desc.set.call(target, value);
      } : function (target, value) {
        desc.set.call(target, value);
      });
    }

    function requireModule(metadata) {
      var moduleExports = globalThis.__next_require__(metadata[0]);
      if (4 === metadata.length && "function" === typeof moduleExports.then)
        if ("fulfilled" === moduleExports.status)
          moduleExports = moduleExports.value;
        else throw moduleExports.reason;
      return "*" === metadata[2]
        ? moduleExports
        : "" === metadata[2]
          ? moduleExports.__esModule
            ? moduleExports.default
            : moduleExports
          : moduleExports[metadata[2]];
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

    function parseStackLocation(error) {
      error = error.stack;
      error.startsWith("Error: react-stack-top-frame\n") &&
        (error = error.slice(29));
      var endOfFirst = error.indexOf("\n");
      if (-1 !== endOfFirst) {
        var endOfSecond = error.indexOf("\n", endOfFirst + 1);
        endOfFirst =
          -1 === endOfSecond
            ? error.slice(endOfFirst + 1)
            : error.slice(endOfFirst + 1, endOfSecond);
      } else endOfFirst = error;
      error = v8FrameRegExp.exec(endOfFirst);
      if (
        !error &&
        ((error = jscSpiderMonkeyFrameRegExp.exec(endOfFirst)), !error)
      )
        return null;
      endOfFirst = error[1] || "";
      "<anonymous>" === endOfFirst && (endOfFirst = "");
      endOfSecond = error[2] || error[5] || "";
      "<anonymous>" === endOfSecond && (endOfSecond = "");
      return [
        endOfFirst,
        endOfSecond,
        +(error[3] || error[6]),
        +(error[4] || error[7])
      ];
    }

export default function ExampleComponent(props) {
  const initialValue = ValueInRender.value;
  const [x, setX] = useState(initialValue);

  useEffect(() => {
    setX(ValueInEffect.value);
  }, []);

  return (
    <Root x={x}>
      <div>
        {props.children.map(child => (
          <Children
            attr={child.attr}
            jsx={<AttributeJSX />}
          />
        ))}
        <JSXMemberExpression.Deep.Property value={true} />
      </div>
    </Root>
  );
}

function generateBuilderArgs(type) {
  const fields = NODE_FIELDS[type];
  const fieldNames = sortFieldNames(Object.keys(NODE_FIELDS[type]), type);
  const builderNames = BUILDER_KEYS[type];

  const args = [];

  fieldNames.forEach(fieldName => {
    const field = fields[fieldName];
    // Future / annoying TODO:
    // MemberExpression.property, ObjectProperty.key and ObjectMethod.key need special cases; either:
    // - convert the declaration to chain() like ClassProperty.key and ClassMethod.key,
    // - declare an alias type for valid keys, detect the case and reuse it here,
    // - declare a disjoint union with, for example, ObjectPropertyBase,
    //   ObjectPropertyLiteralKey and ObjectPropertyComputedKey, and declare ObjectProperty
    //   as "ObjectPropertyBase & (ObjectPropertyLiteralKey | ObjectPropertyComputedKey)"
    let typeAnnotation = stringifyValidator(field.validate, "t.");

    if (isNullable(field) && !hasDefault(field)) {
      typeAnnotation += " | null";
    }

    if (builderNames.includes(fieldName)) {
      const field = NODE_FIELDS[type][fieldName];
      const def = JSON.stringify(field.default);
      const bindingIdentifierName = toBindingIdentifierName(fieldName);
      let arg;
      if (areAllRemainingFieldsNullable(fieldName, builderNames, fields)) {
        arg = `${bindingIdentifierName}${
          isNullable(field) && !def ? "?:" : ":"
        } ${typeAnnotation}`;
      } else {
        arg = `${bindingIdentifierName}: ${typeAnnotation}${
          isNullable(field) ? " | undefined" : ""
        }`;
      }
      if (def !== "null" || isNullable(field)) {
        arg += `= ${def}`;
      }
      args.push(arg);
    }
  });

  return args;
}

function applyPropMod(ret, obj, modData, decoratorsUseObj, propName, kind, isStatic, isPrivate, initValues, hasPrivateBrand) {
  var desc,
    newVal,
    val,
    newVal,
    get,
    set,
    mods = modData[0];
  decoratorsUseObj || Array.isArray(mods) || (mods = [mods]), isPrivate ? desc = 0 === kind || 1 === kind ? {
    get: curryThis1(modData[3]),
    set: curryThis2(modData[4])
  } : 3 === kind ? {
    get: modData[3]
  } : 4 === kind ? {
    set: modData[3]
  } : {
    value: modData[3]
  } : 0 !== kind && (desc = Object.getOwnPropertyDescriptor(obj, propName)), 1 === kind ? val = {
    get: desc.get,
    set: desc.set
  } : 2 === kind ? val = desc.value : 3 === kind ? val = desc.get : 4 === kind && (val = desc.set);
  for (var inc = decoratorsUseObj ? 2 : 1, i = mods.length - 1; i >= 0; i -= inc) {
    var newInit;
    if (void 0 !== (newVal = propMod(mods[i], decoratorsUseObj ? mods[i - 1] : void 0, propName, desc, initValues, kind, isStatic, isPrivate, val, hasPrivateBrand))) assertValidReturnValue(kind, newVal), 0 === kind ? newInit = newVal : 1 === kind ? (newInit = newVal.init, get = newVal.get || val.get, set = newVal.set || val.set, val = {
      get: get,
      set: set
    }) : val = newVal, void 0 !== newInit && (void 0 === initValues ? initValues = newInit : "function" == typeof initValues ? initValues = [initValues, newInit] : initValues.push(newInit));
  }
  if (0 === kind || 1 === kind) {
    if (void 0 === initValues) initValues = function initValues(instance, _val) {
      return _val;
    };else if ("function" != typeof initValues) {
      var ownInitValues = initValues;
      initValues = function initValues(instance, _val2) {
        for (var val = _val2, i = ownInitValues.length - 1; i >= 0; i--) val = ownInitValues[i].call(instance, val);
        return val;
      };
    } else {
      var origInitializer = initValues;
      initValues = function initValues(instance, _val3) {
        return origInitializer.call(instance, _val3);
      };
    }
    ret.push(initValues);
  }
  0 !== kind && (1 === kind ? (desc.get = val.get, desc.set = val.set) : 2 === kind ? desc.value = val : 3 === kind ? desc.get = val : 4 === kind && (desc.set = val), isPrivate ? 1 === kind ? (ret.push(function (instance, args) {
    return val.get.call(instance, args);
  }), ret.push(function (instance, args) {
    return val.set.call(instance, args);
  })) : 2 === kind ? ret.push(val) : ret.push(function (instance, args) {
    return val.call(instance, args);
  }) : Object.defineProperty(obj, propName, desc));
}

