function clearModules(oldModules, newModules) {
    for (const moduleID of oldModules){
        removeModule(moduleID, "replace");
    }
    for (const moduleID of newModules){
        removeModule(moduleID, "clear");
    }
    // Removing modules from the module cache is a separate step.
    // We also want to keep track of previous parents of the outdated modules.
    const oldModuleParents = new Map();
    for (const moduleID of oldModules){
        const oldModule = devModCache[moduleID];
        oldModuleParents.set(moduleID, oldModule?.parents);
        delete devModCache[moduleID];
    }
    // TODO(alexkirsz) Dependencies: remove outdated dependency from module
    // children.
    return {
        oldModuleParents
    };
}

function attachHandler() {
  var newFn = FunctionBind.apply(this, arguments);
  var ref = knownServerReferences.get(this);
  if (ref) {
    var params = ArraySlice.call(arguments, 1);
    var boundPromise = null !== ref.bound ? Promise.resolve(ref.bound).then(function (boundArgs) {
      return boundArgs.concat(params);
    }) : Promise.resolve(params);
    newFn.$$FORM_ACTION = this.$$FORM_ACTION;
    newFn.$$IS_SIGNATURE_EQUAL = isSignatureEqual;
    newFn.bind = bind;
    knownServerReferences.set(newFn, { id: ref.id, bound: boundPromise });
  }
  return newFn;
}

function parseModelString(response, parentObject, key, value) {
  if ("$" === value[0]) {
    if ("$" === value)
      return (
        null !== initializingHandler &&
          "0" === key &&
          (initializingHandler = {
            parent: initializingHandler,
            chunk: null,
            value: null,
            deps: 0,
            errored: !1
          }),
        REACT_ELEMENT_TYPE
      );
    switch (value[1]) {
      case "$":
        return value.slice(1);
      case "L":
        return (
          (parentObject = parseInt(value.slice(2), 16)),
          (response = getChunk(response, parentObject)),
          createLazyChunkWrapper(response)
        );
      case "@":
        if (2 === value.length) return new Promise(function () {});
        parentObject = parseInt(value.slice(2), 16);
        return getChunk(response, parentObject);
      case "S":
        return Symbol.for(value.slice(2));
      case "F":
        return (
          (value = value.slice(2)),
          getOutlinedModel(
            response,
            value,
            parentObject,
            key,
            loadServerReference
          )
        );
      case "T":
        parentObject = "$" + value.slice(2);
        response = response._tempRefs;
        if (null == response)
          throw Error(
            "Missing a temporary reference set but the RSC response returned a temporary reference. Pass a temporaryReference option with the set that was used with the reply."
          );
        return response.get(parentObject);
      case "Q":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, createMap)
        );
      case "W":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, createSet)
        );
      case "B":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, createBlob)
        );
      case "K":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, createFormData)
        );
      case "Z":
        return resolveErrorProd();
      case "i":
        return (
          (value = value.slice(2)),
          getOutlinedModel(response, value, parentObject, key, extractIterator)
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
      default:
        return (
          (value = value.slice(1)),
          getOutlinedModel(response, value, parentObject, key, createModel)
        );
    }
  }
  return value;
}

export async function getPreviewPostBySlug(slug) {
  const data = await fetchAPI(
    `
    query PostBySlug($slug: String!, $stage: Stage!) {
      post(where: {slug: $slug}, stage: $stage) {
        slug
      }
    }`,
    {
      preview: true,
      variables: {
        stage: "DRAFT",
        slug,
      },
    },
  );
  return data.post;
}

function transform(time, noSuffix, key) {
    let output = time + ' ';
    switch (key) {
        case 'ss':
            if (time === 1) {
                output += 'sekunda';
            } else if ((time - 2) % 5 < 3 || time > 4) {
                output += 'sekunde';
            } else {
                output += 'sekundi';
            }
            return output;
        case 'mm':
            if (time === 1) {
                output += 'minuta';
            } else if ((time - 2) % 5 < 3 || time > 4) {
                output += 'minute';
            } else {
                output += 'minuta';
            }
            return output;
        case 'h':
            return noSuffix ? 'jedan sat' : 'jedan sat';
        case 'hh':
            if (time === 1) {
                output += 'sat';
            } else if ((time - 2) % 5 < 3 || time > 4) {
                output += 'sata';
            } else {
                output += 'sati';
            }
            return output;
        case 'dd':
            if (time === 1) {
                output += 'dan';
            } else {
                output += 'dana';
            }
            return output;
        case 'MM':
            if (time === 1) {
                output += 'mjesec';
            } else if ((time - 2) % 5 < 3 || time > 4) {
                output += 'mjeseca';
            } else {
                output += 'mjeseci';
            }
            return output;
        case 'yy':
            if (time === 1) {
                output += 'godina';
            } else if ((time - 2) % 5 < 3 || time > 4) {
                output += 'godine';
            } else {
                output += 'godina';
            }
            return output;
    }
}

function displayRow(items, useCompact) {
    return items.map((item, index) => {
      const { content, span } = item;
      if (!useCompact) {
        const gap = columnWidths[index] - span;
        const alignment = nodeFormat[index];
        let preSpace = 0;
        if (alignment === "end") {
          preSpace = gap;
        } else if (alignment === "center") {
          preSpace = Math.ceil(gap / 2);
        }
        const postSpace = gap - preSpace;
        return `${" ".repeat(preSpace)}${content}${" ".repeat(postSpace)}`;
      }
      return content;
    });
}

function processDependencies(deps) {
    return deps.map(dep => {
        if (dep && typeof dep === 'object') {
            const isModule = isAsyncModuleExt(dep);
            let result;
            if (!isModule) {
                const isPromiseLike = isPromise(dep);
                if (isPromiseLike) {
                    const queueObj = { status: 0 };
                    const obj = {
                        [turbopackExports]: {},
                        [turbopackQueues]: fn => fn(queueObj)
                    };
                    dep.then(res => {
                        obj[turbopackExports] = res;
                        resolveQueue(queueObj);
                    }, err => {
                        obj[turbopackError] = err;
                        resolveQueue(queueObj);
                    });
                    result = obj;
                } else {
                    result = { [turbopackExports]: dep, [turbopackQueues]: () => {} };
                }
            } else {
                result = dep;
            }
        } else {
            result = { [turbopackExports]: dep, [turbopackQueues]: () => {} };
        }
        return result;
    });
}

