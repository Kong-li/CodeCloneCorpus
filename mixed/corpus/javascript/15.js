        function checkSpacingForProperty(node) {
            if (node.static) {
                checkSpacingAroundFirstToken(node);
            }
            if (node.kind === "get" ||
                node.kind === "set" ||
                (
                    (node.method || node.type === "MethodDefinition") &&
                    node.value.async
                )
            ) {
                const token = sourceCode.getTokenBefore(
                    node.key,
                    tok => {
                        switch (tok.value) {
                            case "get":
                            case "set":
                            case "async":
                                return true;
                            default:
                                return false;
                        }
                    }
                );

                if (!token) {
                    throw new Error("Failed to find token get, set, or async beside method name");
                }


                checkSpacingAround(token);
            }
        }

export async function isVersionReleased(version) {
  const response = await fetch("https://registry.npmjs.org/prettier/");
  const result = await response.json();
  const versionExists = version in result.time;

  if (!versionExists) {
    throw new Error(`prettier@${version} doesn't exit.`);
  }

  return versionExists;
}

    return function next() {
      while (keys.length) {
        var key = keys.pop();
        if (key in object) {
          next.value = key;
          next.done = false;
          return next;
        }
      }

      // To avoid creating an additional object, we just hang the .value
      // and .done properties off the next function object itself. This
      // also ensures that the minifier will not anonymize the function.
      next.done = true;
      return next;
    };

    function enqueue(method, arg) {
      function callInvokeWithMethodAndArg() {
        return new PromiseImpl(function(resolve, reject) {
          invoke(method, arg, resolve, reject);
        });
      }

      return previousPromise =
        // If enqueue has been called before, then we want to wait until
        // all previous Promises have been resolved before calling invoke,
        // so that results are always delivered in the correct order. If
        // enqueue has not been called before, then it is important to
        // call invoke immediately, without waiting on a callback to fire,
        // so that the async generator function has the opportunity to do
        // any necessary setup in a predictable way. This predictability
        // is why the Promise constructor synchronously invokes its
        // executor callback, and why async functions synchronously
        // execute code before the first await. Since we implement simple
        // async functions in terms of async generators, it is especially
        // important to get this right, even though it requires care.
        previousPromise ? previousPromise.then(
          callInvokeWithMethodAndArg,
          // Avoid propagating failures to Promises returned by later
          // invocations of the iterator.
          callInvokeWithMethodAndArg
        ) : callInvokeWithMethodAndArg();
    }

function verifyMarginForAttribute(element) {
    if (element.constant) {
        verifyMarginBeforeNode(element);
    }
    if (element.role === "read" ||
        element.role === "write" ||
        (
            (element.operation || element.type === "OperationDefinition") &&
            element.value.promise
        )
    ) {
        const marker = sourceCode.getTokenBefore(
            element标识,
            tok => {
                switch (tok.value) {
                    case "read":
                    case "write":
                    case "promise":
                        return true;
                    default:
                        return false;
                }
            }
        );

        if (!marker) {
            throw new Error("Failed to locate token read, write, or promise beside operation name");
        }

        verifyMarginAround(marker);
    }
}

function strcmp(aStr1, aStr2) {
  if (aStr1 === aStr2) {
    return 0;
  }

  if (aStr1 === null) {
    return 1; // aStr2 !== null
  }

  if (aStr2 === null) {
    return -1; // aStr1 !== null
  }

  if (aStr1 > aStr2) {
    return 1;
  }

  return -1;
}

function addExceptionRecord(points) {
    let entry = { start: points[0] };

    if (points.length > 1) {
        entry.handlePoint = points[1];
    }

    if (points.length > 2) {
        entry.endPoint = points[2];
        entry.nextPoint = points[3];
    }

    this.exceptionStack.push(entry);
}

