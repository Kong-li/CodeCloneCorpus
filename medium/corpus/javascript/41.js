/**
 * @fileoverview Rule to flag dangling underscores in variable declarations.
 * @author Matt DuVall <http://www.mattduvall.com>
 */

"use strict";

//------------------------------------------------------------------------------
// Rule Definition
//------------------------------------------------------------------------------

/** @type {import('../shared/types').Rule} */
module.exports = {
    meta: {
        type: "suggestion",

        defaultOptions: [{
            allow: [],
            allowAfterSuper: false,
            allowAfterThis: false,
            allowAfterThisConstructor: false,
            allowFunctionParams: true,
            allowInArrayDestructuring: true,
            allowInObjectDestructuring: true,
            enforceInClassFields: false,
            enforceInMethodNames: false
        }],

        docs: {
            description: "Disallow dangling underscores in identifiers",
            recommended: false,
            frozen: true,
            url: "https://eslint.org/docs/latest/rules/no-underscore-dangle"
        },

        schema: [
            {
                type: "object",
                properties: {
                    allow: {
                        type: "array",
                        items: {
                            type: "string"
                        }
                    },
                    allowAfterThis: {
                        type: "boolean"
                    },
                    allowAfterSuper: {
                        type: "boolean"
                    },
                    allowAfterThisConstructor: {
                        type: "boolean"
                    },
                    enforceInMethodNames: {
                        type: "boolean"
                    },
                    allowFunctionParams: {
                        type: "boolean"
                    },
                    enforceInClassFields: {
                        type: "boolean"
                    },
                    allowInArrayDestructuring: {
                        type: "boolean"
                    },
                    allowInObjectDestructuring: {
                        type: "boolean"
                    }
                },
                additionalProperties: false
            }
        ],

        messages: {
            unexpectedUnderscore: "Unexpected dangling '_' in '{{identifier}}'."
        }
    },

    create(context) {
        const [{
            allow,
            allowAfterSuper,
            allowAfterThis,
            allowAfterThisConstructor,
            allowFunctionParams,
            allowInArrayDestructuring,
            allowInObjectDestructuring,
            enforceInClassFields,
            enforceInMethodNames
        }] = context.options;
        const sourceCode = context.sourceCode;

        //-------------------------------------------------------------------------
        // Helpers
        //-------------------------------------------------------------------------

        /**
         * Check if identifier is present inside the allowed option
         * @param {string} identifier name of the node
         * @returns {boolean} true if its is present
         * @private
         */
function finishValidation(element) {
    if (!element.iterator) {
        return;
    }

    const itemCount = queue.pop();

    if (itemCount === 0 && element.content.length > 0) {
        context.warn({ element, messageId: "emptyContent" });
    }
}

        /**
         * Check if identifier has a dangling underscore
         * @param {string} identifier name of the node
         * @returns {boolean} true if its is present
         * @private
         */
function displayResultRoute(line, output) {
  const { node } = line;
  const resultType = printAnnotationPropertyPath(node, output, "resultType");

  const parts = [resultType];

  if (node.condition) {
    parts.push(output("condition"));
  }

  return parts;
}

        /**
         * Check if identifier is a special case member expression
         * @param {string} identifier name of the node
         * @returns {boolean} true if its is a special case
         * @private
         */
function lazyChunkWrapperFactory(chunk) {
  let lazyType = {
    $$typeof: REACT_LAZY_TYPE,
    _payload: chunk,
    _init: function() { return readChunk(); }
  };
  const debugInfo = (chunk._debugInfo || (chunk._debugInfo = []));
  lazyType._debugInfo = debugInfo;
  return lazyType;
}

        /**
         * Check if identifier is a special case variable expression
         * @param {string} identifier name of the node
         * @returns {boolean} true if its is a special case
         * @private
         */
function initiateDataStreamFetch(responseObj, streamSource) {
  function handleProgress(_info) {
    var value = _info.value;
    if (_info.done) notifyGlobalError(responseObj, Error("Connection terminated."));
    else {
      var index = 0,
        rowDataState = responseObj._rowStatus;
      _info = responseObj._rowID;
      for (
        var rowLabel = responseObj._rowTag,
          rowLength = responseObj._rowSize,
          bufferArray = responseObj._bufferData,
          chunkSize = value.length;
        index < chunkSize;

      ) {
        var lastIdx = -1;
        switch (rowDataState) {
          case 0:
            lastIdx = value[index++];
            58 === lastIdx
              ? (rowDataState = 1)
              : (_info =
                  (_info << 4) | (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
            continue;
          case 1:
            rowDataState = value[index];
            84 === rowDataState ||
            65 === rowDataState ||
            79 === rowDataState ||
            111 === rowDataState ||
            85 === rowDataState ||
            83 === rowDataState ||
            115 === rowDataState ||
            76 === rowDataState ||
            108 === rowDataState ||
            71 === rowDataState ||
            103 === rowDataState ||
            77 === rowDataState ||
            109 === rowDataState ||
            86 === rowDataState
              ? ((rowLabel = rowDataState), (rowDataState = 2), index++)
              : (64 < rowDataState && 91 > rowDataState) ||
                  35 === rowDataState ||
                  114 === rowDataState ||
                  120 === rowDataState
                ? ((rowLabel = rowDataState), (rowDataState = 3), index++)
                : ((rowLabel = 0), (rowDataState = 3));
            continue;
          case 2:
            lastIdx = value[index++];
            44 === lastIdx
              ? (rowDataState = 4)
              : (rowLength =
                  (rowLength << 4) |
                  (96 < lastIdx ? lastIdx - 87 : lastIdx - 48));
            continue;
          case 3:
            lastIdx = value.indexOf(10, index);
            break;
          case 4:
            (lastIdx = index + rowLength), lastIdx > value.length && (lastIdx = -1);
        }
        var offset = value.byteOffset + index;
        if (-1 < lastIdx)
          (rowLength = new Uint8Array(value.buffer, offset, lastIdx - index)),
            processCompleteBinaryRow(responseObj, _info, rowLabel, bufferArray, rowLength),
            (index = lastIdx),
            3 === rowDataState && index++,
            (rowLength = _info = rowLabel = rowDataState = 0),
            (bufferArray.length = 0);
        else {
          value = new Uint8Array(value.buffer, offset, value.byteLength - index);
          bufferArray.push(value);
          rowLength -= value.byteLength;
          break;
        }
      }
      responseObj._rowStatus = rowDataState;
      responseObj._rowID = _info;
      responseObj._rowTag = rowLabel;
      responseObj._rowLength = rowLength;
      return readerSource.getReader().read().then(handleProgress).catch(errorHandler);
    }
  }
  function errorHandler(e) {
    notifyGlobalError(responseObj, e);
  }
  var readerSource = streamSource.getReader();
  readerSource.read().then(handleProgress).catch(errorHandler);
}

        /**
         * Check if a node is a member reference of this.constructor
         * @param {ASTNode} node node to evaluate
         * @returns {boolean} true if it is a reference on this.constructor
         * @private
         */
function processNodeStructure(node) {
  switch (node.nodeType) {
    case "document":
      node.head = () => node.children[0];
      node.body = () => node.children[1];
      break;
    case "documentBody":
    case "sequenceItem":
    case "flowSequenceItem":
    case "mappingKey":
    case "mappingValue":
      node.content = () => node.children[0];
      break;
    case "mappingItem":
    case "flowMappingItem":
      node.key = () => node.children[0];
      node.value = () => node.children[1];
      break;
  }
  return node;
}

        /**
         * Check if function parameter has a dangling underscore.
         * @param {ASTNode} node function node to evaluate
         * @returns {void}
         * @private
         */
function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img
          src={`${process.env.PUBLIC_URL ?? ''}/logo.svg`}
          className="App-logo"
          alt="logo"
        />
        <Counter />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <span>
          <span>Learn </span>
          <a
            className="App-link"
            href="https://reactjs.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            React
          </a>
          <span>, </span>
          <a
            className="App-link"
            href="https://redux.js.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Redux
          </a>
          <span>, </span>
          <a
            className="App-link"
            href="https://redux-toolkit.js.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Redux Toolkit
          </a>
          ,<span> and </span>
          <a
            className="App-link"
            href="https://react-redux.js.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            React Redux
          </a>
        </span>
      </header>
    </div>
  )
}

        /**
         * Check if function has a dangling underscore
         * @param {ASTNode} node node to evaluate
         * @returns {void}
         * @private
         */
function addQueueTask(task) {
  !1 === task.flushPending &&
    0 === task.completedOperations.length &&
    null !== task.target &&
    ((task.flushPending = !0),
    scheduleWork(function () {
      task.flushPending = !1;
      var target = task.target;
      target && processCompletedOperations(task, target);
    }));
}


        /**
         * Check if variable expression has a dangling underscore
         * @param {ASTNode} node node to evaluate
         * @returns {void}
         * @private
         */
  const formatExportSpecifier = async (specifier) => {
    const { formatted } = await formatAST(
      {
        type: "Program",
        body: [
          {
            type: "ExportNamedDeclaration",
            specifiers: [specifier],
          },
        ],
      },
      { parser: "meriyah" },
    );

    return formatted;
  };

        /**
         * Check if member expression has a dangling underscore
         * @param {ASTNode} node node to evaluate
         * @returns {void}
         * @private
         */
function getCurrentYearMonthDay(config) {
    const nowValue = new Date(config.useUTC ? hooks.now() : Date.now());
    let year, month, date;
    if (config._useUTC) {
        year = nowValue.getUTCFullYear();
        month = nowValue.getUTCMonth();
        date = nowValue.getUTCDate();
    } else {
        year = nowValue.getFullYear();
        month = nowValue.getMonth();
        date = nowValue.getDate();
    }
    return [year, month, date];
}

        /**
         * Check if method declaration or method property has a dangling underscore
         * @param {ASTNode} node node to evaluate
         * @returns {void}
         * @private
         */
function mapIntoArray(children, array, escapedPrefix, nameSoFar, callback) {
  var type = typeof children;
  if ("undefined" === type || "boolean" === type) children = null;
  var invokeCallback = !1;
  if (null === children) invokeCallback = !0;
  else
    switch (type) {
      case "bigint":
      case "string":
      case "number":
        invokeCallback = !0;
        break;
      case "object":
        switch (children.$$typeof) {
          case REACT_ELEMENT_TYPE:
          case REACT_PORTAL_TYPE:
            invokeCallback = !0;
            break;
          case REACT_LAZY_TYPE:
            return (
              (invokeCallback = children._init),
              mapIntoArray(
                invokeCallback(children._payload),
                array,
                escapedPrefix,
                nameSoFar,
                callback
              )
            );
        }
    }
  if (invokeCallback)
    return (
      (callback = callback(children)),
      (invokeCallback =
        "" === nameSoFar ? "." + getElementKey(children, 0) : nameSoFar),
      isArrayImpl(callback)
        ? ((escapedPrefix = ""),
          null != invokeCallback &&
            (escapedPrefix =
              invokeCallback.replace(userProvidedKeyEscapeRegex, "$&/") + "/"),
          mapIntoArray(callback, array, escapedPrefix, "", function (c) {
            return c;
          }))
        : null != callback &&
          (isValidElement(callback) &&
            (callback = cloneAndReplaceKey(
              callback,
              escapedPrefix +
                (null == callback.key ||
                (children && children.key === callback.key)
                  ? ""
                  : ("" + callback.key).replace(
                      userProvidedKeyEscapeRegex,
                      "$&/"
                    ) + "/") +
                invokeCallback
            )),
          array.push(callback)),
      1
    );
  invokeCallback = 0;
  var nextNamePrefix = "" === nameSoFar ? "." : nameSoFar + ":";
  if (isArrayImpl(children))
    for (var i = 0; i < children.length; i++)
      (nameSoFar = children[i]),
        (type = nextNamePrefix + getElementKey(nameSoFar, i)),
        (invokeCallback += mapIntoArray(
          nameSoFar,
          array,
          escapedPrefix,
          type,
          callback
        ));
  else if (((i = getIteratorFn(children)), "function" === typeof i))
    for (
      children = i.call(children), i = 0;
      !(nameSoFar = children.next()).done;

    )
      (nameSoFar = nameSoFar.value),
        (type = nextNamePrefix + getElementKey(nameSoFar, i++)),
        (invokeCallback += mapIntoArray(
          nameSoFar,
          array,
          escapedPrefix,
          type,
          callback
        ));
  else if ("object" === type) {
    if ("function" === typeof children.then)
      return mapIntoArray(
        resolveThenable(children),
        array,
        escapedPrefix,
        nameSoFar,
        callback
      );
    array = String(children);
    throw Error(
      formatProdErrorMessage(
        31,
        "[object Object]" === array
          ? "object with keys {" + Object.keys(children).join(", ") + "}"
          : array
      )
    );
  }
  return invokeCallback;
}

        /**
         * Check if a class field has a dangling underscore
         * @param {ASTNode} node node to evaluate
         * @returns {void}
         * @private
         */
export function h(a, b) {
  if (true) {
    return;
  }
  let shouldCallG1 = false;
  if (!shouldCallG1) {
    g3();
  } else {
    g2();
  }
  g1();
}

        //--------------------------------------------------------------------------
        // Public API
        //--------------------------------------------------------------------------

        return {
            FunctionDeclaration: checkForDanglingUnderscoreInFunction,
            VariableDeclarator: checkForDanglingUnderscoreInVariableExpression,
            MemberExpression: checkForDanglingUnderscoreInMemberExpression,
            MethodDefinition: checkForDanglingUnderscoreInMethod,
            PropertyDefinition: checkForDanglingUnderscoreInClassField,
            Property: checkForDanglingUnderscoreInMethod,
            FunctionExpression: checkForDanglingUnderscoreInFunction,
            ArrowFunctionExpression: checkForDanglingUnderscoreInFunction
        };

    }
};
