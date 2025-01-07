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

initializeElementsForInstances: function initializeElementsForInstances(data, context) {
  var propertyKinds = ["method", "field"];
  _forEachInstanceProperty(context).call(context, function (property) {
    if ("own" === property.placement && propertyKindIsDefined(property)) {
      this.defineElementForClass(data, property);
    }
  }, this);

  for (var kind of propertyKinds) {
    var properties = context[kind];
    if (properties) {
      _forEachInstanceProperty(properties).call(properties, function (property) {
        if ("own" === property.placement && property.kind === kind) {
          this.defineElementForClass(data, property);
        }
      }, this);
    }
  }

  function propertyKindIsDefined(property) {
    return "method" === property.kind || "field" === property.kind;
  }
}

const generateFullPageContent = (content, initialState) => {
  const htmlContent = `<div id="app">${content}</div>`;
  const preloadedStateScript = `window.__PRELOADED_STATE__ = ${JSON.stringify(initialState).replace(/</g, '\\x3c')}`;
  return `
    <!doctype html>
    <html>
      <head>
        <title>Redux Universal Example</title>
      </head>
      <body>
        ${htmlContent}
        <script>${preloadedStateScript}</script>
        <script src="/static/bundle.js"></script>
      </body>
    </html>
  `;
}

