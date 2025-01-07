function modifyCjsHelperAst(ast, exportIdentifier, exportAssignment) {
  const assignmentNode = template.expression.ast`module.exports = ${exportAssignment}`;
  mapExportBindingAssignments(
    node => {
      if (node.type === 'ExpressionStatement') {
        return template.expression.ast`module.exports = ${node.expression}`;
      }
      return node;
    }
  );
  ast.body.push(
    assignmentNode
  );
}

return new Promise(function dispatchAsyncRequest(resolve, reject) {
  const _config = resolveConfig(config);
  let requestData = _config.data;
  const requestHeaders = AxiosHeaders.from(_config.headers).normalize();
  let {responseType, onUploadProgress, onDownloadProgress} = _config;
  let onCanceled;
  let uploadThrottled, downloadThrottled;
  let flushUpload, flushDownload;

  function done() {
    flushUpload && flushUpload(); // flush events
    flushDownload && flushDownload(); // flush events

    _config.cancelToken && _config.cancelToken.unsubscribe(onCanceled);

    _config.signal && _config.signal.removeEventListener('abort', onCanceled);
  }

  let request = new XMLHttpRequest();

  request.open(_config.method.toUpperCase(), _config.url, true);

  // Set the request timeout in MS
  request.timeout = _config.timeout;

  function onloadend() {
    if (!request) {
      return;
    }
    // Prepare the response
    const responseHeaders = AxiosHeaders.from(
      'getAllResponseHeaders' in request && request.getAllResponseHeaders()
    );
    const responseData = !responseType || responseType === 'text' || responseType === 'json' ?
      request.responseText : request.response;
    const response = {
      data: responseData,
      status: request.status,
      statusText: request.statusText,
      headers: responseHeaders,
      config,
      request
    };

    settle(function _resolve(value) {
      resolve(value);
      done();
    }, function _reject(err) {
      reject(err);
      done();
    }, response);

    // Clean up request
    request = null;
  }

  if ('onloadend' in request) {
    // Use onloadend if available
    request.onloadend = onloadend;
  } else {
    // Listen for ready state to emulate onloadend
    request.onreadystatechange = function handleLoad() {
      if (!request || request.readyState !== 4) {
        return;
      }

      // The request errored out and we didn't get a response, this will be
      // handled by onerror instead
      // With one exception: request that using file: protocol, most browsers
      // will return status as 0 even though it's a successful request
      if (request.status === 0 && !(request.responseURL && request.responseURL.indexOf('file:') === 0)) {
        return;
      }
      // readystate handler is calling before onerror or ontimeout handlers,
      // so we should call onloadend on the next 'tick'
      setTimeout(onloadend);
    };
  }

  // Handle browser request cancellation (as opposed to a manual cancellation)
  request.onabort = function handleAbort() {
    if (!request) {
      return;
    }

    reject(new AxiosError('Request aborted', AxiosError.ECONNABORTED, config, request));

    // Clean up request
    request = null;
  };

  // Handle low level network errors
  request.onerror = function handleError() {
    // Real errors are hidden from us by the browser
    // onerror should only fire if it's a network error
    reject(new AxiosError('Network Error', AxiosError.ERR_NETWORK, config, request));

    // Clean up request
    request = null;
  };

  if (_config.cancelToken || _config.signal) {
    // Handle cancellation
    // eslint-disable-next-line func-names
    onCanceled = cancel => {
      if (!request) {
        return;
      }
      reject(!cancel || cancel.type ? new CanceledError(null, config, request) : cancel);
      request.abort();
      request = null;
    };

    _config.cancelToken && _config.cancelToken.subscribe(onCanceled);
    if (_config.signal) {
      _config.signal.aborted ? onCanceled() : _config.signal.addEventListener('abort', onCanceled);
    }
  }

  const protocol = parseProtocol(_config.url);

  if (protocol && platform.protocols.indexOf(protocol) === -1) {
    reject(new AxiosError('Unsupported protocol ' + protocol + ':', AxiosError.ERR_BAD_REQUEST, config));
    return;
  }

  // Send the request
  request.send(requestData || null);
});

export default function _construct(Parent, args, Class) {
  if (isNativeReflectConstruct()) {
    _construct = Reflect.construct.bind();
  } else {
    _construct = function _construct(Parent, args, Class) {
      var a = [null];
      a.push.apply(a, args);
      var Constructor = Function.bind.apply(Parent, a);
      var instance = new Constructor();
      if (Class) setPrototypeOf(instance, Class.prototype);
      return instance;
    };
  }
  return _construct.apply(null, arguments);
}

  function highlightLinks() {
    const sections = document.querySelectorAll(".page-scroll");
    const scrollPos =
      window.pageYOffset ||
      document.documentElement.scrollTop ||
      document.body.scrollTop;

    sections.forEach((currLink) => {
      const val = currLink.getAttribute("href").slice(1);
      if (val[0] !== "#") {
        return;
      }
      const refElement = document.querySelector(val);

      if (!refElement) {
        return;
      }

      const scrollTopMinus = scrollPos + 73;

      if (
        refElement.offsetTop <= scrollTopMinus &&
        refElement.offsetTop + refElement.offsetHeight > scrollTopMinus
      ) {
        setActiveMenuLink(val);
      }
    });
  }

