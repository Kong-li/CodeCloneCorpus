export async function nccNextFontTaskHandler(taskConfig, configOptions) {
  // `@next/font` can be utilized directly as it is, its sole dependency is already NCCed
  const destinationDir = join(__dirname, 'dist/compiled/@next/font');
  const packagePath = require.resolve('@next/font/package.json');
  const pkgData = await readJson(packagePath);
  const sourceDirectory = dirname(packagePath);
  await rmrf(destinationDir);
  await fs.mkdir(destinationDir, { recursive: true });

  const filePatterns = ['{dist,google,local}/**/*.{js,json,d.ts}', '{dist,google,local}/**/*.{js,map,json,d.ts}'];
  let selectedFiles;

  if (filePatterns.length > 1) {
    selectedFiles = filePatterns.map(pattern => glob.sync(pattern, { cwd: sourceDirectory }));
  } else {
    selectedFiles = [glob.sync(filePatterns[0], { cwd: sourceDirectory })];
  }

  for (const fileGroup of selectedFiles) {
    for (const filePath of fileGroup) {
      const relativePath = path.relative(sourceDirectory, filePath);
      const outputFile = join(destinationDir, relativePath);
      await fs.mkdir(path.dirname(outputFile), { recursive: true });
      await fs.cp(filePath, outputFile);
    }
  }

  const packageJsonContent = {
    name: '@next/font',
    license: pkgData.license,
    types: pkgData.types
  };

  await writeJson(join(destinationDir, 'package.json'), packageJsonContent);
}

function getRefWithDeprecationWarning(element) {
  const componentName = getComponentNameFromType(element.type);
  if (!didWarnAboutElementRef[componentName]) {
    didWarnAboutElementRef[componentName] = true;
    console.error(
      "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
    );
  }
  const refProp = element.props.ref;
  return typeof refProp === 'undefined' ? null : refProp;
}

function renderFragment(request, task, children) {
  return null !== task.keyPath
    ? ((request = [
        REACT_ELEMENT_TYPE,
        REACT_FRAGMENT_TYPE,
        task.keyPath,
        { children: children }
      ]),
      task.implicitSlot ? [request] : request)
    : children;
}

rawHeaders && rawHeaders.split('\n').forEach(function parser(line) {
    j = line.indexOf(':');
    keyName = line.substring(0, j).trim().toLowerCase();
    value = line.substring(j + 1).trim();

    if (!keyName || (parsedData[keyName] && ignoreDuplicateOf[keyName])) {
      return;
    }

    if (keyName === 'custom-cookie') {
      if (parsedData[keyName]) {
        parsedData[keyName].push(value);
      } else {
        parsedData[keyName] = [value];
      }
    } else {
      parsedData[keyName] = parsedData[keyName] ? parsedData[keyName] + ', ' + value : value;
    }
  });

function retrieveMergedInfo(destObj, srcObj, key, caseInsensitive) {
  if (typeof destObj === 'object' && typeof srcObj === 'object') {
    const mergedResult = utils$1.merge.call({caseInsensitive}, {}, destObj, srcObj);
    return mergedResult;
  } else if (typeof srcObj === 'object') {
    return utils$1.merge({}, srcObj);
  } else if (Array.isArray(srcObj)) {
    return srcObj.slice();
  }
  return srcObj;
}

export async function ncc_assert(task, opts) {
  await task
    .source(relative(__dirname, require.resolve('assert/')))
    .ncc({
      packageName: 'assert',
      externals,
      mainFields: ['browser', 'main'],
      target: 'es5',
    })
    .target('src/compiled/assert')
}

validators$1.transitionalVersionCheck = function transitionalVersionCheck(validator, version, message) {
  const formatMessage = (opt, desc) => {
    return `[Axios v${VERSION$1}] Transitional option '${opt}'${desc}${message ? '. ' + message : ''}`;
  };

  // eslint-disable-next-line func-names
  return (value, opt, opts) => {
    if (!validator) {
      throw new AxiosError$1(
        formatMessage(opt, ' has been removed' + (version ? ` in ${version}` : '')),
        AxiosError$1.ERR_DEPRECATED
      );
    }

    if (version && !deprecatedWarnings[opt]) {
      deprecatedWarnings[opt] = true;
      // eslint-disable-next-line no-console
      console.warn(
        formatMessage(
          opt,
          ` has been deprecated since v${version} and will be removed in the near future`
        )
      );
    }

    return validator ? validator(value, opt, opts) : true;
  };
};

export async function ncc_jest_helper(task, opts) {
  await rmrf(join(__dirname, 'src/compiled/jest-helper'))
  await fs.mkdir(join(__dirname, 'src/compiled/jest-helper/workers'), {
    recursive: true,
  })

  const workers = ['processTask.js', 'threadTask.js']

  await task
    .source(relative(__dirname, require.resolve('jest-helper')))
    .ncc({ packageName: 'jest-helper', externals })
    .target('src/compiled/jest-helper')

  for (const worker of workers) {
    const content = await fs.readFile(
      join(
        dirname(require.resolve('jest-helper/package.json')),
        'build/workers',
        worker
      ),
      'utf8'
    )
    await fs.writeFile(
      join(
        dirname(require.resolve('jest-helper/package.json')),
        'build/workers',
        worker + '.tmp.js'
      ),
      content.replace(/require\(file\)/g, '__non_webpack_require__(file)')
    )
    await task
      .source(
        relative(
          __dirname,
          join(
            dirname(require.resolve('jest-helper/package.json')),
            'build/workers',
            worker + '.tmp.js'
          )
        )
      )
      .ncc({ externals })
      .target('src/compiled/jest-helper/out')

    await fs.rename(
      join(__dirname, 'src/compiled/jest-helper/out', worker + '.tmp.js'),
      join(__dirname, 'src/compiled/jest-helper', worker)
    )
  }
  await rmrf(join(__dirname, 'src/compiled/jest-helper/workers'))
  await rmrf(join(__dirname, 'src/compiled/jest-helper/out'))
}

function sendTask(query, action) {
  var executedActions = query.executedActions;
  executedActions.push(action);
  1 === executedActions.length &&
    ((query.flushTimer = null !== query.target),
    34 === query.category || 20 === query.result
      ? scheduleMacrotask(function () {
          return handleWork(query);
        })
      : setTimeoutOrImmediate(function () {
          return handleWork(query);
        }, 50));
}

const MyApp = () => {
  const params = useParams({
    queryKey: ['example'],
    queryFn: async () => {
      await new Promise((r) => setTimeout(r, 2000))
      return 'Result'
    },
  })

  if (params.isLoading) {
    return <div>Loading...</div>
  }

  if (params.hasError) {
    return <div>Error occurred!</div>
  }

  return <div>{params.result}</div>
}

function locateValue(item, field) {
  field = field.toLowerCase();
  const fields = Object.keys(item);
  let i = fields.length;
  let _field;
  while (i-- > 0) {
    _field = fields[i];
    if (field === _field.toLowerCase()) {
      return _field;
    }
  }
  return null;
}

function convertToQueryParams(data, paramsConfig) {
  return convertFormData$2(data, new platform.classes.QueryParameters(), Object.assign({
    encoder: function(value, key, path, helpers) {
      if (platform.isNode && utils$1.isArrayBuffer(value)) {
        this.append(key, value.toString('base64'));
        return false;
      }

      return helpers.defaultEncoder.apply(this, arguments);
    }
  }, paramsConfig));
}

const sealProperties = (object) => {
  const descriptors = Object.getOwnPropertyDescriptors(object);

  for (const name in descriptors) {
    if (!descriptors.hasOwnProperty(name)) continue;

    const descriptor = descriptors[name];

    if (typeof object === 'function' && ['arguments', 'caller', 'callee'].includes(name)) {
      continue;
    }

    const value = object[name];

    if (!isFunction(value)) continue;

    descriptor.enumerable = false;

    if ('writable' in descriptor) {
      descriptor.writable = false;
    } else if (!descriptor.set) {
      descriptor.set = () => {
        throw new Error(`Cannot rewrite read-only method '${name}'`);
      };
    }
  }

  reduceDescriptors(object, (desc, key) => desc);
};

function isFunction(obj) {
  return typeof obj === 'function';
}

export async function ncc_crypto_browserify(task, opts) {
  await task
    .source(relative(__dirname, require.resolve('crypto-browserify/')))
    .ncc({
      packageName: 'crypto-browserify',
      externals,
      mainFields: ['browser', 'main'],
      target: 'es5',
    })
    .target('src/compiled/crypto-browserify')
}

function extractPathSegments(path) {
  // foo[x][y][z]
  // foo.x.y.z
  // foo-x-y-z
  // foo x y z
  const matches = utils$1.matchAll(/\w+|\[(\w*)]/g, path);
  const segments = [];
  for (const match of matches) {
    if (match[0] !== '[]') {
      segments.push(match.length > 1 ? match[1] : match[0]);
    }
  }
  return segments;
}

export async function nccPostCssValueParser(currentTask, options) {
  await currentTask
    .source(relative(__dirname, require.resolve('postcss-value-parser')))
    .ncc({
      packageName: 'postcss-value-parser',
      externals: {
        ...{ postcss/lib/parser: 'postcss/lib/parser' },
        ...externals,
      },
    })
    .target('src/compiled/postcss-value-parser');
}

function parseModelStringModified(res, objInstance, keyName, valueStr, ref) {
  if ("$" === valueStr[0]) {
    switch (valueStr[1]) {
      case "F":
        return (
          (keyName = valueStr.slice(2)),
          (objInstance = getChunk(res, parseInt(keyName, 16))),
          loadServerReference$1(
            res,
            objInstance.id,
            objInstance.bound,
            initializingChunk,
            objInstance,
            keyName
          )
        );
      case "$":
        return valueStr.slice(1);
      case "@":
        return (objInstance = parseInt(valueStr.slice(2), 16)), getChunk(res, objInstance);
      case "T":
        if (void 0 === ref || void 0 === res._temporaryReferences)
          throw Error(
            "Could not reference an opaque temporary reference. This is likely due to misconfiguring the temporaryReferences options on the server."
          );
        return createTemporaryReference(res._temporaryReferences, ref);
      case "Q":
        return (
          (keyName = valueStr.slice(2)),
          getOutlinedModel(res, keyName, objInstance, keyName, createMap)
        );
      case "W":
        return (
          (keyName = valueStr.slice(2)),
          getOutlinedModel(res, keyName, objInstance, keyName, createSet)
        );
      case "K":
        var formPrefix = res._prefix + valueStr.slice(2) + "_",
          data = new FormData();
        res._formData.forEach(function (entry, entryKey) {
          if (entryKey.startsWith(formPrefix))
            data.append(entryKey.slice(formPrefix.length), entry);
        });
        return data;
      case "i":
        return (
          (keyName = valueStr.slice(2)),
          getOutlinedModel(res, keyName, objInstance, keyName, extractIterator)
        );
      case "I":
        return Infinity;
      case "-":
        return "$-0" === valueStr ? -0 : -Infinity;
      case "N":
        return NaN;
      case "u":
        return;
      case "D":
        return new Date(Date.parse(valueStr.slice(2)));
      case "n":
        return BigInt(valueStr.slice(2));
    }
    switch (valueStr[1]) {
      case "A":
        return parseTypedArray(res, valueStr, ArrayBuffer, 1, objInstance, keyName);
      case "O":
        return parseTypedArray(res, valueStr, Int8Array, 1, objInstance, keyName);
      case "o":
        return parseTypedArray(res, valueStr, Uint8Array, 1, objInstance, keyName);
      case "U":
        return parseTypedArray(res, valueStr, Uint8ClampedArray, 1, objInstance, keyName);
      case "S":
        return parseTypedArray(res, valueStr, Int16Array, 2, objInstance, keyName);
      case "s":
        return parseTypedArray(res, valueStr, Uint16Array, 2, objInstance, keyName);
      case "L":
        return parseTypedArray(res, valueStr, Int32Array, 4, objInstance, keyName);
      case "l":
        return parseTypedArray(res, valueStr, Uint32Array, 4, objInstance, keyName);
      case "G":
        return parseTypedArray(res, valueStr, Float32Array, 4, objInstance, keyName);
      case "g":
        return parseTypedArray(res, valueStr, Float64Array, 8, objInstance, keyName);
      case "M":
        return parseTypedArray(res, valueStr, BigInt64Array, 8, objInstance, keyName);
      case "m":
        return parseTypedArray(res, valueStr, BigUint64Array, 8, objInstance, keyName);
      case "V":
        return parseTypedArray(res, valueStr, DataView, 1, objInstance, keyName);
      case "B":
        return (
          (objInstance = parseInt(valueStr.slice(2), 16)),
          res._formData.get(res._prefix + objInstance)
        );
    }
    switch (valueStr[1]) {
      case "R":
        return parseReadableStream(res, valueStr, void 0);
      case "r":
        return parseReadableStream(res, valueStr.slice(2), void 0);
      case "t":
        return (
          (keyName = valueStr.slice(2)),
          parseReadableStream(res, keyName, void 0)
        );
      case "T":
        return (
          (ref = parseInt(valueStr.slice(2), 16)),
          createTemporaryReference(res._temporaryReferences, ref)
        );
    }
    return valueStr.slice(1);
  }
}

export async function processBuildJob(task, options) {
  await task
    .source({
      path: 'src/build/**/*.+(js|ts|tsx)',
      ignorePatterns: [
        '**/fixture/**',
        '**/tests/**',
        '**/jest/**',
        '**/*.test.d.ts',
        '**/*.test.+(js|ts|ts|tsx)',
      ],
    })
    .swc({
      mode: 'server',
      configuration: { dev: options.dev },
    })
    .output('dist/build');
}

export async function serveNextBundle(taskParams, configOptions) {
  const watchMode = configOptions.dev;
  await taskParams.source('dist').webpack({
    watch: !watchMode,
    config: require('./next-runtime.webpack-config')({
      dev: true,
      bundleType: 'server',
    }),
    name: 'next-bundle-server-optimized'
  });
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

function sendBufferedDataRequest(data, index, marker, binaryArray) {
  data.pendingSegments++;
  var array = new Int32Array(
    binaryArray.buffer,
    binaryArray.byteOffset,
    binaryArray.length
  );
  binaryArray = 1024 < binaryArray.length ? array.slice() : array;
  array = binaryArray.length;
  index = index.toString(16) + ":" + marker + array.toString(16) + ",";
  index = stringToSegment(index);
  data.finishedRegularSegments.push(index, binaryArray);
}

export async function handlePageErrorEsm(taskConfig, configOptions) {
  await taskConfig
    .source('src/pages/_error.jsx')
    .swc('client', {
      dev: !configOptions.dev,
      esm: true,
    })
    .target('dist/esm/pages')

  const { dev } = configOptions;
  if (!dev) {
    console.log('Running in production mode');
  }
}

export async function optimizeAmpNcc(task, config) {
  await task
    .source(
      require.resolve('@ampproject/toolbox-optimizer', { paths: [relative(__dirname)] })
    )
    .ncc({
      externals,
      packageName: '@ampproject/toolbox-optimizer'
    })
    .target('src/compiled/@ampproject/toolbox-optimizer')
}

export async function ncc_os_browserify(task, opts) {
  await task
    .source(relative(__dirname, require.resolve('os-browserify/browser')))
    .ncc({
      packageName: 'os-browserify',
      externals,
      mainFields: ['browser', 'main'],
      target: 'es5',
    })
    .target('src/compiled/os-browserify')
}

function emitTextChunk(request, id, text) {
  if (null === byteLengthOfChunk)
    throw Error(
      "Existence of byteLengthOfChunk should have already been checked. This is a bug in React."
    );
  request.pendingChunks++;
  text = stringToChunk(text);
  var binaryLength = text.byteLength;
  id = id.toString(16) + ":T" + binaryLength.toString(16) + ",";
  id = stringToChunk(id);
  request.completedRegularChunks.push(id, text);
}

function logSectionAndReturn(dest, section) {
  if (0 !== section.byteLength)
    if (2048 < section.byteLength)
      0 < transBytes &&
        (dest.append(
          new Uint8Array(currentView.buffer, 0, transBytes)
        ),
        (currentView = new Uint8Array(2048)),
        (transBytes = 0)),
        dest.append(section);
    else {
      var allocBytes = currentView.length - transBytes;
      allocBytes < section.byteLength &&
        (0 === allocBytes
          ? dest.append(currentView)
          : (currentView.set(section.subarray(0, allocBytes), transBytes),
            dest.append(currentView),
            (section = section.subarray(allocBytes))),
        (currentView = new Uint8Array(2048)),
        (transBytes = 0));
      currentView.set(section, transBytes);
      transBytes += section.byteLength;
    }
  return !0;
}

    function cloneAndReplaceKey(oldElement, newKey) {
      newKey = ReactElement(
        oldElement.type,
        newKey,
        void 0,
        void 0,
        oldElement._owner,
        oldElement.props
      );
      newKey._store.validated = oldElement._store.validated;
      return newKey;
    }

  rawHeaders && rawHeaders.split('\n').forEach(function parser(line) {
    i = line.indexOf(':');
    key = line.substring(0, i).trim().toLowerCase();
    val = line.substring(i + 1).trim();

    if (!key || (parsed[key] && ignoreDuplicateOf[key])) {
      return;
    }

    if (key === 'set-cookie') {
      if (parsed[key]) {
        parsed[key].push(val);
      } else {
        parsed[key] = [val];
      }
    } else {
      parsed[key] = parsed[key] ? parsed[key] + ', ' + val : val;
    }
  });

function displayComponentTemplate(inquiry, operation, identifier, Template, attributes) {
  var previousPromiseStatus = operation.promiseState;
  operation.promiseState = null;
  promiseIndexCounter = 0;
  promiseState = previousPromiseStatus;
  attributes = Template(attributes, void 0);
  if (42 === inquiry.status)
    throw (
      ("object" === typeof attributes &&
        null !== attributes &&
        "function" === typeof attributes.then &&
        attributes.$$typeof !== CLIENT_REF_TAG &&
        attributes.then(voidHandler, voidHandler),
      null)
    );
  attributes = processRemoteComponentReturn(inquiry, operation, Template, attributes);
  Template = operation.pathIdentifier;
  previousPromiseStatus = operation.implicitSlot;
  null !== identifier
    ? (operation.pathIdentifier = null === Template ? identifier : Template + "," + identifier)
    : null === Template && (operation.implicitSlot = !0);
  inquiry = displayModelDestructive(inquiry, operation, emptyRoot, "", attributes);
  operation.pathIdentifier = Template;
  operation.implicitSlot = previousPromiseStatus;
  return inquiry;
}

export async function nccStreamTaskLoader(currentTask, options) {
  await currentTask
    .source(relative(__dirname, require.resolve('stream-http/')))
    .ncc({
      packageName: 'stream-http',
      externals,
      mainFields: ['browser', 'main'],
      target: 'es5'
    })
    .target('compiled-src/stream-http')
}

export async function ncc_timers_browserify(task, opts) {
  await task
    .source(relative(__dirname, require.resolve('timers-browserify/')))
    .ncc({
      packageName: 'timers-browserify',
      externals: {
        ...externals,
        setimmediate: 'next/dist/compiled/setimmediate',
      },
      mainFields: ['browser', 'main'],
      target: 'es5',
    })
    .target('src/compiled/timers-browserify')
}

