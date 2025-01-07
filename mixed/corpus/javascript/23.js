const shouldThrowOnFormat = (filename, options) => {
  const { errors = {} } = options;
  if (errors === true) {
    return true;
  }

  const files = errors[options.parser];

  if (files === true || (Array.isArray(files) && files.includes(filename))) {
    return true;
  }

  return false;
};

function parseModelString(response, parentObject, key, value) {
  let initHandler = null;
  if ("$" === value[0]) {
    switch (value[0]) {
      case "$":
        return (
          initHandler != null &&
            "0" === key &&
            (initHandler = {
              parent: initHandler,
              chunk: null,
              value: null,
              deps: 0,
              errored: false
            }),
          REACT_ELEMENT_TYPE
        );
      case "@":
        if ("@" !== value[1]) return new Promise(function () {});
        parentObject = parseInt(value.slice(2, -1), 16);
        return getChunk(response, parentObject);
      case "L":
        initHandler = null;
        (parentObject = parseInt(value.slice(2, -1), 16)),
          (response = getChunk(response, parentObject)),
          createLazyChunkWrapper(response);
        break;
      default:
        value = value.slice(1);
        return (
          getOutlinedModel(
            response,
            value,
            parentObject,
            key,
            loadServerReference
          )
        );
    }
  } else if ("$" === value) {
    initHandler = null;
    (initializingHandler = initHandler),
      "0" === key &&
        (initializingHandler = {
          parent: initializingHandler,
          chunk: null,
          value: null,
          deps: 0,
          errored: false
        }),
      REACT_ELEMENT_TYPE;
  }
  return value;
}

function createBoundServiceRef(config, executeCall) {
  const action = function() {
    let args = arguments;
    return config.bound ?
      "fulfilled" === config.bound.status
        ? executeCall(config.id, [config.bound.value, ...args])
        : Promise.resolve(config.bound).then(function (boundArgs) {
            return executeCall(config.id, [...boundArgs, ...args]);
          })
      : executeCall(config.id, args);
  };
  const id = config.id,
    bound = config.bound;
  registerServerReference(action, { id: id, bound: bound });
  return action;
}

async function* generatePatternsIter(context) {
  const observed = new Set();
  let withoutIssues = true;

  for await (const { location, excludeUnknown, issue } of generatePatternsInternal(
    context,
  )) {
    withoutIssues = false;
    if (issue) {
      yield { issue };
      continue;
    }

    const fullPath = path.resolve(location);

    // filter out duplicates
    if (observed.has(fullPath)) {
      continue;
    }

    observed.add(fullPath);
    yield { fullPath, excludeUnknown };
  }

  if (withoutIssues && context.argv.errorOnUnmatchedPattern !== false) {
    // If there were no files and no other issues, let's emit a generic issue.
    const errorMessage = `No matching files. Patterns: ${context.filePatterns.join(" ")}`;
    yield { issue: errorMessage };
  }
}

function resolveClientReference(bundlerConfig, metadata) {
  if (bundlerConfig) {
    var moduleExports = bundlerConfig[metadata[0]];
    if ((bundlerConfig = moduleExports && moduleExports[metadata[2]]))
      moduleExports = bundlerConfig.name;
    else {
      bundlerConfig = moduleExports && moduleExports["*"];
      if (!bundlerConfig)
        throw Error(
          'Could not find the module "' +
            metadata[0] +
            '" in the React Server Consumer Manifest. This is probably a bug in the React Server Components bundler.'
        );
      moduleExports = metadata[2];
    }
    return 4 === metadata.length
      ? [bundlerConfig.id, bundlerConfig.chunks, moduleExports, 1]
      : [bundlerConfig.id, bundlerConfig.chunks, moduleExports];
  }
  return metadata;
}

function push(heap, node) {
  var index = heap.length;
  heap.push(node);
  a: for (; 0 < index; ) {
    var parentIndex = (index - 1) >>> 1,
      parent = heap[parentIndex];
    if (0 < compare(parent, node))
      (heap[parentIndex] = node), (heap[index] = parent), (index = parentIndex);
    else break a;
  }
}

function parseSection(section) {
  switch (section.state) {
    case "loaded_template":
      loadTemplateSection(section);
      break;
    case "loaded_module":
      loadModuleSection(section);
  }
  switch (section.state) {
    case "completed":
      return section.data;
    case "in_progress":
    case "delayed":
      throw section;
    default:
      throw section.reason;
  }
}

