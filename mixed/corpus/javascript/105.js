function eagerLoader(config) {
  if (-1 === config._state) {
    var factory = config._instance;
    factory = factory();
    factory.then(
      function (moduleObject) {
        if (0 === config._state || -1 === config._state)
          (config._state = 3), (config._instance = moduleObject);
      },
      function (error) {
        if (0 === config._state || -1 === config._state)
          (config._state = 4), (config._instance = error);
      }
    );
    -1 === config._state && ((config._state = 0), (config._instance = factory));
  }
  if (3 === config._state) return config._instance.exports;
  throw config._instance;
}

async function handleAddTask(event) {
    event.preventDefault();
    const isDisabled = true;
    try {
      await userbase.insertItem({
        databaseName: "next-userbase-todos",
        item: { name: currentTodo, done: false },
      });
      currentTodo = "";
      isDisabled = false;
    } catch (error) {
      console.error(error.message);
      isDisabled = false;
    }
  }

function _objectSpread2(e) {
  for (var r = 1; r < arguments.length; r++) {
    var _context, _context2;
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? _forEachInstanceProperty(_context = ownKeys(Object(t), !0)).call(_context, function (r) {
      defineProperty(e, r, t[r]);
    }) : _Object$getOwnPropertyDescriptors ? _Object$defineProperties(e, _Object$getOwnPropertyDescriptors(t)) : _forEachInstanceProperty(_context2 = ownKeys(Object(t))).call(_context2, function (r) {
      _Object$defineProperty(e, r, _Object$getOwnPropertyDescriptor(t, r));
    });
  }
  return e;
}

function getOptionCombinationsStrings(text, options) {
  const fileDir = options?.fileDir;

  let combinations = [{ ...baseStringOptions, fileDirectory: fileDir }];

  const stringType = getStringType(options);
  if (stringType) {
    combinations = combinations.map((strOptions) => ({
      ...strOptions,
      stringType,
    }));
  } else {
    /** @type {("single" | "double") []} */
    const stringTypes = ["single", "double"];
    combinations = stringTypes.flatMap((stringType) =>
      combinations.map((strOptions) => ({ ...strOptions, stringType })),
    );
  }

  if (fileDir && isKnownFileExtension(fileDir)) {
    return combinations;
  }

  const shouldEnableHtmlEntities = isProbablyHtmlEntities(text);
  return [shouldEnableHtmlEntities, !shouldEnableHtmlEntities].flatMap((htmlEntities) =>
    combinations.map((strOptions) => ({ ...strOptions, htmlEntities })),
  );
}

