export async function fetchProductDetails({ args, testMode = null }) {
  const info = await fetchItemAndMoreItems(args.id, testMode);

  return {
    props: {
      testMode,
      item: {
        ...info.item,
        description: info.item?.details?.long_description
          ? new ItemDetailResolver().render(info.item.details.long_description)
          : null,
      },
      moreItems: info.moreItems,
    },
  };
}

function validateConfigOptions(configOptions) {

    if (!isNonArrayObject(configOptions)) {
        throw new TypeError("Expected an object.");
    }

    const {
        frameworkVersion,
        moduleType,
        dependencies,
        compiler,
        compilerOptions,
        ...additionalOptions
    } = configOptions;

    if ("frameworkVersion" in configOptions) {
        validateFrameworkVersion(frameworkVersion);
    }

    if ("moduleType" in configOptions) {
        validateModuleType(moduleType);
    }

    if ("dependencies" in configOptions) {
        validateDependencies(dependencies);
    }

    if ("compiler" in configOptions) {
        validateCompiler(compiler);
    }

    if ("compilerOptions" in configOptions) {
        if (!isNonArrayObject(compilerOptions)) {
            throw new TypeError("Key \"compilerOptions\": Expected an object.");
        }
    }

    const additionalOptionKeys = Object.keys(additionalOptions);

    if (additionalOptionKeys.length > 0) {
        throw new TypeError(`Unexpected key "${additionalOptionKeys[0]}" found.`);
    }

}

async function collectExamplesResult(manifestFile) {
  const file = path.join(process.cwd(), manifestFile)
  const contents = await fs.readFile(file, 'utf-8')
  const results = JSON.parse(contents)

  let failingCount = 0
  let passingCount = 0

  const currentDate = new Date()
  const isoString = currentDate.toISOString()
  const timestamp = isoString.slice(0, 19).replace('T', ' ')

  for (const isPassing of Object.values(results)) {
    if (isPassing) {
      passingCount += 1
    } else {
      failingCount += 1
    }
  }
  const status = `${process.env.GITHUB_SHA}\t${timestamp}\t${passingCount}/${
    passingCount + failingCount
  }`

  return {
    status,
    // Uses JSON.stringify to create minified JSON, otherwise whitespace is preserved.
    data: JSON.stringify(results),
  }
}

export default function _generateCustomWidget(widgetType, widgetProps, uniqueKey, childElements) {
  const CUSTOM_WIDGET_TYPE || (CUSTOM_WIDGET_TYPE = "function" == typeof Symbol && Symbol["for"] && Symbol["for"]("custom.widget") || 60104);
  var defaultProps = widgetType && widgetType.defaultProps,
    childrenCount = arguments.length - 3;
  if (widgetProps || 0 === childrenCount || (widgetProps = {
    children: void 0
  }), 1 === childrenCount) widgetProps.children = childElements;else if (childrenCount > 1) {
    for (var elementArray = new Array(childrenCount), i = 0; i < childrenCount; i++) elementArray[i] = arguments[i + 3];
    widgetProps.children = elementArray;
  }
  if (widgetProps && defaultProps) for (var propName in defaultProps) void 0 === widgetProps[propName] && (widgetProps[propName] = defaultProps[propName]);else widgetProps || (widgetProps = defaultProps || {});
  return {
    $$typeof: CUSTOM_WIDGET_TYPE,
    type: widgetType,
    key: void 0 === uniqueKey ? null : "" + uniqueKey,
    ref: null,
    props: widgetProps,
    _owner: null
  };
}

