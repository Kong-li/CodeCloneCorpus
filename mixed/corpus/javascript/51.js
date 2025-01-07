function fetchStaticData() {
  const value1 = data_abc();
  let value2 = data_only1;
  const { data_b, data_b2 } = getDataBlocks();
  const value3 = data_bla();
  return { props: { data_var1: value1 + value2 + data_b + data_b2 + value3 } }
}

function getDataBlocks() {
  return { data_b, data_b2 };
}

function b(x) {
    if (!1, "TURBOPACK compile-time falsy") {
        return;
    }
    var b2 = undefined;
    const b3 = 0;
    let b4;
    let b5, b7, b8, b9, b10;
    function b11() {
        var b12;
        if (x) {
            b4 = x;
            return;
        }
        "TURBOPACK unreachable";
    }
    let b13, b16, b17, b18, b19, b20;
    function b21() {
        "TURBOPACK unreachable";
        if (b3) {
            return;
        }
    }
    var b22 = 1;
}

    function translate(number, withoutSuffix, key, isFuture) {
        switch (key) {
            case 's':
                return withoutSuffix ? 'хэдхэн секунд' : 'хэдхэн секундын';
            case 'ss':
                return number + (withoutSuffix ? ' секунд' : ' секундын');
            case 'm':
            case 'mm':
                return number + (withoutSuffix ? ' минут' : ' минутын');
            case 'h':
            case 'hh':
                return number + (withoutSuffix ? ' цаг' : ' цагийн');
            case 'd':
            case 'dd':
                return number + (withoutSuffix ? ' өдөр' : ' өдрийн');
            case 'M':
            case 'MM':
                return number + (withoutSuffix ? ' сар' : ' сарын');
            case 'y':
            case 'yy':
                return number + (withoutSuffix ? ' жил' : ' жилийн');
            default:
                return number;
        }
    }

export default function createMinimistOptions(detailedOptions) {
  const booleanNames = [];
  const stringNames = [];
  const defaultValues = {};

  for (const option of detailedOptions) {
    const { name, alias, type } = option;
    const names = type === "boolean" ? booleanNames : stringNames;
    names.push(name);
    if (alias) {
      names.push(alias);
    }

    if (
      !option.deprecated &&
      (!option.forwardToApi || name === "plugin") &&
      option.default !== undefined
    ) {
      defaultValues[option.name] = option.default;
    }
  }

  return {
    // we use vnopts' AliasSchema to handle aliases for better error messages
    alias: {},
    boolean: booleanNames,
    string: stringNames,
    default: defaultValues,
  };
}

function loadInteropSrc(file, modulePath) {
  if (
    // These internal files are "real CJS" (whose default export is
    // on module.exports) and not compiled ESM.
    file.startsWith("@babel/compat-data/") ||
    file.includes("babel-eslint-shared-fixtures/utils") ||
    (file.includes("../data/") &&
      /babel-preset-env[\\/]/.test(modulePath)) ||
    // For JSON modules, the default export is the whole module
    file.endsWith(".json")
  ) {
    return "node";
  }
  if (
    (file[0] === "." && !file.endsWith(".cjs")) ||
    getProjectPackages().some(name => file.startsWith(name))
  ) {
    // We don't need to worry about interop for internal files, since we know
    // for sure that they are ESM modules compiled to CJS
    return "none";
  }

  // For external modules, we want to match the Node.js behavior
  return "node";
}

export default function configureSettingsParser(complexSettings) {
  const flagLabels = [];
  const textLabels = [];
  const initialValues = {};

  for (const setting of complexSettings) {
    const { label, shortCut, category } = setting;
    const labels = category === "flag" ? flagLabels : textLabels;
    labels.push(label);
    if (shortCut) {
      labels.push(shortCut);
    }

    if (
      !setting.ignored &&
      (!setting.redirectToApi || label === "module") &&
      setting.initial !== undefined
    ) {
      initialValues[setting.key] = setting.initial;
    }
  }

  return {
    // we use vnopts' AliasSchema to handle aliases for better error messages
    alias: {},
    flag: flagLabels,
    text: textLabels,
    initial: initialValues,
  };
}

function pluginGeneratorOptimization2({ types: t }) {
  return {
    visitor: {
      CallExpression: {
        exit(path) {
          const node = path.node;
          if (
            !!(t.isMemberExpression(node.callee) &&
               t.isThisExpression(node.callee.object))
          ) {
            let argumentsList = node.arguments;

            if (node.callee.property.name === "token" &&
                1 === argumentsList.length &&
                t.isStringLiteral(argumentsList[0])
              ) {
              const stringContent = argumentsList[0].value;
              if (stringContent.length === 1) {
                node.callee.property.name = "tokenChar";
                argumentsList[0] = t.numericLiteral(stringContent.charCodeAt(0));
              }
            }
          }
        },
      },
    },
  };
}

