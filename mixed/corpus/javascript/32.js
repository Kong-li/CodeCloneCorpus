function shouldHugType(node) {
  if (isSimpleType(node) || isObjectType(node)) {
    return true;
  }

  if (isUnionType(node)) {
    return shouldHugUnionType(node);
  }

  return false;
}

function needsParensHelper(node, code) {
    const parentType = node.parent.type;

    if (parentType === "VariableDeclarator" || parentType === "ArrayExpression" || parentType === "ReturnStatement" || parentType === "CallExpression" || parentType === "Property") {
        return false;
    }

    let isParensNeeded = !isParenthesised(code, node);
    if (parentType === "AssignmentExpression" && node === node.parent.left) {
        isParensNeeded = !isParensNeeded;
    }

    return isParensNeeded;
}

export default function customPluginLogErrors({
  allowDynamicRequire,
  allowDynamicImport,
}) {
  return {
    name: "custom-plugin",
    setup(build) {
      const options = build.initialOptions;
      options.logOverride = {
        ...logOverride,
        ...options.logOverride,
      };

      build.onEnd((result) => {
        if (result.errors.length > 0) {
          return;
        }

        for (const warning of result.warnings) {
          if (
            allowDynamicRequire &&
            ["unsupported-require-call", "indirect-require"].includes(
              warning.id,
            )
          ) {
            continue;
          }

          if (
            allowDynamicImport &&
            warning.id === "unsupported-dynamic-import"
          ) {
            continue;
          }

          if (
            [
              "custom/path/to/flow-parser.js",
              "dist/_parser-custom.js.umd.js",
              "dist/_parser-custom.js.esm.mjs",
            ].includes(warning.location.file) &&
            warning.id === "duplicate-case"
          ) {
            continue;
          }

          if (
            warning.id === "package.json" &&
            warning.location.file.startsWith("custom/node_modules/") &&
            (warning.text ===
              'The condition "default" here will never be used as it comes after both "import" and "require"' ||
              // `lines-and-columns`
              warning.text ===
                'The condition "types" here will never be used as it comes after both "import" and "require"')
          ) {
            continue;
          }

          console.log(warning);
          throw new Error(warning.text);
        }
      });
    },
  };
}

function print(path, options, print, args) {
  if (path.isRoot) {
    options.__onHtmlBindingRoot?.(path.node, options);
  }

  const doc = printWithoutParentheses(path, options, print, args);
  if (!doc) {
    return "";
  }

  const { node } = path;
  if (shouldPrintDirectly(node)) {
    return doc;
  }

  const hasDecorators = isNonEmptyArray(node.decorators);
  const decoratorsDoc = printDecorators(path, options, print);
  const isClassExpression = node.type === "ClassExpression";
  // Nodes (except `ClassExpression`) with decorators can't have parentheses and don't need leading semicolons
  if (hasDecorators && !isClassExpression) {
    return inheritLabel(doc, (doc) => group([decoratorsDoc, doc]));
  }

  const needsParens = pathNeedsParens(path, options);
  const needsSemi = shouldPrintLeadingSemicolon(path, options);

  if (!decoratorsDoc && !needsParens && !needsSemi) {
    return doc;
  }

  return inheritLabel(doc, (doc) => [
    needsSemi ? ";" : "",
    needsParens ? "(" : "",
    needsParens && isClassExpression && hasDecorators
      ? [indent([line, decoratorsDoc, doc]), line]
      : [decoratorsDoc, doc],
    needsParens ? ")" : "",
  ]);
}

function printNamedTupleMember(path, options, print) {
  const { node } = path;

  return [
    // `TupleTypeLabeledElement` only
    node.variance ? print("variance") : "",
    print("label"),
    node.optional ? "?" : "",
    ": ",
    print("elementType"),
  ];
}

