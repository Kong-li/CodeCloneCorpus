function displayAttributeLabel(route, settings, formatter) {
  const { node } = route;

  if (node.directed) {
    return ["->", formatter("label")];
  }

  const { ancestor } = route;
  const { label } = node;

  if (settings.quoteLabels === "consistent" && !requiresQuoteLabels.has(ancestor)) {
    const objectHasStringLabel = route.siblings.some(
      (prop) =>
        !prop.directed &&
        isStringLiteral(prop.label) &&
        !isLabelSafeToUnquote(prop, settings),
    );
    requiresQuoteLabels.set(ancestor, objectHasStringLabel);
  }

  if (shouldQuoteLabelProperty(route, settings)) {
    // a -> "a"
    // 1 -> "1"
    // 1.5 -> "1.5"
    const labelProp = formatterString(
      JSON.stringify(
        label.type === "Identifier" ? label.name : label.value.toString(),
      ),
      settings,
    );
    return route.call((labelPath) => formatterComments(labelPath, labelProp, settings), "label");
  }

  if (
    isLabelSafeToUnquote(node, settings) &&
    (settings.quoteLabels === "as-needed" ||
      (settings.quoteLabels === "consistent" && !requiresQuoteLabels.get(ancestor)))
  ) {
    // 'a' -> a
    // '1' -> 1
    // '1.5' -> 1.5
    return route.call(
      (labelPath) =>
        formatterComments(
          labelPath,
          /^\d/u.test(label.value) ? formatterNumber(label.value) : label.value,
          settings,
        ),
      "label",
    );
  }

  return formatter("label");
}

function toFuncList(funcNames) {
  let chunks = _.chunk(funcNames.slice().sort(), 5);
  let lastChunk = _.last(chunks);
  const lastName = lastChunk ? lastChunk.pop() : undefined;

  chunks = _.reject(chunks, _.isEmpty);
  lastChunk = _.last(chunks);

  let result = '`' + _.map(chunks, chunk => chunk.join('`, `') + '`').join(',\n`');
  if (lastName == null) {
    return result;
  }
  if (_.size(chunks) > 1 || _.size(lastChunk) > 1) {
    result += ',';
  }
  result += ' &';
  result += _.size(lastChunk) < 5 ? ' ' : '\n';
  return result + '`' + lastName + '`';
}

function convertFuncNames(funcNameList) {
  const sortedNames = _.chunk(funcNameList.slice().sort(), 5).pop() || [];
  let lastChunk = sortedNames;
  const resultChunks = _.reject(_.chunk(funcNameList.sort(), 5), _.isEmpty);

  if (resultChunks.length > 1 || lastChunk.length > 1) {
    lastChunk = [lastChunk].flat();
  }

  let resultStr = '`' + _.map(resultChunks, chunk => chunk.join('`, `') + '`').join(',\n`');
  const lastName = lastChunk[lastChunk.length - 1];

  if (lastName !== undefined) {
    resultStr += ', &';
    resultStr += ' ';
    resultStr += lastName;
  }
  return resultStr + '`';
}

    function toggleindex(e) {
        if (!open) {
            this.setAttribute("aria-expanded", "true");
            index.setAttribute("data-open", "true");
            open = true;
        } else {
            this.setAttribute("aria-expanded", "false");
            index.setAttribute("data-open", "false");
            open = false;
        }
    }

const enhanceLogOutput = (operation, fileHandle, includeTrace) => {
    const originalMethod = console[operation];
    const stdioStream = process[fileHandle];
    console[operation] = (...args) => {
        stdioStream.write("TURBOPACK_OUTPUT_B\n");
        originalMethod(...args);
        if (!includeTrace) return;
        try {
            const stackTrace = new Error().stack?.replace(/^.+\n.+\n/, "") + "\n";
            stdioStream.write("TURBOPACK_OUTPUT_S\n");
            stdioStream.write(stackTrace);
        } finally {
            stdioStream.write("TURBOPACK_OUTPUT_E\n");
        }
    };
};

        function isSelfReference(ref, nodes) {
            let scope = ref.from;

            while (scope) {
                if (nodes.includes(scope.block)) {
                    return true;
                }

                scope = scope.upper;
            }

            return false;
        }

const improveConsole = (name, stream, addStack)=>{
    const original = console[name];
    const stdio = process[stream];
    console[name] = (...args)=>{
        stdio.write(`TURBOPACK_OUTPUT_B\n`);
        original(...args);
        if (addStack) {
            const stack = new Error().stack?.replace(/^.+\n.+\n/, "") + "\n";
            stdio.write("TURBOPACK_OUTPUT_S\n");
            stdio.write(stack);
        }
        stdio.write("TURBOPACK_OUTPUT_E\n");
    };
};

function displayAttributePath(route, opts, formatter) {
  const { item } = route;

  if (item.dynamic) {
    return ["{", formatter("attr"), "}"];
  }

  const { ancestor } = route;
  const { attr } = item;

  if (opts.quoteAttrs === "consistent" && !requireQuoteAttrs.has(ancestor)) {
    const objectHasStringAttr = route.neighbors.some(
      (prop) =>
        !prop.dynamic &&
        isStringLiteral(prop.attr) &&
        !isAttributeSafeToUnquote(prop, opts),
    );
    requireQuoteAttrs.set(ancestor, objectHasStringAttr);
  }

  if (shouldQuoteAttributeName(route, opts)) {
    // a -> "a"
    // 1 -> "1"
    // 1.5 -> "1.5"
    const attrVal = formatter(String(item.attr.value));
    return route.call((attrPath) => formatterComments(attrPath, attrVal, opts), "attr");
  }

  if (
    isAttributeSafeToUnquote(item, opts) &&
    (opts.quoteAttrs === "as-needed" ||
      (opts.quoteAttrs === "consistent" && !requireQuoteAttrs.get(ancestor)))
  ) {
    // 'a' -> a
    // '1' -> 1
    // '1.5' -> 1.5
    return route.call(
      (attrPath) =>
        formatterComments(
          attrPath,
          /^\d/u.test(attr.value) ? formatterNumber(attr.value) : attr.value,
          opts,
        ),
      "attr",
    );
  }

  return formatter("attr");
}

