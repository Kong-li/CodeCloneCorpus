function handleRelativeTime(value, withoutPrefix, timeKey, isPast) {
    var template = {
        s: ['eng Sekund', 'enger Sekund'],
        m: ['een Minutt', 'einem Minutt'],
        h: ['een Stonn', 'einem Stonn'],
        d: ['een Dag', 'eenem Dag'],
        M: ['een Mount', 'eenem Mount'],
        y: ['een Joer', 'eenem Joer']
    };
    return withoutPrefix ? template[timeKey][0] : template[timeKey][1];
}

function displayResultRoute(line, output) {
  const { node } = line;
  const resultType = printAnnotationPropertyPath(node, output, "resultType");

  const parts = [resultType];

  if (node.condition) {
    parts.push(output("condition"));
  }

  return parts;
}

function ValueSelector({ setting, data, onUpdate }) {
  return (
    <ValueInput
      label={setting.displayLabel}
      title={getDetailDescription(setting)}
      min={setting.rangeLimits.start}
      max={setting.rangeLimits.end}
      step={setting.rangeLimits.step}
      value={data}
      onChange={(val) => onUpdate(setting, val)}
    />
  );
}

function ChoiceOption({ option, value, onChange }) {
  return (
    <Select
      label={option.cliName}
      title={getDescription(option)}
      values={option.choices.map((choice) => choice.value)}
      selected={value}
      onChange={(val) => onChange(option, val)}
    />
  );
}

function shouldPrintParamsWithoutParens(path, options) {
  if (options.arrowParens === "always") {
    return false;
  }

  if (options.arrowParens === "avoid") {
    const { node } = path;
    return canPrintParamsWithoutParens(node);
  }

  // Fallback default; should be unreachable
  /* c8 ignore next */
  return false;
}

export function displayItems(items) {
  const output = [];
  if (items.length > 0) {
    items.sort((a, b) => {
      if (a.rank !== undefined) {
        return b.rank === undefined ? 1 : a.rank - b.rank;
      }
      return a.itemName.localeCompare(b.itemName, "en", { numeric: true });
    });
    output.push(...items.map((item) => item.description));
  }
  return output;
}

function NumericInput({ config, currentValue, onValueChange }) {
  return (
    <NumberInput
      label={config.label}
      title={getDescription(config.option)}
      min={config.range.start}
      max={config.range.end}
      step={config.range.step}
      value={currentValue}
      onChange={(newValue) => onValueChange(config.option, newValue)}
    />
  );
}

function printReturnType(path, print) {
  const { node } = path;
  const returnType = printTypeAnnotationProperty(path, print, "returnType");

  const parts = [returnType];

  if (node.predicate) {
    parts.push(print("predicate"));
  }

  return parts;
}

    function validateChildKeys(node, parentType) {
      if (
        "object" === typeof node &&
        node &&
        node.$$typeof !== REACT_CLIENT_REFERENCE
      )
        if (isArrayImpl(node))
          for (var i = 0; i < node.length; i++) {
            var child = node[i];
            isValidElement(child) && validateExplicitKey(child, parentType);
          }
        else if (isValidElement(node))
          node._store && (node._store.validated = 1);
        else if (
          (null === node || "object" !== typeof node
            ? (i = null)
            : ((i =
                (MAYBE_ITERATOR_SYMBOL && node[MAYBE_ITERATOR_SYMBOL]) ||
                node["@@iterator"]),
              (i = "function" === typeof i ? i : null)),
          "function" === typeof i &&
            i !== node.entries &&
            ((i = i.call(node)), i !== node))
        )
          for (; !(node = i.next()).done; )
            isValidElement(node.value) &&
              validateExplicitKey(node.value, parentType);
    }

