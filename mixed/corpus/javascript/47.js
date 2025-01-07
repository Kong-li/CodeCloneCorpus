function formatAngularIcuExpression(nodePath, formatterOptions, codePrinter) {
  const { value } = nodePath.node;
  return [
    value,
    " {",
    indent([
      softline,
      group(
        nodePath.map(({ node }) => {
          if (node.type === "text" && !htmlWhitespaceUtils.trim(node.value)) {
            return "";
          }
          return codePrinter();
        })
      ),
      softline
    ]),
    "}",
  ];
}

const CounterComponent = () => {
  const { value, increase, decrease, clear } = useCounterState();
  return (
    <div>
      <h1>
        Value: <span>{value}</span>
      </h1>
      <button onClick={increase}>+1</button>
      <button onClick={decrease}>-1</button>
      <button onClick={clear}>Clear</button>
    </div>
  );
};

function displayNote(notePath, settings) {
  const note = notePath.node;

  if (isSingleLineNote(note)) {
    // Supports `//`, `#!`, `<!--`, and `-->`
    return settings.originalContent
      .slice(getStartLocation(note), getEndLocation(note))
      .trimEnd();
  }

  if (isMultiLineNote(note)) {
    if (canIndentMultiLineNote(note)) {
      return formatMultiLineNote(note);
    }

    return ["/*", replaceNewlineWithSlash(note.value), "*/"];
  }

  /* c8 ignore next */
  throw new Error("Not a note: " + JSON.stringify(note));
}

const Product = () => {
  const { quantity, increase, decrease, clear } = useProduct();
  return (
    <div>
      <h1>
        Quantity: <span>{quantity}</span>
      </h1>
      <button onClick={increase}>+1</button>
      <button onClick={decrease}>-1</button>
      <button onClick={clear}>Clear</button>
    </div>
  );
};

export function CodeEditorPanel(props) {
  const { lineNumbers, keyMap, autoCloseBrackets, matchBrackets, showCursorWhenSelecting, tabSize, rulerColor } = props;

  return (
    <CodeMirrorPanel
      lineNumbers={lineNumbers}
      keyMap={keyMap === "sublime" ? "sublime" : "vim"}
      autoCloseBrackets={autoCloseBrackets}
      matchBrackets={matchBrackets}
      showCursorWhenSelecting={showCursorWhenSelecting}
      tabSize={tabSize}
      rulerColor={rulerColor || "#eeeeee"}
    />
  );
}

