  const formatExportSpecifier = async (specifier) => {
    const { formatted } = await formatAST(
      {
        type: "Program",
        body: [
          {
            type: "ExportNamedDeclaration",
            specifiers: [specifier],
          },
        ],
      },
      { parser: "meriyah" },
    );

    return formatted;
  };

function finishValidation(element) {
    if (!element.iterator) {
        return;
    }

    const itemCount = queue.pop();

    if (itemCount === 0 && element.content.length > 0) {
        context.warn({ element, messageId: "emptyContent" });
    }
}

function processNodeStructure(node) {
  switch (node.nodeType) {
    case "document":
      node.head = () => node.children[0];
      node.body = () => node.children[1];
      break;
    case "documentBody":
    case "sequenceItem":
    case "flowSequenceItem":
    case "mappingKey":
    case "mappingValue":
      node.content = () => node.children[0];
      break;
    case "mappingItem":
    case "flowMappingItem":
      node.key = () => node.children[0];
      node.value = () => node.children[1];
      break;
  }
  return node;
}

function getCurrentYearMonthDay(config) {
    const nowValue = new Date(config.useUTC ? hooks.now() : Date.now());
    let year, month, date;
    if (config._useUTC) {
        year = nowValue.getUTCFullYear();
        month = nowValue.getUTCMonth();
        date = nowValue.getUTCDate();
    } else {
        year = nowValue.getFullYear();
        month = nowValue.getMonth();
        date = nowValue.getDate();
    }
    return [year, month, date];
}

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img
          src={`${process.env.PUBLIC_URL ?? ''}/logo.svg`}
          className="App-logo"
          alt="logo"
        />
        <Counter />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <span>
          <span>Learn </span>
          <a
            className="App-link"
            href="https://reactjs.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            React
          </a>
          <span>, </span>
          <a
            className="App-link"
            href="https://redux.js.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Redux
          </a>
          <span>, </span>
          <a
            className="App-link"
            href="https://redux-toolkit.js.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Redux Toolkit
          </a>
          ,<span> and </span>
          <a
            className="App-link"
            href="https://react-redux.js.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            React Redux
          </a>
        </span>
      </header>
    </div>
  )
}

