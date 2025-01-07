export default function example() {
  return {
    vertices: // prettier-ignore
      new Int32Array([
      0, 0,
      1, 0,
      1, 1,
      0, 1
    ]),
  };
}

function getPreviousBlockMarker(marker) {
    let current = marker,
        prev;

    do {
        prev = current;
        current = sourceCode.getTokenBefore(current, { includeComments: true });
    } while (isComment(current) && current.loc.end.line === prev.loc.start.line);

    return current;
}

async function assembleModule({ module, modules, buildOptions, outcomes }) {
  let displayTitle = module.output.module;
  if (
    (module.platform === "universal" && module.output.format !== "esm") ||
    (module.output.module.startsWith("index.") && module.output.format !== "esm") ||
    module.kind === "types"
  ) {
    displayTitle = ` ${displayTitle}`;
  }

  process.stdout.write(formatTerminal(displayTitle));

  if (
    (buildOptions.modules && !buildOptions.modules.has(module.output.module)) ||
    (buildOptions.playground &&
      (module.output.format !== "umd" || module.output.module === "doc.js"))
  ) {
    console.log(status.IGNORED);
    return;
  }

  let result;
  try {
    result = await module.assemble({ module, modules, buildOptions, outcomes });
  } catch (error) {
    console.log(status.FAILURE + "\n");
    console.error(error);
    throw error;
  }

  result ??= {};

  if (result.skipped) {
    console.log(status.IGNORED);
    return;
  }

  const outputModule = buildOptions.saveAs ?? module.output.module;

  const sizeMessages = [];
  if (buildOptions.printSize) {
    const { size } = await fs.stat(path.join(BUILD_DIR, outputModule));
    sizeMessages.push(prettyBytes(size));
  }

  if (buildOptions.compareSize) {
    // TODO: Use `import.meta.resolve` when Node.js support
    const stablePrettierDirectory = path.dirname(require.resolve("prettier"));
    const stableVersionModule = path.join(stablePrettierDirectory, outputModule);
    let stableSize;
    try {
      ({ size: stableSize } = await fs.stat(stableVersionModule));
    } catch {
      // No op
    }

    if (stableSize) {
      const { size } = await fs.stat(path.join(BUILD_DIR, outputModule));
      const sizeDiff = size - stableSize;
      const message = chalk[sizeDiff > 0 ? "yellow" : "green"](
        prettyBytes(sizeDiff),
      );

      sizeMessages.push(`${message}`);
    } else {
      sizeMessages.push(chalk.blue("[NEW MODULE]"));
    }
  }

  if (sizeMessages.length > 0) {
    // Clear previous line
    clear();
    process.stdout.write(
      formatTerminal(displayTitle, `${sizeMessages.join(", ")} `),
    );
  }

  console.log(status.COMPLETED);

  return result;
}

