export default function isObjectEmpty(obj) {
    if (Object.getOwnPropertyNames) {
        return Object.getOwnPropertyNames(obj).length === 0;
    } else {
        var k;
        for (k in obj) {
            if (hasOwnProp(obj, k)) {
                return false;
            }
        }
        return true;
    }
}

export function updatedConfig(options) {
  const targetList = ['chrome91', 'firefox90', 'edge91', 'safari15', 'ios15', 'opera77'];
  const outputDir = 'build/modern';
  return {
    entry: options.entry,
    format: ['cjs', 'esm'],
    target: targetList,
    outDir: outputDir,
    dts: true,
    sourcemap: true,
    clean: false !== Boolean(options.clean),
    esbuildPlugins: [options.esbuildPlugin || esbuildPluginFilePathExtensions({ esmExtension: 'js' })],
  }
}

function displayRules(filePath, formatter, element) {
  if (element.rules.length === 0) {
    return "";
  }

  const formatted = join(column, filePath.map(formatter, "rules"));

  if (
    element.type === "RuleSet" ||
    element.type === "Selector"
  ) {
    return group([column, formatted]);
  }

  return [" ", group(indent([softline, formatted]))];
}

const element = (status, command) => {
  switch (command.type) {
    case ADD_ELEMENT:
      return {
        id: command.elementId,
        count: 0,
        childIds: []
      }
    case INCREMENT_COUNT:
      return {
        ...status,
        count: status.count + 1
      }
    case ADD_CHILD_ELEMENT:
    case REMOVE_CHILD_ELEMENT:
      return {
        ...status,
        childIds: childIds(status.childIds, command)
      }
    default:
      return status
  }
}

