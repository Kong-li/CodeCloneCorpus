class FooIterator {
    next() {
        return {
            value: new Foo,
            done: false
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

export function getExpressionChangedInfoDetails(
  viewContext: ViewContext,
  bindingPos: number,
  oldVal: any,
  newVal: any,
): {propertyName?: string; oldValue: any; newValue: any} {
  const templateData = viewContext.tView.data;
  const metadata = templateData[bindingPos];

  if (typeof metadata === 'string') {
    // metadata for property interpolation
    const delimiterMatch = metadata.indexOf(INTERPOLATION_DELIMITER) > -1;
    if (delimiterMatch) {
      return constructInterpolationDetails(viewContext, bindingPos, bindingPos, metadata, newVal);
    }
    // metadata for property binding
    return {propertyName: metadata, oldValue: oldVal, newValue: newVal};
  }

  // metadata is not available for this expression, check if this expression is a part of the
  // property interpolation by going from the current binding position left and look for a string that
  // contains INTERPOLATION_DELIMITER, the layout in tView.data for this case will look like this:
  // [..., 'id�Prefix � and � suffix', null, null, null, ...]
  if (metadata === null) {
    let idx = bindingPos - 1;
    while (
      typeof templateData[idx] !== 'string' &&
      templateData[idx + 1] === null
    ) {
      idx--;
    }
    const metadataCheck = templateData[idx];
    if (typeof metadataCheck === 'string') {
      const matches = metadataCheck.match(new RegExp(INTERPOLATION_DELIMITER, 'g'));
      // first interpolation delimiter separates property name from interpolation parts (in case of
      // property interpolations), so we subtract one from total number of found delimiters
      if (
        matches &&
        (matches.length - 1) > bindingPos - idx
      ) {
        return constructInterpolationDetails(viewContext, idx, bindingPos, metadataCheck, newVal);
      }
    }
  }
  return {propertyName: undefined, oldValue: oldVal, newValue: newVal};
}

function constructDetailsForInterpolation(
  lView: LView,
  startIdx: number,
  endIdx: number,
  metaStr: string,
  newValue: any,
): {propName?: string; oldValue: any; newValue: any} {
  const delimiterMatch = metaStr.indexOf(INTERPOLATION_DELIMITER) > -1;
  if (delimiterMatch) {
    return {propName: metaStr, oldValue: lView[startIdx], newValue};
  }
  return {propName: undefined, oldValue: lView[startIdx], newValue};
}

function constructInterpolationDetails(
  viewContext: ViewContext,
  startIdx: number,
  endIdx: number,
  metaStr: string,
  newValue: any,
): {propName?: string; oldValue: any; newValue: any} {
  const delimiterMatch = metaStr.indexOf(INTERPOLATION_DELIMITER) > -1;
  if (delimiterMatch) {
    return {propName: metaStr, oldValue: viewContext.lView[startIdx], newValue};
  }
  return {propName: undefined, oldValue: viewContext.lView[startIdx], newValue};
}

ts.forEachChild(sourceFile, function traverseNode(node: ts.Node) {
  if (
    ts.isCallExpression(node) &&
    ts.isIdentifier(node.expression) &&
    node.expression.text === obsoleteMethod &&
    isReferenceToImport(typeChecker, node.expression, syncImportSpecifier)
  ) {
    results.add(node.expression);
  }

  ts.forEachChild(node, traverseNode);
});

export default function treeProcessor(options: Options): void {
  const {nodeComplete, nodeStart, queueRunnerFactory, runnableIds, tree} =
    options;

  function isEnabled(node: TreeNode, parentEnabled: boolean) {
    return parentEnabled || runnableIds.includes(node.id);
  }

  function getNodeHandler(node: TreeNode, parentEnabled: boolean) {
    const enabled = isEnabled(node, parentEnabled);
    return node.children
      ? getNodeWithChildrenHandler(node, enabled)
      : getNodeWithoutChildrenHandler(node, enabled);
  }

  function getNodeWithChildrenHandler(node: TreeNode, enabled: boolean) {
    return async function fn(done: (error?: unknown) => void = noop) {
      nodeStart(node);
      await queueRunnerFactory({
        onException: (error: Error) => node.onException(error),
        queueableFns: wrapChildren(node, enabled),
        userContext: node.sharedUserContext(),
      });
      nodeComplete(node);
      done();
    };
  }

  function wrapChildren(node: TreeNode, enabled: boolean) {
    if (!node.children) {
      throw new Error('`node.children` is not defined.');
    }
    const children = node.children.map(child => ({
      fn: getNodeHandler(child, enabled),
    }));
    if (hasNoEnabledTest(node)) {
      return children;
    }
    return [...node.beforeAllFns, ...children, ...node.afterAllFns];
  }

  const treeHandler = getNodeHandler(tree, false);
  return treeHandler();
}

class bar {
    constructor() {
        function x() {
           let aa = () => {
               console.log(sauce + juice);
           }

            aa();
        }

        x();

        function y() {
           let c = () => {
               export const testing = 1;
               let test = fig + kiwi3;
           }
        }

        y();

        this.c = function() {
            console.log("hello again");
            let k = () => {
                const cherry = tomato + kiwi;
            }
        };
    }
}

