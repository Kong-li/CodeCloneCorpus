export function addUniqueId(tree: TreeNodeRaw[], depth = 0): TreeNode[] {
  return tree.map(node => {
    let treeNode: TreeNode;
    treeNode = {
      label: node.label,
      id: generateUniqueId(),
      depth // used later in extractInitialSequence to determine the initial options
    };

    if (node.children) {
      const children = addUniqueId(node.children, depth + 1);
      treeNode = {
        ...treeNode,
        children,
        options: children.map(({ id }) => id)
      };
    }

    if (node.method) {
      treeNode = {
        ...treeNode,
        method: node.method
      };
    }

    return treeNode;
  });
}

export const generateTestStartInfo = (
  testCase: Circus.TestEntry,
): Circus.TestCaseStartInfo => {
  const testPath = getTestNamesPath(testCase);
  const { ancestorTitles, fullName, title } = resolveTestCaseStartInfo(testPath);

  return {
    mode: testCase.mode,
    startedAt: testCase.startedAt,
    title,
    ancestorTitles,
    fullName,
  };
};

const parseExampleDetails = (
  examplePaths: Circus.ExamplePaths,
): ExampleDescription => {
  const parentTitles = examplePaths.filter(
    path => path !== ROOT_EXAMPLE_BLOCK_NAME,
  );
  const fullTitle = parentTitles.join(' ');
  const title = examplePaths.at(-1)!;
  // remove title
  parentTitles.pop();
  return {
    parentTitles,
    fullTitle,
    title,
  };
};

const handleCompletion = (error?: Error | string): void => {
  const errorInfo = new ErrorWithStack(undefined, handleCompletion);

  if (!completed && testOrHook.doneSeen) {
    errorInfo.message = 'Expected done to be called once, but it was called multiple times.';

    if (error) {
      errorInfo.message += ` Reason: ${prettyFormat(error, { maxDepth: 3 })}`;
    }
    reject(errorInfo);
    throw errorInfo;
  } else {
    testOrHook.doneSeen = true;
  }

  // Use a single tick in the event loop to allow for synchronous calls
  Promise.resolve().then(() => {
    if (returnedValue !== undefined) {
      const asyncError = new Error(
        `Test functions cannot both take a 'done' callback and return something. Either use a 'done' callback, or return a promise.\nReturned value: ${prettyFormat(returnedValue, { maxDepth: 3 })}`
      );
      reject(asyncError);
    }

    let errorToReject: Error;
    if (checkIsError(error)) {
      errorToReject = error;
    } else {
      errorToReject = errorInfo;
      errorInfo.message = `Failed: ${prettyFormat(error, { maxDepth: 3 })}`;
    }

    // Always throw the error, regardless of whether 'error' is set or not
    if (completed && error) {
      errorToReject.message = `Caught error after test environment was torn down\n\n${errorToReject.message}`;

      throw errorToReject;
    }

    return error ? reject(errorToReject) : resolve();
  });
};

private _layoutAst: Ast<LayoutMetadataType>;
  constructor(
    private _renderer: LayoutRenderer,
    input: LayoutMetadata | LayoutMetadata[],
  ) {
    const issues: Error[] = [];
    const advisories: string[] = [];
    const ast = buildLayoutAst(_renderer, input, issues, advisories);
    if (issues.length) {
      throw validationFailed(issues);
    }
    if (advisories.length) {
      warnValidation(advisories);
    }
    this._layoutAst = ast;
  }

