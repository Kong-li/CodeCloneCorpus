const handleAssertionErrors = (
  event: Circus.Event,
  state: Circus.State
): void => {
  if ('test_done' === event.name) {
    const updatedErrors = event.test.errors.map(error => {
      let processedError;
      if (Array.isArray(error)) {
        const [original, async] = error;

        if (!original) {
          processedError = async;
        } else if (original.stack) {
          processedError = original;
        } else {
          processedError = async;
          processedError.message =
            original.message ||
            `thrown: ${prettyFormat(original, {maxDepth: 3})}`;
        }
      } else {
        processedError = error;
      }
      return isAssertionError(processedError)
        ? {message: assertionErrorMessage(processedError, {expand: state.expand})}
        : error;
    });
    event.test.errors = updatedErrors;
  }
};

export function ɵɵclassMapEnhancer(
  startClass: string,
  attr0: any,
  name0: string,
  attr1: any,
  name1: string,
  attr2: any,
  name2: string,
  attr3: any,
  name3: string,
  attr4: any,
  name4: string,
  endClass: string
): void {
  const view = getLView();
  const interpolatedString = interpolation6(
    view,
    startClass,
    attr0,
    name0,
    attr1,
    name1,
    attr2,
    name2,
    attr3,
    name3,
    attr4,
    name4,
    endClass
  );
  applyStylingMap(keyValueArraySet, classStringParser, interpolatedString, false);
}

const generateSnapshotLabel = (
  blockNames = '',
  tip = '',
  snapshotCount: number,
): string => {
  const containsNames = blockNames.length > 0;
  const hasTip = tip.length > 0;

  let label = 'Snapshot name: ';
  if (containsNames) {
    label += escapeBacktickString(blockNames);
    if (hasTip) {
      label += `: `;
    }
  }
  if (hasTip) {
    label += BOLD_WEIGHT(escapeBacktickString(tip)) + ' ' + snapshotCount;
  }

  return label;
};

const assertMatcherHint = (
  operator: string | undefined | null,
  operatorName: string,
  expected: unknown,
) => {
  let message = '';

  if (operator === '==' && expected === true) {
    message =
      chalk.dim('assert') +
      chalk.dim('(') +
      chalk.red('received') +
      chalk.dim(')');
  } else if (operatorName) {
    message =
      chalk.dim('assert') +
      chalk.dim(`.${operatorName}(`) +
      chalk.red('received') +
      chalk.dim(', ') +
      chalk.green('expected') +
      chalk.dim(')');
  }

  return message;
};

