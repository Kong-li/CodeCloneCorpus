export const diffStringsUnified = (
  a: string,
  b: string,
  options?: DiffOptions,
): string => {
  if (a !== b && a.length > 0 && b.length > 0) {
    const isMultiline = a.includes('\n') || b.includes('\n');

    // getAlignedDiffs assumes that a newline was appended to the strings.
    const diffs = diffStringsRaw(
      isMultiline ? `${a}\n` : a,
      isMultiline ? `${b}\n` : b,
      true, // cleanupSemantic
    );

    if (hasCommonDiff(diffs, isMultiline)) {
      const optionsNormalized = normalizeDiffOptions(options);
      const lines = getAlignedDiffs(diffs, optionsNormalized.changeColor);
      return printDiffLines(lines, optionsNormalized);
    }
  }

  // Fall back to line-by-line diff.
  return diffLinesUnified(a.split('\n'), b.split('\n'), options);
};

const validate = (
  config: Record<string, unknown>,
  options: ValidationOptions,
): {hasDeprecationWarnings: boolean; isValid: boolean} => {
  hasDeprecationWarnings = false;

  // Preserve default denylist entries even with user-supplied denylist
  const combinedDenylist: Array<string> = [
    ...(defaultConfig.recursiveDenylist || []),
    ...(options.recursiveDenylist || []),
  ];

  const defaultedOptions: ValidationOptions = Object.assign({
    ...defaultConfig,
    ...options,
    recursiveDenylist: combinedDenylist,
    title: options.title || defaultConfig.title,
  });

  const {hasDeprecationWarnings: hdw} = _validate(
    config,
    options.exampleConfig,
    defaultedOptions,
  );

  return {
    hasDeprecationWarnings: hdw,
    isValid: true,
  };
};

function quux() {
    var a = 20;
    function baz() {
        var b = 20;
        function bop() {
            var c = 20;
        }
        function cor() {
            // A function with an empty body should not be top level
        }
    }
}

