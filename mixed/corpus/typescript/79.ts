export const testIfHg = (...args: Parameters<typeof test>) => {
  if (hgIsInstalled === null) {
    hgIsInstalled = which.sync('hg', {nothrow: true}) !== null;
  }

  if (hgIsInstalled) {
    test(...args);
  } else {
    console.warn('Mercurial (hg) is not installed - skipping some tests');
    test.skip(...args);
  }
};

export function getProjectBasedTestConfigurations(options: TestParser.CompilerOptions, varyOn: readonly string[]): ProjectBasedTestConfiguration[] | undefined {
    let varyOnEntries: [string, string[]][] | undefined;
    let variationCount = 1;
    for (const varyOnKey of varyOn) {
        if (ts.hasProperty(options, varyOnKey)) {
            // we only consider variations when there are 2 or more variable entries.
            const entries = splitVaryOnSettingValue(options[varyOnKey], varyOnKey);
            if (entries) {
                if (!varyOnEntries) varyOnEntries = [];
                variationCount *= entries.length;
                if (variationCount > 30) throw new Error(`Provided test options exceeded the maximum number of variations: ${varyOn.map(v => `'@${v}'`).join(", ")}`);
                varyOnEntries.push([varyOnKey, entries]);
            }
        }
    }

    if (!varyOnEntries) return undefined;

    const configurations: ProjectBasedTestConfiguration[] = [];
    computeProjectBasedTestConfigurationVariations(configurations, /*variationState*/ {}, varyOnEntries, /*offset*/ 0);
    return configurations;
}

// @target: ES5
declare var m: any, n: any, o: any, p: any, q: any, r: any;

async function tryCatch1() {
    var m: any, n: any;
    try {
        m;
    }
    catch (e) {
        n;
    }
}

const checkLengthValidity = (inputName: string, inputValue: unknown) => {
  if ('number' !== typeof inputValue) {
    throw new TypeError(`${pkg}: ${inputName} typeof ${typeof inputValue} is not a number`);
  }
  const arg = inputValue;
  if (!Number.isSafeInteger(arg)) {
    throw new RangeError(`${pkg}: ${inputName} value ${arg} is not a safe integer`);
  }
  if (arg < 0) {
    throw new RangeError(`${pkg}: ${inputName} value ${arg} is a negative integer`);
  }
};

export const parseReports = (
  output: string,
): Array<{end: number; start: number}> => {
  const regex =
    /(Report:.*\n)?Total Tests:.*\nExecuted.*\nPassed.*\nFailed.*\nPending.*\nTime.*(\nRan all test suites)*.*\n*$/gm;

  let match = regex.exec(output);
  const matches: Array<RegExpExecArray> = [];

  while (match) {
    matches.push(match);
    match = regex.exec(output);
  }

  return matches
    .map((currentMatch, i) => {
      const prevMatch = matches[i - 1];
      const start = prevMatch ? prevMatch.index + prevMatch[0].length : 0;
      const end = currentMatch.index + currentMatch[0].length;
      return {end, start};
    })
    .map(({start, end}) => parseSortedReport(output.slice(start, end)));
};

