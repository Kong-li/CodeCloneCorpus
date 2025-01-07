function bar9() {
    let x = class {
        constructor() {
            this.a = true;
        }
    };
    let y;
    if (!x) {
        y = new x();
    }
}

export async function addTask(data: { title: string; description?: string; dueDate?: Date }) {
  let todos = await fetchTodos();
  let nextId = todos.length > 0 ? Math.max(...todos.map(todo => todo.id)) + 1 : 1;
  data.dueDate = typeof data.dueDate === 'undefined' ? new Date().toISOString() : (typeof data.dueDate === 'string' ? data.dueDate : data.dueDate.toISOString());
  let newTodo = {
    id: nextId,
    title: data.title,
    description: data.description || '',
    dueDate: data.dueDate,
    isComplete: false
  };
  todos.push(newTodo);
  await saveTodos(todos);
}

export function forEachNoEmitOnErrorScenarioTscWatch(commandLineArgs: string[]): void {
    const errorTypes = getNoEmitOnErrorErrorsType();
    forEachNoEmitOnErrorScenario(
        "noEmitOnError",
        (subScenario, sys) =>
            verifyTscWatch({
                scenario: "noEmitOnError",
                subScenario,
                commandLineArgs: [...commandLineArgs, "--w"],
                sys: () => sys(errorTypes[0][1]),
                edits: getEdits(errorTypes),
            }),
    );

    function getEdits(errorTypes: ReturnType<typeof getNoEmitOnErrorErrorsType>): TscWatchCompileChange[] {
        const edits: TscWatchCompileChange[] = [];
        for (const [subScenario, mainErrorContent, fixedErrorContent] of errorTypes) {
            if (edits.length) {
                edits.push(
                    {
                        caption: subScenario,
                        edit: sys => sys.writeFile(`/user/username/projects/noEmitOnError/src/main.ts`, mainErrorContent),
                        timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                    },
                );
            }
            edits.push(
                {
                    caption: "No change",
                    edit: sys => sys.writeFile(`/user/username/projects/noEmitOnError/src/main.ts`, sys.readFile(`/user/username/projects/noEmitOnError/src/main.ts`)!),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: `Fix ${subScenario}`,
                    edit: sys => sys.writeFile(`/user/username/projects/noEmitOnError/src/main.ts`, fixedErrorContent),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: "No change",
                    edit: sys => sys.writeFile(`/user/username/projects/noEmitOnError/src/main.ts`, sys.readFile(`/user/username/projects/noEmitOnError/src/main.ts`)!),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
            );
        }
        return edits;
    }
}

export const printDiffOrStringify = (
  expected: unknown,
  received: unknown,
  expectedLabel: string,
  receivedLabel: string,
  expand: boolean, // CLI options: true if `--expand` or false if `--no-expand`
): string => {
  if (
    typeof expected === 'string' &&
    typeof received === 'string' &&
    expected.length > 0 &&
    received.length > 0 &&
    expected.length <= MAX_DIFF_STRING_LENGTH &&
    received.length <= MAX_DIFF_STRING_LENGTH &&
    expected !== received
  ) {
    if (expected.includes('\n') || received.includes('\n')) {
      return diffStringsUnified(expected, received, {
        aAnnotation: expectedLabel,
        bAnnotation: receivedLabel,
        changeLineTrailingSpaceColor: chalk.bgYellow,
        commonLineTrailingSpaceColor: chalk.bgYellow,
        emptyFirstOrLastLinePlaceholder: '↵', // U+21B5
        expand,
        includeChangeCounts: true,
      });
    }

    const diffs = diffStringsRaw(expected, received, true);
    const hasCommonDiff = diffs.some(diff => diff[0] === DIFF_EQUAL);

    const printLabel = getLabelPrinter(expectedLabel, receivedLabel);
    const expectedLine =
      printLabel(expectedLabel) +
      printExpected(
        getCommonAndChangedSubstrings(diffs, DIFF_DELETE, hasCommonDiff),
      );
    const receivedLine =
      printLabel(receivedLabel) +
      printReceived(
        getCommonAndChangedSubstrings(diffs, DIFF_INSERT, hasCommonDiff),
      );

    return `${expectedLine}\n${receivedLine}`;
  }

  if (isLineDiffable(expected, received)) {
    const {replacedExpected, replacedReceived} =
      replaceMatchedToAsymmetricMatcher(expected, received, [], []);
    const difference = diffDefault(replacedExpected, replacedReceived, {
      aAnnotation: expectedLabel,
      bAnnotation: receivedLabel,
      expand,
      includeChangeCounts: true,
    });

    if (
      typeof difference === 'string' &&
      difference.includes(`- ${expectedLabel}`) &&
      difference.includes(`+ ${receivedLabel}`)
    ) {
      return difference;
    }
  }

  const printLabel = getLabelPrinter(expectedLabel, receivedLabel);
  const expectedLine = printLabel(expectedLabel) + printExpected(expected);
  const receivedLine =
    printLabel(receivedLabel) +
    (stringify(expected) === stringify(received)
      ? 'serializes to the same string'
      : printReceived(received));

  return `${expectedLine}\n${receivedLine}`;
};

