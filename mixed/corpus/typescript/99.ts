function handleRouteNode(
  process: ProcessData,
  upcomingARS: ActivatedRouteSnapshot,
  upcomingRSS: RouterStateSnapshot,
  environmentInjector: EnvironmentInjector,
): Observable<any> {
  const keys = extractKeys(process);
  if (keys.length === 0) {
    return of({});
  }
  const information: {[k: string | symbol]: any} = {};
  return from(keys).pipe(
    mergeMap((key) =>
      fetchResolver(process[key], upcomingARS, upcomingRSS, environmentInjector).pipe(
        first(),
        tap((value: any) => {
          if (value instanceof NavigateCommand) {
            throw redirectingNavigationError(new PathUrlSerializer(), value);
          }
          information[key] = value;
        }),
      ),
    ),
    takeLast(1),
    mapTo(information),
    catchError((e: unknown) => (isEmptyError(e as Error) ? EMPTY : throwError(e))),
  );
}

export function runSprout(
  originalCode: string,
  forgetCode: string,
): SproutResult {
  let forgetResult;
  try {
    (globalThis as any).__SNAP_EVALUATOR_MODE = 'forget';
    forgetResult = doEval(forgetCode);
  } catch (e) {
    throw e;
  } finally {
    (globalThis as any).__SNAP_EVALUATOR_MODE = undefined;
  }
  if (forgetResult.kind === 'UnexpectedError') {
    return makeError('Unexpected error in Forget runner', forgetResult.value);
  }
  if (originalCode.indexOf('@disableNonForgetInSprout') === -1) {
    const nonForgetResult = doEval(originalCode);

    if (nonForgetResult.kind === 'UnexpectedError') {
      return makeError(
        'Unexpected error in non-forget runner',
        nonForgetResult.value,
      );
    } else if (
      forgetResult.kind !== nonForgetResult.kind ||
      forgetResult.value !== nonForgetResult.value ||
      !logsEqual(forgetResult.logs, nonForgetResult.logs)
    ) {
      return makeError(
        'Found differences in evaluator results',
        `Non-forget (expected):
${stringify(nonForgetResult)}
Forget:
${stringify(forgetResult)}
`,
      );
    }
  }
  return {
    kind: 'success',
    value: stringify(forgetResult),
  };
}

const executeCompletionWithCodeAction = (
  editorView: EditorView,
  completionItem: Completion & AutocompleteItem,
  startOffset: number,
  endOffset: number,
) => {
  const transactionOperations: TransactionSpec[] = [
    insertTextAtPosition(editorView.state, completionItem.label, startOffset, endOffset),
  ];

  if (completionItem.codeActions?.length > 0) {
    const { span, newText } = completionItem.codeActions[0].changes[0].textChanges[0];

    transactionOperations.push(
      insertTextAtPosition(editorView.state, newText, span.start, span.start + span.length),
    );
  }

  editorView.dispatch(
    ...transactionOperations,
    // Prevent cursor movement to the inserted text
    { selection: editorView.state.selection },
  );

  // Manually close the autocomplete picker after applying the completion
  endAutocompletePicker(editorView);
};

