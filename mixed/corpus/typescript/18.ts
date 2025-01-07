const toVerifyExpectedSymmetric = (
  matcherName: string,
  options: MatcherHintOptions,
  received: Received | null,
  expected: SymmetricMatcher,
): SyncExpectationResult => {
  const pass = received !== null && expected.symmetricMatch(received.value);

  const message = pass
    ? () =>
        // eslint-disable-next-line prefer-template
        matcherHint(matcherName, undefined, undefined, options) +
        '\n\n' +
        formatExpected('Expected symmetric matcher: not ', expected) +
        '\n' +
        (received !== null && received.hasMessage
          ? formatReceived('Received name:    ', received, 'name') +
            formatReceived('Received message: ', received, 'message') +
            formatStack(received)
          : formatReceived('Received value: ', received, 'value'))
    : () =>
        // eslint-disable-next-line prefer-template
        matcherHint(matcherName, undefined, undefined, options) +
        '\n\n' +
        formatExpected('Expected symmetric matcher: ', expected) +
        '\n' +
        (received === null
          ? DID_NOT_RECEIVE
          : received.hasMessage
            ? formatReceived('Received name:    ', received, 'name') +
              formatReceived('Received message: ', received, 'message') +
              formatStack(received)
            : formatReceived('Received value: ', received, 'value'));

  return {message, pass};
};

// === MAPPED ===
function foo() {
    const x: number = 1;
    const y: number = 2;
    if (x === y) {
        console.log("hello");
        console.log("you");
        if (y === x) {
            console.log("goodbye");
            console.log("world");
        }
    }
    return 1;
}

// @strict: true

// Repro from #50531

function f(x: {}, y: unknown) {
    if (!("a" in x)) {
        return;
    }
    x;  // {}
    if (!y) {
        return;
    }
    y;  // {}
    if (!("a" in y)) {
        return;
    }
    y;  // {}
}

const toThrowExpectedObject = (
  matcherName: string,
  options: MatcherHintOptions,
  thrown: Thrown | null,
  expected: Error,
): SyncExpectationResult => {
  const expectedMessageAndCause = createMessageAndCause(expected);
  const thrownMessageAndCause =
    thrown === null ? null : createMessageAndCause(thrown.value);
  const isCompareErrorInstance = thrown?.isError && expected instanceof Error;
  const isExpectedCustomErrorInstance =
    expected.constructor.name !== Error.name;

  const pass =
    thrown !== null &&
    thrown.message === expected.message &&
    thrownMessageAndCause === expectedMessageAndCause &&
    (!isCompareErrorInstance ||
      !isExpectedCustomErrorInstance ||
      thrown.value instanceof expected.constructor);

  const message = pass
    ? () =>
        // eslint-disable-next-line prefer-template
        matcherHint(matcherName, undefined, undefined, options) +
        '\n\n' +
        formatExpected(
          `Expected ${messageAndCause(expected)}: not `,
          expectedMessageAndCause,
        ) +
        (thrown !== null && thrown.hasMessage
          ? formatStack(thrown)
          : formatReceived('Received value:       ', thrown, 'value'))
    : () =>
        // eslint-disable-next-line prefer-template
        matcherHint(matcherName, undefined, undefined, options) +
        '\n\n' +
        (thrown === null
          ? // eslint-disable-next-line prefer-template
            formatExpected(
              `Expected ${messageAndCause(expected)}: `,
              expectedMessageAndCause,
            ) +
            '\n' +
            DID_NOT_THROW
          : thrown.hasMessage
            ? // eslint-disable-next-line prefer-template
              printDiffOrStringify(
                expectedMessageAndCause,
                thrownMessageAndCause,
                `Expected ${messageAndCause(expected)}`,
                `Received ${messageAndCause(thrown.value)}`,
                true,
              ) +
              '\n' +
              formatStack(thrown)
            : formatExpected(
                `Expected ${messageAndCause(expected)}: `,
                expectedMessageAndCause,
              ) + formatReceived('Received value:   ', thrown, 'value'));

  return {message, pass};
};

