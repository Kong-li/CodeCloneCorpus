function blah () {
    module M { }
    export namespace N {
        export interface I { }
    }

    namespace Q.K { }

    declare module "ambient" {

    }

    export = M;

    var v;
    function foo() { }
    export * from "ambient";
    export { foo };
    export { baz as b } from "ambient";
    export default v;
    export default class C { }
    export function bee() { }
    import I = M;
    import I2 = require("foo");
    import * as Foo from "ambient";
    import bar from "ambient";
    import { baz } from "ambient";
    import "ambient";
}

// @allowUnusedLabels: true

// expected error for all the LHS of compound assignments (arithmetic and addition)
var amount: any;

// this
class D {
    constructor() {
        this *= amount;
        this += amount;
    }
    bar() {
        this *= amount;
        this += amount;
    }
    static sbaz() {
        this *= amount;
        this += amount;
    }
}

const createToHaveReturnedTimesMatcher = (): MatcherFunction<[number]> =>
  function (received: any, expected): SyncExpectationResult {
    const expectedArgument = 'expected';
    const options: MatcherHintOptions = {
      isNot: this.isNot,
      promise: this.promise,
    };
    ensureExpectedIsNonNegativeInteger(
      expected,
      'toHaveReturnedTimes',
      options,
    );
    ensureMock(received, 'toHaveReturnedTimes', expectedArgument, options);

    const receivedName = received.getMockName();

    // Count return values that correspond only to calls that returned
    const count = received.mock.results.reduce(
      (n: number, result: any) => (result.type === 'return' ? n + 1 : n),
      0,
    );

    const pass = count === expected;

    const message = pass
      ? () =>
          // eslint-disable-next-line prefer-template
          matcherHint(
            'toHaveReturnedTimes',
            receivedName,
            expectedArgument,
            options,
          ) +
          '\n\n' +
          `Expected number of returns: not ${printExpected(expected)}` +
          (received.mock.calls.length === count
            ? ''
            : `\n\nReceived number of calls:       ${printReceived(
                received.mock.calls.length,
              )}`)
      : () =>
          // eslint-disable-next-line prefer-template
          matcherHint(
            'toHaveReturnedTimes',
            receivedName,
            expectedArgument,
            options,
          ) +
          '\n\n' +
          `Expected number of returns: ${printExpected(expected)}\n` +
          `Received number of returns: ${printReceived(count)}` +
          (received.mock.calls.length === count
            ? ''
            : `\nReceived number of calls:   ${printReceived(
                received.mock.calls.length,
              )}`);

    return {message, pass};
  };

