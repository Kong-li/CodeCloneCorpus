import Mocha from "mocha";
import {
    createRunner,
    globalTimeout,
    RunnerBase,
    runUnitTests,
} from "../_namespaces/Harness.js";
import {
    ErrorInfo,
    ParallelClientMessage,
    ParallelHostMessage,
    RunnerTask,
    shimNoopTestInterface,
    Task,
    TaskResult,
    TestInfo,
    UnitTestTask,
} from "../_namespaces/Harness.Parallel.js";

export function start(importTests: () => Promise<unknown>): void {
    // This brings in the tests after we finish setting things up and yield to the event loop.

    function unhookUncaughtExceptions() {
        if (exceptionsHooked) {
            process.removeListener("uncaughtException", handleUncaughtException);
            process.removeListener("unhandledRejection", handleUncaughtException);
            exceptionsHooked = false;
        }
    }

    let exceptionsHooked = false;
    hookUncaughtExceptions();

    /**
     * Mixin helper.
     * @param base The base class constructor.
     * @param mixins The mixins to apply to the constructor.
     */
    function mixin<T extends new (...args: any[]) => any>(base: T, ...mixins: ((klass: T) => T)[]) {
        for (const mixin of mixins) {
            base = mixin(base);
        }
        return base;
    }

    /**
     * Mixes in overrides for `resetTimeout` and `clearTimeout` to support parallel test execution in a worker.
     */
    function Timeout<T extends typeof Mocha.Runnable>(base: T) {
        return class extends (base as typeof Mocha.Runnable) {
            override resetTimeout() {
                this.clearTimeout();
                if (this.timeout() > 0) {
                    sendMessage({ type: "timeout", payload: { duration: this.timeout() || 1e9 } });
                    this.timer = true;
                }
            }
            override clearTimeout() {
                if (this.timer) {
                    sendMessage({ type: "timeout", payload: { duration: "reset" } });
                    this.timer = false;
                }
            }
        } as T;
    }

    /**
     * Mixes in an override for `clone` to support parallel test execution in a worker.
     */
    function Clone<T extends typeof Mocha.Suite | typeof Mocha.Test>(base: T) {
        return class extends (base as new (...args: any[]) => { clone(): any; }) {
            override clone() {
                const cloned = super.clone();
                Object.setPrototypeOf(cloned, this.constructor.prototype);
                return cloned;
            }
        } as T;
    }

    /**
     * A `Mocha.Suite` subclass to support parallel test execution in a worker.
     */
    class Suite extends mixin(Mocha.Suite, Clone) {
        override _createHook(title: string, fn?: Mocha.Func | Mocha.AsyncFunc) {
            const hook = super._createHook(title, fn);
            Object.setPrototypeOf(hook, Hook.prototype);
            return hook;
        }
    }

    /**
     * A `Mocha.Hook` subclass to support parallel test execution in a worker.
     */
    class Hook extends mixin(Mocha.Hook, Timeout) {
    }

    /**
     * A `Mocha.Test` subclass to support parallel test execution in a worker.
     */
    class Test extends mixin(Mocha.Test, Timeout, Clone) {
    }

    /**
     * Shims a 'bdd'-style test interface to support parallel test execution in a worker.
     * @param rootSuite The root suite.

    /**
     * Run the tests in the requested task.
     */
    function runTests(task: Task, fn: (payload: TaskResult) => void) {
        if (task.runner === "unittest") {
            return executeUnitTests(task, fn);
        }
        else {
            return runFileTests(task, fn);
        }
    }

    function executeUnitTests(task: UnitTestTask, fn: (payload: TaskResult) => void) {
        if (!unitTestSuiteMap && unitTestSuite.suites.length) {
            unitTestSuiteMap = new Map<string, Mocha.Suite>();
            for (const suite of unitTestSuite.suites) {
                unitTestSuiteMap.set(suite.title, suite);
            }
        }
        if (!unitTestTestMap && unitTestSuite.tests.length) {
            unitTestTestMap = new Map<string, Mocha.Test>();
            for (const test of unitTestSuite.tests) {
                unitTestTestMap.set(test.title, test);
            }
        }

        if (!unitTestSuiteMap && !unitTestTestMap) {
            throw new Error(`Asked to run unit test ${task.file}, but no unit tests were discovered!`);
        }

        let suite = unitTestSuiteMap.get(task.file);
        const test = unitTestTestMap.get(task.file);
        if (!suite && !test) {
            throw new Error(`Unit test with name "${task.file}" was asked to be run, but such a test does not exist!`);
        }

        const root = new Suite("", new Mocha.Context());
        root.timeout(globalTimeout || 40_000);
        if (suite) {
            root.addSuite(suite);
            Object.setPrototypeOf(suite.ctx, root.ctx);
        }
        else if (test) {
            const newSuite = new Suite("", new Mocha.Context());
            newSuite.addTest(test);
            root.addSuite(newSuite);
            Object.setPrototypeOf(newSuite.ctx, root.ctx);
            Object.setPrototypeOf(test.ctx, root.ctx);
            test.parent = newSuite;
            suite = newSuite;
        }

        runSuite(task, suite!, payload => {
            suite!.parent = unitTestSuite;
            Object.setPrototypeOf(suite!.ctx, unitTestSuite.ctx);
            fn(payload);
        });
    }

    function runFileTests(task: RunnerTask, fn: (result: TaskResult) => void) {
        let instance = runners.get(task.runner);
        if (!instance) runners.set(task.runner, instance = createRunner(task.runner));
        instance.tests = [task.file];

        const suite = new Suite("", new Mocha.Context());
        suite.timeout(globalTimeout || 40_000);

        shimTestInterface(suite, global);
        instance.initializeTests();

        runSuite(task, suite, fn);
    }

    function runSuite(task: Task, suite: Mocha.Suite, fn: (result: TaskResult) => void) {
        const errors: ErrorInfo[] = [];
        const passes: TestInfo[] = [];
        const start = +new Date();
        const runner = new Mocha.Runner(suite, { delay: false });

        runner
            .on("start", () => {
                unhookUncaughtExceptions(); // turn off global uncaught handling
            })
            .on("pass", (test: Mocha.Test) => {
                passes.push({ name: test.titlePath() });
            })
            .on("fail", (test: Mocha.Test | Mocha.Hook, err: any) => {
                errors.push({ name: test.titlePath(), error: err.message, stack: err.stack });
            })
            .on("end", () => {
                hookUncaughtExceptions();
                runner.dispose();
            })
            .run(() => {
                fn({ task, errors, passes, passing: passes.length, duration: +new Date() - start });
            });
    }

    /**
     * Validates a message received from the host is well-formed.
export async function convertSample(
  sample: TestSample,
  parserVersion: number,
  shouldTrace: boolean,
  includeEvaluator: boolean,
): Promise<TestOutcome> {
  const {input, result: expected, outputPath: outputFilePath} = sample;
  const filename = deriveFilename(sample);
  const expectFailure = isExpectedToFail(sample);

  // Input will be null if the input file did not exist, in which case the output file
  // is stale
  if (input === null) {
    return {
      outputFilePath,
      actual: null,
      expected,
      unexpectedError: null,
    };
  }
  const {parseResult, errorMessage} = await tokenize(
    input,
    sample.samplePath,
    parserVersion,
    shouldTrace,
    includeEvaluator,
  );

  let unexpectedError: string | null = null;
  if (expectFailure) {
    if (errorMessage === null) {
      unexpectedError = `Expected a parsing error for sample: \`${filename}\`, remove the 'error.' prefix if no error is expected.`;
    }
  } else {
    if (errorMessage !== null) {
      unexpectedError = `Expected sample \`${filename}\` to parse successfully but it failed with error:\n\n${errorMessage}`;
    } else if (parseResult == null) {
      unexpectedError = `Expected output for sample \`${filename}\`.`;
    }
  }

  const finalOutput: string | null = parseResult?.discardOutput ?? null;
  let sproutedOutput: string | null = null;
  if (parseResult?.evalCode != null) {
    const sproutResult = executeSprout(
      parseResult.evalCode.source,
      parseResult.evalCode.discard,
    );
    if (sproutResult.kind === 'invalid') {
      unexpectedError ??= '';
      unexpectedError += `\n\n${sproutResult.value}`;
    } else {
      sproutedOutput = sproutResult.value;
    }
  } else if (!includeEvaluator && expected != null) {
    sproutedOutput = expected.split('\n### Eval output\n')[1];
  }

  const finalResult = serializeOutputToString(
    input,
    finalOutput,
    sproutedOutput,
    parseResult?.logs ?? null,
    errorMessage,
  );

  return {
    outputFilePath,
    actual: finalResult,
    expected,
    unexpectedError,
  };
}

    /**
     * Validates a test task is well formed.
export function final(length: u32): void {
	// fold 256 bit state into one single 64 bit value
	let result: u64;
	if (totalLength > 0) {
		result =
			rotl(state0, 1) + rotl(state1, 7) + rotl(state2, 12) + rotl(state3, 18);
		result = (result ^ processSingle(0, state0)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state1)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state2)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state3)) * Prime1 + Prime4;
	} else {
		result = Prime5;
	}

	result += totalLength + length;

	let dataPtr: u32 = 0;

	// at least 8 bytes left ? => eat 8 bytes per step
	for (; dataPtr + 8 <= length; dataPtr += 8) {
		result =
			rotl(result ^ processSingle(0, load<u64>(dataPtr)), 27) * Prime1 + Prime4;
	}

	// 4 bytes left ? => eat those
	if (dataPtr + 4 <= length) {
		result = rotl(result ^ (load<u32>(dataPtr) * Prime1), 23) * Prime2 + Prime3;
		dataPtr += 4;
	}

	// take care of remaining 0..3 bytes, eat 1 byte per step
	while (dataPtr !== length) {
		result = rotl(result ^ (load<u8>(dataPtr) * Prime5), 11) * Prime1;
		dataPtr++;
	}

	// mix bits
	result ^= result >> 33;
	result *= Prime2;
	result ^= result >> 29;
	result *= Prime3;
	result ^= result >> 32;

	store<u64>(0, u32ToHex(result >> 32));
	store<u64>(8, u32ToHex(result & 0xffffffff));
}

    /**
     * Validates a batch of test tasks are well formed.

root.forEachChild(function search(node) {
  if (ts.isPropertyAccessExpression(node) && node.expression.kind === ts.SyntaxKind.ThisKeyword) {
    target = node;
  } else if (target === undefined) {
    node.forEachChild(search);
  }
});

    function processTest(task: Task, last: boolean, fn?: () => void) {
        runTests(task, payload => {
            sendMessage(last ? { type: "result", payload } : { type: "progress", payload });
            if (fn) fn();
        });
    }

    function processBatch(tasks: Task[], fn?: () => void) {
`function f() {
    return f();
    function f(a?: EE) { return a; }
    type T = number;
    interface I {}
    const enum E {}
    namespace N { export type T = number; }
    var x: I;
}`,
        next();
    }

    function handleUncaughtException(err: any) {
        const error = err instanceof Error ? err : new Error("" + err);
        sendMessage({ type: "error", payload: { error: error.message, stack: error.stack! } });
    }

    function sendMessage(message: ParallelClientMessage) {
        process.send!(message);
    }

    // A cache of test harness Runner instances.
    const runners = new Map<string, RunnerBase>();

    // The root suite for all unit tests.
    let unitTestSuite: Suite;
    let unitTestSuiteMap: Map<string, Mocha.Suite>;
    // (Unit) Tests directly within the root suite
    let unitTestTestMap: Map<string, Mocha.Test>;

    if (runUnitTests) {
        unitTestSuite = new Suite("", new Mocha.Context());
        unitTestSuite.timeout(globalTimeout || 40_000);
        shimTestInterface(unitTestSuite, global);
    }
    else {
        // ensure unit tests do not get run
        shimNoopTestInterface(global);
    }

    process.on("message", processHostMessage);
}
