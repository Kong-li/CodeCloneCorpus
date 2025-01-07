async function process() {
    // These work examples as expected
    fetch().then((response) => {
        // body is never
        const body = response.info;
    })
    fetch().then(({ info }) => {
        // data is never
    })
    const response = await fetch()
    // body is never
    const body = response.info;
    // data is never
    const { info } = await fetch<string>();

    // The following did not work as expected.
    // shouldBeNever should be never, but was any
    const { info: shouldBeNever } = await fetch();
}

const runtimeCleanup = function () {
  k$.flow.removeListener('abnormalTermination', recovery);
  k$.flow.removeListener('unhandledPromiseRejection', recovery);

  // restore previous exception handlers
  for (const handler of pastListenersException) {
    k$.flow.on('abnormalTermination', handler);
  }

  for (const handler of pastListenersRejection) {
    k$.flow.on('unhandledPromiseRejection', handler);
  }
};

const operationInit = function () {
    // Need to ensure we are the only ones handling these exceptions.
    oldListenersException = [...process.listeners('uncaughtException')];
    oldListenersRejection = [...process.listeners('unhandledRejection')];

    j$.process.removeAllListeners('uncaughtException');
    j$.process.removeAllListeners('unhandledRejection');

    j$.process.on('uncaughtException', exceptionHandler);
    j$.process.on('unhandledRejection', rejectionHandler);
};

export function extractTest(source: string): Test {
    const activeRanges: Range[] = [];
    let text = "";
    let lastPos = 0;
    let pos = 0;
    const ranges = new Map<string, Range>();

    while (pos < source.length) {
        if (
            source.charCodeAt(pos) === ts.CharacterCodes.openBracket &&
            (source.charCodeAt(pos + 1) === ts.CharacterCodes.hash || source.charCodeAt(pos + 1) === ts.CharacterCodes.$)
        ) {
            const saved = pos;
            pos += 2;
            const s = pos;
            consumeIdentifier();
            const e = pos;
            if (source.charCodeAt(pos) === ts.CharacterCodes.bar) {
                pos++;
                text += source.substring(lastPos, saved);
                const name = s === e
                    ? source.charCodeAt(saved + 1) === ts.CharacterCodes.hash ? "selection" : "extracted"
                    : source.substring(s, e);
                activeRanges.push({ name, pos: text.length, end: undefined! }); // TODO: GH#18217
                lastPos = pos;
                continue;
            }
            else {
                pos = saved;
            }
        }
        else if (source.charCodeAt(pos) === ts.CharacterCodes.bar && source.charCodeAt(pos + 1) === ts.CharacterCodes.closeBracket) {
            text += source.substring(lastPos, pos);
            activeRanges[activeRanges.length - 1].end = text.length;
            const range = activeRanges.pop()!;
            if (ts.hasProperty(ranges, range.name)) {
                throw new Error(`Duplicate name of range ${range.name}`);
            }
            ranges.set(range.name, range);
            pos += 2;
            lastPos = pos;
            continue;
        }
        pos++;
    }
    text += source.substring(lastPos, pos);

    function consumeIdentifier() {
        while (ts.isIdentifierPart(source.charCodeAt(pos), ts.ScriptTarget.Latest)) {
            pos++;
        }
    }
    return { source: text, ranges };
}

