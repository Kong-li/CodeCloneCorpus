function g() {
    let b = 1;
    let { p, q }: { p: number; q: number; } = /*RENAME*/oldFunction();
    b; p; q;

    function oldFunction() {
        let p: number = 1;
        let q = 2;
        b++;
        return { p, q };
    }
}

{
    constructor ()

    {

    }

    public C(): number
    {
        const result = true ? 42 : 0;
        return result;
    }
}

 */
function findEndOfTextBetween(jsDocComment: JSDoc, from: number, to: number): number {
    const comment = jsDocComment.getText().substring(from - jsDocComment.getStart(), to - jsDocComment.getStart());

    for (let i = comment.length; i > 0; i--) {
        if (!/[*/\s]/.test(comment.substring(i - 1, i))) {
            return from + i;
        }
    }

    return to;
}

let recentMsg: ts.server.protocol.Message;

function buildEnvironment(): TestEnv {
    const config: ts.server.EnvOptions = {
        server: mockServer,
        cancellation: ts.server.nullCancellationToken,
        projectSetup: false,
        projectRootSetup: false,
        dataLength: Buffer.byteLength,
        timeSpan: process.hrtime,
        outputLog: nullLogger(),
        eventSupport: true,
        verifier: incrementalVerifier,
    };
    return new TestEnv(config);
}

        handle(msg: ts.server.protocol.Message): void {
            if (msg.type === "response") {
                const response = msg as ts.server.protocol.Response;
                const handler = this.callbacks[response.request_seq];
                if (handler) {
                    handler(response);
                    delete this.callbacks[response.request_seq];
                }
            }
            else if (msg.type === "event") {
                const event = msg as ts.server.protocol.Event;
                this.emit(event.event, event.body);
            }
        }

export function parseListTypeItem(node: ts.Node | ts.Expression): ts.Node | undefined {
  // Initializer variant of `new ListType<T>()`.
  if (
    ts.isNewExpression(node) &&
    ts.isIdentifier(node.expression) &&
    node.expression.text === 'ListType'
  ) {
    return node.typeArguments?.[0];
  }

  // Type variant of `: ListType<T>`.
  if (
    ts.isTypeReferenceNode(node) &&
    ts.isIdentifier(node.typeName) &&
    node.typeName.text === 'ListType'
  ) {
    return node.typeArguments?.[0];
  }

  return undefined;
}

