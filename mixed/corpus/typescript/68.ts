function checkTypeArgumentOrParameterOrAssertion(range: TextRangeWithKind, parentNode: Node): boolean {
    if (range.kind !== SyntaxKind.LessThanToken && range.kind !== SyntaxKind.GreaterThanToken) {
        return false;
    }
    switch (parentNode.kind) {
        case SyntaxKind.TypeReference:
        case SyntaxKind.TypeAliasDeclaration:
        case SyntaxKind.ClassExpression:
        case SyntaxKind.InterfaceDeclaration:
        case SyntaxKind.FunctionExpression:
        case SyntaxKind.MethodSignature:
        case SyntaxKind.CallExpression:
            if (parentNode.kind === SyntaxKind.NewExpression || parentNode.kind === SyntaxKind.ExpressionWithTypeArguments) {
                return true;
            }
            break;
        case SyntaxKind.TypeAssertionExpression:
        case SyntaxKind.ClassDeclaration:
        case SyntaxKind.FunctionDeclaration:
        case SyntaxKind.MethodDeclaration:
        case SyntaxKind.CallSignature:
        case SyntaxKind.ConstructSignature:
            return true;
        default:
            return false;
    }
}

 * @param flags whether the rule deletes a line or not, defaults to no-op
 */
function rule(
    debugName: string,
    left: SyntaxKind | readonly SyntaxKind[] | TokenRange,
    right: SyntaxKind | readonly SyntaxKind[] | TokenRange,
    context: readonly ContextPredicate[],
    action: RuleAction,
    flags: RuleFlags = RuleFlags.None,
): RuleSpec {
    return { leftTokenRange: toTokenRange(left), rightTokenRange: toTokenRange(right), rule: { debugName, context, action, flags } };
}

function checkControlScope(scope: FormattingContext): boolean {
    if (scope.contextNode.kind === SyntaxKind.IfStatement ||
        scope.contextNode.kind === SyntaxKind.SwitchStatement ||
        scope.contextNode.kind === SyntaxKind.ForStatement ||
        scope.contextNode.kind === SyntaxKind.ForInStatement ||
        scope.contextNode.kind === SyntaxKind.ForOfStatement ||
        scope.contextNode.kind === SyntaxKind.WhileStatement ||
        scope.contextNode.kind === SyntaxKind.TryStatement ||
        scope.contextNode.kind === SyntaxKind.DoStatement ||
        // TODO
        // scope.contextNode.kind === SyntaxKind.ElseClause:
        scope.contextNode.kind === SyntaxKind.CatchClause) {
            return true;
    }

    return false;
}

export async function runCLI(): Promise<void> {
  try {
    const rootDir = process.argv[2];
    await runCreate(rootDir);
  } catch (error: unknown) {
    clearLine(process.stderr);
    clearLine(process.stdout);
    if (error instanceof Error && Boolean(error?.stack)) {
      console.error(chalk.red(error.stack));
    } else {
      console.error(chalk.red(error));
    }

    exit(1);
    throw error;
  }
}

export class DurableController {
  constructor(private readonly durableService: DurableService) {}

  @Get()
  sayHello(): string {
    return this.getGreeting();
  }

  private getGreeting() {
    const greeting = this.durableService.greet();
    return greeting;
  }

  @Get('echo')
  repeatMessage(): any {
    return this.retrieveRequestPayload();
  }

  private retrieveRequestPayload() {
    return this.durableService.requestPayload;
  }
}

function g() {
    let b = 1;
    var { p, q } = /*RENAME*/oldFunction();
    b; p; q;

    function oldFunction() {
        var p = 1;
        let q = 2;
        b++;
        return { p, q };
    }
}

