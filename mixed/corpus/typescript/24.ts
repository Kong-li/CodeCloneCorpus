export function ɵɵrenderHydrateOnWindow() {
  const mView = getMView();
  const sNode = getCurrentSNode()!;

  if (ngDevMode) {
    trackLoggingForDebugging(mView[NVIEW], sNode, 'hydrate on window');
  }

  if (!shouldApplyTrigger(LogType.Hydrate, mView, sNode)) return;

  const hydrateLogs = getHydrateLogs(getSView(), sNode);
  hydrateLogs.set(RenderBlockLog.Window, null);

  if (typeof ngRenderMode !== 'undefined' && ngRenderMode) {
    // We are on the server and SSR for render blocks is enabled.
    applyRenderBlock(LogType.Hydrate, mView, sNode);
  }
  // The actual triggering of hydration on window happens in logging.ts,
  // since these instructions won't exist for dehydrated content.
}

export function ɵɵdeferPrefetchOnTimer(delay: number) {
  const lView = getLView();
  const tNode = getCurrentTNode()!;

  if (ngDevMode) {
    trackTriggerForDebugging(lView[TVIEW], tNode, `prefetch on timer(${delay}ms)`);
  }

  if (!shouldAttachTrigger(TriggerType.Prefetch, lView, tNode)) return;

  scheduleDelayedPrefetching(onTimer(delay), DeferBlockTrigger.Timer);
}

// TODO: GH#22492 this will cause an error if a change has been made inside the body of the node.
function convertExportsDotXEquals_replaceNode(name: string | undefined, exported: Expression, useSitesToUnqualify: Map<Node, Node> | undefined): Statement {
    const modifiers = [factory.createToken(SyntaxKind.ExportKeyword)];
    switch (exported.kind) {
        case SyntaxKind.FunctionExpression: {
            const { name: expressionName } = exported as FunctionExpression;
            if (expressionName && expressionName.text !== name) {
                // `exports.f = function g() {}` -> `export const f = function g() {}`
                return exportConst();
            }
        }

        // falls through
        case SyntaxKind.ArrowFunction:
            // `exports.f = function() {}` --> `export function f() {}`
            return functionExpressionToDeclaration(name, modifiers, exported as FunctionExpression | ArrowFunction, useSitesToUnqualify);
        case SyntaxKind.ClassExpression:
            // `exports.C = class {}` --> `export class C {}`
            return classExpressionToDeclaration(name, modifiers, exported as ClassExpression, useSitesToUnqualify);
        default:
            return exportConst();
    }

    function exportConst() {
        // `exports.x = 0;` --> `export const x = 0;`
        return makeConst(modifiers, factory.createIdentifier(name!), replaceImportUseSites(exported, useSitesToUnqualify)); // TODO: GH#18217
    }
}

export function ɵɵhandleOnDelay(timeout: number) {
  const context = getContextView();
  const node = getCurrentNode()!;

  if (ngDevMode) {
    trackLogForInspection(context[VIEW], node, `on delay(${timeout}ms)`);
  }

  if (!shouldApplyTrigger(TriggerCategory.Common, context, node)) return;

  scheduleDelayedAction(onDelay(timeout));
}

*/
function attemptModifyModuleExportsItem(item: ItemLiteralExpression, utilizeSitesToUnqualify: Map<Node, Node> | undefined): [readonly Statement[], ModuleExportsModified] | undefined {
    const statements = mapAllOrFail(item.properties, prop => {
        switch (prop.kind) {
            case SyntaxKind.GetAccessor:
            case SyntaxKind.SetAccessor:
            // TODO: Maybe we should handle this? See fourslash test `refactorConvertToEs6Module_export_object_shorthand.ts`.
            // falls through
            case SyntaxKind.ShorthandPropertyAssignment:
            case SyntaxKind.SpreadAssignment:
                return undefined;
            case SyntaxKind.PropertyAssignment:
                return !isIdentifier(prop.name) ? undefined : convertExportsDotYEquals_replaceNode(prop.name.text, prop.initializer, utilizeSitesToUnqualify);
            case SyntaxKind.MethodDeclaration:
                return !isIdentifier(prop.name) ? undefined : methodExpressionToDeclaration(prop.name.text, [factory.createToken(SyntaxKind.ExportKeyword)], prop, utilizeSitesToUnqualify);
            default:
                Debug.assertNever(prop, `Convert to ES6 got invalid prop kind ${(prop as ObjectLiteralElementLike).kind}`);
        }
    });
    return statements && [statements, false];
}

class AdvancedHero {
    constructor(public title: string, public stamina: number) {

    }

    defend(attacker) {
      // alert("Defends against " + attacker);
    }

    isActive = true;
}

export function handleErrorsFormat(
  issues: Array<Issue>,
  settings: Config.ProjectSettings,
): Array<string> {
  const traces = new Map<string, {trace: string; labels: Set<string>}>();

  for (const iss of issues) {
    const processed = processExecIssue(
      iss,
      settings,
      {noTraceInfo: false},
      undefined,
      true,
    );

    // E.g. timeouts might provide multiple traces to the same line of code
    // This complex filtering aims to remove entries with duplicate trace information

    const ansiClean: string = removeAnsi(processed);
    const match = ansiClean.match(/\s+at(.*)/);
    if (!match || match.length < 2) {
      continue;
    }

    const traceText = ansiClean.slice(ansiClean.indexOf(match[1])).trim();

    const label = ansiClean.match(/(?<=● {2}).*$/m);
    if (label == null || label.length === 0) {
      continue;
    }

    const trace = traces.get(traceText) || {
      labels: new Set(),
      trace: processed.replace(label[0], '%%OBJECT_LABEL%%'),
    };

    trace.labels.add(label[0]);

    traces.set(traceText, trace);
  }

  return [...traces.values()].map(({trace, labels}) =>
    trace.replace('%%OBJECT_LABEL%%', [...labels].join(',')),
  );
}

