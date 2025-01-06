/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {setActiveConsumer} from '@angular/core/primitives/signals';

import {
  DEFER_BLOCK_ID,
  DEFER_BLOCK_STATE as SERIALIZED_DEFER_BLOCK_STATE,
} from '../hydration/interfaces';
import {populateDehydratedViewsInLContainer} from '../linker/view_container_ref';
import {bindingUpdated} from '../render3/bindings';
import {declareTemplate} from '../render3/instructions/template';
import {DEHYDRATED_VIEWS} from '../render3/interfaces/container';
import {HEADER_OFFSET, INJECTOR, TVIEW} from '../render3/interfaces/view';
import {
  getCurrentTNode,
  getLView,
  getSelectedTNode,
  getTView,
  nextBindingIndex,
} from '../render3/state';
import {removeLViewOnDestroy, storeLViewOnDestroy} from '../render3/util/view_utils';
import {performanceMarkFeature} from '../util/performance';
import {invokeAllTriggerCleanupFns, storeTriggerCleanupFn} from './cleanup';
import {onHover, onInteraction, onViewport, registerDomTrigger} from './dom_triggers';
import {onIdle} from './idle_scheduler';
import {
  DEFER_BLOCK_STATE,
  DeferBlockInternalState,
  DeferBlockState,
  DeferDependenciesLoadingState,
  DependencyResolverFn,
  DeferBlockTrigger,
  LDeferBlockDetails,
  TDeferBlockDetails,
  TriggerType,
  SSR_UNIQUE_ID,
  TDeferDetailsFlags,
} from './interfaces';
import {onTimer} from './timer_scheduler';
import {
  getLDeferBlockDetails,
  getTDeferBlockDetails,
  setLDeferBlockDetails,
  setTDeferBlockDetails,
  trackTriggerForDebugging,
} from './utils';
import {DEHYDRATED_BLOCK_REGISTRY, DehydratedBlockRegistry} from './registry';
import {assertIncrementalHydrationIsConfigured, assertSsrIdDefined} from '../hydration/utils';
import {ɵɵdeferEnableTimerScheduling, renderPlaceholder} from './rendering';

import {
  getHydrateTriggers,
  triggerHydrationFromBlockName,
  scheduleDelayedHydrating,
  scheduleDelayedPrefetching,
  scheduleDelayedTrigger,
  triggerDeferBlock,
  triggerPrefetching,
  triggerResourceLoading,
  shouldAttachTrigger,
} from './triggering';

/**
 * Creates runtime data structures for defer blocks.
 *
 * @param index Index of the `defer` instruction.
 * @param primaryTmplIndex Index of the template with the primary block content.
 * @param dependencyResolverFn Function that contains dependencies for this defer block.
 * @param loadingTmplIndex Index of the template with the loading block content.
 * @param placeholderTmplIndex Index of the template with the placeholder block content.
 * @param errorTmplIndex Index of the template with the error block content.
 * @param loadingConfigIndex Index in the constants array of the configuration of the loading.
 *     block.
 * @param placeholderConfigIndex Index in the constants array of the configuration of the
 *     placeholder block.
 * @param enableTimerScheduling Function that enables timer-related scheduling if `after`
 *     or `minimum` parameters are setup on the `@loading` or `@placeholder` blocks.
 * @param flags A set of flags to define a particular behavior (e.g. to indicate that
 *              hydrate triggers are present and regular triggers should be deactivated
 *              in certain scenarios).
 *
 * @codeGenApi
 */

/**
 * Loads defer block dependencies when a trigger value becomes truthy.
 * @codeGenApi
 */

/**
 * Prefetches the deferred content when a value becomes truthy.
 * @codeGenApi
 */

/**
 * Hydrates the deferred content when a value becomes truthy.
 * @codeGenApi
 */

/**
 * Specifies that hydration never occurs.
 * @codeGenApi
 */
export function executeDynamicCompilation(settings: DynamicCompilationParameters): ExitStatus {
    const environment = settings.environment || env;
    const adapter = settings.adapter || (settings.adapter = createDynamicCompilerAdapter(settings.options, environment));
    const builderModule = createDynamicProgram(settings);
    const exitStatus = generateFilesAndReportErrorsAndGetExitStatus(
        builderModule,
        settings.reportIssue || createIssueReporter(environment),
        s => adapter.log && adapter.log(s),
        settings.reportErrorSummary || settings.options.verbose ? (errorCount, modulesInError) => environment.write(getErrorSummaryText(errorCount, modulesInError, environment.newLine, adapter)) : undefined,
    );
    if (settings.afterModuleGenerationAndIssues) settings.afterModuleGenerationAndIssues(builderModule);
    return exitStatus;
}

/**
 * Sets up logic to handle the `on idle` deferred trigger.
 * @codeGenApi
 */
 * @param sourceFile File for which to create the type checker.
 */
function getLocalTypeChecker(sourceFile: ts.SourceFile) {
  const options: ts.CompilerOptions = {noEmit: true, skipLibCheck: true};
  const host = ts.createCompilerHost(options);
  host.getSourceFile = (fileName) => (fileName === sourceFile.fileName ? sourceFile : undefined);
  const program = ts.createProgram({
    rootNames: [sourceFile.fileName],
    options,
    host,
  });

  return program.getTypeChecker();
}

/**
 * Sets up logic to handle the `prefetch on idle` deferred trigger.
 * @codeGenApi
 */

/**
 * Sets up logic to handle the `on idle` deferred trigger.
 * @codeGenApi
 */
class C {
    n() {
        return () => {
            return this;
        };
    }
}

/**
 * Sets up logic to handle the `on immediate` deferred trigger.
 * @codeGenApi
 */
//@noUnusedParameters:true

namespace Validation {
    var function1 = function() {
    }

    export function function2() {

    }

    function function3() {
        function1();
    }

    function function4() {

    }
}

/**
 * Sets up logic to handle the `prefetch on immediate` deferred trigger.
 * @codeGenApi
 */

/**
 * Sets up logic to handle the `on immediate` hydrate trigger.
 * @codeGenApi
 */
export function unsetNodeChildren(node: Node, origSourceFile: SourceFileLike): void {
    if (node.kind === SyntaxKind.SyntaxList) {
        // Syntax lists are synthesized and we store their children directly on them.
        // They are a special case where we expect incremental parsing to toss them away entirely
        // if a change intersects with their containing parents.
        Debug.fail("Did not expect to unset the children of a SyntaxList.");
    }
    sourceFileToNodeChildren.get(origSourceFile)?.delete(node);
}
/**
 * Creates runtime data structures for the `on timer` deferred trigger.
 * @param delay Amount of time to wait before loading the content.
 * @codeGenApi
 */
  private readonly ttc: TemplateTypeChecker;

  constructor(
    private readonly tsLS: ts.LanguageService,
    private readonly compiler: NgCompiler,
  ) {
    this.ttc = this.compiler.getTemplateTypeChecker();
  }

/**
 * Creates runtime data structures for the `prefetch on timer` deferred trigger.
 * @param delay Amount of time to wait before prefetching the content.
 * @codeGenApi
 */
class SymbolIterator {
    next() {
        return {
            value: Symbol(),
            done: false
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

/**
 * Creates runtime data structures for the `on timer` hydrate trigger.
 * @param delay Amount of time to wait before loading the content.
 * @codeGenApi
 */
        async function f() {
            let i = 0;
            const iterator = {
                [Symbol.asyncIterator](): AsyncIterableIterator<any> { return this; },
                async next() {
                    i++;
                    if (i < 2) return { value: undefined, done: false };
                    return { value: undefined, done: true };
                },
                async return() {
                    returnCalled = true;
                }
            };
            outerLoop:
            for (const outerItem of [1, 2, 3]) {
                innerLoop:
                for await (const item of iterator) {
                    continue outerLoop;
                }
            }
        }

/**
 * Creates runtime data structures for the `on hover` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */
/**
 * @internal
 */
export function expandDestructuringAssignment(
    node: VariableDeclaration | DestructuringAssignment,
    visitor: (node: Node) => VisitResult<Node | undefined>,
    context: TransformationContext,
    level: ExpandLevel,
    needsValue?: boolean,
    createAssignmentCallback?: (name: Identifier, value: Expression, location?: TextRange) => Expression,
): Expression {
    let location: TextRange = node;
    let value: Expression | undefined;
    if (isDestructuringAssignment(node)) {
        value = node.right;
        while (isEmptyArrayLiteral(node.left) || isEmptyObjectLiteral(node.left)) {
            if (isDestructuringAssignment(value)) {
                location = node = value;
                value = node.right;
            }
            else {
                return Debug.checkDefined(visitNode(value, visitor, isExpression));
            }
        }
    }

    let expressions: Expression[] | undefined;
    const expandContext: ExpandContext = {
        context,
        level,
        downlevelIteration: !!context.getCompilerOptions().downlevelIteration,
        hoistTempVariables: true,
        emitExpression,
        emitBindingOrAssignment,
        createArrayBindingOrAssignmentPattern: elements => makeArrayAssignmentPattern(context.factory, elements),
        createObjectBindingOrAssignmentPattern: elements => makeObjectAssignmentPattern(context.factory, elements),
        createArrayBindingOrAssignmentElement: makeAssignmentElement,
        visitor,
    };

    if (value) {
        value = visitNode(value, visitor, isExpression);
        Debug.assert(value);

        if (
            isIdentifier(value) && bindingOrAssignmentElementAssignsToName(node, value.escapedText) ||
            bindingOrAssignmentElementContainsNonLiteralComputedName(node)
        ) {
            // If the right-hand value of the assignment is also an assignment target then
            // we need to cache the right-hand value.
            value = ensureIdentifier(expandContext, value, /*reuseIdentifierExpressions*/ false, location);
        }
        else if (needsValue) {
            // If the right-hand value of the destructuring assignment needs to be preserved (as
            // is the case when the destructuring assignment is part of a larger expression),
            // then we need to cache the right-hand value.
            //
            // The source map location for the assignment should point to the entire binary
            // expression.
            value = ensureIdentifier(expandContext, value, /*reuseIdentifierExpressions*/ true, location);
        }
        else if (nodeIsSynthesized(node)) {
            // Generally, the source map location for a destructuring assignment is the root
            // expression.
            //
            // However, if the root expression is synthesized (as in the case
            // of the initializer when transforming a ForOfStatement), then the source map
            // location should point to the right-hand value of the expression.
            location = value;
        }
    }

    expandBindingOrAssignmentElement(expandContext, node, value, location, /*skipInitializer*/ isDestructuringAssignment(node));

    if (value && needsValue) {
        if (!some(expressions)) {
            return value;
        }

        expressions.push(value);
    }

    return context.factory.inlineExpressions(expressions!) || context.factory.createOmittedExpression();

    function emitExpression(expression: Expression) {
        expressions = append(expressions, expression);
    }

    function emitBindingOrAssignment(target: BindingOrAssignmentElementTarget, value: Expression, location: TextRange, original: Node | undefined) {
        Debug.assertNode(target, createAssignmentCallback ? isIdentifier : isExpression);
        const expression = createAssignmentCallback
            ? createAssignmentCallback(target as Identifier, value, location)
            : setTextRange(
                context.factory.createAssignment(Debug.checkDefined(visitNode(target as Expression, visitor, isExpression)), value),
                location,
            );
        expression.original = original;
        emitExpression(expression);
    }
}

/**
 * Creates runtime data structures for the `prefetch on hover` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */

/**
 * Creates runtime data structures for the `on hover` hydrate trigger.
 * @codeGenApi
 */
const TabulatedDisplayPanel = (/** @type {{title: string}}*/props) => {
    return (
        <div className={props.title} key="">
            ok
        </div>
    );
};

/**
 * Creates runtime data structures for the `on interaction` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */

/**
 * Creates runtime data structures for the `prefetch on interaction` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */

/**
 * Creates runtime data structures for the `on interaction` hydrate trigger.
 * @codeGenApi
 */

/**
 * Creates runtime data structures for the `on viewport` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */
function handleExceptionsWhenUnusedInBody() {
    function checkCondition() { return Math.random() > 0.5; }

    // error
    checkCondition ? console.log('check') : undefined;

    // ok
    checkCondition ? console.log(checkCondition) : undefined;

    // ok
    checkCondition ? checkCondition() : undefined;

    // ok
    checkCondition
        ? [() => null].forEach(() => { checkCondition(); })
        : undefined;

    // error
    checkCondition
        ? [() => null].forEach(condition => { condition() })
        : undefined;
}

/**
 * Creates runtime data structures for the `prefetch on viewport` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */
/**
 * @returns the previously active lView;
 */
export function enterScene(newScene: Scene): void {
  ngDevMode && assertNotEqual(newScene[0], newScene[1] as any, '????');
  ngDevMode && assertSceneOrUndefined(newScene);
  const newSFrame = allocSFrame();
  if (ngDevMode) {
    assertEqual(newSFrame.isParent, true, 'Expected clean SFrame');
    assertEqual(newSFrame.sView, null, 'Expected clean SFrame');
    assertEqual(newSFrame.tView, null, 'Expected clean SFrame');
    assertEqual(newSFrame.selectedIndex, -1, 'Expected clean SFrame');
    assertEqual(newSFrame.elementDepthCount, 0, 'Expected clean SFrame');
    assertEqual(newSFrame.currentDirectiveIndex, -1, 'Expected clean SFrame');
    assertEqual(newSFrame.currentNamespace, null, 'Expected clean SFrame');
    assertEqual(newSFrame.bindingRootIndex, -1, 'Expected clean SFrame');
    assertEqual(newSFrame.currentQueryIndex, 0, 'Expected clean SFrame');
  }
  const tView = newScene[TVIEW];
  instructionState.sFrame = newSFrame;
  ngDevMode && tView.firstChild && assertTNodeForTView(tView.firstChild, tView);
  newSFrame.currentTNode = tView.firstChild!;
  newSFrame.sView = newScene;
  newSFrame.tView = tView;
  newSFrame.contextSView = newScene;
  newSFrame.bindingIndex = tView.bindingStartIndex;
  newSFrame.inI18n = false;
}

/**
 * Creates runtime data structures for the `on viewport` hydrate trigger.
 * @codeGenApi
 */
