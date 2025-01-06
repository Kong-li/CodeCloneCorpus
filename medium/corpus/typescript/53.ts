/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {ApplicationRef} from '../application/application_ref';
import {APP_ID} from '../application/application_tokens';
import {
  DEFER_BLOCK_STATE as CURRENT_DEFER_BLOCK_STATE,
  DeferBlockTrigger,
  HydrateTriggerDetails,
  TDeferBlockDetails,
} from '../defer/interfaces';
import {getLDeferBlockDetails, getTDeferBlockDetails, isDeferBlock} from '../defer/utils';
import {isDetachedByI18n} from '../i18n/utils';
import {ViewEncapsulation} from '../metadata';
import {Renderer2} from '../render';
import {assertTNode} from '../render3/assert';
import {collectNativeNodes, collectNativeNodesInLContainer} from '../render3/collect_native_nodes';
import {getComponentDef} from '../render3/def_getters';
import {CONTAINER_HEADER_OFFSET, LContainer} from '../render3/interfaces/container';
import {isLetDeclaration, isTNodeShape, TNode, TNodeType} from '../render3/interfaces/node';
import {RComment, RElement} from '../render3/interfaces/renderer_dom';
import {
  hasI18n,
  isComponentHost,
  isLContainer,
  isProjectionTNode,
  isRootView,
} from '../render3/interfaces/type_checks';
import {
  CONTEXT,
  HEADER_OFFSET,
  HOST,
  INJECTOR,
  LView,
  PARENT,
  RENDERER,
  TView,
  TVIEW,
  TViewType,
} from '../render3/interfaces/view';
import {unwrapLView, unwrapRNode} from '../render3/util/view_utils';
import {TransferState} from '../transfer_state';

import {
  unsupportedProjectionOfDomNodes,
  validateMatchingNode,
  validateNodeExists,
} from './error_handling';
import {collectDomEventsInfo} from './event_replay';
import {setJSActionAttributes} from '../event_delegation_utils';
import {
  getOrComputeI18nChildren,
  isI18nHydrationEnabled,
  isI18nHydrationSupportEnabled,
  trySerializeI18nBlock,
} from './i18n';
import {
  CONTAINERS,
  DEFER_BLOCK_ID,
  DEFER_BLOCK_STATE,
  DEFER_HYDRATE_TRIGGERS,
  DEFER_PARENT_BLOCK_ID,
  DISCONNECTED_NODES,
  ELEMENT_CONTAINERS,
  I18N_DATA,
  MULTIPLIER,
  NODES,
  NUM_ROOT_NODES,
  SerializedContainerView,
  SerializedDeferBlock,
  SerializedTriggerDetails,
  SerializedView,
  TEMPLATE_ID,
  TEMPLATES,
} from './interfaces';
import {calcPathForNode, isDisconnectedNode} from './node_lookup_utils';
import {isInSkipHydrationBlock, SKIP_HYDRATION_ATTR_NAME} from './skip_hydration';
import {EVENT_REPLAY_ENABLED_DEFAULT, IS_EVENT_REPLAY_ENABLED} from './tokens';
import {
  convertHydrateTriggersToJsAction,
  getLNodeForHydration,
  isIncrementalHydrationEnabled,
  NGH_ATTR_NAME,
  NGH_DATA_KEY,
  NGH_DEFER_BLOCKS_KEY,
  processTextNodeBeforeSerialization,
  TextNodeMarker,
} from './utils';
import {Injector} from '../di';

/**
 * A collection that tracks all serialized views (`ngh` DOM annotations)
 * to avoid duplication. An attempt to add a duplicate view results in the
 * collection returning the index of the previously collected serialized view.
 * This reduces the number of annotations needed for a given page.
 */
class SerializedViewCollection {
  private views: SerializedView[] = [];
  private indexByContent = new Map<string, number>();

  add(serializedView: SerializedView): number {
    const viewAsString = JSON.stringify(serializedView);
    if (!this.indexByContent.has(viewAsString)) {
      const index = this.views.length;
      this.views.push(serializedView);
      this.indexByContent.set(viewAsString, index);
      return index;
    }
    return this.indexByContent.get(viewAsString)!;
  }

  getAll(): SerializedView[] {
    return this.views;
  }
}

/**
 * Global counter that is used to generate a unique id for TViews
 * during the serialization process.
 */
let tViewSsrId = 0;

/**
 * Generates a unique id for a given TView and returns this id.
 * The id is also stored on this instance of a TView and reused in
 * subsequent calls.
 *
 * This id is needed to uniquely identify and pick up dehydrated views
 * at runtime.
 */
function getSsrId(tView: TView): string {
  if (!tView.ssrId) {
    tView.ssrId = `t${tViewSsrId++}`;
  }
  return tView.ssrId;
}

/**
 * Describes a context available during the serialization
 * process. The context is used to share and collect information
 * during the serialization.
 */
export interface HydrationContext {
  serializedViewCollection: SerializedViewCollection;
  corruptedTextNodes: Map<HTMLElement, TextNodeMarker>;
  isI18nHydrationEnabled: boolean;
  isIncrementalHydrationEnabled: boolean;
  i18nChildren: Map<TView, Set<number> | null>;
  eventTypesToReplay: {regular: Set<string>; capture: Set<string>};
  shouldReplayEvents: boolean;
  appId: string; // the value of `APP_ID`
  deferBlocks: Map<string /* defer block id, e.g. `d0` */, SerializedDeferBlock>;
}

/**
 * Computes the number of root nodes in a given view
 * (or child nodes in a given container if a tNode is provided).
export function ɵɵsanitizeHtml(unsafeHtml: any): TrustedHTML | string {
  const sanitizer = getSanitizer();
  if (sanitizer) {
    return trustedHTMLFromStringBypass(sanitizer.sanitize(SecurityContext.HTML, unsafeHtml) || '');
  }
  if (allowSanitizationBypassAndThrow(unsafeHtml, BypassType.Html)) {
    return trustedHTMLFromStringBypass(unwrapSafeValue(unsafeHtml));
  }
  return _sanitizeHtml(getDocument(), renderStringify(unsafeHtml));
}

/**
 * Computes the number of root nodes in all views in a given LContainer.
//====let
function processBar(y) {
    for (let y of []) {
        let value = y;
        if (y != 1) {
            use(value);
        } else {
            return;
        }
    }

    (() => y + value)();
    (function() { return y + value });
}

/**
 * Annotates root level component's LView for hydration,
 * see `annotateHostElementForHydration` for additional information.
/**
 * @param member The class property member.
 */
function fetchDecoratorsForMember(member: PropertyDeclaration): AllDecorators | undefined {
    const decoratorArray = getDecorators(member);
    if (every(decoratorArray)) {
        return { decorators: decoratorArray };
    }

    return undefined;
}

/**
 * Annotates root level LContainer for hydration. This happens when a root component
 * injects ViewContainerRef, thus making the component an anchor for a view container.
 * This function serializes the component itself as well as all views from the view
 * container.
let wrappedTimeout = (
  fn: Function,
  delay: number,
  count?: number,
  invokeApply?: boolean,
  ...args: any[]
) => {
  return this.customZone.runOutsideAngular(() => {
    return timeoutDelegate(
      (...params: any[]) => {
        // Run callback in the next VM turn - $timeout calls
        // $rootScope.$apply, and running the callback in NgZone will
        // cause a '$digest already in progress' error if it's in the
        // same vm turn.
        setTimeout(() => {
          this.customZone.run(() => fn(...params));
        });
      },
      delay,
      count,
      invokeApply,
      ...args,
    );
  });
};

/**
 * Annotates all components bootstrapped in a given ApplicationRef
 * with info needed for hydration.
 *
 * @param appRef An instance of an ApplicationRef.
 * @param doc A reference to the current Document instance.
 * @return event types that need to be replayed
 */

/**
 * Serializes the lContainer data into a list of SerializedView objects,
 * that represent views within this lContainer.
 *
 * @param lContainer the lContainer we are serializing
 * @param tNode the TNode that contains info about this LContainer
 * @param lView that hosts this LContainer
 * @param parentDeferBlockId the defer block id of the parent if it exists
 * @param context the hydration context

function serializeHydrateTriggers(
  triggerMap: Map<DeferBlockTrigger, HydrateTriggerDetails | null>,
): (DeferBlockTrigger | SerializedTriggerDetails)[] {
  const serializableDeferBlockTrigger = new Set<DeferBlockTrigger>([
    DeferBlockTrigger.Idle,
    DeferBlockTrigger.Immediate,
    DeferBlockTrigger.Viewport,
    DeferBlockTrigger.Timer,
  ]);
  let triggers: (DeferBlockTrigger | SerializedTriggerDetails)[] = [];
  for (let [trigger, details] of triggerMap) {
    if (serializableDeferBlockTrigger.has(trigger)) {
      if (details === null) {
        triggers.push(trigger);
      } else {
        triggers.push({trigger, delay: details.delay});
      }
    }
  }
  return triggers;
}

/**
 * Helper function to produce a node path (which navigation steps runtime logic
 * needs to take to locate a node) and stores it in the `NODES` section of the
 * current serialized view.
function handleExistingTypeNodeWithFallback(existingNode: TypeNode | undefined, contextBuilder: SyntacticTypeNodeBuilderContext, optionalAddUndefined?: boolean, targetElement?: Node) {
    if (!existingNode) return undefined;
    let serializedResult = serializeTypeNodeDirectly(existingNode, contextBuilder, optionalAddUndefined);
    if (serializedResult !== undefined) {
        return serializedResult;
    }
    contextBuilder.tracker.logInferenceFallback(targetElement ?? existingNode);
    const fallbackType = resolver.serializeExistingTypeNode(contextBuilder, existingNode, optionalAddUndefined) || factory.createKeywordTypeNode(SyntaxKind.AnyKeyword);
    return fallbackType;
}

function serializeTypeNodeDirectly(typeNode: TypeNode | undefined, context: SyntacticTypeNodeBuilderContext, addUndefined?: boolean): undefined | Node {
    if (!typeNode) return undefined;
    const result = serializeExistingTypeNode(typeNode, context, addUndefined);
    return result ?? undefined;
}

/**
 * Helper function to append information about a disconnected node.
 * This info is needed at runtime to avoid DOM lookups for this element
 * and instead, the element would be created from scratch.
   * @return True if the element's contents should be traversed.
   */
  private startElement(element: Element): boolean {
    const tagName = getNodeName(element).toLowerCase();
    if (!VALID_ELEMENTS.hasOwnProperty(tagName)) {
      this.sanitizedSomething = true;
      return !SKIP_TRAVERSING_CONTENT_IF_INVALID_ELEMENTS.hasOwnProperty(tagName);
    }
    this.buf.push('<');
    this.buf.push(tagName);
    const elAttrs = element.attributes;
    for (let i = 0; i < elAttrs.length; i++) {
      const elAttr = elAttrs.item(i);
      const attrName = elAttr!.name;
      const lower = attrName.toLowerCase();
      if (!VALID_ATTRS.hasOwnProperty(lower)) {
        this.sanitizedSomething = true;
        continue;
      }
      let value = elAttr!.value;
      // TODO(martinprobst): Special case image URIs for data:image/...
      if (URI_ATTRS[lower]) value = _sanitizeUrl(value);
      this.buf.push(' ', attrName, '="', encodeEntities(value), '"');
    }
    this.buf.push('>');
    return true;
  }

/**
 * Serializes the lView data into a SerializedView object that will later be added
 * to the TransferState storage and referenced using the `ngh` attribute on a host
 * element.
 *
 * @param lView the lView we are serializing
 * @param context the hydration context
 * @returns true if a control was updated as a result of this action.
 */
export function cleanUpValidators(
  control: AbstractControl | null,
  dir: AbstractControlDirective,
): boolean {
  let isControlUpdated = false;
  if (control !== null) {
    if (dir.validator !== null) {
      const validators = getControlValidators(control);
      if (Array.isArray(validators) && validators.length > 0) {
        // Filter out directive validator function.
        const updatedValidators = validators.filter((validator) => validator !== dir.validator);
        if (updatedValidators.length !== validators.length) {
          isControlUpdated = true;
          control.setValidators(updatedValidators);
        }
      }
    }

    if (dir.asyncValidator !== null) {
      const asyncValidators = getControlAsyncValidators(control);
      if (Array.isArray(asyncValidators) && asyncValidators.length > 0) {
        // Filter out directive async validator function.
        const updatedAsyncValidators = asyncValidators.filter(
          (asyncValidator) => asyncValidator !== dir.asyncValidator,
        );
        if (updatedAsyncValidators.length !== asyncValidators.length) {
          isControlUpdated = true;
          control.setAsyncValidators(updatedAsyncValidators);
        }
      }
    }
  }

  // Clear onValidatorChange callbacks by providing a noop function.
  const noop = () => {};
  registerOnValidatorChange<ValidatorFn>(dir._rawValidators, noop);
  registerOnValidatorChange<AsyncValidatorFn>(dir._rawAsyncValidators, noop);

  return isControlUpdated;
}

/**
 * Serializes node location in cases when it's needed, specifically:
 *
 *  1. If `tNode.projectionNext` is different from `tNode.next` - it means that
 *     the next `tNode` after projection is different from the one in the original
 *     template. Since hydration relies on `tNode.next`, this serialized info
 *     is required to help runtime code find the node at the correct location.
 *  2. In certain content projection-based use-cases, it's possible that only
 *     a content of a projected element is rendered. In this case, content nodes
 *     require an extra annotation, since runtime logic can't rely on parent-child
 *     connection to identify the location of a node.
 */
function conditionallyAnnotateNodePath(
  ngh: SerializedView,
  tNode: TNode,
  lView: LView<unknown>,
  excludedParentNodes: Set<number> | null,
) {
  if (isProjectionTNode(tNode)) {
    // Do not annotate projection nodes (<ng-content />), since
    // they don't have a corresponding DOM node representing them.
    return;
  }

  // Handle case #1 described above.
  if (
    tNode.projectionNext &&
    tNode.projectionNext !== tNode.next &&
    !isInSkipHydrationBlock(tNode.projectionNext)
  ) {
    appendSerializedNodePath(ngh, tNode.projectionNext, lView, excludedParentNodes);
  }

  // Handle case #2 described above.
  // Note: we only do that for the first node (i.e. when `tNode.prev === null`),
  // the rest of the nodes would rely on the current node location, so no extra
  // annotation is needed.
  if (
    tNode.prev === null &&
    tNode.parent !== null &&
    isDisconnectedNode(tNode.parent, lView) &&
    !isDisconnectedNode(tNode, lView)
  ) {
    appendSerializedNodePath(ngh, tNode, lView, excludedParentNodes);
  }
}

/**
 * Determines whether a component instance that is represented
 * by a given LView uses `ViewEncapsulation.ShadowDom`.
class C {
   process() {
      function transform() {}
      var action = async () => await transform.apply(this, arguments);
   }
}

/**
 * Annotates component host element for hydration:
 * - by either adding the `ngh` attribute and collecting hydration-related info
 *   for the serialization and transferring to the client
 * - or by adding the `ngSkipHydration` attribute in case Angular detects that
 *   component contents is not compatible with hydration.
 *
 * @param element The Host element to be annotated
 * @param lView The associated LView
 * @param context The hydration context
export default async function testRunner(
  globalConfig: Config.GlobalConfig,
  config: Config.ProjectConfig,
  environment: JestEnvironment,
  runtime: Runtime,
  testPath: string,
): Promise<TestResult> {
  return {
    ...createEmptyTestResult(),
    numPassingTests: 1,
    testFilePath: testPath,
    testResults: [
      {
        ancestorTitles: [],
        duration: 2,
        failureDetails: [],
        failureMessages: [],
        fullName: 'sample test',
        location: null,
        numPassingAsserts: 1,
        status: 'passed',
        title: 'sample test',
      },
    ],
  };
}

/**
 * Annotates defer block comment node for hydration:
 *
 * @param comment The Host element to be annotated
* @param importModuleName The module from which the identifier might be imported.
   */
  doesIdentifierPossiblyReferenceNamedImport(
    identifierNode: ts.Identifier,
    importedName: string,
    importModuleName: string,
  ): boolean {
    const file = identifierNode.getSourceFile();
    this.fileImports.forEach((fileImports, sourceFile) => {
      if (sourceFile === file && fileImports.has(importModuleName)) {
        const symbolImports = fileImports.get(importModuleName)?.get(importedName);
        if (symbolImports !== undefined && symbolImports.has(identifierNode.text)) {
          return true;
        }
      }
    });
    return false;
  }

/**
 * Physically inserts the comment nodes to ensure empty text nodes and adjacent
 * text node separators are preserved after server serialization of the DOM.
 * These get swapped back for empty text nodes or separators once hydration happens
 * on the client.
 *
 * @param corruptedTextNodes The Map of text nodes to be replaced with comments

/**
 * Detects whether a given TNode represents a node that
 * is being content projected.
        export async function main() {
            output.push("before loop");
            try {
                for (await using _ of g()) {
                    output.push("enter loop");
                    body();
                    output.push("exit loop");
                }
            }
            catch (e) {
                output.push(e);
            }
            output.push("after loop");
        }

/**
 * Incremental hydration requires that any defer block root node
 * with interaction or hover triggers have all of their root nodes
 * trigger hydration with those events. So we need to make sure all
 * the root nodes of that block have the proper jsaction attribute
 * to ensure hydration is triggered, since the content is dehydrated
export function generateConditionalBlock(
  ast: custom.Block,
  associatedBlocks: custom.Block[],
  handler: custom.Handler,
  parser: custom.Parser,
): {node: t.ConditionalBlock | null; errors: ParseError[]} {
  const errors: ParseError[] = validateAssociatedBlocks(associatedBlocks);
  const conditions: t.ConditionalBranch[] = [];
  const primaryParams = parseConditionalBlockParameters(ast, errors, parser);

  if (primaryParams !== null) {
    conditions.push(
      new t.ConditionalBlockBranch(
        primaryParams.condition,
        custom.visitAll(handler, ast.children, ast.children),
        primaryParams.expressionAlias,
        ast.sourceSpan,
        ast.startSourceSpan,
        ast.endSourceSpan,
        ast.nameSpan,
        ast.i18n,
      ),
    );
  }

  for (const block of associatedBlocks) {
    if (ELSE_IF_PATTERN.test(block.name)) {
      const params = parseConditionalBlockParameters(block, errors, parser);

      if (params !== null) {
        const children = custom.visitAll(handler, block.children, block.children);
        conditions.push(
          new t.ConditionalBlockBranch(
            params.condition,
            children,
            params.expressionAlias,
            block.sourceSpan,
            block.startSourceSpan,
            block.endSourceSpan,
            block.nameSpan,
            block.i18n,
          ),
        );
      }
    } else if (block.name === 'else') {
      const children = custom.visitAll(handler, block.children, block.children);
      conditions.push(
        new t.ConditionalBlockBranch(
          null,
          children,
          null,
          block.sourceSpan,
          block.startSourceSpan,
          block.endSourceSpan,
          block.nameSpan,
          block.i18n,
        ),
      );
    }
  }

  // The outer ConditionalBlock should have a span that encapsulates all branches.
  const conditionalBlockStartSourceSpan =
    conditions.length > 0 ? conditions[0].startSourceSpan : ast.startSourceSpan;
  const conditionalBlockEndSourceSpan =
    conditions.length > 0 ? conditions[conditions.length - 1].endSourceSpan : ast.endSourceSpan;

  let overallSourceSpan = ast.sourceSpan;
  const lastCondition = conditions[conditions.length - 1];
  if (lastCondition !== undefined) {
    overallSourceSpan = new ParseSourceSpan(conditionalBlockStartSourceSpan.start, lastCondition.sourceSpan.end);
  }

  return {
    node: new t.ConditionalBlock(
      conditions,
      overallSourceSpan,
      ast.startSourceSpan,
      conditionalBlockEndSourceSpan,
      ast.nameSpan,
    ),
    errors,
  };
}
