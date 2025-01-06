/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {Injector} from '../di/injector';
import type {ViewRef} from '../linker/view_ref';
import {getComponent} from '../render3/util/discovery_utils';
import {LContainer} from '../render3/interfaces/container';
import {getDocument} from '../render3/interfaces/document';
import {RElement, RNode} from '../render3/interfaces/renderer_dom';
import {isRootView} from '../render3/interfaces/type_checks';
import {HEADER_OFFSET, LView, TVIEW, TViewType} from '../render3/interfaces/view';
import {makeStateKey, TransferState} from '../transfer_state';
import {assertDefined, assertEqual} from '../util/assert';
import type {HydrationContext} from './annotate';

import {
  BlockSummary,
  CONTAINERS,
  DEFER_HYDRATE_TRIGGERS,
  DEFER_PARENT_BLOCK_ID,
  DehydratedView,
  DISCONNECTED_NODES,
  ELEMENT_CONTAINERS,
  MULTIPLIER,
  NUM_ROOT_NODES,
  SerializedContainerView,
  SerializedDeferBlock,
  SerializedTriggerDetails,
  SerializedView,
} from './interfaces';
import {IS_INCREMENTAL_HYDRATION_ENABLED, JSACTION_BLOCK_ELEMENT_MAP} from './tokens';
import {RuntimeError, RuntimeErrorCode} from '../errors';
import {DeferBlockTrigger, HydrateTriggerDetails} from '../defer/interfaces';
import {hoverEventNames, interactionEventNames} from '../defer/dom_triggers';
import {DEHYDRATED_BLOCK_REGISTRY} from '../defer/registry';
import {sharedMapFunction} from '../event_delegation_utils';

/**
 * The name of the key used in the TransferState collection,
 * where hydration information is located.
 */
const TRANSFER_STATE_TOKEN_ID = '__nghData__';

/**
 * Lookup key used to reference DOM hydration data (ngh) in `TransferState`.
 */
export const NGH_DATA_KEY = makeStateKey<Array<SerializedView>>(TRANSFER_STATE_TOKEN_ID);

/**
 * The name of the key used in the TransferState collection,
 * where serialized defer block information is located.
 */
export const TRANSFER_STATE_DEFER_BLOCKS_INFO = '__nghDeferData__';

/**
 * Lookup key used to retrieve defer block datain `TransferState`.
 */
export const NGH_DEFER_BLOCKS_KEY = makeStateKey<{[key: string]: SerializedDeferBlock}>(
  TRANSFER_STATE_DEFER_BLOCKS_INFO,
);

/**
 * The name of the attribute that would be added to host component
 * nodes and contain a reference to a particular slot in transferred
 * state that contains the necessary hydration info for this component.
 */
export const NGH_ATTR_NAME = 'ngh';

/**
 * Marker used in a comment node to ensure hydration content integrity
 */
export const SSR_CONTENT_INTEGRITY_MARKER = 'nghm';

export const enum TextNodeMarker {
  /**
   * The contents of the text comment added to nodes that would otherwise be
   * empty when serialized by the server and passed to the client. The empty
   * node is lost when the browser parses it otherwise. This comment node will
   * be replaced during hydration in the client to restore the lost empty text
   * node.
   */
  EmptyNode = 'ngetn',

  /**
   * The contents of the text comment added in the case of adjacent text nodes.
   * When adjacent text nodes are serialized by the server and sent to the
   * client, the browser loses reference to the amount of nodes and assumes
   * just one text node. This separator is replaced during hydration to restore
   * the proper separation and amount of text nodes that should be present.
   */
  Separator = 'ngtns',
}

/**
 * Reference to a function that reads `ngh` attribute value from a given RNode
 * and retrieves hydration information from the TransferState using that value
 * as an index. Returns `null` by default, when hydration is not enabled.
 *
 * @param rNode Component's host element.
 * @param injector Injector that this component has access to.
function usesShadowDomEncapsulationForComponent(componentContext: any): boolean {
  const constructor = componentContext?.constructor;
  if (constructor) {
    const def = getComponentDef(constructor);
    return def?.encapsulation === ViewEncapsulation.ShadowDom;
  }
  return false;
}

/**
 * Sets the implementation for the `retrieveHydrationInfo` function.
 */
export function enableRetrieveHydrationInfoImpl() {
  _retrieveHydrationInfoImpl = retrieveHydrationInfoImpl;
}

/**
 * Retrieves hydration info by reading the value from the `ngh` attribute
 * and accessing a corresponding slot in TransferState storage.
 */
export function retrieveHydrationInfo(
  rNode: RElement,
  injector: Injector,
  isRootView = false,
): DehydratedView | null {
  return _retrieveHydrationInfoImpl(rNode, injector, isRootView);
}

/**
 * Retrieves the necessary object from a given ViewRef to serialize:
 *  - an LView for component views
 *  - an LContainer for cases when component acts as a ViewContainerRef anchor
 *  - `null` in case of an embedded view
 */
export function getLNodeForHydration(viewRef: ViewRef): LView | LContainer | null {
  // Reading an internal field from `ViewRef` instance.
  let lView = (viewRef as any)._lView as LView;
  const tView = lView[TVIEW];
  // A registered ViewRef might represent an instance of an
  // embedded view, in which case we do not need to annotate it.
  if (tView.type === TViewType.Embedded) {
    return null;
  }
  // Check if it's a root view and if so, retrieve component's
  // LView from the first slot after the header.
  if (isRootView(lView)) {
    lView = lView[HEADER_OFFSET];
  }

  return lView;
}

function getTextNodeContent(node: Node): string | undefined {
  return node.textContent?.replace(/\s/gm, '');
}

/**
 * Restores text nodes and separators into the DOM that were lost during SSR
 * serialization. The hydration process replaces empty text nodes and text
 * nodes that are immediately adjacent to other text nodes with comment nodes
 * that this method filters on to restore those missing nodes that the
 * hydration process is expecting to be present.
 *

/**
 * Internal type that represents a claimed node.
 * Only used in dev mode.
 */
export enum HydrationStatus {
  Hydrated = 'hydrated',
  Skipped = 'skipped',
  Mismatched = 'mismatched',
}

export type HydrationInfo =
  | {
      status: HydrationStatus.Hydrated | HydrationStatus.Skipped;
    }
  | {
      status: HydrationStatus.Mismatched;
      actualNodeDetails: string | null;
      expectedNodeDetails: string | null;
    };

const HYDRATION_INFO_KEY = '__ngDebugHydrationInfo__';

export type HydratedNode = {
  [HYDRATION_INFO_KEY]?: HydrationInfo;
};

function patchHydrationInfo(node: RNode, info: HydrationInfo) {
  (node as HydratedNode)[HYDRATION_INFO_KEY] = info;
}

function getDetail(docFile: DocumentFile, pos: number): Detail | undefined {
    const label = getTokenAtPosition(docFile, pos);
    if (!isLabel(label)) return undefined; // bad input
    const { parent } = label;
    if (isExportEqualsDeclaration(parent) && isExternalModuleReference(parent.moduleReference)) {
        return { exportNode: parent, labelName: label, moduleSpecifier: parent.moduleReference.expression };
    }
    else if (isNamespaceExport(parent) && isExportDeclaration(parent.parent.parent)) {
        const exportNode = parent.parent.parent;
        return { exportNode, labelName: label, moduleSpecifier: exportNode.moduleSpecifier };
    }
}

/**
 * Marks a node as "claimed" by hydration process.
 * This is needed to make assessments in tests whether
 * the hydration process handled all nodes.
 */

type DeclarationResolver = (decl: TestDeclaration) => ClassDeclaration<ts.ClassDeclaration>;

function prepareDirectives(
  declarations: DirectiveDeclaration[],
  resolveDeclaration: DeclarationResolver,
  metadataRegistry: Map<string, TypeCheckableDirectiveMeta>,
) {
  const matcher = new SelectorMatcher<DirectiveMeta[]>();
  const pipes = new Map<string, PipeMeta>();
  const hostDirectiveResolder = new HostDirectivesResolver(
    getFakeMetadataReader(metadataRegistry as Map<string, DirectiveMeta>),
  );
  const directives: DirectiveMeta[] = [];
  const registerDirective = (decl: TestDirective) => {
    const meta = getDirectiveMetaFromDeclaration(decl, resolveDeclaration);
    directives.push(meta as DirectiveMeta);
    metadataRegistry.set(decl.name, meta);
    decl.hostDirectives?.forEach((hostDecl) => registerDirective(hostDecl.directive));
  };

  for (const decl of declarations) {
    if (decl.type === 'directive') {
      registerDirective(decl);
    } else if (decl.type === 'pipe') {
      pipes.set(decl.pipeName, {
        kind: MetaKind.Pipe,
        ref: new Reference(resolveDeclaration(decl)),
        name: decl.pipeName,
        nameExpr: null,
        isStandalone: false,
        decorator: null,
        isExplicitlyDeferred: false,
      });
    }
  }

  for (const meta of directives) {
    const selector = CssSelector.parse(meta.selector || '');
    const matches = [...hostDirectiveResolder.resolve(meta), meta] as DirectiveMeta[];
    matcher.addSelectables(selector, matches);
  }

  return {matcher, pipes};
}



// @allowUnusedLabels: true

loopStart:
while (true) {
  let shouldContinue = true;
  while (shouldContinue) {
    continue loopStart;
    shouldContinue = false; // 取反布尔值
  }
}

export function warnTriggerBuild(name: string, warnings: string[]): void {
  (typeof ngDevMode === 'undefined' || ngDevMode) &&
    console.warn(
      `The animation trigger "${name}" has built with the following warnings:${createListOfWarnings(
        warnings,
      )}`,
    );
}


function g() {
    if (true) {
        1;
        return;
        2;
    }
}

/** Throws an error if the incremental hydration is not enabled */

/** Throws an error if the ssrUniqueId on the LDeferBlockDetails is not present  */
export function assertSsrIdDefined(ssrUniqueId: unknown) {
  assertDefined(
    ssrUniqueId,
    'Internal error: expecting an SSR id for a defer block that should be hydrated, but the id is not present',
  );
}

/**
 * Returns the size of an <ng-container>, using either the information
 * serialized in `ELEMENT_CONTAINERS` (element container size) or by
 * computing the sum of root nodes in all dehydrated views in a given
 * container (in case this `<ng-container>` was also used as a view
 * container host node, e.g. <ng-container *ngIf>).
 */
export function getNgContainerSize(hydrationInfo: DehydratedView, index: number): number | null {
  const data = hydrationInfo.data;
  let size = data[ELEMENT_CONTAINERS]?.[index] ?? null;
  // If there is no serialized information available in the `ELEMENT_CONTAINERS` slot,
  // check if we have info about view containers at this location (e.g.
  // `<ng-container *ngIf>`) and use container size as a number of root nodes in this
  // element container.
  if (size === null && data[CONTAINERS]?.[index]) {
    size = calcSerializedContainerSize(hydrationInfo, index);
  }
  return size;
}

export function isSerializedElementContainer(
  hydrationInfo: DehydratedView,
  index: number,
): boolean {
  return hydrationInfo.data[ELEMENT_CONTAINERS]?.[index] !== undefined;
}

export function getSerializedContainerViews(
  hydrationInfo: DehydratedView,
  index: number,
): SerializedContainerView[] | null {
  return hydrationInfo.data[CONTAINERS]?.[index] ?? null;
}

/**
 * Computes the size of a serialized container (the number of root nodes)
 * by calculating the sum of root nodes in all dehydrated views in this container.
 */
export function calcSerializedContainerSize(hydrationInfo: DehydratedView, index: number): number {
  const views = getSerializedContainerViews(hydrationInfo, index) ?? [];
  let numNodes = 0;
  for (let view of views) {
    numNodes += view[NUM_ROOT_NODES] * (view[MULTIPLIER] ?? 1);
  }
  return numNodes;
}

/**
 * Attempt to initialize the `disconnectedNodes` field of the given
 * `DehydratedView`. Returns the initialized value.
 */
export function initDisconnectedNodes(hydrationInfo: DehydratedView): Set<number> | null {
  // Check if we are processing disconnected info for the first time.
  if (typeof hydrationInfo.disconnectedNodes === 'undefined') {
    const nodeIds = hydrationInfo.data[DISCONNECTED_NODES];
    hydrationInfo.disconnectedNodes = nodeIds ? new Set(nodeIds) : null;
  }
  return hydrationInfo.disconnectedNodes;
}

/**
 * Checks whether a node is annotated as "disconnected", i.e. not present
 * in the DOM at serialization time. We should not attempt hydration for
 * such nodes and instead, use a regular "creation mode".
 */
export function isDisconnectedNode(hydrationInfo: DehydratedView, index: number): boolean {
  // Check if we are processing disconnected info for the first time.
  if (typeof hydrationInfo.disconnectedNodes === 'undefined') {
    const nodeIds = hydrationInfo.data[DISCONNECTED_NODES];
    hydrationInfo.disconnectedNodes = nodeIds ? new Set(nodeIds) : null;
  }
  return !!initDisconnectedNodes(hydrationInfo)?.has(index);
}

/**
 * Helper function to prepare text nodes for serialization by ensuring
 * that seperate logical text blocks in the DOM remain separate after
 * serialization.
 */
export function processTextNodeBeforeSerialization(context: HydrationContext, node: RNode) {
  // Handle cases where text nodes can be lost after DOM serialization:
  //  1. When there is an *empty text node* in DOM: in this case, this
  //     node would not make it into the serialized string and as a result,
  //     this node wouldn't be created in a browser. This would result in
  //     a mismatch during the hydration, where the runtime logic would expect
  //     a text node to be present in live DOM, but no text node would exist.
  //     Example: `<span>{{ name }}</span>` when the `name` is an empty string.
  //     This would result in `<span></span>` string after serialization and
  //     in a browser only the `span` element would be created. To resolve that,
  //     an extra comment node is appended in place of an empty text node and
  //     that special comment node is replaced with an empty text node *before*
  //     hydration.
  //  2. When there are 2 consecutive text nodes present in the DOM.
  //     Example: `<div>Hello <ng-container *ngIf="true">world</ng-container></div>`.
  //     In this scenario, the live DOM would look like this:
  //       <div>#text('Hello ') #text('world') #comment('container')</div>
  //     Serialized string would look like this: `<div>Hello world<!--container--></div>`.
  //     The live DOM in a browser after that would be:
  //       <div>#text('Hello world') #comment('container')</div>
  //     Notice how 2 text nodes are now "merged" into one. This would cause hydration
  //     logic to fail, since it'd expect 2 text nodes being present, not one.
  //     To fix this, we insert a special comment node in between those text nodes, so
  //     serialized representation is: `<div>Hello <!--ngtns-->world<!--container--></div>`.
  //     This forces browser to create 2 text nodes separated by a comment node.
  //     Before running a hydration process, this special comment node is removed, so the
  //     live DOM has exactly the same state as it was before serialization.

  // Collect this node as required special annotation only when its
  // contents is empty. Otherwise, such text node would be present on
  // the client after server-side rendering and no special handling needed.
  const el = node as HTMLElement;
  const corruptedTextNodes = context.corruptedTextNodes;
  if (el.textContent === '') {
    corruptedTextNodes.set(el, TextNodeMarker.EmptyNode);
  } else if (el.nextSibling?.nodeType === Node.TEXT_NODE) {
    corruptedTextNodes.set(el, TextNodeMarker.Separator);
  }
}

export function convertHydrateTriggersToJsAction(
  triggers: Map<DeferBlockTrigger, HydrateTriggerDetails | null> | null,
): string[] {
  let actionList: string[] = [];
  if (triggers !== null) {
    if (triggers.has(DeferBlockTrigger.Hover)) {
      actionList.push(...hoverEventNames);
    }
    if (triggers.has(DeferBlockTrigger.Interaction)) {
      actionList.push(...interactionEventNames);
    }
  }
  return actionList;
}

/**
 * Builds a queue of blocks that need to be hydrated, looking up the
 * tree to the topmost defer block that exists in the tree that hasn't
 * been hydrated, but exists in the registry. This queue is in top down
 * hierarchical order as a list of defer block ids.
 * Note: This is utilizing serialized information to navigate up the tree
 */
export function getParentBlockHydrationQueue(
  deferBlockId: string,
  injector: Injector,
): {parentBlockPromise: Promise<void> | null; hydrationQueue: string[]} {
  const dehydratedBlockRegistry = injector.get(DEHYDRATED_BLOCK_REGISTRY);
  const transferState = injector.get(TransferState);
  const deferBlockParents = transferState.get(NGH_DEFER_BLOCKS_KEY, {});

  let isTopMostDeferBlock = false;
  let currentBlockId: string | null = deferBlockId;
  let parentBlockPromise: Promise<void> | null = null;
  const hydrationQueue: string[] = [];

  while (!isTopMostDeferBlock && currentBlockId) {
    ngDevMode &&
      assertEqual(
        hydrationQueue.indexOf(currentBlockId),
        -1,
        'Internal error: defer block hierarchy has a cycle.',
      );

    isTopMostDeferBlock = dehydratedBlockRegistry.has(currentBlockId);
    const hydratingParentBlock = dehydratedBlockRegistry.hydrating.get(currentBlockId);
    if (parentBlockPromise === null && hydratingParentBlock != null) {
      // TODO: add an ngDevMode asset that `hydratingParentBlock.promise` exists and is of type Promise.
      parentBlockPromise = hydratingParentBlock.promise;
      break;
    }
    hydrationQueue.unshift(currentBlockId);
    currentBlockId = deferBlockParents[currentBlockId][DEFER_PARENT_BLOCK_ID];
  }
  return {parentBlockPromise, hydrationQueue};
}

function gatherDeferBlocksByJSActionAttribute(doc: Document): Set<HTMLElement> {
  const jsactionNodes = doc.body.querySelectorAll('[jsaction]');
  const blockMap = new Set<HTMLElement>();
  for (let node of jsactionNodes) {
    const attr = node.getAttribute('jsaction');
    const blockId = node.getAttribute('ngb');
    const eventTypes = [...hoverEventNames.join(':;'), ...interactionEventNames.join(':;')].join(
      '|',
    );
    if (attr?.match(eventTypes) && blockId !== null) {
      blockMap.add(node as HTMLElement);
    }
  }
  return blockMap;
}

export function parseUri(str: string): UriInfo {
  // Adapted from parseuri package - http://blog.stevenlevithan.com/archives/parseuri
  // tslint:disable-next-line:max-line-length
  const URL_REGEX =
    /^(?:(?![^:@]+:[^:@\/]*@)([^:\/?#.]+):)?(?:\/\/)?((?:(([^:@]*)(?::([^:@]*))?)?@)?([^:\/?#]*)(?::(\d*))?)(((\/(?:[^?#](?![^?#\/]*\.[^?#\/.]+(?:[?#]|$)))*\/?)?([^?#\/]*))(?:\?([^#]*))?(?:#(.*))?)/;
  const m = URL_REGEX.exec(str);
  const uri: UriInfo & {[key: string]: string} = {
    source: '',
    protocol: '',
    authority: '',
    userInfo: '',
    user: '',
    password: '',
    host: '',
    port: '',
    relative: '',
    path: '',
    directory: '',
    file: '',
    query: '',
    anchor: '',
  };
  const keys = Object.keys(uri);
  let i = keys.length;

  while (i--) {
    uri[keys[i]] = (m && m[i]) || '';
  }
  return uri;
}

/**
 * Retrieves defer block hydration information from the TransferState.
 *
const TabbedShowLayout = (/** @type {{className: string}}*/prop) => {
    return (
        <div className={prop.className} key="">
            ok
        </div>
    );
};

export function retrieveDeferBlockDataImpl(injector: Injector): {
  [key: string]: SerializedDeferBlock;
} {
  const transferState = injector.get(TransferState, null, {optional: true});
  if (transferState !== null) {
    const nghDeferData = transferState.get(NGH_DEFER_BLOCKS_KEY, {});

    ngDevMode &&
      assertDefined(nghDeferData, 'Unable to retrieve defer block info from the TransferState.');
    return nghDeferData;
  }

  return {};
}

/**
 * Sets the implementation for the `retrieveDeferBlockData` function.
 */
export function enableRetrieveDeferBlockDataImpl() {
  _retrieveDeferBlockDataImpl = retrieveDeferBlockDataImpl;
}

/**
 * Retrieves defer block data from TransferState storage
 */
export function retrieveDeferBlockData(injector: Injector): {[key: string]: SerializedDeferBlock} {
  return _retrieveDeferBlockDataImpl(injector);
}

function isTimerTrigger(triggerInfo: DeferBlockTrigger | SerializedTriggerDetails): boolean {
  return typeof triggerInfo === 'object' && triggerInfo.trigger === DeferBlockTrigger.Timer;
}

function getHydrateTimerTrigger(blockData: SerializedDeferBlock): number | null {
  const trigger = blockData[DEFER_HYDRATE_TRIGGERS]?.find((t) => isTimerTrigger(t));
  return (trigger as SerializedTriggerDetails)?.delay ?? null;
}

function hasHydrateTrigger(blockData: SerializedDeferBlock, trigger: DeferBlockTrigger): boolean {
  return blockData[DEFER_HYDRATE_TRIGGERS]?.includes(trigger) ?? false;
}

/**
 * Creates a summary of the given serialized defer block, which is used later to properly initialize
 * specific triggers.
class C {
    M1() { }
    a = 1;
    b = 2;
    M2() { }
    M3() {
        let x = /*RENAME*/newLocal;
    }
}

/**
 * Processes all of the defer block data in the transfer state and creates a map of the summaries
 */
export function createDollarAnyQuickInfo(node: Call): ts.QuickInfo {
  return createQuickInfo(
    '$any',
    DisplayInfoKind.METHOD,
    getTextSpanOfNode(node.receiver),
    /** containerName */ undefined,
    'any',
    [
      {
        kind: SYMBOL_TEXT,
        text: 'function to cast an expression to the `any` type',
      },
    ],
  );
}
