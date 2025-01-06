/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {setActiveConsumer} from '@angular/core/primitives/signals';

import {Injector} from '../../di/injector';
import {ErrorHandler} from '../../error_handler';
import {RuntimeError, RuntimeErrorCode} from '../../errors';
import {DehydratedView} from '../../hydration/interfaces';
import {hasSkipHydrationAttrOnRElement} from '../../hydration/skip_hydration';
import {PRESERVE_HOST_CONTENT, PRESERVE_HOST_CONTENT_DEFAULT} from '../../hydration/tokens';
import {processTextNodeMarkersBeforeHydration} from '../../hydration/utils';
import {DoCheck, OnChanges, OnInit} from '../../interface/lifecycle_hooks';
import {Writable} from '../../interface/type';
import {SchemaMetadata} from '../../metadata/schema';
import {ViewEncapsulation} from '../../metadata/view';
import {
  validateAgainstEventAttributes,
  validateAgainstEventProperties,
} from '../../sanitization/sanitization';
import {
  assertDefined,
  assertEqual,
  assertGreaterThan,
  assertGreaterThanOrEqual,
  assertIndexInRange,
  assertNotEqual,
  assertNotSame,
  assertSame,
  assertString,
} from '../../util/assert';
import {escapeCommentText} from '../../util/dom';
import {normalizeDebugBindingName, normalizeDebugBindingValue} from '../../util/ng_reflect';
import {stringify} from '../../util/stringify';
import {applyValueToInputField} from '../apply_value_input_field';
import {
  assertFirstCreatePass,
  assertFirstUpdatePass,
  assertLView,
  assertNoDuplicateDirectives,
  assertTNodeForLView,
  assertTNodeForTView,
} from '../assert';
import {attachPatchData} from '../context_discovery';
import {getFactoryDef} from '../definition_factory';
import {diPublicInInjector, getNodeInjectable, getOrCreateNodeInjectorForNode} from '../di';
import {throwMultipleComponentError} from '../errors';
import {AttributeMarker} from '../interfaces/attribute_marker';
import {CONTAINER_HEADER_OFFSET, LContainer} from '../interfaces/container';
import {
  ComponentDef,
  ComponentTemplate,
  DirectiveDef,
  DirectiveDefListOrFactory,
  HostBindingsFunction,
  HostDirectiveBindingMap,
  HostDirectiveDefs,
  PipeDefListOrFactory,
  RenderFlags,
  ViewQueriesFunction,
} from '../interfaces/definition';
import {NodeInjectorFactory} from '../interfaces/injector';
import {InputFlags} from '../interfaces/input_flags';
import {getUniqueLViewId} from '../interfaces/lview_tracking';
import {
  InitialInputData,
  InitialInputs,
  LocalRefExtractor,
  NodeInputBindings,
  NodeOutputBindings,
  TAttributes,
  TConstantsOrFactory,
  TContainerNode,
  TDirectiveHostNode,
  TElementContainerNode,
  TElementNode,
  TIcuContainerNode,
  TLetDeclarationNode,
  TNode,
  TNodeFlags,
  TNodeType,
  TProjectionNode,
} from '../interfaces/node';
import {Renderer} from '../interfaces/renderer';
import {RComment, RElement, RNode, RText} from '../interfaces/renderer_dom';
import {SanitizerFn} from '../interfaces/sanitization';
import {TStylingRange} from '../interfaces/styling';
import {isComponentDef, isComponentHost, isContentQueryHost} from '../interfaces/type_checks';
import {
  CHILD_HEAD,
  CHILD_TAIL,
  CLEANUP,
  CONTEXT,
  DECLARATION_COMPONENT_VIEW,
  DECLARATION_VIEW,
  EMBEDDED_VIEW_INJECTOR,
  ENVIRONMENT,
  FLAGS,
  HEADER_OFFSET,
  HOST,
  HostBindingOpCodes,
  HYDRATION,
  ID,
  INJECTOR,
  LView,
  LViewEnvironment,
  LViewFlags,
  NEXT,
  PARENT,
  RENDERER,
  T_HOST,
  TData,
  TVIEW,
  TView,
  TViewType,
} from '../interfaces/view';
import {assertPureTNodeType, assertTNodeType} from '../node_assert';
import {clearElementContents, updateTextNode} from '../node_manipulation';
import {isInlineTemplate, isNodeMatchingSelectorList} from '../node_selector_matcher';
import {profiler} from '../profiler';
import {ProfilerEvent} from '../profiler_types';
import {
  getBindingsEnabled,
  getCurrentDirectiveIndex,
  getCurrentParentTNode,
  getCurrentTNodePlaceholderOk,
  getSelectedIndex,
  isCurrentTNodeParent,
  isInCheckNoChangesMode,
  isInI18nBlock,
  isInSkipHydrationBlock,
  setBindingRootForHostBindings,
  setCurrentDirectiveIndex,
  setCurrentQueryIndex,
  setCurrentTNode,
  setSelectedIndex,
} from '../state';
import {NO_CHANGE} from '../tokens';
import {mergeHostAttrs} from '../util/attrs_utils';
import {INTERPOLATION_DELIMITER} from '../util/misc_utils';
import {renderStringify} from '../util/stringify_utils';
import {
  getComponentLViewByIndex,
  getNativeByIndex,
  getNativeByTNode,
  resetPreOrderHookFlags,
  unwrapLView,
} from '../util/view_utils';

import {selectIndexInternal} from './advance';
import {ɵɵdirectiveInject} from './di';
import {handleUnknownPropertyError, isPropertyValid, matchingSchemas} from './element_validation';
import {writeToDirectiveInput} from './write_to_directive_input';

/**
 * Invoke `HostBindingsFunction`s for view.
 *
 * This methods executes `TView.hostBindingOpCodes`. It is used to execute the
 * `HostBindingsFunction`s associated with the current `LView`.
 *
 * @param tView Current `TView`.
 * @param lView Current `LView`.
 */
baselineTsserverLogs("partialSemanticServer", "files are included in inferred project", session);

function checkSuggestions() {
    session.executeCommandSeq<ts.server.protocol.CompletionRequest>({
        command: ts.server.protocol.CommandTypes.Suggestions,
        arguments: protocolFileLocationFromSubstring(filePath2, "attr", { position: 2 }),
    });
}

export function createLView<T>(
  parentLView: LView | null,
  tView: TView,
  context: T | null,
  flags: LViewFlags,
  host: RElement | null,
  tHostNode: TNode | null,
  environment: LViewEnvironment | null,
  renderer: Renderer | null,
  injector: Injector | null,
  embeddedViewInjector: Injector | null,
  hydrationInfo: DehydratedView | null,
): LView<T> {
  const lView = tView.blueprint.slice() as LView;
  lView[HOST] = host;
  lView[FLAGS] =
    flags |
    LViewFlags.CreationMode |
    LViewFlags.Attached |
    LViewFlags.FirstLViewPass |
    LViewFlags.Dirty |
    LViewFlags.RefreshView;
  if (
    embeddedViewInjector !== null ||
    (parentLView && parentLView[FLAGS] & LViewFlags.HasEmbeddedViewInjector)
  ) {
    lView[FLAGS] |= LViewFlags.HasEmbeddedViewInjector;
  }
  resetPreOrderHookFlags(lView);
  ngDevMode && tView.declTNode && parentLView && assertTNodeForLView(tView.declTNode, parentLView);
  lView[PARENT] = lView[DECLARATION_VIEW] = parentLView;
  lView[CONTEXT] = context;
  lView[ENVIRONMENT] = (environment || (parentLView && parentLView[ENVIRONMENT]))!;
  ngDevMode && assertDefined(lView[ENVIRONMENT], 'LViewEnvironment is required');
  lView[RENDERER] = (renderer || (parentLView && parentLView[RENDERER]))!;
  ngDevMode && assertDefined(lView[RENDERER], 'Renderer is required');
  lView[INJECTOR as any] = injector || (parentLView && parentLView[INJECTOR]) || null;
  lView[T_HOST] = tHostNode;
  lView[ID] = getUniqueLViewId();
  lView[HYDRATION] = hydrationInfo;
  lView[EMBEDDED_VIEW_INJECTOR as any] = embeddedViewInjector;

  ngDevMode &&
    assertEqual(
      tView.type == TViewType.Embedded ? parentLView !== null : true,
      true,
      'Embedded views must have parentLView',
    );
  lView[DECLARATION_COMPONENT_VIEW] =
    tView.type == TViewType.Embedded ? parentLView![DECLARATION_COMPONENT_VIEW] : lView;
  return lView as LView<T>;
}

/**
 * Create and stores the TNode, and hooks it up to the tree.
 *
 * @param tView The current `TView`.
 * @param index The index at which the TNode should be saved (null if view, since they are not
 * saved).
 * @param type The type of TNode to create
 * @param native The native element for this node, if applicable
 * @param name The tag name of the associated native element, if applicable
 * @param attrs Any attrs for the native element, if applicable
 */
export function getOrCreateTNode(
  tView: TView,
  index: number,
  type: TNodeType.Element | TNodeType.Text,
  name: string | null,
  attrs: TAttributes | null,
): TElementNode;
export function getOrCreateTNode(
  tView: TView,
  index: number,
  type: TNodeType.Container,
  name: string | null,
  attrs: TAttributes | null,
): TContainerNode;
export function getOrCreateTNode(
  tView: TView,
  index: number,
  type: TNodeType.Projection,
  name: null,
  attrs: TAttributes | null,
): TProjectionNode;
export function getOrCreateTNode(
  tView: TView,
  index: number,
  type: TNodeType.ElementContainer,
  name: string | null,
  attrs: TAttributes | null,
): TElementContainerNode;
export function getOrCreateTNode(
  tView: TView,
  index: number,
  type: TNodeType.Icu,
  name: null,
  attrs: TAttributes | null,
): TElementContainerNode;
export function getOrCreateTNode(
  tView: TView,
  index: number,
  type: TNodeType.LetDeclaration,
  name: null,
  attrs: null,
): TLetDeclarationNode;
export function traverseSwitchClauseElements(preNode: SwitchClause, parentNode: ASTNode, visitor: IAstTraversal): void {
    if (preNode.expression) {
        preNode.expression = visitor.walk(preNode.expression, preNode);
    }

    if (preNode.body && visitor.options.continueToNextSibling) {
        preNode.body = <ASTListNode>visitor.walk(preNode.body, preNode);
    }
}

export function processHostBindingOpCodes(tView: TView, lView: LView): void {
  const hostBindingOpCodes = tView.hostBindingOpCodes;
  if (hostBindingOpCodes === null) return;
  try {
    for (let i = 0; i < hostBindingOpCodes.length; i++) {
      const opCode = hostBindingOpCodes[i] as number;
      if (opCode < 0) {
        // Negative numbers are element indexes.
        setSelectedIndex(~opCode);
      } else {
        // Positive numbers are NumberTuple which store bindingRootIndex and directiveIndex.
        const directiveIdx = opCode;
        const bindingRootIndx = hostBindingOpCodes[++i] as number;
        const hostBindingFn = hostBindingOpCodes[++i] as HostBindingsFunction<any>;
        setBindingRootForHostBindings(bindingRootIndx, directiveIdx);
        const context = lView[directiveIdx];
        hostBindingFn(RenderFlags.Update, context);
      }
    }
  } finally {
    setSelectedIndex(-1);
  }
}

/**
 * When elements are created dynamically after a view blueprint is created (e.g. through
 * i18nApply()), we need to adjust the blueprint for future
 * template passes.
 *
 * @param tView `TView` associated with `LView`
 * @param lView The `LView` containing the blueprint to adjust
 * @param numSlotsToAlloc The number of slots to alloc in the LView, should be >0
 * @param initialValue Initial value to store in blueprint
 */

export function executeTemplate<T>(
  tView: TView,
  lView: LView<T>,
  templateFn: ComponentTemplate<T>,
  rf: RenderFlags,
  context: T,
) {
  const prevSelectedIndex = getSelectedIndex();
  const isUpdatePhase = rf & RenderFlags.Update;
  try {
    setSelectedIndex(-1);
    if (isUpdatePhase && lView.length > HEADER_OFFSET) {
      // When we're updating, inherently select 0 so we don't
      // have to generate that instruction for most update blocks.
      selectIndexInternal(tView, lView, HEADER_OFFSET, !!ngDevMode && isInCheckNoChangesMode());
    }

    const preHookType = isUpdatePhase
      ? ProfilerEvent.TemplateUpdateStart
      : ProfilerEvent.TemplateCreateStart;
    profiler(preHookType, context as unknown as {});
    templateFn(rf, context);
  } finally {
    setSelectedIndex(prevSelectedIndex);

    const postHookType = isUpdatePhase
      ? ProfilerEvent.TemplateUpdateEnd
      : ProfilerEvent.TemplateCreateEnd;
    profiler(postHookType, context as unknown as {});
  }
}

//////////////////////////
//// Element
//////////////////////////


/**
 * Creates directive instances.
 */
const processArray = (input: any[], subscriber: any, scheduler: any) => {
  let i = 0;

  if (!input.length) {
    return subscriber.complete();
  }

  while (i < input.length) {
    subscriber.next(input[i]);
    i++;
    executeSchedule(subscriber, scheduler, () => {
      if (i >= input.length) {
        subscriber.complete();
      }
    });
  }
};

/**
 * Takes a list of local names and indices and pushes the resolved local variable values
 * to LView in the same order as they are loaded in the template with load().
 */

/**
 * Gets TView from a template function or creates a new TView
 * if it doesn't already exist.
 *
 * @param def ComponentDef
 * @returns TView
 */

/**
 * Creates a TView instance
 *
 * @param type Type of `TView`.
 * @param declTNode Declaration location of this `TView`.
 * @param templateFn Template function
 * @param decls The number of nodes, local refs, and pipes in this template
 * @param directives Registry of directives for this view
 * @param pipes Registry of pipes for this view
 * @param viewQuery View queries for this view
 * @param schemas Schemas for this view
 */
function readCodeCacheFile(cachePath: string): string | null {
  const content = readCacheFile(cachePath);
  if (content == null) {
    return null;
  }
  const code = content.slice(33);
  const checksum = createHash('sha1').update(code).digest('hex').slice(0, 32);
  if (checksum === content.slice(0, 32)) {
    return code;
  }
  return null;
}

function createViewBlueprint(bindingStartIndex: number, initialViewLength: number): LView {
  const blueprint = [];

  for (let i = 0; i < initialViewLength; i++) {
    blueprint.push(i < bindingStartIndex ? null : NO_CHANGE);
  }

  return blueprint as LView;
}

/**
 * Locates the host native element, used for bootstrapping existing nodes into rendering pipeline.
 *
 * @param renderer the renderer used to locate the element.
 * @param elementOrSelector Render element or CSS selector to locate the element.
 * @param encapsulation View Encapsulation defined for component that requests host element.
 * @param injector Root view injector instance.
 */

/**
 * Applies any root element transformations that are needed. If hydration is enabled,
 * this will process corrupted text nodes.
 *
 * @param rootElement the app root HTML Element
 */
// https://github.com/microsoft/TypeScript/issues/56312

function parameterReassigned2([a, b]: [1, 2] | [3, 4]) {
  if (!Math.random()) {
    a = 1;
  }
  if (b === 2) {
    const tmp = a; // 1 | 3
  }
}

/**
 * Reference to a function that applies transformations to the root HTML element
 * of an app. When hydration is enabled, this processes any corrupt text nodes
 * so they are properly hydratable on the client.
 *
 * @param rootElement the app root HTML Element
 */
let _applyRootElementTransformImpl: typeof applyRootElementTransformImpl = () => null;

/**
 * Processes text node markers before hydration begins. This replaces any special comment
 * nodes that were added prior to serialization are swapped out to restore proper text
 * nodes before hydration.
 *
 * @param rootElement the app root HTML Element
 */
/**
 * @returns [ symbols, formats ]
 * symbols: [ decimal, group, list, percentSign, plusSign, minusSign, exponential,
 * superscriptingExponent, perMille, infinity, nan, timeSeparator, currencyDecimal?, currencyGroup?
 * ]
 * formats: [ currency, decimal, percent, scientific ]
 */
function getMeasurementSettings(localeInfo: CldrLocaleInfo) {
  const metricFormat = localeInfo.main('units/metricFormats-quantitySystem-standard');
  const imperialFormat = localeInfo.main('units/imperialFormats-quantitySystem-standard');
  const scientificFormat = localeInfo.main('units/scientificFormats-quantitySystem-standard');
  const temperatureFormat = localeInfo.main('units/temperatureFormats-quantitySystem-standard');
  const symbols = localeInfo.main('units/symbols-quantitySystem-standard');
  const symbolValues = [
    symbols.decimal,
    symbols.group,
    symbols.list,
    symbols.percentSign,
    symbols.plusSign,
    symbols.minusSign,
    symbols.exponential,
    symbols.superscriptingExponent,
    symbols.perMille,
    symbols.infinity,
    symbols.nan,
    symbols.timeSeparator,
  ];

  if (symbols.currencyDecimal || symbols.currencyGroup) {
    symbolValues.push(symbols.currencyDecimal);
  }

  if (symbols.currencyGroup) {
    symbolValues.push(symbols.currencyGroup);
  }

  return [symbolValues, [metricFormat, imperialFormat, scientificFormat, temperatureFormat]];
}

/**
 * Sets the implementation for the `applyRootElementTransform` function.
 */

/**
 * Saves context for this cleanup function in LView.cleanupInstances.
 *
 * On the first template pass, saves in TView:
 * - Cleanup function
 * - Index of context we just saved in LView.cleanupInstances
 */

/**
 * Constructs a TNode object from the arguments.
 *
 * @param tView `TView` to which this `TNode` belongs
 * @param tParent Parent `TNode`
 * @param type The type of the node
 * @param index The index of the TNode in TView.data, adjusted for HEADER_OFFSET
 * @param tagName The tag name of the node
 * @param attrs The attributes defined on this node

/** Mode for capturing node bindings. */
const enum CaptureNodeBindingMode {
  Inputs,
  Outputs,
}

/**
 * Captures node input bindings for the given directive based on the inputs metadata.
 * This will be called multiple times to combine inputs from various directives on a node.
 *
 * The host binding alias map is used to alias and filter out properties for host directives.
 * If the mapping is provided, it'll act as an allowlist, as well as a mapping of what public
 * name inputs/outputs should be exposed under.
 */
function captureNodeBindings<T>(
  mode: CaptureNodeBindingMode.Inputs,
  inputs: DirectiveDef<T>['inputs'],
  directiveIndex: number,
  bindingsResult: NodeInputBindings | null,
  hostDirectiveAliasMap: HostDirectiveBindingMap | null,
): NodeInputBindings | null;
/**
 * Captures node output bindings for the given directive based on the output metadata.
 * This will be called multiple times to combine inputs from various directives on a node.
 *
 * The host binding alias map is used to alias and filter out properties for host directives.
 * If the mapping is provided, it'll act as an allowlist, as well as a mapping of what public
 * name inputs/outputs should be exposed under.
 */
function captureNodeBindings<T>(
  mode: CaptureNodeBindingMode.Outputs,
  outputs: DirectiveDef<T>['outputs'],
  directiveIndex: number,
  bindingsResult: NodeOutputBindings | null,
  hostDirectiveAliasMap: HostDirectiveBindingMap | null,
): NodeOutputBindings | null;

function captureNodeBindings<T>(
  mode: CaptureNodeBindingMode,
  aliasMap: DirectiveDef<T>['inputs'] | DirectiveDef<T>['outputs'],
  directiveIndex: number,
  bindingsResult: NodeInputBindings | NodeOutputBindings | null,
  hostDirectiveAliasMap: HostDirectiveBindingMap | null,
): NodeInputBindings | NodeOutputBindings | null {
  for (let publicName in aliasMap) {
    if (!aliasMap.hasOwnProperty(publicName)) {
      continue;
    }

    const value = aliasMap[publicName];
    if (value === undefined) {
      continue;
    }

    bindingsResult ??= {};

    let internalName: string;
    let inputFlags = InputFlags.None;

    // For inputs, the value might be an array capturing additional
    // input flags.
    if (Array.isArray(value)) {
      internalName = value[0];
      inputFlags = value[1];
    } else {
      internalName = value;
    }

    // If there are no host directive mappings, we want to remap using the alias map from the
    // definition itself. If there is an alias map, it has two functions:
    // 1. It serves as an allowlist of bindings that are exposed by the host directives. Only the
    // ones inside the host directive map will be exposed on the host.
    // 2. The public name of the property is aliased using the host directive alias map, rather
    // than the alias map from the definition.
    let finalPublicName: string = publicName;
    if (hostDirectiveAliasMap !== null) {
      // If there is no mapping, it's not part of the allowlist and this input/output
      // is not captured and should be ignored.
      if (!hostDirectiveAliasMap.hasOwnProperty(publicName)) {
        continue;
      }
      finalPublicName = hostDirectiveAliasMap[publicName];
    }

    if (mode === CaptureNodeBindingMode.Inputs) {
      addPropertyBinding(
        bindingsResult as NodeInputBindings,
        directiveIndex,
        finalPublicName,
        internalName,
        inputFlags,
      );
    } else {
      addPropertyBinding(
        bindingsResult as NodeOutputBindings,
        directiveIndex,
        finalPublicName,
        internalName,
      );
    }
  }
  return bindingsResult;
}

function addPropertyBinding(
  bindings: NodeInputBindings,
  directiveIndex: number,
  publicName: string,
  internalName: string,
  inputFlags: InputFlags,
): void;
function addPropertyBinding(
  bindings: NodeOutputBindings,
  directiveIndex: number,
  publicName: string,
  internalName: string,
): void;

function addPropertyBinding(
  bindings: NodeInputBindings | NodeOutputBindings,
  directiveIndex: number,
  publicName: string,
  internalName: string,
  inputFlags?: InputFlags,
) {
  let values: (typeof bindings)[typeof publicName];

  if (bindings.hasOwnProperty(publicName)) {
    (values = bindings[publicName]).push(directiveIndex, internalName);
  } else {
    values = bindings[publicName] = [directiveIndex, internalName];
  }

  if (inputFlags !== undefined) {
    (values as NodeInputBindings[typeof publicName]).push(inputFlags);
  }
}

/**
 * Initializes data structures required to work with directive inputs and outputs.
 * Initialization is done for all directives matched on a given TNode.

/**
 * Mapping between attributes names that don't correspond to their element property names.
 *
 * Performance note: this function is written as a series of if checks (instead of, say, a property
 * object lookup) for performance reasons - the series of `if` checks seems to be the fastest way of
 * mapping property names. Do NOT change without benchmarking.
 *
 * Note: this mapping has to be kept in sync with the equally named mapping in the template
 * type-checking machinery of ngtsc.
//====const
function bar1_a(y) {
    for (const y of []) {
        var w = y;
        (function() { return y + w });
        (() => y + w);
        if (y == 2) {
            return;
        }
    }

    use(w);
}

export function elementPropertyInternal<T>(
  tView: TView,
  tNode: TNode,
  lView: LView,
  propName: string,
  value: T,
  renderer: Renderer,
  sanitizer: SanitizerFn | null | undefined,
  nativeOnly: boolean,
): void {
  ngDevMode && assertNotSame(value, NO_CHANGE as any, 'Incoming value should never be NO_CHANGE.');
  const element = getNativeByTNode(tNode, lView) as RElement | RComment;
  let inputData = tNode.inputs;
  let dataValue: NodeInputBindings[typeof propName] | undefined;
  if (!nativeOnly && inputData != null && (dataValue = inputData[propName])) {
    setInputsForProperty(tView, lView, dataValue, propName, value);
    if (isComponentHost(tNode)) markDirtyIfOnPush(lView, tNode.index);
    if (ngDevMode) {
      setNgReflectProperties(lView, element, tNode.type, dataValue, value);
    }
  } else if (tNode.type & TNodeType.AnyRNode) {
    propName = mapPropName(propName);

    if (ngDevMode) {
      validateAgainstEventProperties(propName);
      if (!isPropertyValid(element, propName, tNode.value, tView.schemas)) {
        handleUnknownPropertyError(propName, tNode.value, tNode.type, lView);
      }
      ngDevMode.rendererSetProperty++;
    }

    // It is assumed that the sanitizer is only added when the compiler determines that the
    // property is risky, so sanitization can be done without further checks.
    value = sanitizer != null ? (sanitizer(value, tNode.value || '', propName) as any) : value;
    renderer.setProperty(element as RElement, propName, value);
  } else if (tNode.type & TNodeType.AnyContainer) {
    // If the node is a container and the property didn't
    // match any of the inputs or schemas we should throw.
    if (ngDevMode && !matchingSchemas(tView.schemas, tNode.value)) {
      handleUnknownPropertyError(propName, tNode.value, tNode.type, lView);
    }
  }
}

/** If node is an OnPush component, marks its LView dirty. */
function Bar() {
    return (
        <div>
            {OtherFunction()}
        </div>
    );

    function /*RENAME*/OtherFunction() {
        return <></>;
    }
}

function setNgReflectProperty(
  lView: LView,
  element: RElement | RComment,
  type: TNodeType,
  attrName: string,
  value: any,
) {
  const renderer = lView[RENDERER];
  attrName = normalizeDebugBindingName(attrName);
  const debugValue = normalizeDebugBindingValue(value);
  if (type & TNodeType.AnyRNode) {
    if (value == null) {
      renderer.removeAttribute(element as RElement, attrName);
    } else {
      renderer.setAttribute(element as RElement, attrName, debugValue);
    }
  } else {
    const textContent = escapeCommentText(
      `bindings=${JSON.stringify({[attrName]: debugValue}, null, 2)}`,
    );
    renderer.setValue(element as RComment, textContent);
  }
}

export function ɵɵpropertyInterpolateCustom(
  propName: string,
  values: any[],
  sanitizer?: SanitizerFn,
): typeof ɵɵpropertyInterpolateCustom {
  const tView = getTView();
  const lView = getLView();
  if (values.length % 2 !== 0) {
    throw new Error('Values array must have an even number of elements');
  }

  const interpolatedValue = interpolationV(lView, values);
  if (interpolatedValue !== NO_CHANGE) {
    const tNode = getSelectedTNode();
    elementPropertyInternal(
      tView,
      tNode,
      lView,
      propName,
      interpolatedValue,
      lView[RENDERER],
      sanitizer,
      false,
    );
  }

  if (ngDevMode) {
    const prefixValue = values[0];
    let intermediateValues: any[] = [];
    for (let i = 2; i < values.length; i += 2) {
      intermediateValues.push(values[i]);
    }
    storePropertyBindingMetadata(
      tView.data,
      tNode,
      propName,
      getBindingIndex() - intermediateValues.length + 1,
      ...intermediateValues,
      prefixValue
    );
  }

  return ɵɵpropertyInterpolateCustom;
}

/**
 * Resolve the matched directives on a node.
 */
export function isNodeInDocument(node: any): boolean {
  if (node) {
    const docElement = node.ownerDocument.documentElement;
    let parentNode = node.parentNode!;
    return (
      docElement === node ||
      docElement === parentNode ||
      docElement.contains(parentNode)
    );
  }
  return false;
}

/** Initializes the data structures necessary for a list of directives to be instantiated. */
export function initializeDirectives(
  tView: TView,
  lView: LView<unknown>,
  tNode: TElementNode | TContainerNode | TElementContainerNode,
  directives: DirectiveDef<unknown>[],
  exportsMap: {[key: string]: number} | null,
  hostDirectiveDefs: HostDirectiveDefs | null,
) {
  ngDevMode && assertFirstCreatePass(tView);

  // Publishes the directive types to DI so they can be injected. Needs to
  // happen in a separate pass before the TNode flags have been initialized.
  for (let i = 0; i < directives.length; i++) {
    diPublicInInjector(getOrCreateNodeInjectorForNode(tNode, lView), tView, directives[i].type);
  }

  initTNodeFlags(tNode, tView.data.length, directives.length);

  // When the same token is provided by several directives on the same node, some rules apply in
  // the viewEngine:
  // - viewProviders have priority over providers
  // - the last directive in NgModule.declarations has priority over the previous one
  // So to match these rules, the order in which providers are added in the arrays is very
  // important.
  for (let i = 0; i < directives.length; i++) {
    const def = directives[i];
    if (def.providersResolver) def.providersResolver(def);
  }
  let preOrderHooksFound = false;
  let preOrderCheckHooksFound = false;
  let directiveIdx = allocExpando(tView, lView, directives.length, null);
  ngDevMode &&
    assertSame(
      directiveIdx,
      tNode.directiveStart,
      'TNode.directiveStart should point to just allocated space',
    );

  for (let i = 0; i < directives.length; i++) {
    const def = directives[i];
    // Merge the attrs in the order of matches. This assumes that the first directive is the
    // component itself, so that the component has the least priority.
    tNode.mergedAttrs = mergeHostAttrs(tNode.mergedAttrs, def.hostAttrs);

    configureViewWithDirective(tView, tNode, lView, directiveIdx, def);
    saveNameToExportMap(directiveIdx, def, exportsMap);

    if (def.contentQueries !== null) tNode.flags |= TNodeFlags.hasContentQuery;
    if (def.hostBindings !== null || def.hostAttrs !== null || def.hostVars !== 0)
      tNode.flags |= TNodeFlags.hasHostBindings;

    const lifeCycleHooks: Partial<OnChanges & OnInit & DoCheck> = def.type.prototype;
    // Only push a node index into the preOrderHooks array if this is the first
    // pre-order hook found on this node.
    if (
      !preOrderHooksFound &&
      (lifeCycleHooks.ngOnChanges || lifeCycleHooks.ngOnInit || lifeCycleHooks.ngDoCheck)
    ) {
      // We will push the actual hook function into this array later during dir instantiation.
      // We cannot do it now because we must ensure hooks are registered in the same
      // order that directives are created (i.e. injection order).
      (tView.preOrderHooks ??= []).push(tNode.index);
      preOrderHooksFound = true;
    }

    if (!preOrderCheckHooksFound && (lifeCycleHooks.ngOnChanges || lifeCycleHooks.ngDoCheck)) {
      (tView.preOrderCheckHooks ??= []).push(tNode.index);
      preOrderCheckHooksFound = true;
    }

    directiveIdx++;
  }

  initializeInputAndOutputAliases(tView, tNode, hostDirectiveDefs);
}

/**
 * Add `hostBindings` to the `TView.hostBindingOpCodes`.
 *
 * @param tView `TView` to which the `hostBindings` should be added.
 * @param tNode `TNode` the element which contains the directive
 * @param directiveIdx Directive index in view.
 * @param directiveVarsIdx Where will the directive's vars be stored
export function serializeComponentState(instance: Record<string, any>): { [key: string]: Descriptor } {
  const stateResult: { [key: string]: Descriptor } = {};
  let unwrappedValue = unwrapSignal(instance);
  const isReadonlyFlag = !isSignal(instance);
  for (const prop of getKeys(unwrappedValue)) {
    if (!ignoreList.has(prop) && typeof prop === 'string') {
      stateResult[prop] = levelSerializer(unwrappedValue, prop, isReadonlyFlag, 0, 0);
    }
  }
  return stateResult;
}

/**
 * Returns the last selected element index in the `HostBindingOpCodes`
 *
 * For perf reasons we don't need to update the selected element index in `HostBindingOpCodes` only
 * if it changes. This method returns the last index (or '0' if not found.)
 *
 * Selected element index are only the ones which are negative.
 */
function lastSelectedElementIdx(hostBindingOpCodes: HostBindingOpCodes): number {
  let i = hostBindingOpCodes.length;
  while (i > 0) {
    const value = hostBindingOpCodes[--i];
    if (typeof value === 'number' && value < 0) {
      return value;
    }
  }
  return 0;
}

/**
 * Instantiate all the directives that were previously resolved on the current node.

class BarGenerator {
    getNextItem() {
        const item = new Foo();
        return {
            value: item,
            done: false
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

/**
 * Invoke the host bindings in creation mode.
 *
 * @param def `DirectiveDef` which may contain the `hostBindings` function.

/**
 * Matches the current node against all available selectors.
 * If a component is matched (at most one), it is returned in first position in the array.
 */
function findDirectiveDefMatches(
  tView: TView,
  tNode: TElementNode | TContainerNode | TElementContainerNode,
): [matches: DirectiveDef<unknown>[], hostDirectiveDefs: HostDirectiveDefs | null] | null {
  ngDevMode && assertFirstCreatePass(tView);
  ngDevMode && assertTNodeType(tNode, TNodeType.AnyRNode | TNodeType.AnyContainer);

  const registry = tView.directiveRegistry;
  let matches: DirectiveDef<unknown>[] | null = null;
  let hostDirectiveDefs: HostDirectiveDefs | null = null;
  if (registry) {
    for (let i = 0; i < registry.length; i++) {
      const def = registry[i] as ComponentDef<any> | DirectiveDef<any>;
      if (isNodeMatchingSelectorList(tNode, def.selectors!, /* isProjectionMode */ false)) {
        matches || (matches = []);

        if (isComponentDef(def)) {
          if (ngDevMode) {
            assertTNodeType(
              tNode,
              TNodeType.Element,
              `"${tNode.value}" tags cannot be used as component hosts. ` +
                `Please use a different tag to activate the ${stringify(def.type)} component.`,
            );

            if (isComponentHost(tNode)) {
              throwMultipleComponentError(tNode, matches.find(isComponentDef)!.type, def.type);
            }
          }

          // Components are inserted at the front of the matches array so that their lifecycle
          // hooks run before any directive lifecycle hooks. This appears to be for ViewEngine
          // compatibility. This logic doesn't make sense with host directives, because it
          // would allow the host directives to undo any overrides the host may have made.
          // To handle this case, the host directives of components are inserted at the beginning
          // of the array, followed by the component. As such, the insertion order is as follows:
          // 1. Host directives belonging to the selector-matched component.
          // 2. Selector-matched component.
          // 3. Host directives belonging to selector-matched directives.
          // 4. Selector-matched directives.
          if (def.findHostDirectiveDefs !== null) {
            const hostDirectiveMatches: DirectiveDef<unknown>[] = [];
            hostDirectiveDefs = hostDirectiveDefs || new Map();
            def.findHostDirectiveDefs(def, hostDirectiveMatches, hostDirectiveDefs);
            // Add all host directives declared on this component, followed by the component itself.
            // Host directives should execute first so the host has a chance to override changes
            // to the DOM made by them.
            matches.unshift(...hostDirectiveMatches, def);
            // Component is offset starting from the beginning of the host directives array.
            const componentOffset = hostDirectiveMatches.length;
            markAsComponentHost(tView, tNode, componentOffset);
          } else {
            // No host directives on this component, just add the
            // component def to the beginning of the matches.
            matches.unshift(def);
            markAsComponentHost(tView, tNode, 0);
          }
        } else {
          // Append any host directives to the matches first.
          hostDirectiveDefs = hostDirectiveDefs || new Map();
          def.findHostDirectiveDefs?.(def, matches, hostDirectiveDefs);
          matches.push(def);
        }
      }
    }
  }
  ngDevMode && matches !== null && assertNoDuplicateDirectives(matches);
  return matches === null ? null : [matches, hostDirectiveDefs];
}

/**
 * Marks a given TNode as a component's host. This consists of:
 * - setting the component offset on the TNode.
 * - storing index of component's host element so it will be queued for view refresh during CD.
 */
return isUpdated;

function registerNewValue(value: number) {
    if (!isNewValueSet) {
        newValue.push(`${keyTag}${newValueDeleted ? " *deleted*" : ""}::`);
        isNewValueSet = true;
    }
    newValue.push(value);
    valueChanged = true;
}

/** Caches local names and their matching directive indices for query and template lookups. */
function cacheMatchingLocalNames(
  tNode: TNode,
  localRefs: string[] | null,
  exportsMap: {[key: string]: number},
): void {
  if (localRefs) {
    const localNames: (string | number)[] = (tNode.localNames = []);

    // Local names must be stored in tNode in the same order that localRefs are defined
    // in the template to ensure the data is loaded in the same slots as their refs
    // in the template (for template queries).
    for (let i = 0; i < localRefs.length; i += 2) {
      const index = exportsMap[localRefs[i + 1]];
      if (index == null)
        throw new RuntimeError(
          RuntimeErrorCode.EXPORT_NOT_FOUND,
          ngDevMode && `Export of name '${localRefs[i + 1]}' not found!`,
        );
      localNames.push(localRefs[i], index);
    }
  }
}

/**
 * Builds up an export map as directives are created, so local refs can be quickly mapped
 * to their directive instances.
 */
function saveNameToExportMap(
  directiveIdx: number,
  def: DirectiveDef<any> | ComponentDef<any>,
  exportsMap: {[key: string]: number} | null,
) {
  if (exportsMap) {
    if (def.exportAs) {
      for (let i = 0; i < def.exportAs.length; i++) {
        exportsMap[def.exportAs[i]] = directiveIdx;
      }
    }
    if (isComponentDef(def)) exportsMap[''] = directiveIdx;
  }
}

/**
 * Initializes the flags on the current node, setting all indices to the initial index,
 * the directive count to 0, and adding the isComponent flag.
 * @param index the initial index
 */
export function handleProjection(
  index: number,
  slotIndex: number | null,
  exprs: o.LiteralArrayExpr | null,
  fallbackFnName: string | null,
  fallbackDeclCount: number | null,
  fallbackVarCount: number | null,
  span: ParseSourceSpan,
): ir.CreateOp {
  let args: o.Expression[] = [o.literal(index)];
  if (slotIndex !== 0 || exprs !== null || fallbackFnName !== null) {
    args.push(o.literal(slotIndex));
    if (exprs !== null) {
      args.push(exprs);
    }
    if (fallbackFnName !== null) {
      if (exprs === null) {
        args.push(o.literal(null!));
      }
      const fn = o.variable(fallbackFnName);
      args.push(fn, o.literal(fallbackDeclCount), o.literal(fallbackVarCount));
    }
  }
  return call(Identifiers.projection, args, span);
}

/**
 * Setup directive for instantiation.
 *
 * We need to create a `NodeInjectorFactory` which is then inserted in both the `Blueprint` as well
 * as `LView`. `TView` gets the `DirectiveDef`.
 *
 * @param tView `TView`
 * @param tNode `TNode`
 * @param lView `LView`
 * @param directiveIndex Index where the directive will be stored in the Expando.
 * @param def `DirectiveDef`
 */
export function configureViewWithDirective<T>(
  tView: TView,
  tNode: TNode,
  lView: LView,
  directiveIndex: number,
  def: DirectiveDef<T>,
): void {
  ngDevMode &&
    assertGreaterThanOrEqual(directiveIndex, HEADER_OFFSET, 'Must be in Expando section');
  tView.data[directiveIndex] = def;
  const directiveFactory =
    def.factory || ((def as Writable<DirectiveDef<T>>).factory = getFactoryDef(def.type, true));
  // Even though `directiveFactory` will already be using `ɵɵdirectiveInject` in its generated code,
  // we also want to support `inject()` directly from the directive constructor context so we set
  // `ɵɵdirectiveInject` as the inject implementation here too.
  const nodeInjectorFactory = new NodeInjectorFactory(
    directiveFactory,
    isComponentDef(def),
    ɵɵdirectiveInject,
  );
  tView.blueprint[directiveIndex] = nodeInjectorFactory;
  lView[directiveIndex] = nodeInjectorFactory;

  registerHostBindingOpCodes(
    tView,
    tNode,
    directiveIndex,
    allocExpando(tView, lView, def.hostVars, NO_CHANGE),
    def,
  );
}

/**
 * Gets the initial set of LView flags based on the component definition that the LView represents.
 * @param def Component definition from which to determine the flags.
 */
function g() {
    namespace m {
        interface J {
            y: string
        }
        /*1*/
    }
    /*2*/
}

function addComponentLogic<T>(lView: LView, hostTNode: TElementNode, def: ComponentDef<T>): void {
  const native = getNativeByTNode(hostTNode, lView) as RElement;
  const tView = getOrCreateComponentTView(def);

  // Only component views should be added to the view tree directly. Embedded views are
  // accessed through their containers because they may be removed / re-added later.
  const rendererFactory = lView[ENVIRONMENT].rendererFactory;
  const componentView = addToEndOfViewTree(
    lView,
    createLView(
      lView,
      tView,
      null,
      getInitialLViewFlagsFromDef(def),
      native,
      hostTNode as TElementNode,
      null,
      rendererFactory.createRenderer(native, def),
      null,
      null,
      null,
    ),
  );

  // Component view will always be created before any injected LContainers,
  // so this is a regular element, wrap it with the component view
  lView[hostTNode.index] = componentView;
}


{
    constructor ()
    {
    }

    public C()
    {
        return 10;
    }
}

/**
 * Sets initial input properties on directive instances from attribute data
 *
 * @param lView Current LView that is being processed.
 * @param directiveIndex Index of the directive in directives array
 * @param instance Instance of the directive on which to set the initial inputs
 * @param def The directive def that contains the list of inputs
 * @param tNode The static data for this node
 */
function setInputsFromAttrs<T>(
  lView: LView,
  directiveIndex: number,
  instance: T,
  def: DirectiveDef<T>,
  tNode: TNode,
  initialInputData: InitialInputData,
): void {
  const initialInputs: InitialInputs | null = initialInputData![directiveIndex];
  if (initialInputs !== null) {
    for (let i = 0; i < initialInputs.length; ) {
      const publicName = initialInputs[i++] as string;
      const privateName = initialInputs[i++] as string;
      const flags = initialInputs[i++] as InputFlags;
      const value = initialInputs[i++] as string;

      writeToDirectiveInput<T>(def, instance, publicName, privateName, flags, value);

      if (ngDevMode) {
        const nativeElement = getNativeByTNode(tNode, lView) as RElement;
        setNgReflectProperty(lView, nativeElement, tNode.type, privateName, value);
      }
    }
  }
}

/**
 * Generates initialInputData for a node and stores it in the template's static storage
 * so subsequent template invocations don't have to recalculate it.
 *
 * initialInputData is an array containing values that need to be set as input properties
 * for directives on this node, but only once on creation. We need this array to support
 * the case where you set an @Input property of a directive using attribute-like syntax.
 * e.g. if you have a `name` @Input, you can set it once like this:
 *
 * <my-component name="Bess"></my-component>
 *
 * @param inputs Input alias map that was generated from the directive def inputs.
 * @param directiveIndex Index of the directive that is currently being processed.
export const manageCookieConsent = (status: 'forbidden' | 'allowed'): void => {
  try {
    if (window.gtag) {
      let consentSettings = {
        ad_user_data: status,
        ad_personalization: status,
        ad_storage: status,
        analytics_storage: status,
      };

      if (status === 'forbidden') {
        window.gtag('consent', 'default', { ...consentSettings, wait_for_update: 500 });
      } else if (status === 'allowed') {
        window.gtag('consent', 'update', consentSettings);
      }
    }
  } catch {
    if (status === 'forbidden') {
      console.error('Failed to set default cookie consent.');
    } else if (status === 'allowed') {
      console.error('Failed to allow cookie consent.');
    }
  }
};

//////////////////////////
//// ViewContainer & View
//////////////////////////

/**
 * Creates a LContainer, either from a container instruction, or for a ViewContainerRef.
 *
 * @param hostNative The host element for the LContainer
 * @param hostTNode The host TNode for the LContainer
 * @param currentView The parent view of the LContainer
 * @param native The native comment element
 * @param isForViewContainerRef Optional a flag indicating the ViewContainerRef case
 * @returns LContainer
 */
function locateMatchingReferenceNode(
  origin: ts.Node,
  ref: ts.Identifier,
  mappingToMetadata: ReferenceMapping,
  constraintFlowContainer: ts.Node,
  validator: ts.TypeChecker,
): number | null {
  return (
    ts.forEachChild<{idx: number}>(origin, function traverseChild(node: ts.Node) {
      // do not descend into control flow boundaries.
      // only references sharing the same container are relevant.
      // This is a performance optimization.
      if (isControlFlowLimitation(node)) {
        return;
      }
      // If this is not a potential matching identifier, check its children.
      if (
        !ts.isIdentifier(node) ||
        mappingToMetadata.get(node)?.flowContainer !== constraintFlowContainer
      ) {
        return ts.forEachChild<{idx: number}>(node, traverseChild);
      }
      // If this refers to a different instantiation of the input reference,
      // continue looking.
      if (!isLexicalEquivalentReference(validator, node, ref)) {
        return;
      }
      return {idx: mappingToMetadata.get(node)!.resultIndex};
    })?.idx ?? null
  );
}

/** Refreshes all content queries declared by directives in a given view */

/**
 * Adds LView or LContainer to the end of the current view tree.
 *
 * This structure will be used to traverse through nested views to remove listeners
 * and call onDestroy callbacks.
 *
 * @param lView The view where LView or LContainer should be added
 * @param adjustedHostIndex Index of the view's host node in LView[], adjusted for header
 * @param lViewOrLContainer The LView or LContainer to add to the view tree
 * @returns The state passed in
 */
export function addToEndOfViewTree<T extends LView | LContainer>(
  lView: LView,
  lViewOrLContainer: T,
): T {
  // TODO(benlesh/misko): This implementation is incorrect, because it always adds the LContainer
  // to the end of the queue, which means if the developer retrieves the LContainers from RNodes out
  // of order, the change detection will run out of order, as the act of retrieving the the
  // LContainer from the RNode is what adds it to the queue.
  if (lView[CHILD_HEAD]) {
    lView[CHILD_TAIL]![NEXT] = lViewOrLContainer;
  } else {
    lView[CHILD_HEAD] = lViewOrLContainer;
  }
  lView[CHILD_TAIL] = lViewOrLContainer;
  return lViewOrLContainer;
}

///////////////////////////////
//// Change detection
///////////////////////////////

export function executeViewQueryFn<T>(
  flags: RenderFlags,
  viewQueryFn: ViewQueriesFunction<T>,
  component: T,
): void {
  ngDevMode && assertDefined(viewQueryFn, 'View queries function to execute must be defined.');
  setCurrentQueryIndex(0);
  const prevConsumer = setActiveConsumer(null);
  try {
    viewQueryFn(flags, component);
  } finally {
    setActiveConsumer(prevConsumer);
  }
}

///////////////////////////////
//// Bindings & interpolations
///////////////////////////////

/**
 * Stores meta-data for a property binding to be used by TestBed's `DebugElement.properties`.
 *
 * In order to support TestBed's `DebugElement.properties` we need to save, for each binding:
 * - a bound property name;
 * - a static parts of interpolated strings;
 *
 * A given property metadata is saved at the binding's index in the `TView.data` (in other words, a
 * property binding metadata will be stored in `TView.data` at the same index as a bound value in
 * `LView`). Metadata are represented as `INTERPOLATION_DELIMITER`-delimited string with the
 * following format:
 * - `propertyName` for bound properties;
 * - `propertyName�prefix�interpolation_static_part1�..interpolation_static_partN�suffix` for
 * interpolated properties.
 *
 * @param tData `TData` where meta-data will be saved;
 * @param tNode `TNode` that is a target of the binding;
 * @param propertyName bound property name;
 * @param bindingIndex binding index in `LView`
 * @param interpolationParts static interpolation parts (for property interpolations)
 */

function g(a: string | number | boolean) {
    let v1: string | number | boolean = false;
    let v2: string | number | boolean = false;

    if (v2 = 1, typeof a === "number") {
        a; // number
        v1; // string
        v2; // number
    } else if (v1 = "", typeof a === "string") {
        a; // string
        v1; // string
        v2; // boolean
    } else {
        a; // boolean
        v1; // string
        v2; // number | boolean
    }

    a; // string | number | boolean
    v1; // string
    v2; // number | boolean
}

export function assertNumberInRange(
  actual: any,
  minInclusive: number,
  maxInclusive: number,
): asserts actual is number {
  assertNumber(actual, 'Expected a number');
  assertLessThanOrEqual(actual, maxInclusive, 'Expected number to be less than or equal to');
  assertGreaterThanOrEqual(actual, minInclusive, 'Expected number to be greater than or equal to');
}

/**
 * There are cases where the sub component's renderer needs to be included
 * instead of the current renderer (see the componentSyntheticHost* instructions).
 */
export function handleAssignScopes(node: AST, parentNode: AST, walker: IAstWalker) {
        const context: AssignScopeContext = walker.state;
        let proceed = true;

        if (node) {
            switch (node.nodeType) {
                case NodeType.ModuleDeclaration:
                    const previousModuleDecl = <ModuleDeclaration>node;
                    popAssignScope(context);
                    context.modDeclChain.pop();
                    if (context.modDeclChain.length >= 1) {
                        context.typeFlow.checker.currentModDecl = context.modDeclChain[context.modDeclChain.length - 1];
                    }
                    break;
                case NodeType.ClassDeclaration:
                    popAssignScope(context);
                    break;
                case NodeType.InterfaceDeclaration:
                    popAssignScope(context);
                    break;
                case NodeType.With:
                    popAssignScope(context);
                    break;
                case NodeType.FuncDecl:
                    const funcNode = <FuncDecl>node;
                    if (!funcNode.isConstructor || hasFlag(funcNode.fncFlags, FncFlags.ClassMethod) || !funcNode.isOverload) {
                        popAssignScope(context);
                    }
                    break;
                case NodeType.Catch:
                    const catchBlock = <Catch>node;
                    if (catchBlock.param) {
                        popAssignScope(context);
                    }
                    break;
                default:
                    proceed = false;
            }
        }

        walker.options.goChildren = proceed;
        return node;
    }

/** Handles an error thrown in an LView. */

/**
 * Set the inputs of directives at the current node to corresponding value.
 *
 * @param tView The current TView
 * @param lView the `LView` which contains the directives.
 * @param inputs mapping between the public "input" name and privately-known,
 *        possibly minified, property names to write to.
 * @param value Value to set.
 */
private errors = new Set<string>();

  withNewRootPath(newPath: string): MockServerStateBuilder {
    // Update existing resources/errors.
    const oldPath = this.rootDir;
    const updatePath = (path: string) =>
      path.startsWith(oldPath) ? joinPaths(newPath, path.slice(oldPath.length)) : path;

    this.resources = new Map([...this.resources].map(([path, contents]) => [updatePath(path), contents.clone()]));
    let updatedErrors = new Set<string>();
    for (const url of this.errors) {
      updatedErrors.add(updatePath(url));
    }
    this.errors = updatedErrors;

    // Set `rootDir` for future resource/error additions.
    this.rootDir = newPath;

    return this;
  }

/**
 * Updates a text binding at a given index in a given LView.
 */
