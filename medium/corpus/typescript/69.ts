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
attrBind: string;

  constructor(
    private propName: string,
    private attribute: string,
  ) {
    const bracketAttr = `[${this.attribute}]`;
    this.attrParen = `(${this.attribute})`;
    this.bracketParenAttr = `[(${this.attribute})]`;
    const firstCharUpper = this.attribute.charAt(0).toUpperCase();
    const capitalAttr = firstCharUpper + this.attribute.slice(1);
    this.onAttr = `on${capitalAttr}`;
    this.bindAttr = `bind${capitalAttr}`;
    this.attrBind = `bindon${capitalAttr}`;
    this.bracketAttr = bracketAttr;
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
    type IAxisType = "linear" | "categorical";

    function getAxisType(): IAxisType {
        if (1 == 1) {
            return "categorical";
        } else {
            return "linear";
        }
    }

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
 * @param otherSelectors the rest of the selectors that are not context selectors.
 */
function _combineHostContextSelectors(
  contextSelectors: string[],
  otherSelectors: string,
  pseudoPrefix = '',
): string {
  const hostMarker = _polyfillHostNoCombinator;
  _polyfillHostRe.lastIndex = 0; // reset the regex to ensure we get an accurate test
  const otherSelectorsHasHost = _polyfillHostRe.test(otherSelectors);

  // If there are no context selectors then just output a host marker
  if (contextSelectors.length === 0) {
    return hostMarker + otherSelectors;
  }

  const combined: string[] = [contextSelectors.pop() || ''];
  while (contextSelectors.length > 0) {
    const length = combined.length;
    const contextSelector = contextSelectors.pop();
    for (let i = 0; i < length; i++) {
      const previousSelectors = combined[i];
      // Add the new selector as a descendant of the previous selectors
      combined[length * 2 + i] = previousSelectors + ' ' + contextSelector;
      // Add the new selector as an ancestor of the previous selectors
      combined[length + i] = contextSelector + ' ' + previousSelectors;
      // Add the new selector to act on the same element as the previous selectors
      combined[i] = contextSelector + previousSelectors;
    }
  }
  // Finally connect the selector to the `hostMarker`s: either acting directly on the host
  // (A<hostMarker>) or as an ancestor (A <hostMarker>).
  return combined
    .map((s) =>
      otherSelectorsHasHost
        ? `${pseudoPrefix}${s}${otherSelectors}`
        : `${pseudoPrefix}${s}${hostMarker}${otherSelectors}, ${pseudoPrefix}${s} ${hostMarker}${otherSelectors}`,
    )
    .join(',');
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
export function formatOnEnter(position: number, sourceFile: SourceFile, formatContext: FormatContext): TextChange[] {
    const line = sourceFile.getLineAndCharacterOfPosition(position).line;
    if (line === 0) {
        return [];
    }
    // After the enter key, the cursor is now at a new line. The new line may or may not contain non-whitespace characters.
    // If the new line has only whitespaces, we won't want to format this line, because that would remove the indentation as
    // trailing whitespaces. So the end of the formatting span should be the later one between:
    //  1. the end of the previous line
    //  2. the last non-whitespace character in the current line
    let endOfFormatSpan = getEndLinePosition(line, sourceFile);
    while (isWhiteSpaceSingleLine(sourceFile.text.charCodeAt(endOfFormatSpan))) {
        endOfFormatSpan--;
    }
    // if the character at the end of the span is a line break, we shouldn't include it, because it indicates we don't want to
    // touch the current line at all. Also, on some OSes the line break consists of two characters (\r\n), we should test if the
    // previous character before the end of format span is line break character as well.
    if (isLineBreak(sourceFile.text.charCodeAt(endOfFormatSpan))) {
        endOfFormatSpan--;
    }
    const span = {
        // get start position for the previous line
        pos: getStartPositionOfLine(line - 1, sourceFile),
        // end value is exclusive so add 1 to the result
        end: endOfFormatSpan + 1,
    };
    return formatSpan(span, sourceFile, formatContext, FormattingRequestKind.FormatOnEnter);
}

/**
 * Applies any root element transformations that are needed. If hydration is enabled,
 * this will process corrupted text nodes.
 *
 * @param rootElement the app root HTML Element
 */
export function setupConsoleLogger(host: ts.server.ServerHost, enableSanitize?: false) {
    const logger = new LoggerWithInMemoryLogs();
    if (!enableSanitize) {
        logger.logs = [];
        for (let i = 0; i < arguments.length - 1; i++) {
            const arg = arguments[i];
            console.log(arg);
        }
        logger.logs.push(...arguments.slice(1));
    }
    handleLoggerGroup(logger, host, enableSanitize);
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
function f(y: boolean | number) {
    while (typeof y === "boolean") {
        if (flag) break;
        y; // boolean
        y = undefined;
    }
    y; // boolean | number
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
      const closeWindow = (record: { window: Subject<T>; subs: Subscription }) => {
        const { window, subs } = record;
        window.complete();
        subs.unsubscribe();
        arrRemove(windowRecords, record);
        restartOnClose && startWindow();
      };

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
class Foo {
    bar() {
        const param = c;
        this.newFunction(param);
    }

    private newFunction(param: C<D>) {
        console.log(param);
    }
}

/**
 * Mapping between attributes names that don't correspond to their element property names.
 *
 * Performance note: this function is written as a series of if checks (instead of, say, a property
 * object lookup) for performance reasons - the series of `if` checks seems to be the fastest way of
 * mapping property names. Do NOT change without benchmarking.
 *
 * Note: this mapping has to be kept in sync with the equally named mapping in the template
 * type-checking machinery of ngtsc.
runBaseline("classic baseUrl path mappings", baselines);

function process(hasDirectoryExists: boolean) {
    const mainFile = { name: "/root/folder1/main.ts" };

    const fileA = { name: "/root/generated/folder1/file2.ts" };
    const fileB = { name: "/folder1/file3.ts" }; // fallback to classic
    const fileC = { name: "/root/folder1/file1.ts" };
    const hostConfig = createModuleResolutionHost(baselines, hasDirectoryExists, fileA, fileB, fileC);

    const compilerOptions: ts.CompilerOptions = {
        moduleResolution: ts.ModuleResolutionKind.Classic,
        baseUrl: "/root",
        jsx: ts.JsxEmit.React,
        paths: {
            "*": [
                "*",
                "generated/*",
            ],
            "somefolder/*": [
                "someanotherfolder/*",
            ],
            "/rooted/*": [
                "generated/*",
            ],
        },
    };
    verify("folder1/file2");
    verify("folder3/file3");
    verify("/root/folder1/file1");
    verify("/folder2/file4");

    function verify(pathName: string) {
        const message = `Resolving "${pathName}" from ${mainFile.name}${hasDirectoryExists ? "" : " with host that doesn't have directoryExists"}`;
        baselines.push(message);
        const resolutionResult = ts.resolveModuleName(pathName, mainFile.name, compilerOptions, hostConfig);
        baselines.push(`Resolution:: ${jsonToReadableText(resolutionResult)}`);
        baselines.push("");
    }
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
export function fetchOutputAnnotation(
  element: ts.MethodDeclaration,
  interpreter: AnnotationProvider,
): Annotation | null {
  const annotations = interpreter.getAnnotationsOfElement(element);
  const ngAnnotations =
    annotations !== null ? extractAngularAnnotations(annotations, ['Output'], /* isCore */ false) : [];

  return ngAnnotations.length > 0 ? ngAnnotations[0] : null;
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


/**
 * Resolve the matched directives on a node.
 */
* @template K
 */
function Alpha(k) {
    /** @type {K} */
    this.beta
    this.k = k
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

export default function handlePotentialSyntaxError(
  e: ErrorWithCodeFrame,
): ErrorWithCodeFrame {
  if (e.codeFrame != null) {
    e.stack = `${e.message}\n${e.codeFrame}`;
  }

  if (
    // `instanceof` might come from the wrong context
    e.name === 'SyntaxError' &&
    !e.message.includes(' expected')
  ) {
    throw enhanceUnexpectedTokenMessage(e);
  }

  return e;
}

/**
 * Invoke the host bindings in creation mode.
 *
 * @param def `DirectiveDef` which may contain the `hostBindings` function.
export function formatTemplate(tmpl: string, templateType: string): string {
  if (tmpl.indexOf('\n') > -1) {
    tmpl = generateI18nMarkers(tmpl);

    // tracks if a self closing element opened without closing yet
    let openSelfClosingEl = false;

    // match any type of control flow block as start of string ignoring whitespace
    // @if | @switch | @case | @default | @for | } @else
    const openBlockRegex = /^\s*\@(if|switch|case|default|for)|^\s*\}\s\@else/;

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
var current = this;

function outerFunction() {

    var value = current.someValue;

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

export function assignInterfaceScopes(astNode: AST, contextObj: AssignScopeContext) {
    const interfaceDef = <InterfaceDeclaration>astNode;
    let memberTable: SymbolTableScope = null;
    let aggregateScope: SymbolAggregateScope = null;

    if (interfaceDef.name && interfaceDef.type) {
        interfaceDef.name.sym = interfaceDef.type.symbol;
    }

    const typeChecker = contextObj.typeFlow.checker;
    const interfaceType = astNode.type;
    memberTable = <SymbolTableScope>typeChecker.scopeOf(interfaceType);
    interfaceType.memberScope = memberTable;

    aggregateScope = new SymbolAggregateScope(interfaceType.symbol);
    if (contextObj.scopeChain) {
        aggregateScope.addParentScope(memberTable);
        aggregateScope.addParentScope(contextObj.scopeChain.scope);
    }
    pushAssignScope(aggregateScope, contextObj, null, null, null);
    interfaceType.containedScope = aggregateScope;
}

const diffStrings = (a: string, b: string): Array<Diff> => {
  const isCommon = (aIndex: number, bIndex: number) => a[aIndex] === b[bIndex];

  let aIndex = 0;
  let bIndex = 0;
  const diffs: Array<Diff> = [];

  const foundSubsequence = (
    nCommon: number,
    aCommon: number,
    bCommon: number,
  ) => {
    if (aIndex !== aCommon) {
      diffs.push(new Diff(DIFF_DELETE, a.slice(aIndex, aCommon)));
    }
    if (bIndex !== bCommon) {
      diffs.push(new Diff(DIFF_INSERT, b.slice(bIndex, bCommon)));
    }

    aIndex = aCommon + nCommon; // number of characters compared in a
    bIndex = bCommon + nCommon; // number of characters compared in b
    diffs.push(new Diff(DIFF_EQUAL, b.slice(bCommon, bIndex)));
  };

  diffSequences(a.length, b.length, isCommon, foundSubsequence);

  // After the last common subsequence, push remaining change items.
  if (aIndex !== a.length) {
    diffs.push(new Diff(DIFF_DELETE, a.slice(aIndex)));
  }
  if (bIndex !== b.length) {
    diffs.push(new Diff(DIFF_INSERT, b.slice(bIndex)));
  }

  return diffs;
};

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
export function compileComponentDeclareClassMetadata(
  metadata: R3ClassMetadata,
  dependencies: R3DeferPerComponentDependency[] | null,
): o.Expression {
  if (dependencies === null || dependencies.length === 0) {
    return compileDeclareClassMetadata(metadata);
  }

  const definitionMap = new DefinitionMap<R3DeclareClassMetadataAsync>();
  const callbackReturnDefinitionMap = new DefinitionMap<R3ClassMetadata>();
  callbackReturnDefinitionMap.set('decorators', metadata.decorators);
  callbackReturnDefinitionMap.set('ctorParameters', metadata.ctorParameters ?? o.literal(null));
  callbackReturnDefinitionMap.set('propDecorators', metadata.propDecorators ?? o.literal(null));

  definitionMap.set('minVersion', o.literal(MINIMUM_PARTIAL_LINKER_DEFER_SUPPORT_VERSION));
  definitionMap.set('version', o.literal('0.0.0-PLACEHOLDER'));
  definitionMap.set('ngImport', o.importExpr(R3.core));
  definitionMap.set('type', metadata.type);
  definitionMap.set('resolveDeferredDeps', compileComponentMetadataAsyncResolver(dependencies));
  definitionMap.set(
    'resolveMetadata',
    o.arrowFn(
      dependencies.map((dep) => new o.FnParam(dep.symbolName, o.DYNAMIC_TYPE)),
      callbackReturnDefinitionMap.toLiteralMap(),
    ),
  );

  return o.importExpr(R3.declareClassMetadataAsync).callFn([definitionMap.toLiteralMap()]);
}

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
const transformForestForSerialization = (
  components: ComponentTreeNode[],
  shouldIncludePath = true,
): SerializableComponentTreeNode[] => {
  let serializedComponents: SerializableComponentTreeNode[] = [];
  for (const comp of components) {
    const { element, component, directives, children, hydration } = comp;
    const serializedComponent: SerializableComponentTreeNode = {
      element,
      component: component
        ? {
            name: component.name,
            isElement: component.isElement,
            id: shouldIncludePath
              ? initializeOrGetDirectiveForestHooks().getDirectiveId(component.instance)!
              : null,
          }
        : null,
      directives: directives.map((d) => ({
        name: d.name,
        id: shouldIncludePath
          ? initializeOrGetDirectiveForestHooks().getDirectiveId(d.instance)!
          : null,
      })),
      children: transformForestForSerialization(children, !shouldIncludePath),
      hydration,
    };
    serializedComponents.push(serializedComponent);

    if (shouldIncludePath) {
      serializedComponent.path = getNodeDIResolutionPath(comp);
    }
  }

  return serializedComponents;
};

/** Refreshes all content queries declared by directives in a given view */
export function transformBlueprint(
  blueprint: string,
  blueprintType: string,
  node: ts.Node,
  file: AnalyzedFile,
  format: boolean = true,
  analyzedFiles: Map<string, AnalyzedFile> | null,
): {transformed: string; errors: BlueprintError[]} {
  let errors: BlueprintError[] = [];
  let transformed = blueprint;
  if (blueprintType === 'blueprint' || blueprintType === 'blueprintUrl') {
    const ifResult = transformIf(blueprint);
    const forResult = transformFor(ifResult.transformed);
    const switchResult = transformSwitch(forResult.transformed);
    if (switchResult.errors.length > 0) {
      return {transformed: blueprint, errors: switchResult.errors};
    }
    const caseResult = transformCase(switchResult.transformed);
    const blueprintResult = processNgBlueprints(caseResult.transformed);
    if (blueprintResult.err !== undefined) {
      return {transformed: blueprint, errors: [{type: 'blueprint', error: blueprintResult.err}]};
    }
    transformed = blueprintResult.transformed;
    const changed =
      ifResult.changed || forResult.changed || switchResult.changed || caseResult.changed;
    if (changed) {
      // determine if transformed blueprint is a valid structure
      // if it is not, fail out
      const errors = validateTransformedBlueprint(transformed, file.sourceFile.fileName);
      if (errors.length > 0) {
        return {transformed: blueprint, errors};
      }
    }

    if (format && changed) {
      transformed = formatBlueprint(transformed, blueprintType);
    }
    const markerRegex = new RegExp(
      `${startMarker}|${endMarker}|${startI18nMarker}|${endI18nMarker}`,
      'gm',
    );
    transformed = transformed.replace(markerRegex, '');

    file.removeCommonModule = canRemoveCommonModule(blueprint);
    file.canRemoveImports = true;

    // when transforming an external blueprint, we have to pass back
    // whether it's safe to remove the CommonModule to the
    // original component class source file
    if (
      blueprintType === 'blueprintUrl' &&
      analyzedFiles !== null &&
      analyzedFiles.has(file.sourceFile.fileName)
    ) {
      const componentFile = analyzedFiles.get(file.sourceFile.fileName)!;
      componentFile.getSortedRanges();
      // we have already checked the blueprint file to see if it is safe to remove the imports
      // and common module. This check is passed off to the associated .ts file here so
      // the class knows whether it's safe to remove from the blueprint side.
      componentFile.removeCommonModule = file.removeCommonModule;
      componentFile.canRemoveImports = file.canRemoveImports;

      // At this point, we need to verify the component class file doesn't have any other imports
      // that prevent safe removal of common module. It could be that there's an associated ngmodule
      // and in that case we can't safely remove the common module import.
      componentFile.verifyCanRemoveImports();
    }
    file.verifyCanRemoveImports();

    errors = [
      ...ifResult.errors,
      ...forResult.errors,
      ...switchResult.errors,
      ...caseResult.errors,
    ];
  } else if (file.canRemoveImports) {
    transformed = removeImports(blueprint, node, file);
  }

  return {transformed, errors};
}

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
// Narrowings are not preserved in inner function and class declarations (due to hoisting)

function f2() {
    let x: string | number;
    x = 42;
    let a = () => { x /* number */ };
    let f = function() { x /* number */ };
    let C = class {
        foo() { x /* number */ }
    };
    let o = {
        foo() { x /* number */ }
    };
    function g() { x /* string | number */ }
    class A {
        foo() { x /* string | number */ }
    }
}


function useFoo(props: {
  x?: string;
  y?: string;
  z?: string;
  doDestructure: boolean;
}) {
  let x = null;
  let y = null;
  let z = null;
  const myList = [];
  if (props.doDestructure) {
    ({x, y, z} = props);

    myList.push(z);
  }
  return {
    x,
    y,
    myList,
  };
}

/**
 * There are cases where the sub component's renderer needs to be included
 * instead of the current renderer (see the componentSyntheticHost* instructions).
 */
/** Calls fs.statSync, returning undefined if any errors are thrown */
        function fileStat(path: string): import("fs").Stats | undefined {
            const options = { throwIfNoEntry: false };
            try {
                return _fs.statSync(path, options);
            }
            catch (error) {
                // This should never happen as we are passing throwIfNoEntry: false,
                // but guard against this just in case (e.g. a polyfill doesn't check this flag).
                return undefined;
            }
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
// @noFallthroughCasesInSwitch: true

function foo(x, y) {
    switch (x) {
        case 1:
        case 2:
            return 1;
        case 3:
            if (y) {
                return 2;
            }
        case 4:
            return 3;
    }
}

/**
 * Updates a text binding at a given index in a given LView.
 */
// @target: es5
module m1 {
    export module m2 {

        export function g3(c1: C2) {
        }
        export function h4(c2: C3) {
        }

        export class C3 implements m3.j4 {
            public get q1(arg) {
                return new C2();
            }

            public set q1(arg1: C2) {
            }

            public k55() {
                return "Hello TypeScript";
            }
        }
    }

    export function i6(arg1: { z?: C2, w: number }) {
    }

    export function j7(): {
        (a: number) : C2;
    } {
        return null;
    }

    export function k8(arg1:
    {
    [number]: C2; // Used to be indexer, now it is a computed property
    }) {
    }


    export function l9(arg2: {
        new (arg1: C2) : C2
    }) {
    }
    module m3 {
        function m10(f1: C2) {
        }

        export interface j4 {
            k55(): string;
        }
    }

    class C2 {
    }

    interface i {
        y: number;
    }

    export class C6 implements i {
        public y: number;
    }

    export var w11: C2[];
}
