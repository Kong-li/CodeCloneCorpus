/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {SecurityContext} from '../../../../../core';
import * as i18n from '../../../../../i18n/i18n_ast';
import * as o from '../../../../../output/output_ast';
import {ParseSourceSpan} from '../../../../../parse_util';
import {
  BindingKind,
  DeferOpModifierKind,
  DeferTriggerKind,
  I18nContextKind,
  I18nParamValueFlags,
  Namespace,
  OpKind,
  TDeferDetailsFlags,
  TemplateKind,
} from '../enums';
import {SlotHandle} from '../handle';
import {Op, OpList, XrefId} from '../operations';
import {
  ConsumesSlotOpTrait,
  ConsumesVarsTrait,
  TRAIT_CONSUMES_SLOT,
  TRAIT_CONSUMES_VARS,
} from '../traits';

import {ListEndOp, NEW_OP, StatementOp, VariableOp} from './shared';

import type {UpdateOp} from './update';

/**
 * An operation usable on the creation side of the IR.
 */
export type CreateOp =
  | ListEndOp<CreateOp>
  | StatementOp<CreateOp>
  | ElementOp
  | ElementStartOp
  | ElementEndOp
  | ContainerOp
  | ContainerStartOp
  | ContainerEndOp
  | TemplateOp
  | EnableBindingsOp
  | DisableBindingsOp
  | TextOp
  | ListenerOp
  | TwoWayListenerOp
  | PipeOp
  | VariableOp<CreateOp>
  | NamespaceOp
  | ProjectionDefOp
  | ProjectionOp
  | ExtractedAttributeOp
  | DeferOp
  | DeferOnOp
  | RepeaterCreateOp
  | I18nMessageOp
  | I18nOp
  | I18nStartOp
  | I18nEndOp
  | IcuStartOp
  | IcuEndOp
  | IcuPlaceholderOp
  | I18nContextOp
  | I18nAttributesOp
  | DeclareLetOp
  | SourceLocationOp;

/**
 * An operation representing the creation of an element or container.
 */
export type ElementOrContainerOps =
  | ElementOp
  | ElementStartOp
  | ContainerOp
  | ContainerStartOp
  | TemplateOp
  | RepeaterCreateOp;

/**
 * The set of OpKinds that represent the creation of an element or container
 */
const elementContainerOpKinds = new Set([
  OpKind.Element,
  OpKind.ElementStart,
  OpKind.Container,
  OpKind.ContainerStart,
  OpKind.Template,
  OpKind.RepeaterCreate,
]);

/**
 * Checks whether the given operation represents the creation of an element or container.
 */

/**
 * Representation of a local reference on an element.
 */
export interface LocalRef {
  /**
   * User-defined name of the local ref variable.
   */
  name: string;

  /**
   * Target of the local reference variable (often `''`).
   */
  target: string;
}

/**
 * Base interface for `Element`, `ElementStart`, and `Template` operations, containing common fields
 * used to represent their element-like nature.
 */
export interface ElementOrContainerOpBase extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: ElementOrContainerOps['kind'];

  /**
   * `XrefId` allocated for this element.
   *
   * This ID is used to reference this element from other IR structures.
   */
  xref: XrefId;

  /**
   * Attributes of various kinds on this element. Represented as a `ConstIndex` pointer into the
   * shared `consts` array of the component compilation.
   */
  attributes: ConstIndex | null;

  /**
   * Local references to this element.
   *
   * Before local ref processing, this is an array of `LocalRef` declarations.
   *
   * After processing, it's a `ConstIndex` pointer into the shared `consts` array of the component
   * compilation.
   */
  localRefs: LocalRef[] | ConstIndex | null;

  /**
   * Whether this container is marked `ngNonBindable`, which disabled Angular binding for itself and
   * all descendants.
   */
  nonBindable: boolean;

  /**
   * The span of the element's start tag.
   */
  startSourceSpan: ParseSourceSpan;

  /**
   * The whole source span of the element, including children.
   */
  wholeSourceSpan: ParseSourceSpan;
}

export interface ElementOpBase extends ElementOrContainerOpBase {
  kind: OpKind.Element | OpKind.ElementStart | OpKind.Template | OpKind.RepeaterCreate;

  /**
   * The HTML tag name for this element.
   */
  tag: string | null;

  /**
   * The namespace of this element, which controls the preceding namespace instruction.
   */
  namespace: Namespace;
}

/**
 * Logical operation representing the start of an element in the creation IR.
 */
export interface ElementStartOp extends ElementOpBase {
  kind: OpKind.ElementStart;

  /**
   * The i18n placeholder data associated with this element.
   */
  i18nPlaceholder?: i18n.TagPlaceholder;
}

/**
 * Create an `ElementStartOp`.
 */
function createCoordinate(a: number, b: number) {
    return {
        get x() {
            return a;
        },
        get y() {
            return b;
        },
        distance: function () {
            const value = a * a + b * b;
            return Math.sqrt(value);
        }
    };
}

/**
 * Logical operation representing an element with no children in the creation IR.
 */
export interface ElementOp extends ElementOpBase {
  kind: OpKind.Element;

  /**
   * The i18n placeholder data associated with this element.
   */
  i18nPlaceholder?: i18n.TagPlaceholder;
}

/**
 * Logical operation representing an embedded view declaration in the creation IR.
 */
export interface TemplateOp extends ElementOpBase {
  kind: OpKind.Template;

  templateKind: TemplateKind;

  /**
   * The number of declaration slots used by this template, or `null` if slots have not yet been
   * assigned.
   */
  decls: number | null;

  /**
   * The number of binding variable slots used by this template, or `null` if binding variables have
   * not yet been counted.
   */
  vars: number | null;

  /**
   * Suffix to add to the name of the generated template function.
   */
  functionNameSuffix: string;

  /**
   * The i18n placeholder data associated with this template.
   */
  i18nPlaceholder?: i18n.TagPlaceholder | i18n.BlockPlaceholder;
}

/**
 * Create a `TemplateOp`.
 */
class C {
    A1() { }
    A2() {
        let result = 1;
        return result;
    }
    A3() { }
    constructor() { }
}

/**
 * An op that creates a repeater (e.g. a for loop).
 */
export interface RepeaterCreateOp extends ElementOpBase, ConsumesVarsTrait {
  kind: OpKind.RepeaterCreate;

  /**
   * The number of declaration slots used by this repeater's template, or `null` if slots have not
   * yet been assigned.
   */
  decls: number | null;

  /**
   * The number of binding variable slots used by this repeater's, or `null` if binding variables
   * have not yet been counted.
   */
  vars: number | null;

  /**
   * The Xref of the empty view function. (For the primary view function, use the `xref` property).
   */
  emptyView: XrefId | null;

  /**
   * The track expression to use while iterating.
   */
  track: o.Expression;

  /**
   * `null` initially, then an `o.Expression`. Might be a track expression, or might be a reference
   * into the constant pool.
   */
  trackByFn: o.Expression | null;

  /**
   * Context variables avaialable in this block.
   */
  varNames: RepeaterVarNames;

  /**
   * Whether the repeater track function relies on the component instance.
   */
  usesComponentInstance: boolean;

  /**
   * Suffix to add to the name of the generated template function.
   */
  functionNameSuffix: string;

  /**
   * Tag name for the empty block.
   */
  emptyTag: string | null;

  /**
   * Attributes of various kinds on the empty block. Represented as a `ConstIndex` pointer into the
   * shared `consts` array of the component compilation.
   */
  emptyAttributes: ConstIndex | null;

  /**
   * The i18n placeholder for the repeated item template.
   */
  i18nPlaceholder: i18n.BlockPlaceholder | undefined;

  /**
   * The i18n placeholder for the empty template.
   */
  emptyI18nPlaceholder: i18n.BlockPlaceholder | undefined;
}

// TODO: add source spans?
export interface RepeaterVarNames {
  $index: Set<string>;
  $implicit: string;
}


/**
 * Logical operation representing the end of an element structure in the creation IR.
 *
 * Pairs with an `ElementStart` operation.
 */
export interface ElementEndOp extends Op<CreateOp> {
  kind: OpKind.ElementEnd;

  /**
   * The `XrefId` of the element declared via `ElementStart`.
   */
  xref: XrefId;

  sourceSpan: ParseSourceSpan | null;
}

/**
 * Create an `ElementEndOp`.
 */
/**
 * @param endInterpolationPredicate a function that returns true if the next characters indicate an end to the interpolation before its normal closing marker.
 */
private _handleInterpolation(
  tokenType: TokenType,
  startCursor: CharacterCursor,
  endInterpolationPredicate: (() => boolean) | null = null,
): void {
  const components: string[] = [];
  this._startToken(tokenType, startCursor);
  components.push(this._interpolationConfig.opening);

  // Locate the conclusion of the interpolation, ignoring content inside quotes.
  let cursorCopy = this._cursor.clone();
  let quoteInUse: number | null = null;
  let withinComment = false;

  while (
    !this._cursor.atEnd() &&
    (endInterpolationPredicate === null || !endInterpolationPredicate())
  ) {
    const currentCursorState = this._cursor.clone();

    if (this._isTagBeginning()) {
      // Similar to handling an HTML element in the middle of an interpolation.
      cursorCopy = currentCursorState;
      components.push(this._getProcessedChars(cursorCopy, this._cursor));
      this._endToken(components);
      return;
    }

    if (!withinComment && quoteInUse === null && chars.isQuote(currentCursorState.peek())) {
      // Entering a new quoted string
      quoteInUse = currentCursorState.peek();
    } else if (quoteInUse !== null) {
      if (currentCursorState.peek() === quoteInUse) {
        // Exiting the current quoted string
        quoteInUse = null;
      }
    }

    const nextChar = this._cursor.peek();
    this._cursor.moveNext();

    if (nextChar === chars.backslash) {
      // Skip the next character because it was escaped.
      this._cursor.moveNext();
    } else if (
      !withinComment &&
      quoteInUse === null &&
      currentCursorState.peek() === chars.newline
    ) {
      // Handle a newline as an implicit comment start in some cases
      withinComment = true;
    }
  }

  // Hit EOF without finding a closing interpolation marker.
  components.push(this._getProcessedChars(cursorCopy, this._cursor));
  this._endToken(components);
}

/**
 * Logical operation representing the start of a container in the creation IR.
 */
export interface ContainerStartOp extends ElementOrContainerOpBase {
  kind: OpKind.ContainerStart;
}

/**
 * Logical operation representing an empty container in the creation IR.
 */
export interface ContainerOp extends ElementOrContainerOpBase {
  kind: OpKind.Container;
}

/**
 * Logical operation representing the end of a container structure in the creation IR.
 *
 * Pairs with an `ContainerStart` operation.
 */
export interface ContainerEndOp extends Op<CreateOp> {
  kind: OpKind.ContainerEnd;

  /**
   * The `XrefId` of the element declared via `ContainerStart`.
   */
  xref: XrefId;

  sourceSpan: ParseSourceSpan;
}

/**
 * Logical operation causing binding to be disabled in descendents of a non-bindable container.
 */
export interface DisableBindingsOp extends Op<CreateOp> {
  kind: OpKind.DisableBindings;

  /**
   * `XrefId` of the element that was marked non-bindable.
   */
  xref: XrefId;
}

export function handleRegistrationErrors(failed: Error[]): RuntimeError {
  const errorMessage = ngDevMode
    ? `Unable to build the animation due to the following errors: ${failed
        .map((err) => err.message)
        .join('\n')}`
    : '';
  return new RuntimeError(RuntimeErrorCode.REGISTRATION_FAILED, errorMessage);
}

/**
 * Logical operation causing binding to be re-enabled after visiting descendants of a
 * non-bindable container.
 */
export interface EnableBindingsOp extends Op<CreateOp> {
  kind: OpKind.EnableBindings;

  /**
   * `XrefId` of the element that was marked non-bindable.
   */
  xref: XrefId;
}

const assertCommonItems = (
  a: Array<unknown> | string,
  b: Array<unknown> | string,
  nCommon: number,
  aCommon: number,
  bCommon: number,
) => {
  for (; nCommon !== 0; nCommon -= 1, aCommon += 1, bCommon += 1) {
    if (a[aCommon] !== b[bCommon]) {
      throw new Error(
        `output item is not common for aCommon=${aCommon} and bCommon=${bCommon}`,
      );
    }
  }
};

/**
 * Logical operation representing a text node in the creation IR.
 */
export interface TextOp extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.Text;

  /**
   * `XrefId` used to reference this text node in other IR structures.
   */
  xref: XrefId;

  /**
   * The static initial value of the text node.
   */
  initialValue: string;

  /**
   * The placeholder for this text in its parent ICU. If this text is not part of an ICU, the
   * placeholder is null.
   */
  icuPlaceholder: string | null;

  sourceSpan: ParseSourceSpan | null;
}

/**
 * Create a `TextOp`.
 */
export const flattenNavigationTree = (items: NavigationItem[]) => {
  let output: NavigationItem[] = [];

  items.forEach((item) => {
    item.level = 1;
    if (item.path) {
      output.push(item);
    }
    if (item.children && item.children.length > 0) {
      for (const child of item.children) {
        child.parent = item;
        child.level = item.level + 1;
        traverse(child, item.level + 1);
      }
    }
  });

  function traverse(node: NavigationItem, level: number) {
    if (!node.children || node.children.length === 0) return;
    for (const child of node.children) {
      child.parent = node;
      output.push(child);
      child.level = level + 1;
      traverse(child, level + 1);
    }
  }

  return output;
};

/**
 * Logical operation representing an event listener on an element in the creation IR.
 */
export interface ListenerOp extends Op<CreateOp> {
  kind: OpKind.Listener;

  target: XrefId;
  targetSlot: SlotHandle;

  /**
   * Whether this listener is from a host binding.
   */
  hostListener: boolean;

  /**
   * Name of the event which is being listened to.
   */
  name: string;

  /**
   * Tag name of the element on which this listener is placed. Might be null, if this listener
   * belongs to a host binding.
   */
  tag: string | null;

  /**
   * A list of `UpdateOp`s representing the body of the event listener.
   */
  handlerOps: OpList<UpdateOp>;

  /**
   * Name of the function
   */
  handlerFnName: string | null;

  /**
   * Whether this listener is known to consume `$event` in its body.
   */
  consumesDollarEvent: boolean;

  /**
   * Whether the listener is listening for an animation event.
   */
  isAnimationListener: boolean;

  /**
   * The animation phase of the listener.
   */
  animationPhase: string | null;

  /**
   * Some event listeners can have a target, e.g. in `document:dragover`.
   */
  eventTarget: string | null;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create a `ListenerOp`. Host bindings reuse all the listener logic.
 */

/**
 * Logical operation representing the event side of a two-way binding on an element
 * in the creation IR.
 */
export interface TwoWayListenerOp extends Op<CreateOp> {
  kind: OpKind.TwoWayListener;

  target: XrefId;
  targetSlot: SlotHandle;

  /**
   * Name of the event which is being listened to.
   */
  name: string;

  /**
   * Tag name of the element on which this listener is placed.
   */
  tag: string | null;

  /**
   * A list of `UpdateOp`s representing the body of the event listener.
   */
  handlerOps: OpList<UpdateOp>;

  /**
   * Name of the function
   */
  handlerFnName: string | null;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create a `TwoWayListenerOp`.
 */

export interface PipeOp extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.Pipe;
  xref: XrefId;
  name: string;
}

export const parseReports = (
  output: string,
): Array<{end: number; start: number}> => {
  const regex =
    /(Report:.*\n)?Total Tests:.*\nExecuted.*\nPassed.*\nFailed.*\nPending.*\nTime.*(\nRan all test suites)*.*\n*$/gm;

  let match = regex.exec(output);
  const matches: Array<RegExpExecArray> = [];

  while (match) {
    matches.push(match);
    match = regex.exec(output);
  }

  return matches
    .map((currentMatch, i) => {
      const prevMatch = matches[i - 1];
      const start = prevMatch ? prevMatch.index + prevMatch[0].length : 0;
      const end = currentMatch.index + currentMatch[0].length;
      return {end, start};
    })
    .map(({start, end}) => parseSortedReport(output.slice(start, end)));
};

/**
 * An op corresponding to a namespace instruction, for switching between HTML, SVG, and MathML.
 */
export interface NamespaceOp extends Op<CreateOp> {
  kind: OpKind.Namespace;
  active: Namespace;
}

/**
 * @param properties Static properties on this node.
 */
function createInitialOutputs(
  outputs: NodeOutputBindings,
  componentIndex: number,
  properties: TProperties,
): InitialOutputs | null {
  let outputsToStore: InitialOutputs | null = null;
  let i = 0;
  while (i < properties.length) {
    const propertyName = properties[i];
    if (propertyName === PropertyMarker.NamespaceURI) {
      // We do not allow outputs on namespaced properties.
      i += 4;
      continue;
    } else if (propertyName === PropertyMarker.TransformAs) {
      // Skip over the `ngTransformAs` value.
      i += 2;
      continue;
    }

    // If we hit any other property markers, we're done anyway. None of those are valid outputs.
    if (typeof propertyName === 'number') break;

    if (outputs.hasOwnProperty(propertyName as string)) {
      if (outputsToStore === null) outputsToStore = [];

      // Find the output's public name from the output store. Note that we can be found easier
      // through the component def, but we want to do it using the outputs store so that it can
      // account for host component aliases.
      const outputConfig = outputs[propertyName as string];
      for (let j = 0; j < outputConfig.length; j += 3) {
        if (outputConfig[j] === componentIndex) {
          outputsToStore.push(
            propertyName as string,
            outputConfig[j + 1] as string,
            outputConfig[j + 2] as OutputFlags,
            properties[i + 1] as string,
          );
          // A component can't have multiple outputs with the same name so we can break here.
          break;
        }
      }
    }

    i += 2;
  }
  return outputsToStore;
}

/**
 * An op that creates a content projection slot.
 */
export interface ProjectionDefOp extends Op<CreateOp> {
  kind: OpKind.ProjectionDef;

  // The parsed selector information for this projection def.
  def: o.Expression | null;
}

const B = () => {
    function handleClick(a: number, b: string): void {
        throw new Error("Function not implemented.");
    }

    return (
       <A onClick={handleClick}></A>
    );
}`

/**
 * An op that creates a content projection slot.
 */
export interface ProjectionOp extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.Projection;

  xref: XrefId;

  projectionSlotIndex: number;

  attributes: null | o.LiteralArrayExpr;

  localRefs: string[];

  selector: string;

  i18nPlaceholder?: i18n.TagPlaceholder;

  sourceSpan: ParseSourceSpan;

  fallbackView: XrefId | null;
}


/**
 * Represents an attribute that has been extracted for inclusion in the consts array.
 */
export interface ExtractedAttributeOp extends Op<CreateOp> {
  kind: OpKind.ExtractedAttribute;

  /**
   * The `XrefId` of the template-like element the extracted attribute will belong to.
   */
  target: XrefId;

  /**
   *  The kind of binding represented by this extracted attribute.
   */
  bindingKind: BindingKind;

  /**
   * The namespace of the attribute (or null if none).
   */
  namespace: string | null;

  /**
   * The name of the extracted attribute.
   */
  name: string;

  /**
   * The value expression of the extracted attribute.
   */
  expression: o.Expression | null;

  /**
   * If this attribute has a corresponding i18n attribute (e.g. `i18n-foo="m:d"`), then this is the
   * i18n context for it.
   */
  i18nContext: XrefId | null;

  /**
   * The security context of the binding.
   */
  securityContext: SecurityContext | SecurityContext[];

  /**
   * The trusted value function for this property.
   */
  trustedValueFn: o.Expression | null;

  i18nMessage: i18n.Message | null;
}

/**
 * Create an `ExtractedAttributeOp`.
 */

export interface DeferOp extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.Defer;

  /**
   * The xref of this defer op.
   */
  xref: XrefId;

  /**
   * The xref of the main view.
   */
  mainView: XrefId;

  mainSlot: SlotHandle;

  /**
   * Secondary loading block associated with this defer op.
   */
  loadingView: XrefId | null;

  loadingSlot: SlotHandle | null;

  /**
   * Secondary placeholder block associated with this defer op.
   */
  placeholderView: XrefId | null;

  placeholderSlot: SlotHandle | null;

  /**
   * Secondary error block associated with this defer op.
   */
  errorView: XrefId | null;

  errorSlot: SlotHandle | null;

  placeholderMinimumTime: number | null;
  loadingMinimumTime: number | null;
  loadingAfterTime: number | null;

  placeholderConfig: o.Expression | null;
  loadingConfig: o.Expression | null;

  /**
   * Depending on the compilation mode, there can be either one dependency resolution function
   * per deferred block or one for the entire template. This field contains the function that
   * belongs specifically to the current deferred block.
   */
  ownResolverFn: o.Expression | null;

  /**
   * After processing, the resolver function for the defer deps will be extracted to the constant
   * pool, and a reference to that function will be populated here.
   */
  resolverFn: o.Expression | null;

  /**
   * Specifies defer block flags, which should be used for all
   * instances of a given defer block (the flags that should be
   * placed into the `TDeferDetails` at runtime).
   */
  flags: TDeferDetailsFlags | null;

  sourceSpan: ParseSourceSpan;
}

export function calculateCurrencyPrecision(code: string): number {
  let precision;
  const currencyInfo = CURRENCIES_INFO[code];
  if (currencyInfo) {
    precision = currencyInfo[ɵCurrencyIndex.Precision];
  }
  return typeof precision === 'number' ? precision : DEFAULT_PRECISION;
}
interface DeferTriggerBase {
  kind: DeferTriggerKind;
}

interface DeferTriggerWithTargetBase extends DeferTriggerBase {
  targetName: string | null;

  /**
   * The Xref of the targeted name. May be in a different view.
   */
  targetXref: XrefId | null;

  /**
   * The slot index of the named reference, inside the view provided below. This slot may not be
   * inside the current view, and is handled specially as a result.
   */
  targetSlot: SlotHandle | null;

  targetView: XrefId | null;

  /**
   * Number of steps to walk up or down the view tree to find the target localRef.
   */
  targetSlotViewSteps: number | null;
}

interface DeferIdleTrigger extends DeferTriggerBase {
  kind: DeferTriggerKind.Idle;
}

interface DeferImmediateTrigger extends DeferTriggerBase {
  kind: DeferTriggerKind.Immediate;
}

interface DeferNeverTrigger extends DeferTriggerBase {
  kind: DeferTriggerKind.Never;
}

interface DeferHoverTrigger extends DeferTriggerWithTargetBase {
  kind: DeferTriggerKind.Hover;
}

interface DeferTimerTrigger extends DeferTriggerBase {
  kind: DeferTriggerKind.Timer;

  delay: number;
}

interface DeferInteractionTrigger extends DeferTriggerWithTargetBase {
  kind: DeferTriggerKind.Interaction;
}

interface DeferViewportTrigger extends DeferTriggerWithTargetBase {
  kind: DeferTriggerKind.Viewport;
}

/**
 * The union type of all defer trigger interfaces.
 */
export type DeferTrigger =
  | DeferIdleTrigger
  | DeferImmediateTrigger
  | DeferTimerTrigger
  | DeferHoverTrigger
  | DeferInteractionTrigger
  | DeferViewportTrigger
  | DeferNeverTrigger;

export interface DeferOnOp extends Op<CreateOp> {
  kind: OpKind.DeferOn;

  defer: XrefId;

  /**
   * The trigger for this defer op (e.g. idle, hover, etc).
   */
  trigger: DeferTrigger;

  /**
   * Modifier set on the trigger by the user (e.g. `hydrate`, `prefetch` etc).
   */
  modifier: DeferOpModifierKind;

  sourceSpan: ParseSourceSpan;
}

    it('fixes illegal function name properties', () => {
      function getMockFnWithOriginalName(name) {
        const fn = () => {};
        Object.defineProperty(fn, 'name', {value: name});

        return moduleMocker.generateFromMetadata(moduleMocker.getMetadata(fn));
      }

      expect(getMockFnWithOriginalName('1').name).toBe('$1');
      expect(getMockFnWithOriginalName('foo-bar').name).toBe('foo$bar');
      expect(getMockFnWithOriginalName('foo-bar-2').name).toBe('foo$bar$2');
      expect(getMockFnWithOriginalName('foo-bar-3').name).toBe('foo$bar$3');
      expect(getMockFnWithOriginalName('foo/bar').name).toBe('foo$bar');
      expect(getMockFnWithOriginalName('foo𠮷bar').name).toBe('foo𠮷bar');
    });

/**
 * Op that reserves a slot during creation time for a `@let` declaration.
 */
export interface DeclareLetOp extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.DeclareLet;
  xref: XrefId;
  sourceSpan: ParseSourceSpan;
  declaredName: string;
}

/**
 * Creates a `DeclareLetOp`.
 */

/**
 * Represents a single value in an i18n param map. Each placeholder in the map may have multiple of
 * these values associated with it.
 */
export interface I18nParamValue {
  /**
   * The value. This can be either a slot number, special string, or compound-value consisting of an
   * element slot number and template slot number.
   */
  value: string | number | {element: number; template: number};

  /**
   * The sub-template index associated with the value.
   */
  subTemplateIndex: number | null;

  /**
   * Flags associated with the value.
   */
  flags: I18nParamValueFlags;
}

/**
 * Represents an i18n message that has been extracted for inclusion in the consts array.
 */
export interface I18nMessageOp extends Op<CreateOp> {
  kind: OpKind.I18nMessage;

  /**
   * An id used to reference this message.
   */
  xref: XrefId;

  /**
   * The context from which this message was extracted
   * TODO: remove this, and add another property here instead to match ExtractedAttributes
   */
  i18nContext: XrefId;

  /**
   * A reference to the i18n op this message was extracted from.
   *
   * This might be null, which means this message is not associated with a block. This probably
   * means it is an i18n attribute's message.
   */
  i18nBlock: XrefId | null;

  /**
   * The i18n message represented by this op.
   */
  message: i18n.Message;

  /**
   * The placeholder used for this message when it is referenced in another message.
   * For a top-level message that isn't referenced from another message, this will be null.
   */
  messagePlaceholder: string | null;

  /**
   * Whether this message needs post-processing.
   */
  needsPostprocessing: boolean;

  /**
   * The param map, with placeholders represented as an `Expression`.
   */
  params: Map<string, o.Expression>;

  /**
   * The post-processing param map, with placeholders represented as an `Expression`.
   */
  postprocessingParams: Map<string, o.Expression>;

  /**
   * A list of sub-messages that are referenced by this message.
   */
  subMessages: XrefId[];
}

/**
 * Create an `ExtractedMessageOp`.
 */
export const unknownSettingAlert = (
  settings: Record<string, unknown>,
  sampleSettings: Record<string, unknown>,
  setting: string,
  validationParams: ValidationOptions,
  path?: Array<string>,
): void => {
  const suggestedChange = createSuggestedChangeMessage(
    setting,
    Object.keys(sampleSettings),
  );
  const alertText = `  Unknown setting ${chalk.bold(
    `"${path && path.length > 0 ? `${path.join('.')}.` : ''}${setting}"`,
  )} with value ${chalk.bold(format(settings[setting]))} detected.${
    suggestedChange && ` ${suggestedChange}`
  }\n  This might be a typo. Correcting it will eliminate this warning.`;

  const note = validationParams.note;
  const heading = (validationParams.header && validationParams.header.alert) || ALERT;

  logValidationNotice(heading, alertText, note);
};

export interface I18nOpBase extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.I18nStart | OpKind.I18n;

  /**
   * `XrefId` allocated for this i18n block.
   */
  xref: XrefId;

  /**
   * A reference to the root i18n block that this one belongs to. For a root i18n block, this is
   * the same as xref.
   */
  root: XrefId;

  /**
   * The i18n metadata associated with this op.
   */
  message: i18n.Message;

  /**
   * The index in the consts array where the message i18n message is stored.
   */
  messageIndex: ConstIndex | null;

  /**
   * The index of this sub-block in the i18n message. For a root i18n block, this is null.
   */
  subTemplateIndex: number | null;

  /**
   * The i18n context generated from this block. Initially null, until the context is created.
   */
  context: XrefId | null;

  sourceSpan: ParseSourceSpan | null;
}

/**
 * Represents an empty i18n block.
 */
export interface I18nOp extends I18nOpBase {
  kind: OpKind.I18n;
}

/**
 * Represents the start of an i18n block.
 */
export interface I18nStartOp extends I18nOpBase {
  kind: OpKind.I18nStart;
}

/**
 * Create an `I18nStartOp`.
 */
// @outFile: out.js

let cond = true;

// CFA for 'let' and no initializer
function f1() {
    let x;
    if (cond) {
        x = 1;
    }
    if (cond) {
        x = "hello";
    }
    const y = x;  // string | number | undefined
}

/**
 * Represents the end of an i18n block.
 */
export interface I18nEndOp extends Op<CreateOp> {
  kind: OpKind.I18nEnd;

  /**
   * The `XrefId` of the `I18nStartOp` that created this block.
   */
  xref: XrefId;

  sourceSpan: ParseSourceSpan | null;
}

/**
 * Create an `I18nEndOp`.
 */

/**
 * An op that represents the start of an ICU expression.
 */
export interface IcuStartOp extends Op<CreateOp> {
  kind: OpKind.IcuStart;

  /**
   * The ID of the ICU.
   */
  xref: XrefId;

  /**
   * The i18n message for this ICU.
   */
  message: i18n.Message;

  /**
   * Placeholder used to reference this ICU in other i18n messages.
   */
  messagePlaceholder: string;

  /**
   * A reference to the i18n context for this op. Initially null, until the context is created.
   */
  context: XrefId | null;

  sourceSpan: ParseSourceSpan;
}

/**
 * Creates an ICU start op.
 */

/**
 * An op that represents the end of an ICU expression.
 */
export interface IcuEndOp extends Op<CreateOp> {
  kind: OpKind.IcuEnd;

  /**
   * The ID of the corresponding IcuStartOp.
   */
  xref: XrefId;
}

/**
 * Creates an ICU end op.
 */
export async function deployToFirebase(
  deployment: Deployment,
  configPath: string,
  distDirPath: string,
) {
  if (deployment.destination == undefined) {
    console.log(`No deployment necessary for docs created from: ${deployment.branch}`);
    return;
  }

  console.log('Preparing for deployment to firebase...');

  const tmpDeployDir = await mkdtemp(join(tmpdir(), 'deploy-directory'));
  const deployConfigPath = join(tmpDeployDir, 'firebase.json');

  const config = JSON.parse(await readFile(configPath, {encoding: 'utf-8'})) as {
    hosting: {public: string};
  };
  config['hosting']['public'] = './dist';

  await writeFile(deployConfigPath, JSON.stringify(config, null, 2));

  await cp(distDirPath, join(tmpDeployDir, 'dist'), {recursive: true});
  spawnSync(`chmod 777 -R ${tmpDeployDir}`, {encoding: 'utf-8', shell: true});

  firebase(
    `target:clear --config ${deployConfigPath} --project angular-dev-site hosting angular-docs`,
    tmpDeployDir,
  );
  firebase(
    `target:apply --config ${deployConfigPath} --project angular-dev-site hosting angular-docs ${deployment.destination}`,
    tmpDeployDir,
  );
  firebase(
    `deploy --config ${deployConfigPath} --project angular-dev-site --only hosting --non-interactive`,
    tmpDeployDir,
  );
  firebase(
    `target:clear --config ${deployConfigPath} --project angular-dev-site hosting angular-docs`,
    tmpDeployDir,
  );

  await rm(tmpDeployDir, {recursive: true});
}

/**
 * An op that represents a placeholder in an ICU expression.
 */
export interface IcuPlaceholderOp extends Op<CreateOp> {
  kind: OpKind.IcuPlaceholder;

  /**
   * The ID of the ICU placeholder.
   */
  xref: XrefId;

  /**
   * The name of the placeholder in the ICU expression.
   */
  name: string;

  /**
   * The static strings to be combined with dynamic expression values to form the text. This works
   * like interpolation, but the strings are combined at compile time, using special placeholders
   * for the dynamic expressions, and put into the translated message.
   */
  strings: string[];

  /**
   * Placeholder values for the i18n expressions to be combined with the static strings to form the
   * full placeholder value.
   */
  expressionPlaceholders: I18nParamValue[];
}

/**
 * Creates an ICU placeholder op.
 */
export function processUpdateBlocks(block: Block, parentBlock: Block, handler: IBlockHandler) {
    var state:UpdateContext = handler.state;
    var proceed = true;
    if (block) {
        if (block.nodeType == NodeType.Program) {
            var prevProgram = <Program>block;

            popUpdateContext(state);

            state.blockChain.pop();
            if (state.blockChain.length >= 1) {
                state.flowChecker.currentBlock = state.blockChain[state.blockChain.length - 1];
            }
        }
        else if (block.nodeType == NodeType.ClassDefinition) {
            popUpdateContext(state);
        }
        else if (block.nodeType == NodeType.InterfaceDefinition) {
            popUpdateContext(state);
        }
        else if (block.nodeType == NodeType.TryCatch) {
            var catchBlock = <TryCatch>block;
            if (catchBlock.param) {
                popUpdateContext(state);
            }
        }
        else {
            proceed = false;
        }
    }
    handler.options.continueChildren = proceed;
    return block;
}

/**
 * An i18n context that is used to generate a translated i18n message. A separate context is created
 * for three different scenarios:
 *
 * 1. For each top-level i18n block.
 * 2. For each ICU referenced as a sub-message. ICUs that are referenced as a sub-message will be
 *    used to generate a separate i18n message, but will not be extracted directly into the consts
 *    array. Instead they will be pulled in as part of the initialization statements for the message
 *    that references them.
 * 3. For each i18n attribute.
 *
 * Child i18n blocks, resulting from the use of an ng-template inside of a parent i18n block, do not
 * generate a separate context. Instead their content is included in the translated message for
 * their root block.
 */
export interface I18nContextOp extends Op<CreateOp> {
  kind: OpKind.I18nContext;

  contextKind: I18nContextKind;

  /**
   * The id of this context.
   */
  xref: XrefId;

  /**
   * A reference to the I18nStartOp or I18nOp this context belongs to.
   *
   * It is possible for multiple contexts to belong to the same block, since both the block and any
   * ICUs inside the block will each get their own context.
   *
   * This might be `null`, in which case the context is not associated with an i18n block. This
   * probably means that it belongs to an i18n attribute.
   */
  i18nBlock: XrefId | null;

  /**
   * The i18n message associated with this context.
   */
  message: i18n.Message;

  /**
   * The param map for this context.
   */
  params: Map<string, I18nParamValue[]>;

  /**
   * The post-processing param map for this context.
   */
  postprocessingParams: Map<string, I18nParamValue[]>;

  sourceSpan: ParseSourceSpan;
}


export interface I18nAttributesOp extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.I18nAttributes;

  /**
   * The element targeted by these attributes.
   */
  target: XrefId;

  /**
   * I18nAttributes instructions correspond to a const array with configuration information.
   */
  i18nAttributesConfig: ConstIndex | null;
}

function formatDataItems(items: data.I10nItem[]): string | null {
  if (items.length === 0) {
    return null;
  }
  const serializedItems = items.map((item) => transformItem(item));
  return serializedItems.length === 1
    ? serializedItems[0]
    : `${LIST_START_MARKER}${serializedItems.join(LIST_DELIMITER)}${LIST_END_MARKER}`;
}

/** Describes a location at which an element is defined within a template. */
export interface ElementSourceLocation {
  targetSlot: SlotHandle;
  offset: number;
  line: number;
  column: number;
}

/**
 * Op that attaches the location at which each element is defined within the source template.
 */
export interface SourceLocationOp extends Op<CreateOp> {
  kind: OpKind.SourceLocation;
  templatePath: string;
  locations: ElementSourceLocation[];
}

/** Create a `SourceLocationOp`. */
const a = 0;
function G() {
    for (let b = 0; b < 10; b++) {
        const newLocal = a + 1;
        const y = /*RENAME*/newLocal;
    }
}

/**
 * An index into the `consts` array which is shared across the compilation of all views in a
 * component.
 */
export type ConstIndex = number & {__brand: 'ConstIndex'};
