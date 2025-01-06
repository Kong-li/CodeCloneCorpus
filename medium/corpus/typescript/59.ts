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
 */

function isForwardRefCallback(path: NodePath<t.Expression>): boolean {
  return !!(
    path.parentPath.isCallExpression() &&
    path.parentPath.get('callee').isExpression() &&
    isReactAPI(path.parentPath.get('callee'), 'forwardRef')
  );
}

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
export class HeroData {
  createDb() {
    const heroes = [
      {id: 1, name: 'Windstorm'},
      {id: 2, name: 'Bombasto'},
      {id: 3, name: 'Magneta'},
      {id: 4, name: 'Tornado'},
    ];
    return {heroes};
  }
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

const populateDataRecord = (component: ComponentInstanceType | DirectiveInstanceType) => {
    const {instance, name} = component;
    const metadata = getComponentMetadata(instance);
    metadata.dependencies = getDependenciesForComponent(
      injector,
      resolutionPathWithProviders,
      instance.constructor,
    );

    if (query.propertyQuery.type === PropertyQueryTypes.All) {
      componentProperties[name] = {
        props: serializeComponentState(instance),
        metadata,
      };
    }

    if (query.propertyQuery.type === PropertyQueryTypes.Specified) {
      componentProperties[name] = {
        props: deeplySerializeSelectedProperties(
          instance,
          query.propertyQuery.properties[name] || [],
        ),
        metadata,
      };
    }
  };

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
const filterItemsByLabel = (label: string) => {
    return items.filter(item => {
        if (typeof item === 'string') {
            return item !== label;
        }

        return item.label !== label;
    })
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

export function openTreeBenchmark() {
  browser.rootEl = '#root';
  openBrowser({
    url: '',
    ignoreBrowserSynchronization: true,
    params: [{name: 'depth', value: 4}],
  });
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
export async function loadSettings(filePath: string): Promise<SettingsData> {
  const settingsPath = join(filePath, 'settings.json');

  if (!existsSync(settingsPath)) {
    throw Error(`Unable to find settings.json file at: ${filePath}`);
  }

  const data = await getFileContents<string>(settingsPath);
  return JSON.parse(data) as SettingsData;
}

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
const JSX_TEXT_CHILD_REQUIRES_EXPR_CONTAINER_PATTERN = /[<>&]/;
function codegenJsxElement(
  cx: Context,
  place: Place,
):
  | t.JSXText
  | t.JSXExpressionContainer
  | t.JSXSpreadChild
  | t.JSXElement
  | t.JSXFragment {
  const value = codegenPlace(cx, place);
  switch (value.type) {
    case 'JSXText': {
      if (JSX_TEXT_CHILD_REQUIRES_EXPR_CONTAINER_PATTERN.test(value.value)) {
        return createJsxExpressionContainer(
          place.loc,
          createStringLiteral(place.loc, value.value),
        );
      }
      return createJsxText(place.loc, value.value);
    }
    case 'JSXElement':
    case 'JSXFragment': {
      return value;
    }
    default: {
      return createJsxExpressionContainer(place.loc, value);
    }
  }
}

export interface PipeOp extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.Pipe;
  xref: XrefId;
  name: string;
}

  const pipeDefs = () => {
    if (!USE_RUNTIME_DEPS_TRACKER_FOR_JIT) {
      if (cachedPipeDefs === null) {
        cachedPipeDefs = [];
        const seen = new Set<Type<unknown>>();

        for (const rawDep of imports) {
          const dep = resolveForwardRef(rawDep);
          if (seen.has(dep)) {
            continue;
          }
          seen.add(dep);

          if (!!getNgModuleDef(dep)) {
            const scope = transitiveScopesFor(dep);
            for (const pipe of scope.exported.pipes) {
              const def = getPipeDef(pipe);
              if (def && !seen.has(pipe)) {
                seen.add(pipe);
                cachedPipeDefs.push(def);
              }
            }
          } else {
            const def = getPipeDef(dep);
            if (def) {
              cachedPipeDefs.push(def);
            }
          }
        }
      }
      return cachedPipeDefs;
    } else {
      if (ngDevMode) {
        for (const rawDep of imports) {
          verifyStandaloneImport(rawDep, type);
        }
      }

      if (!isComponent(type)) {
        return [];
      }

      const scope = depsTracker.getStandaloneComponentScope(type, imports);

      return [...scope.compilation.pipes].map((p) => getPipeDef(p)!).filter((d) => d !== null);
    }
  };

/**
 * An op corresponding to a namespace instruction, for switching between HTML, SVG, and MathML.
 */
export interface NamespaceOp extends Op<CreateOp> {
  kind: OpKind.Namespace;
  active: Namespace;
}


/**
 * An op that creates a content projection slot.
 */
export interface ProjectionDefOp extends Op<CreateOp> {
  kind: OpKind.ProjectionDef;

  // The parsed selector information for this projection def.
  def: o.Expression | null;
}

  const callback = () => {
    removeLoadListenerFn();
    removeErrorListenerFn();
    const computedStyle = window.getComputedStyle(img);
    let renderedWidth = parseFloat(computedStyle.getPropertyValue('width'));
    let renderedHeight = parseFloat(computedStyle.getPropertyValue('height'));
    const boxSizing = computedStyle.getPropertyValue('box-sizing');

    if (boxSizing === 'border-box') {
      const paddingTop = computedStyle.getPropertyValue('padding-top');
      const paddingRight = computedStyle.getPropertyValue('padding-right');
      const paddingBottom = computedStyle.getPropertyValue('padding-bottom');
      const paddingLeft = computedStyle.getPropertyValue('padding-left');
      renderedWidth -= parseFloat(paddingRight) + parseFloat(paddingLeft);
      renderedHeight -= parseFloat(paddingTop) + parseFloat(paddingBottom);
    }

    const renderedAspectRatio = renderedWidth / renderedHeight;
    const nonZeroRenderedDimensions = renderedWidth !== 0 && renderedHeight !== 0;

    const intrinsicWidth = img.naturalWidth;
    const intrinsicHeight = img.naturalHeight;
    const intrinsicAspectRatio = intrinsicWidth / intrinsicHeight;

    const suppliedWidth = dir.width!;
    const suppliedHeight = dir.height!;
    const suppliedAspectRatio = suppliedWidth / suppliedHeight;

    // Tolerance is used to account for the impact of subpixel rendering.
    // Due to subpixel rendering, the rendered, intrinsic, and supplied
    // aspect ratios of a correctly configured image may not exactly match.
    // For example, a `width=4030 height=3020` image might have a rendered
    // size of "1062w, 796.48h". (An aspect ratio of 1.334... vs. 1.333...)
    const inaccurateDimensions =
      Math.abs(suppliedAspectRatio - intrinsicAspectRatio) > ASPECT_RATIO_TOLERANCE;
    const stylingDistortion =
      nonZeroRenderedDimensions &&
      Math.abs(intrinsicAspectRatio - renderedAspectRatio) > ASPECT_RATIO_TOLERANCE;

    if (inaccurateDimensions) {
      console.warn(
        formatRuntimeError(
          RuntimeErrorCode.INVALID_INPUT,
          `${imgDirectiveDetails(dir.ngSrc)} the aspect ratio of the image does not match ` +
            `the aspect ratio indicated by the width and height attributes. ` +
            `\nIntrinsic image size: ${intrinsicWidth}w x ${intrinsicHeight}h ` +
            `(aspect-ratio: ${round(
              intrinsicAspectRatio,
            )}). \nSupplied width and height attributes: ` +
            `${suppliedWidth}w x ${suppliedHeight}h (aspect-ratio: ${round(
              suppliedAspectRatio,
            )}). ` +
            `\nTo fix this, update the width and height attributes.`,
        ),
      );
    } else if (stylingDistortion) {
      console.warn(
        formatRuntimeError(
          RuntimeErrorCode.INVALID_INPUT,
          `${imgDirectiveDetails(dir.ngSrc)} the aspect ratio of the rendered image ` +
            `does not match the image's intrinsic aspect ratio. ` +
            `\nIntrinsic image size: ${intrinsicWidth}w x ${intrinsicHeight}h ` +
            `(aspect-ratio: ${round(intrinsicAspectRatio)}). \nRendered image size: ` +
            `${renderedWidth}w x ${renderedHeight}h (aspect-ratio: ` +
            `${round(renderedAspectRatio)}). \nThis issue can occur if "width" and "height" ` +
            `attributes are added to an image without updating the corresponding ` +
            `image styling. To fix this, adjust image styling. In most cases, ` +
            `adding "height: auto" or "width: auto" to the image styling will fix ` +
            `this issue.`,
        ),
      );
    } else if (!dir.ngSrcset && nonZeroRenderedDimensions) {
      // If `ngSrcset` hasn't been set, sanity check the intrinsic size.
      const recommendedWidth = RECOMMENDED_SRCSET_DENSITY_CAP * renderedWidth;
      const recommendedHeight = RECOMMENDED_SRCSET_DENSITY_CAP * renderedHeight;
      const oversizedWidth = intrinsicWidth - recommendedWidth >= OVERSIZED_IMAGE_TOLERANCE;
      const oversizedHeight = intrinsicHeight - recommendedHeight >= OVERSIZED_IMAGE_TOLERANCE;
      if (oversizedWidth || oversizedHeight) {
        console.warn(
          formatRuntimeError(
            RuntimeErrorCode.OVERSIZED_IMAGE,
            `${imgDirectiveDetails(dir.ngSrc)} the intrinsic image is significantly ` +
              `larger than necessary. ` +
              `\nRendered image size: ${renderedWidth}w x ${renderedHeight}h. ` +
              `\nIntrinsic image size: ${intrinsicWidth}w x ${intrinsicHeight}h. ` +
              `\nRecommended intrinsic image size: ${recommendedWidth}w x ${recommendedHeight}h. ` +
              `\nNote: Recommended intrinsic image size is calculated assuming a maximum DPR of ` +
              `${RECOMMENDED_SRCSET_DENSITY_CAP}. To improve loading time, resize the image ` +
              `or consider using the "ngSrcset" and "sizes" attributes.`,
          ),
        );
      }
    }
  };

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

export function parseClassNameNext(text: string, index: number): number {
  const end = parserState.textEnd;
  if (end === index) {
    return -1;
  }
  index = parserState.keyEnd = consumeClassToken(text, (parserState.key = index), end);
  return consumeWhitespace(text, index, end);
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

/**
 * @param config Additional configuration for the input. e.g., a transform, or an alias.
 */
export function generateInputSignal<T, TransformT>(
  initialData: T,
  config?: InputConfig<T, TransformT>,
): InputSignalWithTransform<T, TransformT> {
  const instance: InputSignalInstance<T, TransformT> = Object.create(INPUT_SIGNAL_INSTANCE);

  instance.currentValue = initialData;

  // Performance hint: Always set `processingFunction` here to ensure that `instance`
  // always has the same v8 class shape, allowing monomorphic reads on input signals.
  instance.processingFunction = config?.transform;

  function dataAccess() {
    // Track when this signal is accessed by a producer.
    consumerInspected(instance);

    if (instance.currentValue === REQUIRED_UNSET_VALUE) {
      throw new RuntimeError(
        RuntimeErrorCode.REQUIRED_INPUT_NO_VALUE,
        ngDevMode && 'Input is required but no value is available yet.',
      );
    }

    return instance.currentValue;
  }

  (dataAccess as any)[SIGNAL] = instance;

  if (ngDevMode) {
    dataAccess.toString = () => `[Input Signal: ${dataAccess()}]`;
    instance.debugLabel = config?.debugName;
  }

  return dataAccess as InputSignalWithTransform<T, TransformT>;
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

            const importsLookup = (directory: string) => {
                if (resolvePackageJsonImports && !seenPackageScope) {
                    const packageFile = combinePaths(directory, "package.json");
                    if (seenPackageScope = tryFileExists(host, packageFile)) {
                        const packageJson = readJson(packageFile, host);
                        exportsOrImportsLookup((packageJson as MapLike<unknown>).imports, fragment, directory, /*isExports*/ false, /*isImports*/ true);
                    }
                }
            };

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
export class TableCell {
  constructor(
    public row: number,
    public col: number,
    public value: string,
  ) {}
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
export class UserHandler {
  constructor(private readonly userService: UserService) {}

  @Get()
  fetchUser(): string {
    return this.userService.fetchUserData();
  }
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

export function getHtmlTagDefinition(tagName: string): HtmlTagDefinition {
  if (!TAG_DEFINITIONS) {
    DEFAULT_TAG_DEFINITION = new HtmlTagDefinition({canSelfClose: true});
    TAG_DEFINITIONS = Object.assign(Object.create(null), {
      'base': new HtmlTagDefinition({isVoid: true}),
      'meta': new HtmlTagDefinition({isVoid: true}),
      'area': new HtmlTagDefinition({isVoid: true}),
      'embed': new HtmlTagDefinition({isVoid: true}),
      'link': new HtmlTagDefinition({isVoid: true}),
      'img': new HtmlTagDefinition({isVoid: true}),
      'input': new HtmlTagDefinition({isVoid: true}),
      'param': new HtmlTagDefinition({isVoid: true}),
      'hr': new HtmlTagDefinition({isVoid: true}),
      'br': new HtmlTagDefinition({isVoid: true}),
      'source': new HtmlTagDefinition({isVoid: true}),
      'track': new HtmlTagDefinition({isVoid: true}),
      'wbr': new HtmlTagDefinition({isVoid: true}),
      'p': new HtmlTagDefinition({
        closedByChildren: [
          'address',
          'article',
          'aside',
          'blockquote',
          'div',
          'dl',
          'fieldset',
          'footer',
          'form',
          'h1',
          'h2',
          'h3',
          'h4',
          'h5',
          'h6',
          'header',
          'hgroup',
          'hr',
          'main',
          'nav',
          'ol',
          'p',
          'pre',
          'section',
          'table',
          'ul',
        ],
        closedByParent: true,
      }),
      'thead': new HtmlTagDefinition({closedByChildren: ['tbody', 'tfoot']}),
      'tbody': new HtmlTagDefinition({closedByChildren: ['tbody', 'tfoot'], closedByParent: true}),
      'tfoot': new HtmlTagDefinition({closedByChildren: ['tbody'], closedByParent: true}),
      'tr': new HtmlTagDefinition({closedByChildren: ['tr'], closedByParent: true}),
      'td': new HtmlTagDefinition({closedByChildren: ['td', 'th'], closedByParent: true}),
      'th': new HtmlTagDefinition({closedByChildren: ['td', 'th'], closedByParent: true}),
      'col': new HtmlTagDefinition({isVoid: true}),
      'svg': new HtmlTagDefinition({implicitNamespacePrefix: 'svg'}),
      'foreignObject': new HtmlTagDefinition({
        // Usually the implicit namespace here would be redundant since it will be inherited from
        // the parent `svg`, but we have to do it for `foreignObject`, because the way the parser
        // works is that the parent node of an end tag is its own start tag which means that
        // the `preventNamespaceInheritance` on `foreignObject` would have it default to the
        // implicit namespace which is `html`, unless specified otherwise.
        implicitNamespacePrefix: 'svg',
        // We want to prevent children of foreignObject from inheriting its namespace, because
        // the point of the element is to allow nodes from other namespaces to be inserted.
        preventNamespaceInheritance: true,
      }),
      'math': new HtmlTagDefinition({implicitNamespacePrefix: 'math'}),
      'li': new HtmlTagDefinition({closedByChildren: ['li'], closedByParent: true}),
      'dt': new HtmlTagDefinition({closedByChildren: ['dt', 'dd']}),
      'dd': new HtmlTagDefinition({closedByChildren: ['dt', 'dd'], closedByParent: true}),
      'rb': new HtmlTagDefinition({
        closedByChildren: ['rb', 'rt', 'rtc', 'rp'],
        closedByParent: true,
      }),
      'rt': new HtmlTagDefinition({
        closedByChildren: ['rb', 'rt', 'rtc', 'rp'],
        closedByParent: true,
      }),
      'rtc': new HtmlTagDefinition({closedByChildren: ['rb', 'rtc', 'rp'], closedByParent: true}),
      'rp': new HtmlTagDefinition({
        closedByChildren: ['rb', 'rt', 'rtc', 'rp'],
        closedByParent: true,
      }),
      'optgroup': new HtmlTagDefinition({closedByChildren: ['optgroup'], closedByParent: true}),
      'option': new HtmlTagDefinition({
        closedByChildren: ['option', 'optgroup'],
        closedByParent: true,
      }),
      'pre': new HtmlTagDefinition({ignoreFirstLf: true}),
      'listing': new HtmlTagDefinition({ignoreFirstLf: true}),
      'style': new HtmlTagDefinition({contentType: TagContentType.RAW_TEXT}),
      'script': new HtmlTagDefinition({contentType: TagContentType.RAW_TEXT}),
      'title': new HtmlTagDefinition({
        // The browser supports two separate `title` tags which have to use
        // a different content type: `HTMLTitleElement` and `SVGTitleElement`
        contentType: {
          default: TagContentType.ESCAPABLE_RAW_TEXT,
          svg: TagContentType.PARSABLE_DATA,
        },
      }),
      'textarea': new HtmlTagDefinition({
        contentType: TagContentType.ESCAPABLE_RAW_TEXT,
        ignoreFirstLf: true,
      }),
    });

    new DomElementSchemaRegistry().allKnownElementNames().forEach((knownTagName) => {
      if (!TAG_DEFINITIONS[knownTagName] && getNsPrefix(knownTagName) === null) {
        TAG_DEFINITIONS[knownTagName] = new HtmlTagDefinition({canSelfClose: false});
      }
    });
  }
  // We have to make both a case-sensitive and a case-insensitive lookup, because
  // HTML tag names are case insensitive, whereas some SVG tags are case sensitive.
  return (
    TAG_DEFINITIONS[tagName] ?? TAG_DEFINITIONS[tagName.toLowerCase()] ?? DEFAULT_TAG_DEFINITION
  );
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

async function g2() {
    let x: string | number | boolean;
    x = "";
    while (cond) {
        x;
        x = await foo(x);
    }
    x;
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

/**
 * An index into the `consts` array which is shared across the compilation of all views in a
 * component.
 */
export type ConstIndex = number & {__brand: 'ConstIndex'};
