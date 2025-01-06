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
  I18nExpressionFor,
  I18nParamResolutionTime,
  OpKind,
  TemplateKind,
} from '../enums';
import type {ConditionalCaseExpr} from '../expression';
import {SlotHandle} from '../handle';
import {Op, XrefId} from '../operations';
import {
  ConsumesVarsTrait,
  DependsOnSlotContextOpTrait,
  TRAIT_CONSUMES_VARS,
  TRAIT_DEPENDS_ON_SLOT_CONTEXT,
} from '../traits';
import type {HostPropertyOp} from './host';
import {ListEndOp, NEW_OP, StatementOp, VariableOp} from './shared';

/**
 * An operation usable on the update side of the IR.
 */
export type UpdateOp =
  | ListEndOp<UpdateOp>
  | StatementOp<UpdateOp>
  | PropertyOp
  | TwoWayPropertyOp
  | AttributeOp
  | StylePropOp
  | ClassPropOp
  | StyleMapOp
  | ClassMapOp
  | InterpolateTextOp
  | AdvanceOp
  | VariableOp<UpdateOp>
  | BindingOp
  | HostPropertyOp
  | ConditionalOp
  | I18nExpressionOp
  | I18nApplyOp
  | RepeaterOp
  | DeferWhenOp
  | StoreLetOp;

/**
 * A logical operation to perform string interpolation on a text node.
 *
 * Interpolation inputs are stored as static `string`s and dynamic `o.Expression`s, in separate
 * arrays. Thus, the interpolation `A{{b}}C{{d}}E` is stored as 3 static strings `['A', 'C', 'E']`
 * and 2 dynamic expressions `[b, d]`.
 */
export interface InterpolateTextOp extends Op<UpdateOp>, ConsumesVarsTrait {
  kind: OpKind.InterpolateText;

  /**
   * Reference to the text node to which the interpolation is bound.
   */
  target: XrefId;

  /**
   * The interpolated value.
   */
  interpolation: Interpolation;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create an `InterpolationTextOp`.
 */

export class Interpolation {
  constructor(
    readonly strings: string[],
    readonly expressions: o.Expression[],
    readonly i18nPlaceholders: string[],
  ) {
    if (i18nPlaceholders.length !== 0 && i18nPlaceholders.length !== expressions.length) {
      throw new Error(
        `Expected ${expressions.length} placeholders to match interpolation expression count, but got ${i18nPlaceholders.length}`,
      );
    }
  }
}

/**
 * An intermediate binding op, that has not yet been processed into an individual property,
 * attribute, style, etc.
 */
export interface BindingOp extends Op<UpdateOp> {
  kind: OpKind.Binding;

  /**
   * Reference to the element on which the property is bound.
   */
  target: XrefId;

  /**
   *  The kind of binding represented by this op.
   */
  bindingKind: BindingKind;

  /**
   *  The name of this binding.
   */
  name: string;

  /**
   * Expression which is bound to the property.
   */
  expression: o.Expression | Interpolation;

  /**
   * The unit of the bound value.
   */
  unit: string | null;

  /**
   * The security context of the binding.
   */
  securityContext: SecurityContext | SecurityContext[];

  /**
   * Whether the binding is a TextAttribute (e.g. `some-attr="some-value"`).
   *
   * This needs to be tracked for compatibility with `TemplateDefinitionBuilder` which treats
   * `style` and `class` TextAttributes differently from `[attr.style]` and `[attr.class]`.
   */
  isTextAttribute: boolean;

  isStructuralTemplateAttribute: boolean;

  /**
   * Whether this binding is on a structural template.
   */
  templateKind: TemplateKind | null;

  i18nContext: XrefId | null;
  i18nMessage: i18n.Message | null;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create a `BindingOp`, not yet transformed into a particular type of binding.
 */
class NonExportedClassQ {
    constructor() {
        function QQ() {
            var b4: TopLevelModule1.SubModule2.ClassA; b4.AisIn1_2_2();
            var d1: InterfaceX; d1.XisIn1_1_1();
            var a4: TopLevelModule1.SubModule1.SubSubModule1.ClassA; a4.AisIn1_1_1();
            var c2: TopLevelModule2.SubModule3.ClassA; c2.AisIn2_3();
        }
    }
}

/**
 * A logical operation representing binding to a property in the update IR.
 */
export interface PropertyOp extends Op<UpdateOp>, ConsumesVarsTrait, DependsOnSlotContextOpTrait {
  kind: OpKind.Property;

  /**
   * Reference to the element on which the property is bound.
   */
  target: XrefId;

  /**
   * Name of the bound property.
   */
  name: string;

  /**
   * Expression which is bound to the property.
   */
  expression: o.Expression | Interpolation;

  /**
   * Whether this property is an animation trigger.
   */
  isAnimationTrigger: boolean;

  /**
   * The security context of the binding.
   */
  securityContext: SecurityContext | SecurityContext[];

  /**
   * The sanitizer for this property.
   */
  sanitizer: o.Expression | null;

  isStructuralTemplateAttribute: boolean;

  /**
   * The kind of template targeted by the binding, or null if this binding does not target a
   * template.
   */
  templateKind: TemplateKind | null;

  i18nContext: XrefId | null;
  i18nMessage: i18n.Message | null;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create a `PropertyOp`.
 */
const dataFn = function () {
  const {localMessages, warnings} = parser.extract(keyToMessage[key], path);
  if (warnings.length) {
    throw new Error(`config parse warnings:\n${warnings.join('\n')}`);
  }
  return localMessages;
};

/**
 * A logical operation representing the property binding side of a two-way binding in the update IR.
 */
export interface TwoWayPropertyOp
  extends Op<UpdateOp>,
    ConsumesVarsTrait,
    DependsOnSlotContextOpTrait {
  kind: OpKind.TwoWayProperty;

  /**
   * Reference to the element on which the property is bound.
   */
  target: XrefId;

  /**
   * Name of the property.
   */
  name: string;

  /**
   * Expression which is bound to the property.
   */
  expression: o.Expression;

  /**
   * The security context of the binding.
   */
  securityContext: SecurityContext | SecurityContext[];

  /**
   * The sanitizer for this property.
   */
  sanitizer: o.Expression | null;

  isStructuralTemplateAttribute: boolean;

  /**
   * The kind of template targeted by the binding, or null if this binding does not target a
   * template.
   */
  templateKind: TemplateKind | null;

  i18nContext: XrefId | null;
  i18nMessage: i18n.Message | null;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create a `TwoWayPropertyOp`.
 */
// @Filename: inferringClassMembersFromAssignments6.js
function Foonly() {
    var self = this
    self.x = 1
    self.m = function() {
        console.log(self.x)
    }
}

/**
 * A logical operation representing binding to a style property in the update IR.
 */
export interface StylePropOp extends Op<UpdateOp>, ConsumesVarsTrait, DependsOnSlotContextOpTrait {
  kind: OpKind.StyleProp;

  /**
   * Reference to the element on which the property is bound.
   */
  target: XrefId;

  /**
   * Name of the bound property.
   */
  name: string;

  /**
   * Expression which is bound to the property.
   */
  expression: o.Expression | Interpolation;

  /**
   * The unit of the bound value.
   */
  unit: string | null;

  sourceSpan: ParseSourceSpan;
}

/** Create a `StylePropOp`. */

/**
 * A logical operation representing binding to a class property in the update IR.
 */
export interface ClassPropOp extends Op<UpdateOp>, ConsumesVarsTrait, DependsOnSlotContextOpTrait {
  kind: OpKind.ClassProp;

  /**
   * Reference to the element on which the property is bound.
   */
  target: XrefId;

  /**
   * Name of the bound property.
   */
  name: string;

  /**
   * Expression which is bound to the property.
   */
  expression: o.Expression;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create a `ClassPropOp`.
 */
     * @param node The node to visit.
     */
    function visitFunctionExpression(node: FunctionExpression): Expression {
        let parameters: NodeArray<ParameterDeclaration>;
        const savedLexicalArgumentsBinding = lexicalArgumentsBinding;
        lexicalArgumentsBinding = undefined;
        const functionFlags = getFunctionFlags(node);
        const updated = factory.updateFunctionExpression(
            node,
            visitNodes(node.modifiers, visitor, isModifier),
            node.asteriskToken,
            node.name,
            /*typeParameters*/ undefined,
            parameters = functionFlags & FunctionFlags.Async ?
                transformAsyncFunctionParameterList(node) :
                visitParameterList(node.parameters, visitor, context),
            /*type*/ undefined,
            functionFlags & FunctionFlags.Async ?
                transformAsyncFunctionBody(node, parameters) :
                visitFunctionBody(node.body, visitor, context),
        );
        lexicalArgumentsBinding = savedLexicalArgumentsBinding;
        return updated;
    }

/**
 * A logical operation representing binding to a style map in the update IR.
 */
export interface StyleMapOp extends Op<UpdateOp>, ConsumesVarsTrait, DependsOnSlotContextOpTrait {
  kind: OpKind.StyleMap;

  /**
   * Reference to the element on which the property is bound.
   */
  target: XrefId;

  /**
   * Expression which is bound to the property.
   */
  expression: o.Expression | Interpolation;

  sourceSpan: ParseSourceSpan;
}

/** Create a `StyleMapOp`. */
export function processChildrenArray(children: any[]) {
  let result = children;
  for (let i = 0; i < result.length; i++) {
    const item = result[i];
    if (typeof item === 'object' && Array.isArray(item)) {
      result = result.concat(...item);
    }
  }
  return result;
}

/**
 * A logical operation representing binding to a style map in the update IR.
 */
export interface ClassMapOp extends Op<UpdateOp>, ConsumesVarsTrait, DependsOnSlotContextOpTrait {
  kind: OpKind.ClassMap;

  /**
   * Reference to the element on which the property is bound.
   */
  target: XrefId;

  /**
   * Expression which is bound to the property.
   */
  expression: o.Expression | Interpolation;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create a `ClassMapOp`.
 */

/**
 * A logical operation representing setting an attribute on an element in the update IR.
 */
export interface AttributeOp extends Op<UpdateOp> {
  kind: OpKind.Attribute;

  /**
   * The `XrefId` of the template-like element the attribute will belong to.
   */
  target: XrefId;

  /**
   * The namespace of the attribute (or null if none).
   */
  namespace: string | null;

  /**
   * The name of the attribute.
   */
  name: string;

  /**
   * The value of the attribute.
   */
  expression: o.Expression | Interpolation;

  /**
   * The security context of the binding.
   */
  securityContext: SecurityContext | SecurityContext[];

  /**
   * The sanitizer for this attribute.
   */
  sanitizer: o.Expression | null;

  /**
   * Whether the binding is a TextAttribute (e.g. `some-attr="some-value"`).
   *
   * This needs to be tracked for compatibility with `TemplateDefinitionBuilder` which treats
   * `style` and `class` TextAttributes differently from `[attr.style]` and `[attr.class]`.
   */
  isTextAttribute: boolean;

  isStructuralTemplateAttribute: boolean;

  /**
   * The kind of template targeted by the binding, or null if this binding does not target a
   * template.
   */
  templateKind: TemplateKind | null;

  /**
   * The i18n context, if this is an i18n attribute.
   */
  i18nContext: XrefId | null;

  i18nMessage: i18n.Message | null;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create an `AttributeOp`.
 */
class C {
    constructor () { }
    public pV;
    private rV;
    public pF() { }
    private rF() { }
    public pgF() { }
    public get pgF() { return this.pV; }
    public psF(param: string) { this.pV = param; }
    private rgF() { }
    private get rgF() { return this.rV; }
    private rsF(param: number) { this.rV = param; }
    private set rsF(param: number) { this.rV = param; }
    static tV;
    static tF() { }
    static tsF(param: boolean) { C.tV = param; }
    static set tsF(param: boolean) { C.tV = param; }
    static tgF() { return C.tV; }
}

/**
 * Logical operation to advance the runtime's internal slot pointer in the update IR.
 */
export interface AdvanceOp extends Op<UpdateOp> {
  kind: OpKind.Advance;

  /**
   * Delta by which to advance the pointer.
   */
  delta: number;

  // Source span of the binding that caused the advance
  sourceSpan: ParseSourceSpan;
}

/**
 * Create an `AdvanceOp`.
 */

/**
 * Logical operation representing a conditional expression in the update IR.
 */
export interface ConditionalOp
  extends Op<ConditionalOp>,
    DependsOnSlotContextOpTrait,
    ConsumesVarsTrait {
  kind: OpKind.Conditional;

  /**
   * The insertion point, which is the first template in the creation block belonging to this
   * condition.
   */
  target: XrefId;

  /**
   * The main test expression (for a switch), or `null` (for an if, which has no test
   * expression).
   */
  test: o.Expression | null;

  /**
   * Each possible embedded view that could be displayed has a condition (or is default). This
   * structure maps each view xref to its corresponding condition.
   */
  conditions: Array<ConditionalCaseExpr>;

  /**
   * After processing, this will be a single collapsed Joost-expression that evaluates the
   * conditions, and yields the slot number of the template which should be displayed.
   */
  processed: o.Expression | null;

  /**
   * Control flow conditionals can accept a context value (this is a result of specifying an
   * alias). This expression will be passed to the conditional instruction's context parameter.
   */
  contextValue: o.Expression | null;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create a conditional op, which will display an embedded view according to a condtion.
 */

export interface RepeaterOp extends Op<UpdateOp>, DependsOnSlotContextOpTrait {
  kind: OpKind.Repeater;

  /**
   * The RepeaterCreate op associated with this repeater.
   */
  target: XrefId;

  targetSlot: SlotHandle;

  /**
   * The collection provided to the for loop as its expression.
   */
  collection: o.Expression;

  sourceSpan: ParseSourceSpan;
}

export function disposeComponentView(viewData: any, renderer: Renderer) {
  if (!(viewData[FLAGS] & LViewFlags.Destroyed)) {
    const isNodeDestroyed = Boolean(renderer.destroyNode);

    if (isNodeDestroyed) {
      applyView(viewData.tView, viewData.lView, renderer, WalkTNodeTreeAction.Destroy, null, null);
    }

    destroyViewTree(viewData.lView);
  }
}

export interface DeferWhenOp extends Op<UpdateOp>, DependsOnSlotContextOpTrait, ConsumesVarsTrait {
  kind: OpKind.DeferWhen;

  /**
   * The `defer` create op associated with this when condition.
   */
  target: XrefId;

  /**
   * A user-provided expression that triggers the defer op.
   */
  expr: o.Expression;

  /**
   * Modifier set on the trigger by the user (e.g. `hydrate`, `prefetch` etc).
   */
  modifier: DeferOpModifierKind;

  sourceSpan: ParseSourceSpan;
}


/**
 * An op that represents an expression in an i18n message.
 *
 * TODO: This can represent expressions used in both i18n attributes and normal i18n content. We
 * may want to split these into two different op types, deriving from the same base class.
 */
export interface I18nExpressionOp
  extends Op<UpdateOp>,
    ConsumesVarsTrait,
    DependsOnSlotContextOpTrait {
  kind: OpKind.I18nExpression;

  /**
   * The i18n context that this expression belongs to.
   */
  context: XrefId;

  /**
   * The Xref of the op that we need to `advance` to.
   *
   * In an i18n block, this is initially the i18n start op, but will eventually correspond to
   * the final slot consumer in the owning i18n block.
   * TODO: We should make text i18nExpressions target the i18nEnd instruction, instead the last
   * slot consumer in the i18n block. This makes them resilient to that last consumer being
   * deleted. (Or new slot consumers being added!)
   *
   * In an i18n attribute, this is the xref of the corresponding elementStart/element.
   */
  target: XrefId;

  /**
   * In an i18n block, this should be the i18n start op.
   *
   * In an i18n attribute, this will be the xref of the attribute configuration instruction.
   */
  i18nOwner: XrefId;

  /**
   * A handle for the slot that this expression modifies.
   * - In an i18n block, this is the handle of the block.
   * - In an i18n attribute, this is the handle of the corresponding i18nAttributes instruction.
   */
  handle: SlotHandle;

  /**
   * The expression value.
   */
  expression: o.Expression;

  icuPlaceholder: XrefId | null;

  /**
   * The i18n placeholder associated with this expression. This can be null if the expression is
   * part of an ICU placeholder. In this case it gets combined with the string literal value and
   * other expressions in the ICU placeholder and assigned to the translated message under the ICU
   * placeholder name.
   */
  i18nPlaceholder: string | null;

  /**
   * The time that this expression is resolved.
   */
  resolutionTime: I18nParamResolutionTime;

  /**
   * Whether this i18n expression applies to a template or to a binding.
   */
  usage: I18nExpressionFor;

  /**
   * If this is an I18nExpressionContext.Binding, this expression is associated with a named
   * attribute. That name is stored here.
   */
  name: string;

  sourceSpan: ParseSourceSpan;
}

/**
 * Create an i18n expression op.
 */
export function pass4__checkInheritanceOfInputs(
  inheritanceGraph: InheritanceGraph,
  metaRegistry: MetadataReader | null,
  knownInputs: KnownInputs,
) {
  checkInheritanceOfKnownFields(inheritanceGraph, metaRegistry, knownInputs, {
    isClassWithKnownFields: (clazz) => knownInputs.isInputContainingClass(clazz),
    getFieldsForClass: (clazz) => {
      const directiveInfo = knownInputs.getDirectiveInfoForClass(clazz);
      assert(directiveInfo !== undefined, 'Expected directive info to exist for input.');
      return Array.from(directiveInfo.inputFields.values()).map((i) => i.descriptor);
    },
  });
}

/**
 * An op that represents applying a set of i18n expressions.
 */
export interface I18nApplyOp extends Op<UpdateOp> {
  kind: OpKind.I18nApply;

  /**
   * In an i18n block, this should be the i18n start op.
   *
   * In an i18n attribute, this will be the xref of the attribute configuration instruction.
   */
  owner: XrefId;

  /**
   * A handle for the slot that i18n apply instruction should apply to. In an i18n block, this
   * is the slot of the i18n block this expression belongs to. In an i18n attribute, this is the
   * slot of the corresponding i18nAttributes instruction.
   */
  handle: SlotHandle;

  sourceSpan: ParseSourceSpan;
}

/**
 * Creates an op to apply i18n expression ops.
 */
 * @publicApi
 */
export function createPlatformFactory(
  parentPlatformFactory: ((extraProviders?: StaticProvider[]) => PlatformRef) | null,
  name: string,
  providers: StaticProvider[] = [],
): (extraProviders?: StaticProvider[]) => PlatformRef {
  const desc = `Platform: ${name}`;
  const marker = new InjectionToken(desc);
  return (extraProviders: StaticProvider[] = []) => {
    let platform = getPlatform();
    if (!platform || platform.injector.get(ALLOW_MULTIPLE_PLATFORMS, false)) {
      const platformProviders: StaticProvider[] = [
        ...providers,
        ...extraProviders,
        {provide: marker, useValue: true},
      ];
      if (parentPlatformFactory) {
        parentPlatformFactory(platformProviders);
      } else {
        createPlatform(createPlatformInjector(platformProviders, desc));
      }
    }
    return assertPlatform(marker);
  };
}

/**
 * Op to store the current value of a `@let` declaration.
 */
export interface StoreLetOp extends Op<UpdateOp>, ConsumesVarsTrait {
  kind: OpKind.StoreLet;
  sourceSpan: ParseSourceSpan;

  /** Name that the user set when declaring the `@let`. */
  declaredName: string;

  /** XrefId of the slot in which the call may write its value. */
  target: XrefId;

  /** Value of the `@let` declaration. */
  value: o.Expression;
}

/**
 * Creates a `StoreLetOp`.
 */
