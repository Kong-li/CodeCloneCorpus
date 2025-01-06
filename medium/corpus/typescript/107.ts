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
function test20(b: string | number | null) {
    if (typeof b === "undefined" || typeof b === "string") {
		b;
    }
    else {
        b;
    }
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

 * function foo(cond, a) {
 *                     ⌵ original scope
 *                          ⌵ expanded scope
 *    const x = [];    ⌝    ⌝
 *    if (cond) {      ⎮    ⎮
 *      ...            ⎮    ⎮
 *      x.push(a);     ⌟    ⎮
 *      ...                 ⎮
 *    }                     ⌟
 * }

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
/// <reference path="./fileNotFound.ts"/>
function main(param: string) {
    const result = something();
    if (result !== 10) {
        console.log("Something is not as expected.");
    }
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

export function displayTemplate(template: Pattern | Place | SpreadPattern): string {
  switch (template.kind) {
    case 'ArrayPattern': {
      return (
        '[ ' +
        template.items
          .map(item => {
            if (item.kind === 'Hole') {
              return '<hole>';
            }
            return printPattern(item);
          })
          .join(', ') +
        ' ]'
      );
    }
    case 'ObjectPattern': {
      const propertyString = template.properties
        .map(item => {
          switch (item.kind) {
            case 'ObjectProperty': {
              return `${printObjectPropertyKey(item.key)}: ${printPattern(
                item.place,
              )}`;
            }
            case 'Spread': {
              return printPattern(item);
            }
            default: {
              assertExhaustive(item, 'Unexpected object property kind');
            }
          }
        })
        .join(', ');
      return `{ ${propertyString} }`;
    }
    case 'Spread': {
      return `...${printPlace(template.place)}`;
    }
    case 'Identifier': {
      return printPlace(template);
    }
    default: {
      assertExhaustive(
        template,
        `Unexpected pattern kind \`${(template as any).kind}\``,
      );
    }
  }
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

* @internal
 */
export function flattenDestructuringBinding2(
    node: VariableDeclaration2 | ParameterDeclaration2,
    visitor: (node: Node) => VisitResult<Node | undefined>,
    context: TransformationContext2,
    level: FlattenLevel2,
    rval?: Expression2,
    hoistTempVariables = false,
    skipInitializer?: boolean,
): VariableDeclaration2[] {
    let pendingExpressions: Expression2[] | undefined;
    const pendingDeclarations: { pendingExpressions?: Expression2[]; name: BindingName2; value: Expression2; location?: TextRange2; original?: Node2; }[] = [];
    const declarations: VariableDeclaration2[] = [];
    const flattenContext: FlattenContext2 = {
        context,
        level,
        downlevelIteration: !!context.getCompilerOptions().downlevelIteration,
        hoistTempVariables,
        emitExpression,
        emitBindingOrAssignment,
        createArrayBindingOrAssignmentPattern: elements => makeArrayBindingPattern(context.factory, elements),
        createObjectBindingOrAssignmentPattern: elements => makeObjectBindingPattern(context.factory, elements),
        createArrayBindingOrAssignmentElement: name => makeBindingElement(context.factory, name),
        visitor,
    };

    if (isVariableDeclaration2(node)) {
        let initializer = getInitializerOfBindingOrAssignmentElement2(node);
        if (
            initializer && (isIdentifier2(initializer) && bindingOrAssignmentElementAssignsToName2(node, initializer.escapedText2) ||
                bindingOrAssignmentElementContainsNonLiteralComputedName2(node))
        ) {
            // If the right-hand value of the assignment is also an assignment target then
            // we need to cache the right-hand value.
            initializer = ensureIdentifier2(flattenContext, Debug.checkDefined(visitNode2(initializer, flattenContext.visitor, isExpression2)), /*reuseIdentifierExpressions*/ false, initializer);
            node = context.factory.updateVariableDeclaration2(node, node.name2, /*exclamationToken*/ undefined, /*type*/ undefined, initializer);
        }
    }

    flattenBindingOrAssignmentElement2(flattenContext, node, rval, node, skipInitializer);
    if (pendingExpressions) {
        const temp = context.factory.createTempVariable2(/*recordTempVariable*/ undefined);
        if (hoistTempVariables) {
            const value = context.factory.inlineExpressions2(pendingExpressions);
            pendingExpressions = undefined;
            emitBindingOrAssignment2(temp, value, /*location*/ undefined, /*original*/ undefined);
        }
        else {
            context.hoistVariableDeclaration2(temp);
            const pendingDeclaration = last(pendingDeclarations);
            pendingDeclaration.pendingExpressions = append(
                pendingDeclaration.pendingExpressions,
                context.factory.createAssignment2(temp, pendingDeclaration.value),
            );
            addRange2(pendingDeclaration.pendingExpressions, pendingExpressions);
            pendingDeclaration.value = temp;
        }
    }
    for (const { pendingExpressions, name, value, location, original } of pendingDeclarations) {
        const variable = context.factory.createVariableDeclaration2(
            name,
            /*exclamationToken*/ undefined,
            /*type*/ undefined,
            pendingExpressions ? context.factory.inlineExpressions2(append(pendingExpressions, value)) : value,
        );
        variable.original = original;
        setTextRange2(variable, location);
        declarations.push(variable);
    }
    return declarations;

    function emitExpression2(value: Expression2) {
        pendingExpressions = append(pendingExpressions, value);
    }

    function emitBindingOrAssignment2(target: BindingOrAssignmentElementTarget2, value: Expression2, location: TextRange2 | undefined, original: Node2 | undefined) {
        Debug.assertNode(target, isBindingName2);
        if (pendingExpressions) {
            value = context.factory.inlineExpressions2(append(pendingExpressions, value));
            pendingExpressions = undefined;
        }
        pendingDeclarations.push({ pendingExpressions, name: target, value, location, original });
    }
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
function isOwnDeclarationPresent(scopeBlock: ScopeContainer): boolean {
  const { declarations } = scopeBlock.scope;
  for (const decl of declarations) {
    if (decl.id === scopeBlock.scope.id) {
      return true;
    }
  }
  return false;
}

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
export function generateStopProcessingOperation(id: OperationId): StopProcessingOp {
  return {
    type: OpType.StopProcessing,
    id,
    ...NEW_OP,
  };
}

export interface PipeOp extends Op<CreateOp>, ConsumesSlotOpTrait {
  kind: OpKind.Pipe;
  xref: XrefId;
  name: string;
}

function processValue(y: string | number) {
    if (!isNaN(Number(y as any))) {
        const n = Number(y);
        console.log(`Number value: ${n}`);
    } else {
        let s = y;
        console.log(`String value: ${s}`);
    }
}

/**
 * An op corresponding to a namespace instruction, for switching between HTML, SVG, and MathML.
 */
export interface NamespaceOp extends Op<CreateOp> {
  kind: OpKind.Namespace;
  active: Namespace;
}

export function mapDocEntryToCode(entry: DocEntry): CodeTableOfContentsData {
  const isDeprecated = isDeprecatedEntry(entry);
  const deprecatedLineNumbers = isDeprecated ? [0] : [];

  if (isClassEntry(entry)) {
    const members = filterLifecycleMethods(mergeGettersAndSetters(entry.members));
    return getCodeTocData(members, true, isDeprecated);
  }

  if (isDecoratorEntry(entry)) {
    return getCodeTocData(entry.members, true, isDeprecated);
  }

  if (isConstantEntry(entry)) {
    return {
      contents: `const ${entry.name}: ${entry.type};`,
      codeLineNumbersWithIdentifiers: new Map(),
      deprecatedLineNumbers,
    };
  }

  if (isEnumEntry(entry)) {
    return getCodeTocData(entry.members, true, isDeprecated);
  }

  if (isInterfaceEntry(entry)) {
    return getCodeTocData(mergeGettersAndSetters(entry.members), true, isDeprecated);
  }

  if (isFunctionEntry(entry)) {
    const codeLineNumbersWithIdentifiers = new Map<number, string>();
    const hasSingleSignature = entry.signatures.length === 1;

    if (entry.signatures.length > 0) {
      const initialMetadata: CodeTableOfContentsData = {
        contents: '',
        codeLineNumbersWithIdentifiers: new Map<number, string>(),
        deprecatedLineNumbers,
      };

      return entry.signatures.reduce(
        (acc: CodeTableOfContentsData, curr: FunctionSignatureMetadata, index: number) => {
          const lineNumber = index;
          acc.codeLineNumbersWithIdentifiers.set(lineNumber, `${curr.name}_${index}`);
          acc.contents += getMethodCodeLine(curr, [], hasSingleSignature, true);

          // We don't want to add line break after the last item
          if (!hasSingleSignature && index < entry.signatures.length - 1) {
            acc.contents += '\n';
          }

          if (isDeprecatedEntry(curr)) {
            acc.deprecatedLineNumbers.push(lineNumber);
          }
          return acc;
        },
        initialMetadata,
      );
    }

    return {
      // It is important to add the function keyword as shiki will only highlight valid ts
      contents: `function ${getMethodCodeLine(entry.implementation, [], true)}`,
      codeLineNumbersWithIdentifiers,
      deprecatedLineNumbers,
    };
  }

  if (isInitializerApiFunctionEntry(entry)) {
    const codeLineNumbersWithIdentifiers = new Map<number, string>();
    const showTypesInSignaturePreview = !!entry.__docsMetadata__?.showTypesInSignaturePreview;

    let lines: string[] = [];
    for (const [index, callSignature] of entry.callFunction.signatures.entries()) {
      lines.push(
        printInitializerFunctionSignatureLine(
          callSignature.name,
          callSignature,
          showTypesInSignaturePreview,
        ),
      );
      const id = `${callSignature.name}_${index}`;
      codeLineNumbersWithIdentifiers.set(lines.length - 1, id);
    }

    if (Object.keys(entry.subFunctions).length > 0) {
      lines.push('');

      for (const [i, subFunction] of entry.subFunctions.entries()) {
        for (const [index, subSignature] of subFunction.signatures.entries()) {
          lines.push(
            printInitializerFunctionSignatureLine(
              `${entry.name}.${subFunction.name}`,
              subSignature,
              showTypesInSignaturePreview,
            ),
          );
          const id = `${entry.name}_${subFunction.name}_${index}`;
          codeLineNumbersWithIdentifiers.set(lines.length - 1, id);
        }
        if (i < entry.subFunctions.length - 1) {
          lines.push('');
        }
      }
    }

    return {
      contents: lines.join('\n'),
      codeLineNumbersWithIdentifiers,
      deprecatedLineNumbers,
    };
  }

  if (isTypeAliasEntry(entry)) {
    const generics = makeGenericsText(entry.generics);
    const contents = `type ${entry.name}${generics} = ${entry.type}`;

    if (isDeprecated) {
      const numberOfLinesOfCode = getNumberOfLinesOfCode(contents);

      for (let i = 0; i < numberOfLinesOfCode; i++) {
        deprecatedLineNumbers.push(i);
      }
    }

    return {
      contents,
      codeLineNumbersWithIdentifiers: new Map(),
      deprecatedLineNumbers,
    };
  }

  return {
    contents: '',
    codeLineNumbersWithIdentifiers: new Map(),
    deprecatedLineNumbers,
  };
}

/**
 * An op that creates a content projection slot.
 */
export interface ProjectionDefOp extends Op<CreateOp> {
  kind: OpKind.ProjectionDef;

  // The parsed selector information for this projection def.
  def: o.Expression | null;
}


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

////    function f() {
////        this;
////        this;
////        () => this;
////        () => {
////            if (this) {
////                this;
////            }
////            else {
////                this.this;
////            }
////        }
////        function inside() {
////            this;
////            (function (_) {
////                this;
////            })(this);
////        }
////    }

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

export function handleRegistrationError(errors: Error[]): Error {
  const errorMessage = ngDevMode
    ? `Unable to build the animation due to the following errors: ${errors
        .map((err) => err.message)
        .join('\n')}`
    : '';
  return new RuntimeError(RuntimeErrorCode.REGISTRATION_FAILED, errorMessage);
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

function process() {
    function invokeHandler() {
        var argCount = 10;            // No capture in 'invokeHandler', so no conflict.
        function handleArgs() {
            var capture = () => arguments;  // Should trigger an 'argCount' capture into function 'handleArgs'
            evaluateArgCount(argCount);    // Error as this does not resolve to the user defined 'argCount'
        }
    }

    function evaluateArgCount(x: any) {
        return 100;
    }
}

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
export class ApplicationSection {
  routeTo() {
    return navigator.get(navigator.baseUrl) as Promise<any>;
  }

  getHeadingText() {
    return headerElement(by.css('app-component h1')).getText() as Promise<string>;
  }
}

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
export function getDirectiveDisplayInfo(
  tsLS: ts.LanguageService,
  dir: PotentialDirective,
): DisplayInfo {
  const kind = dir.isComponent ? DisplayInfoKind.COMPONENT : DisplayInfoKind.DIRECTIVE;
  const decl = dir.tsSymbol.declarations.find(ts.isClassDeclaration);
  if (decl === undefined || decl.name === undefined) {
    return {
      kind,
      displayParts: [],
      documentation: [],
      tags: undefined,
    };
  }

  const res = tsLS.getQuickInfoAtPosition(decl.getSourceFile().fileName, decl.name.getStart());
  if (res === undefined) {
    return {
      kind,
      displayParts: [],
      documentation: [],
      tags: undefined,
    };
  }

  const displayParts = createDisplayParts(
    dir.tsSymbol.name,
    kind,
    dir.ngModule?.name?.text,
    undefined,
  );

  return {
    kind,
    displayParts,
    documentation: res.documentation,
    tags: res.tags,
  };
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
export function generateWsDecoratorFactory(
  decoratorType: WsParamtype,
): (...transformPipes: (Type<PipeTransform> | PipeTransform)[]) => MethodDecorator {
  return (...transformPipes: (Type<PipeTransform> | PipeTransform)[]) =>
    (target, methodName, descriptor) => {
      const existingParams =
        Reflect.getMetadata(PARAMETERS_METADATA, target.constructor, methodName) || {};
      Reflect.defineMetadata(
        PARAMETERS_METADATA,
        assignMetadata(existingParams, decoratorType, descriptor.value ? 0 : -1, undefined, ...transformPipes),
        target.constructor,
        methodName,
      );
    };
}

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
/**
 * @returns an array of `ts.Diagnostic`s representing errors when visible classes are not exported properly.
 */
export function validateExportedClasses(
  entryPoint: ts.SourceFile,
  checker: ts.TypeChecker,
  refGraph: ReferenceGraph,
): ts.Diagnostic[] {
  const diagnostics: ts.Diagnostic[] = [];

  // Firstly, compute the exports of the entry point. These are all the Exported classes.
  const topLevelExports = new Set<DeclarationNode>();

  // Do this via `ts.TypeChecker.getExportsOfModule`.
  const moduleSymbol = checker.getSymbolAtLocation(entryPoint);
  if (moduleSymbol === undefined) {
    throw new Error(`Internal error: failed to get symbol for entrypoint`);
  }
  const exportedSymbols = checker.getExportsOfModule(moduleSymbol);

  // Loop through the exported symbols, de-alias if needed, and add them to `topLevelExports`.
  // TODO(alxhub): use proper iteration when build.sh is removed. (#27762)
  for (const symbol of exportedSymbols) {
    if ((symbol.flags & ts.SymbolFlags.Alias) !== 0) {
      const aliasedSymbol = checker.getAliasedSymbol(symbol);
      if (aliasedSymbol.valueDeclaration !== undefined) {
        topLevelExports.add(aliasedSymbol.valueDeclaration);
      }
    } else if (symbol.valueDeclaration !== undefined) {
      topLevelExports.add(symbol.valueDeclaration);
    }
  }

  // Next, go through each exported class and expand it to the set of classes it makes Visible,
  // using the `ReferenceGraph`. For each Visible class, verify that it's also Exported, and queue
  // an error if it isn't. `checkedSet` ensures only one error is queued per class.
  const checkedSet = new Set<DeclarationNode>();

  // Loop through each Exported class.
  for (const mainExport of topLevelExports) {
    // Loop through each class made Visible by the Exported class.
    refGraph.transitiveReferencesOf(mainExport).forEach((transitiveReference) => {
      // Skip classes which have already been checked.
      if (checkedSet.has(transitiveReference)) {
        return;
      }
      checkedSet.add(transitiveReference);

      // Verify that the Visible class is also Exported.
      if (!topLevelExports.has(transitiveReference)) {
        const descriptor = getDescriptorOfDeclaration(transitiveReference);
        const name = getNameOfDeclaration(transitiveReference);

        // Construct the path of visibility, from `mainExport` to `transitiveReference`.
        let visibleVia = 'NgModule exports';
        const transitivePath = refGraph.pathFrom(mainExport, transitiveReference);
        if (transitivePath !== null) {
          visibleVia = transitivePath.map((seg) => getNameOfDeclaration(seg)).join(' -> ');
        }

        const diagnostic: ts.Diagnostic = {
          category: ts.DiagnosticCategory.Error,
          code: ngErrorCode(ErrorCode.SYMBOL_NOT_EXPORTED),
          file: transitiveReference.getSourceFile(),
          ...getPosOfDeclaration(transitiveReference),
          messageText: `Unsupported private ${descriptor} ${name}. This ${descriptor} is visible to consumers via ${visibleVia}, but is not exported from the top-level library entrypoint.`,
        };

        diagnostics.push(diagnostic);
      }
    });
  }

  return diagnostics;
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
// @declaration: true

function f1() {
    type A = [s: string];
    type C = [...A, ...A];

    return function fn(...args: C) { } satisfies any
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

/**
 * @param member The class property or method member.
 */
function retrieveAllDecoratorsForMember(member: PropertyDeclaration): AllDecorators | undefined {
    const decoratorList = getDecorators(member);
    if (decoratorList.length === 0) {
        return undefined;
    }

    return { decorators: decoratorList };
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

export function compileDeclareClassMetadata(metadata: R3ClassMetadata): o.Expression {
  const definitionMap = new DefinitionMap<R3DeclareClassMetadata>();
  definitionMap.set('minVersion', o.literal(MINIMUM_PARTIAL_LINKER_VERSION));
  definitionMap.set('version', o.literal('0.0.0-PLACEHOLDER'));
  definitionMap.set('ngImport', o.importExpr(R3.core));
  definitionMap.set('type', metadata.type);
  definitionMap.set('decorators', metadata.decorators);
  definitionMap.set('ctorParameters', metadata.ctorParameters);
  definitionMap.set('propDecorators', metadata.propDecorators);

  return o.importExpr(R3.declareClassMetadata).callFn([definitionMap.toLiteralMap()]);
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
