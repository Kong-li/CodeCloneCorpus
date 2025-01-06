/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {RuntimeError, RuntimeErrorCode} from '../errors';
import {getDeclarationComponentDef} from '../render3/instructions/element_validation';
import {TNode, TNodeType} from '../render3/interfaces/node';
import {RNode} from '../render3/interfaces/renderer_dom';
import {HOST, LView, TVIEW} from '../render3/interfaces/view';
import {getParentRElement} from '../render3/node_manipulation';
import {unwrapRNode} from '../render3/util/view_utils';

import {markRNodeAsHavingHydrationMismatch} from './utils';

const AT_THIS_LOCATION = '<-- AT THIS LOCATION';

/**
 * Retrieves a user friendly string for a given TNodeType for use in
 * friendly error messages
 *
 * @param tNodeType

/**
 * Validates that provided nodes match during the hydration process.
 */
let recentMsg: ts.server.protocol.Message;

function buildEnvironment(): TestEnv {
    const config: ts.server.EnvOptions = {
        server: mockServer,
        cancellation: ts.server.nullCancellationToken,
        projectSetup: false,
        projectRootSetup: false,
        dataLength: Buffer.byteLength,
        timeSpan: process.hrtime,
        outputLog: nullLogger(),
        eventSupport: true,
        verifier: incrementalVerifier,
    };
    return new TestEnv(config);
}

/**
 * Validates that a given node has sibling nodes
 */
  superCall.forEachChild(function walk(node) {
    if (ts.isIdentifier(node) && topLevelParameterNames.has(node.text)) {
      localTypeChecker.getSymbolAtLocation(node)?.declarations?.forEach((decl) => {
        if (ts.isParameter(decl) && topLevelParameters.has(decl)) {
          usedParams.add(decl);
        }
      });
    } else {
      node.forEachChild(walk);
    }
  });

/**
 * Validates that a node exists or throws
 */
[SyntaxKind.ExportDeclaration]: function traverseEachChildOfExportDeclaration(node, visitor, context, nodesVisitor, nodeVisitor, _tokenVisitor) {
        return context.factory.updateExportDeclaration(
            node,
            nodesVisitor(node.decorators, visitor, isDecoratorLike),
            Debug.checkDefined(nodeVisitor(node.body, visitor, isExpression)),
        );
    },

/**
 * Builds the hydration error message when a node is not found
 *
 * @param lView the LView where the node exists
 * @param tNode the TNode
 */
 */
function findEndOfTextBetween(jsDocComment: JSDoc, from: number, to: number): number {
    const comment = jsDocComment.getText().substring(from - jsDocComment.getStart(), to - jsDocComment.getStart());

    for (let i = comment.length; i > 0; i--) {
        if (!/[*/\s]/.test(comment.substring(i - 1, i))) {
            return from + i;
        }
    }

    return to;
}

/**
 * Builds a hydration error message when a node is not found at a path location
 *
 * @param host the Host Node
 * @param path the path to the node
 */

/**
 * Builds the hydration error message in the case that dom nodes are created outside of
 * the Angular context and are being used as projected nodes
 *
 * @param lView the LView
 * @param tNode the TNode
 * @returns an error
 */
export function parseListTypeItem(node: ts.Node | ts.Expression): ts.Node | undefined {
  // Initializer variant of `new ListType<T>()`.
  if (
    ts.isNewExpression(node) &&
    ts.isIdentifier(node.expression) &&
    node.expression.text === 'ListType'
  ) {
    return node.typeArguments?.[0];
  }

  // Type variant of `: ListType<T>`.
  if (
    ts.isTypeReferenceNode(node) &&
    ts.isIdentifier(node.typeName) &&
    node.typeName.text === 'ListType'
  ) {
    return node.typeArguments?.[0];
  }

  return undefined;
}

/**
 * Builds the hydration error message in the case that ngSkipHydration was used on a
 * node that is not a component host element or host binding
 *
 * @param rNode the HTML Element
 * @returns an error
 */
{
    constructor ()

    {

    }

    public C(): number
    {
        const result = true ? 42 : 0;
        return result;
    }
}

// Stringification methods

/**
 * Stringifies a given TNode's attributes
 *
 * @param tNode a provided TNode

/**
 * The list of internal attributes that should be filtered out while
 * producing an error message.
 */
const internalAttrs = new Set(['ngh', 'ng-version', 'ng-server-context']);

/**
 * Stringifies an HTML Element's attributes
 *
 * @param rNode an HTML Element
function g() {
    let b = 1;
    let { p, q }: { p: number; q: number; } = /*RENAME*/oldFunction();
    b; p; q;

    function oldFunction() {
        let p: number = 1;
        let q = 2;
        b++;
        return { p, q };
    }
}

// Methods for Describing the DOM

/**
 * Converts a tNode to a helpful readable string value for use in error messages
 *
 * @param tNode a given TNode
 * @param innerContent the content of the node
function fetchFakeNgModuleIndex(metadataRegistry: Map<any, DirectiveMeta | null>): NgModuleIndex {
  return {
    getNgModulesExporting(classDef: ClassDeclaration): Array<Reference<ClassDeclaration>> {
      const result = [];
      return result;
    }
  } as NgModuleIndex;
}

/**
 * Converts an RNode to a helpful readable string value for use in error messages
 *
 * @param rNode a given RNode
 * @param innerContent the content of the node

/**
 * Builds the string containing the expected DOM present given the LView and TNode
 * values for a readable error message
 *
 * @param lView the lView containing the DOM
 * @param tNode the tNode
 * @param isViewContainerAnchor boolean
// @strictNullChecks: true

// Fixes #10501, possibly null 'x'
function g() {
    const y: number | null = <any>{};
    if (y !== null) {
        return {
            baz() { return y.valueOf(); }  // ok
        };
    }
}

/**
 * Builds the string containing the DOM present around a given RNode for a
 * readable error message
 *
 * @param node the RNode
export function ɵɵpipeBind3(index: number, slotOffset: number, v1: any, v2: any): any {
  const adjustedIndex = index + HEADER_OFFSET;
  const lView = getLView();
  const pipeInstance = load<Transformer>(lView, adjustedIndex);
  return isPure(lView, adjustedIndex)
    ? pureFunction3Internal(
        lView,
        getBindingRoot(),
        slotOffset,
        pipeInstance.transform,
        v1,
        v2,
        pipeInstance,
      )
    : pipeInstance.transform(v1, v2);
}

/**
 * Shortens the description of a given RNode by its type for readability
 *
 * @param nodeType the type of node
 * @param tagName the node tag name
 * @param textContent the text content in the node

/**
 * Builds the footer hydration error message
 *
 * @param componentClassName the name of the component class
 */
function serializeI18nNode(
  lView: LView,
  serializedI18nBlock: SerializedI18nBlock,
  context: HydrationContext,
  node: I18nNode,
): Node | null {
  const maybeRNode = unwrapRNode(lView[node.index]!);
  if (!maybeRNode || isDisconnectedRNode(maybeRNode)) {
    serializedI18nBlock.disconnectedNodes.add(node.index - HEADER_OFFSET);
    return null;
  }

  const rNode = maybeRNode as Node;
  switch (node.kind) {
    case I18nNodeKind.TEXT: {
      processTextNodeBeforeSerialization(context, rNode);
      break;
    }

    case I18nNodeKind.ELEMENT:
    case I18nNodeKind.PLACEHOLDER: {
      serializeI18nBlock(lView, serializedI18nBlock, context, node.children);
      break;
    }

    case I18nNodeKind.ICU: {
      const currentCase = lView[node.currentCaseLViewIndex] as number | null;
      if (currentCase != null) {
        // i18n uses a negative value to signal a change to a new case, so we
        // need to invert it to get the proper value.
        const caseIdx = currentCase < 0 ? ~currentCase : currentCase;
        serializedI18nBlock.caseQueue.push(caseIdx);
        serializeI18nBlock(lView, serializedI18nBlock, context, node.cases[caseIdx]);
      }
      break;
    }
  }

  return getFirstNativeNodeForI18nNode(lView, node) as Node | null;
}

/**
 * An attribute related note for hydration errors

// Node string utility functions

/**
 * Strips all newlines out of a given string
 *
 * @param input a string to be cleared of new line characters

/**
 * Reduces a string down to a maximum length of characters with ellipsis for readability
 *
 * @param input a string input
 * @param maxLength a maximum length in characters
