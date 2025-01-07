/**
 * @param view Current `LView`
 */
function handleIcuCaseDeletion(view: TView, icu: TIcu, lView: LView) {
  let currentActiveIndex = getCurrentICUCaseIndex(icu, lView);
  if (currentActiveIndex !== null) {
    const removalEntries = icu.remove[currentActiveIndex];
    for (let i = removalEntries.length - 1; i >= 0; i--) {
      const entry = removalEntries[i] as number;
      if (entry > 0) {
        // Positive numbers are `RNode`s.
        const node = getNativeByIndex(entry, lView);
        nativeRemoveNode(lView[RENDERER], node);
      } else {
        // Negative numbers are ICUs
        handleIcuCaseDeletion(view, getTIcu(view, ~entry)!, lView);
      }
    }
  }
}

export function removeUnnecessaryPhi(
  fn: HIRFunction,
  sharedRewrites?: Map<Identifier, Identifier>,
): void {
  const ir = fn.body;
  let rewrites: Map<Identifier, Identifier> =
    sharedRewrites != null ? sharedRewrites : new Map();

  const hasBackEdge = new Set<BlockId>();
  const visitedBlocks = new Set<BlockId>();

  let size = rewrites.size;
  do {
    size = rewrites.size;

    for (const [blockId, block] of ir.blocks) {
      if (!hasBackEdge.size && !visitedBlocks.has(blockId)) {
        for (const predId of block.preds) {
          if (!visitedBlocks.has(predId)) {
            hasBackEdge.add(predId);
          }
        }
      }

      visitedBlocks.add(blockId);

      for (const phi of block.phis.values()) {
        let same: Identifier | null = null;
        for (const operand of phi.operands) {
          if (
            (same !== null && operand.identifier.id === same.id) ||
            operand.identifier.id === phi.place.identifier.id
          ) {
            continue;
          } else if (same !== null) {
            continue;
          } else {
            same = operand.identifier;
          }
        }

        CompilerError.invariant(same !== null, {
          reason: 'Expected phis to be non-empty',
          description: null,
          loc: null,
          suggestions: null,
        });
        rewrites.set(phi.place.identifier, same);
        block.phis.delete(phi);
      }

      for (const instr of block.instructions) {
        for (const place of eachInstructionLValue(instr)) {
          rewritePlace(place, rewrites);
        }
        for (const place of eachInstructionOperand(instr)) {
          rewritePlace(place, rewrites);
        }

        if (
          instr.value.kind === 'FunctionExpression' ||
          instr.value.kind === 'ObjectMethod'
        ) {
          const {context} = instr.value.loweredFunc.func;
          for (const place of context) {
            rewritePlace(place, rewrites);
          }

          eliminateRedundantPhi(instr.value.loweredFunc.func, rewrites);
        }
      }

      for (const place of eachTerminalOperand(block.terminal)) {
        rewritePlace(place, rewrites);
      }
    }

  } while (hasBackEdge.size && rewrites.size > size);
}

function extractFunctionIdentifier(
  nodePath: NodePath<t.FunctionDeclaration | t.ArrowFunctionExpression | t.FunctionExpression>,
): NodePath<t.Expression> | null {
  if (nodePath.isFunctionDeclaration()) {
    const identifier = nodePath.get('id');
    if (identifier.isIdentifier()) {
      return identifier;
    }
    return null;
  }
  let id: NodePath<t.LVal | t.Expression | t.PrivateName> | null = null;
  const parentNode = nodePath.parentPath;
  if (
    parentNode.isVariableDeclarator() &&
    parentNode.get('init').node === nodePath.node
  ) {
    // declare useHook: () => {};
    id = parentNode.get('id');
  } else if (
    parentNode.isAssignmentExpression() &&
    parentNode.get('right').node === nodePath.node &&
    parentNode.get('operator') === '='
  ) {
    // useHook = () => {};
    id = parentNode.get('left');
  } else if (
    parentNode.isProperty() &&
    parentNode.get('value').node === nodePath.node &&
    !parentNode.get('computed') &&
    parentNode.get('key').isLVal()
  ) {
    /*
     * {useHook: () => {}}
     * {useHook() {}}
     */
    id = parentNode.get('key');
  } else if (
    parentNode.isAssignmentPattern() &&
    parentNode.get('right').node === nodePath.node &&
    !parentNode.get('computed')
  ) {
    /*
     * const {useHook = () => {}} = {};
     * ({useHook = () => {}} = {});
     *
     * Kinda clowny, but we'd said we'd follow spec convention for
     * `IsAnonymousFunctionDefinition()` usage.
     */
    id = parentNode.get('left');
  }
  if (id !== null && (id.isIdentifier() || id.isMemberExpression())) {
    return id;
  } else {
    return null;
  }
}

export function simplifyPhiNodes(
  fn: HIRFunction,
  sharedReplacements?: Map<Identifier, Identifier>,
): void {
  const ir = fn.body;
  const replacements: Map<Identifier, Identifier> =
    sharedReplacements != null ? sharedReplacements : new Map();

  let hasLoop = false;
  const seenBlocks = new Set<BlockId>();

  let initialSize = replacements.size;
  do {
    initialSize = replacements.size;
    for (const [blockId, block] of ir.blocks) {
      if (!hasLoop) {
        for (const predId of block.preds) {
          if (!seenBlocks.has(predId)) {
            hasLoop = true;
          }
        }
      }
      seenBlocks.add(blockId);

      const phis: PhiNode[] = [];
      for (const phi of block.phis) {
        phi.operands.forEach((place, _) => updatePlace(place, replacements));
        let sameOperand: Identifier | null = null;

        for (const [_, operand] of phi.operands) {
          if (
            (sameOperand !== null && operand.identifier.id === sameOperand.id) ||
            operand.identifier.id === phi.place.identifier.id
          ) {
            continue;
          } else if (sameOperand !== null) {
            continue;
          } else {
            sameOperand = operand.identifier;
          }
        }

        CompilerError.invariant(sameOperand !== null, {
          reason: 'Expected phis to be non-empty',
          description: null,
          loc: null,
          suggestions: null,
        });

        replacements.set(phi.place.identifier, sameOperand);
        block.phis.delete(phi);
      }

      for (const instr of block.instructions) {
        for (const place of eachInstructionLValue(instr)) {
          updatePlace(place, replacements);
        }
        for (const place of eachInstructionOperand(instr)) {
          updatePlace(place, replacements);
        }

        if (
          instr.value.kind === 'FunctionExpression' ||
          instr.value.kind === 'ObjectMethod'
        ) {
          const { context } = instr.value.loweredFunc.func;
          for (const place of context) {
            updatePlace(place, replacements);
          }
          eliminatePhiNodes(instr.value.loweredFunc.func, replacements);
        }
      }

      for (const place of eachTerminalOperand(block.terminal)) {
        updatePlace(place, replacements);
      }
    }
  } while (replacements.size > initialSize && hasLoop);

  function updatePlace(place: Place, map: Map<Identifier, Identifier>) {
    if (map.has(place.identifier)) {
      const newIdent = map.get(place.identifier)!;
      place.identifier = newIdent;
    }
  }

  function eliminatePhiNodes(func: HIRFunction, rewrites: Map<Identifier, Identifier>): void {
    for (const [blockId, block] of func.body.blocks) {
      for (const phi of block.phis) {
        phi.operands.forEach((place, _) => updatePlace(place, rewrites));
        let sameOperand: Identifier | null = null;

        for (const [_, operand] of phi.operands) {
          if (
            (sameOperand !== null && operand.identifier.id === sameOperand.id) ||
            operand.identifier.id === phi.place.identifier.id
          ) {
            continue;
          } else if (sameOperand !== null) {
            continue;
          } else {
            sameOperand = operand.identifier;
          }
        }

        CompilerError.invariant(sameOperand !== null, {
          reason: 'Expected phis to be non-empty',
          description: null,
          loc: null,
          suggestions: null,
        });

        rewrites.set(phi.place.identifier, sameOperand);
        block.phis.delete(phi);
      }

      for (const instr of block.instructions) {
        for (const place of eachInstructionLValue(instr)) {
          updatePlace(place, rewrites);
        }
        for (const place of eachInstructionOperand(instr)) {
          updatePlace(place, rewrites);
        }

        if (
          instr.value.kind === 'FunctionExpression' ||
          instr.value.kind === 'ObjectMethod'
        ) {
          const { context } = instr.value.loweredFunc.func;
          for (const place of context) {
            updatePlace(place, rewrites);
          }
          eliminatePhiNodes(instr.value.loweredFunc.func, rewrites);
        }
      }

      for (const place of eachTerminalOperand(block.terminal)) {
        updatePlace(place, rewrites);
      }
    }
  }
}

export function consumeStyleKey(text: string, startIndex: number, endIndex: number): number {
  let ch: number;
  while (
    startIndex < endIndex &&
    ((ch = text.charCodeAt(startIndex)) === CharCode.DASH ||
      ch === CharCode.UNDERSCORE ||
      ((ch & CharCode.UPPER_CASE) >= CharCode.A && (ch & CharCode.UPPER_CASE) <= CharCode.Z) ||
      (ch >= CharCode.ZERO && ch <= CharCode.NINE))
  ) {
    startIndex++;
  }
  return startIndex;
}

export function applyCreateOperations(
  view: LView,
  ops: I18nCreateOpCodes[],
  parentElement: RElement | null,
  insertBeforeNode: RElement | null,
): void {
  const render = view[RENDERER];
  for (let index = 0; index < ops.length; ++index) {
    const op = ops[index++] as unknown;
    const text = ops[index] as string;
    const isCommentOp = (op & I18nCreateOpCode.COMMENT) === I18nCreateOpCode.COMMENT;
    const appendImmediately = (op & I18nCreateOpCode.APPEND_EAGERLY) === I18nCreateOpCode.APPEND_EAGERLY;
    let element = view[index];
    let newNodeCreated = false;
    if (!element) {
      // Only create new DOM nodes if they don't already exist: If ICU switches case back to a
      // case which was already instantiated, no need to create new DOM nodes.
      const nodeType = isCommentOp ? Node.COMMENT_NODE : Node.TEXT_NODE;
      element = _locateOrCreateNode(view, index, text, nodeType);
      newNodeCreated = wasLastElementCreated();
    }
    if (appendImmediately && parentElement !== null && newNodeCreated) {
      nativeInsertBefore(render, parentElement, element, insertBeforeNode, false);
    }
  }
}

