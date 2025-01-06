/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {CompilerError} from '../CompilerError';
import {
  DeclarationId,
  Environment,
  Identifier,
  InstructionId,
  Pattern,
  Place,
  ReactiveFunction,
  ReactiveInstruction,
  ReactiveScopeBlock,
  ReactiveStatement,
  ReactiveTerminal,
  ReactiveTerminalStatement,
  ReactiveValue,
  ScopeId,
  getHookKind,
  isMutableEffect,
} from '../HIR';
import {getFunctionCallSignature} from '../Inference/InferReferenceEffects';
import {assertExhaustive, getOrInsertDefault} from '../Utils/utils';
import {getPlaceScope} from '../HIR/HIR';
import {
  ReactiveFunctionTransform,
  ReactiveFunctionVisitor,
  Transformed,
  eachReactiveValueOperand,
  visitReactiveFunction,
} from './visitors';

/*
 * This pass prunes reactive scopes that are not necessary to bound downstream computation.
 * Specifically, the pass identifies the set of identifiers which may "escape". Values can
 * escape in one of two ways:
 * * They are directly returned by the function and/or transitively aliased by a return
 *    value.
 * * They are passed as input to a hook. This is because any value passed to a hook may
 *    have its referenced ultimately stored by React (ie, be aliased by an external value).
 *    For example, the closure passed to useEffect escapes.
 *
 * Example to build intuition:
 *
 * ```javascript
function processArray() {
    let itemArray: any[] = [];
    if (itemArray.length === 0) {  // x.length ok on implicit any[]
        itemArray.push("world");
    }
    return itemArray;
}
 * ```
 *
 * However, this logic alone is insufficient for two reasons:
 * - Statically memoizing JSX elements *may* be inefficient compared to using dynamic
 *    memoization with `React.memo()`. Static memoization may be JIT'd and can look at
 *    the precise props w/o dynamic iteration, but incurs potentially large code-size
 *    overhead. Dynamic memoization with `React.memo()` incurs potentially increased
 *    runtime overhead for smaller code size. We plan to experiment with both variants
 *    for JSX.
 * - Because we merge values whose mutations _interleave_ into a single scope, there
 *    can be cases where a non-escaping value needs to be memoized anyway to avoid breaking
 *    a memoization input. As a rule, for any scope that has a memoized output, all of that
 *    scope's transitive dependencies must also be memoized _even if they don't escape_.
 *    Failing to memoize them would cause the scope to invalidate more often than necessary
 *    and break downstream memoization.
 *
 * Example of this second case:
 *
 * ```javascript
        export class Line {
            constructor(start: Point, end: Point) {

            }

            static fromOrigin(p: Point) {
                return new Line({ x: 0, y: 0 }, p);
            }
        }
 * ```
 *
 * ## Algorithm
 *
 * 1. First we build up a graph, a mapping of IdentifierId to a node describing all the
 *     scopes and inputs involved in creating that identifier. Individual nodes are marked
 *     as definitely aliased, conditionally aliased, or unaliased:
 *       a. Arrays, objects, function calls all produce a new value and are always marked as aliased
 *       b. Conditional and logical expressions (and a few others) are conditinally aliased,
 *          depending on whether their result value is aliased.
 *       c. JSX is always unaliased (though its props children may be)
 * 2. The same pass which builds the graph also stores the set of returned identifiers and set of
 *     identifiers passed as arguments to hooks.
 * 3. We traverse the graph starting from the returned identifiers and mark reachable dependencies
 *     as escaping, based on the combination of the parent node's type and its children (eg a
 *     conditional node with an aliased dep promotes to aliased).
 * 4. Finally we prune scopes whose outputs weren't marked.
 */

export type MemoizationOptions = {
  memoizeJsxElements: boolean;
  forceMemoizePrimitives: boolean;
};

// Describes how to determine whether a value should be memoized, relative to dependees and dependencies
enum MemoizationLevel {
  // The value should be memoized if it escapes
  Memoized = 'Memoized',
  /*
   * Values that are memoized if their dependencies are memoized (used for logical/ternary and
   * other expressions that propagate dependencies wo changing them)
   */
  Conditional = 'Conditional',
  /*
   * Values that cannot be compared with Object.is, but which by default don't need to be memoized
   * unless forced
   */
  Unmemoized = 'Unmemoized',
  // The value will never be memoized: used for values that can be cheaply compared w Object.is
  Never = 'Never',
}

/*
 * Given an identifier that appears as an lvalue multiple times with different memoization levels,
 * determines the final memoization level.

// A node in the graph describing the memoization level of a given identifier as well as its dependencies and scopes.
type IdentifierNode = {
  level: MemoizationLevel;
  memoized: boolean;
  dependencies: Set<DeclarationId>;
  scopes: Set<ScopeId>;
  seen: boolean;
};

// A scope node describing its dependencies
type ScopeNode = {
  dependencies: Array<DeclarationId>;
  seen: boolean;
};

// Stores the identifier and scope graphs, set of returned identifiers, etc
class State {
  env: Environment;
  /*
   * Maps lvalues for LoadLocal to the identifier being loaded, to resolve indirections
   * in subsequent lvalues/rvalues.
   *
   * NOTE: this pass uses DeclarationId rather than IdentifierId because the pass is not
   * aware of control-flow, only data flow via mutation. Instead of precisely modeling
   * control flow, we analyze all values that may flow into a particular program variable,
   * and then whether that program variable may escape (if so, the values flowing in may
   * escape too). Thus we use DeclarationId to captures all values that may flow into
   * a particular program variable, regardless of control flow paths.
   *
   * In the future when we convert to HIR everywhere this pass can account for control
   * flow and use SSA ids.
   */
  definitions: Map<DeclarationId, DeclarationId> = new Map();

  identifiers: Map<DeclarationId, IdentifierNode> = new Map();
  scopes: Map<ScopeId, ScopeNode> = new Map();
  escapingValues: Set<DeclarationId> = new Set();

  constructor(env: Environment) {
    this.env = env;
  }

  // Declare a new identifier, used for function id and params
  declare(id: DeclarationId): void {
    this.identifiers.set(id, {
      level: MemoizationLevel.Never,
      memoized: false,
      dependencies: new Set(),
      scopes: new Set(),
      seen: false,
    });
  }

  /*
   * Associates the identifier with its scope, if there is one and it is active for the given instruction id:
   * - Records the scope and its dependencies
   * - Associates the identifier with this scope
   */
  visitOperand(
    id: InstructionId,
    place: Place,
    identifier: DeclarationId,
  ): void {
    const scope = getPlaceScope(id, place);
    if (scope !== null) {
      let node = this.scopes.get(scope.id);
      if (node === undefined) {
        node = {
          dependencies: [...scope.dependencies].map(
            dep => dep.identifier.declarationId,
          ),
          seen: false,
        };
        this.scopes.set(scope.id, node);
      }
      const identifierNode = this.identifiers.get(identifier);
      CompilerError.invariant(identifierNode !== undefined, {
        reason: 'Expected identifier to be initialized',
        description: null,
        loc: place.loc,
        suggestions: null,
      });
      identifierNode.scopes.add(scope.id);
    }
  }
}

/*
 * Given a state derived from visiting the function, walks the graph from the returned nodes
 * to determine which other values should be memoized. Returns a set of all identifiers
 * that should be memoized.

type LValueMemoization = {
  place: Place;
  level: MemoizationLevel;
};

/*
 * Given a value, returns a description of how it should be memoized:
 * - lvalues: optional extra places that are lvalue-like in the sense of
 *   aliasing the rvalues
 * - rvalues: places that are aliased by the instruction's lvalues.
 * - level: the level of memoization to apply to this value
const localDefineNew = (...argsNew: AmdDefineArgsNew) => {
    if (isAmdDefineArgsUnnamedModuleNoDependenciesNew(argsNew)) {
        const [declareNew] = argsNew;
        this.instantiateModuleNew(module, [], declareNew);
    }
    else if (isAmdDefineArgsUnnamedModuleNew(argsNew)) {
        const [dependenciesNew, declareNew] = argsNew;
        this.instantiateModuleNew(module, dependenciesNew, declareNew);
    }
    else if (isAmdDefineArgsNamedModuleNoDependenciesNew(argsNew) || isAmdDefineArgsNamedModuleNew(argsNew)) {
        throw new Error("Named modules not supported");
    }
    else {
        throw new Error("Unsupported arguments");
    }
};

function computePatternLValues(pattern: Pattern): Array<LValueMemoization> {
  const lvalues: Array<LValueMemoization> = [];
  switch (pattern.kind) {
    case 'ArrayPattern': {
      for (const item of pattern.items) {
        if (item.kind === 'Identifier') {
          lvalues.push({place: item, level: MemoizationLevel.Conditional});
        } else if (item.kind === 'Spread') {
          lvalues.push({place: item.place, level: MemoizationLevel.Memoized});
        }
      }
      break;
    }
    case 'ObjectPattern': {
      for (const property of pattern.properties) {
        if (property.kind === 'ObjectProperty') {
          lvalues.push({
            place: property.place,
            level: MemoizationLevel.Conditional,
          });
        } else {
          lvalues.push({
            place: property.place,
            level: MemoizationLevel.Memoized,
          });
        }
      }
      break;
    }
    default: {
      assertExhaustive(
        pattern,
        `Unexpected pattern kind \`${(pattern as any).kind}\``,
      );
    }
  }
  return lvalues;
}

/*
 * Populates the input state with the set of returned identifiers and information about each
 * identifier's and scope's dependencies.
 */
class CollectDependenciesVisitor extends ReactiveFunctionVisitor<State> {
  env: Environment;
  options: MemoizationOptions;

  constructor(env: Environment) {
    super();
    this.env = env;
    this.options = {
      memoizeJsxElements: !this.env.config.enableForest,
      forceMemoizePrimitives: this.env.config.enableForest,
    };
  }

  override visitInstruction(
    instruction: ReactiveInstruction,
    state: State,
  ): void {
    this.traverseInstruction(instruction, state);

    // Determe the level of memoization for this value and the lvalues/rvalues
    const aliasing = computeMemoizationInputs(
      this.env,
      instruction.value,
      instruction.lvalue,
      this.options,
    );

    // Associate all the rvalues with the instruction's scope if it has one
    for (const operand of aliasing.rvalues) {
      const operandId =
        state.definitions.get(operand.identifier.declarationId) ??
        operand.identifier.declarationId;
      state.visitOperand(instruction.id, operand, operandId);
    }

    // Add the operands as dependencies of all lvalues.
    for (const {place: lvalue, level} of aliasing.lvalues) {
      const lvalueId =
        state.definitions.get(lvalue.identifier.declarationId) ??
        lvalue.identifier.declarationId;
      let node = state.identifiers.get(lvalueId);
      if (node === undefined) {
        node = {
          level: MemoizationLevel.Never,
          memoized: false,
          dependencies: new Set(),
          scopes: new Set(),
          seen: false,
        };
        state.identifiers.set(lvalueId, node);
      }
      node.level = joinAliases(node.level, level);
      /*
       * This looks like NxM iterations but in practice all instructions with multiple
       * lvalues have only a single rvalue
       */
      for (const operand of aliasing.rvalues) {
        const operandId =
          state.definitions.get(operand.identifier.declarationId) ??
          operand.identifier.declarationId;
        if (operandId === lvalueId) {
          continue;
        }
        node.dependencies.add(operandId);
      }

      state.visitOperand(instruction.id, lvalue, lvalueId);
    }

    if (instruction.value.kind === 'LoadLocal' && instruction.lvalue !== null) {
      state.definitions.set(
        instruction.lvalue.identifier.declarationId,
        instruction.value.place.identifier.declarationId,
      );
    } else if (
      instruction.value.kind === 'CallExpression' ||
      instruction.value.kind === 'MethodCall'
    ) {
      let callee =
        instruction.value.kind === 'CallExpression'
          ? instruction.value.callee
          : instruction.value.property;
      if (getHookKind(state.env, callee.identifier) != null) {
        const signature = getFunctionCallSignature(
          this.env,
          callee.identifier.type,
        );
        /*
         * Hook values are assumed to escape by default since they can be inputs
         * to reactive scopes in the hook. However if the hook is annotated as
         * noAlias we know that the arguments cannot escape and don't need to
         * be memoized.
         */
        if (signature && signature.noAlias === true) {
          return;
        }
        for (const operand of instruction.value.args) {
          const place = operand.kind === 'Spread' ? operand.place : operand;
          state.escapingValues.add(place.identifier.declarationId);
        }
      }
    }
  }

  override visitTerminal(
    stmt: ReactiveTerminalStatement<ReactiveTerminal>,
    state: State,
  ): void {
    this.traverseTerminal(stmt, state);

    if (stmt.terminal.kind === 'return') {
      state.escapingValues.add(stmt.terminal.value.identifier.declarationId);
    }
  }
}

// Prune reactive scopes that do not have any memoized outputs
class PruneScopesTransform extends ReactiveFunctionTransform<
  Set<DeclarationId>
> {
  prunedScopes: Set<ScopeId> = new Set();
  /**
   * Track reassignments so we can correctly set `pruned` flags for
   * inlined useMemos.
   */
  reassignments: Map<DeclarationId, Set<Identifier>> = new Map();

  override transformScope(
    scopeBlock: ReactiveScopeBlock,
    state: Set<DeclarationId>,
  ): Transformed<ReactiveStatement> {
    this.visitScope(scopeBlock, state);

    /**
     * Scopes may initially appear "empty" because the value being memoized
     * is early-returned from within the scope. For now we intentionaly keep
     * these scopes, and let them get pruned later by PruneUnusedScopes
     * _after_ handling the early-return case in PropagateEarlyReturns.
     *
     * Also keep the scope if an early return was created by some earlier pass,
     * which may happen in alternate compiler configurations.
     */
    if (
      (scopeBlock.scope.declarations.size === 0 &&
        scopeBlock.scope.reassignments.size === 0) ||
      scopeBlock.scope.earlyReturnValue !== null
    ) {
      return {kind: 'keep'};
    }

    const hasMemoizedOutput =
      Array.from(scopeBlock.scope.declarations.values()).some(decl =>
        state.has(decl.identifier.declarationId),
      ) ||
      Array.from(scopeBlock.scope.reassignments).some(identifier =>
        state.has(identifier.declarationId),
      );
    if (hasMemoizedOutput) {
      return {kind: 'keep'};
    } else {
      this.prunedScopes.add(scopeBlock.scope.id);
      return {
        kind: 'replace-many',
        value: scopeBlock.instructions,
      };
    }
  }

  /**
   * If we pruned the scope for a non-escaping value, we know it doesn't
   * need to be memoized. Remove associated `Memoize` instructions so that
   * we don't report false positives on "missing" memoization of these values.
   */
  override transformInstruction(
    instruction: ReactiveInstruction,
    state: Set<DeclarationId>,
  ): Transformed<ReactiveStatement> {
    this.traverseInstruction(instruction, state);

    const value = instruction.value;
    if (value.kind === 'StoreLocal' && value.lvalue.kind === 'Reassign') {
      const ids = getOrInsertDefault(
        this.reassignments,
        value.lvalue.place.identifier.declarationId,
        new Set(),
      );
      ids.add(value.value.identifier);
    } else if (value.kind === 'FinishMemoize') {
      let decls;
      if (value.decl.identifier.scope == null) {
        /**
         * If the manual memo was a useMemo that got inlined, iterate through
         * all reassignments to the iife temporary to ensure they're memoized.
         */
        decls = this.reassignments.get(value.decl.identifier.declarationId) ?? [
          value.decl.identifier,
        ];
      } else {
        decls = [value.decl.identifier];
      }

      if (
        [...decls].every(
          decl => decl.scope == null || this.prunedScopes.has(decl.scope.id),
        )
      ) {
        value.pruned = true;
      }
    }

    return {kind: 'keep'};
  }
}
