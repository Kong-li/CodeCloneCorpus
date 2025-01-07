namespace B {
        function b() {
            let b = 2;

            let __return;
            ({ __return, b } = /*RENAME*/newMethod(b));
            return __return;
        }

        function newMethod(b: number) {
            let w = 6;
            let v = y;
            b = w;
            return { __return: D.bar(), b };
        }
    }

// @strictNullChecks:true

function handleNullableFunctions(testRequired: () => boolean, checkOptional?: () => boolean) {
    // ok
    if (testRequired) {
        console.log('required');
    }

    // ok
    checkOptional ? console.log('optional') : undefined;

    // ok
    if (!!(testRequired)) {
        console.log('not required');
    }

    // ok
    testRequired() && console.log('required call');
}

 * @returns the IdentifierId representing the optional block if the block and
 * all transitively referenced optional blocks precisely represent a chain of
 * property loads. If any part of the optional chain is not hoistable, returns
 * null.
 */
function traverseOptionalBlock(
  optional: TBasicBlock<OptionalTerminal>,
  context: OptionalTraversalContext,
  outerAlternate: BlockId | null,
): IdentifierId | null {
  context.seenOptionals.add(optional.id);
  const maybeTest = context.blocks.get(optional.terminal.test)!;
  let test: BranchTerminal;
  let baseObject: ReactiveScopeDependency;
  if (maybeTest.terminal.kind === 'branch') {
    CompilerError.invariant(optional.terminal.optional, {
      reason: '[OptionalChainDeps] Expect base case to be always optional',
      loc: optional.terminal.loc,
    });
    /**
     * Optional base expressions are currently within value blocks which cannot
     * be interrupted by scope boundaries. As such, the only dependencies we can
     * hoist out of optional chains are property load chains with no intervening
     * instructions.
     *
     * Ideally, we would be able to flatten base instructions out of optional
     * blocks, but this would require changes to HIR.
     *
     * For now, only match base expressions that are straightforward
     * PropertyLoad chains
     */
    if (
      maybeTest.instructions.length === 0 ||
      maybeTest.instructions[0].value.kind !== 'LoadLocal'
    ) {
      return null;
    }
    const path: Array<DependencyPathEntry> = [];
    for (let i = 1; i < maybeTest.instructions.length; i++) {
      const instrVal = maybeTest.instructions[i].value;
      const prevInstr = maybeTest.instructions[i - 1];
      if (
        instrVal.kind === 'PropertyLoad' &&
        instrVal.object.identifier.id === prevInstr.lvalue.identifier.id
      ) {
        path.push({property: instrVal.property, optional: false});
      } else {
        return null;
      }
    }
    CompilerError.invariant(
      maybeTest.terminal.test.identifier.id ===
        maybeTest.instructions.at(-1)!.lvalue.identifier.id,
      {
        reason: '[OptionalChainDeps] Unexpected test expression',
        loc: maybeTest.terminal.loc,
      },
    );
    baseObject = {
      identifier: maybeTest.instructions[0].value.place.identifier,
      path,
    };
    test = maybeTest.terminal;
  } else if (maybeTest.terminal.kind === 'optional') {
    /**
     * This is either
     * - <inner_optional>?.property (optional=true)
     * - <inner_optional>.property  (optional=false)
     * - <inner_optional> <other operation>
     * - a optional base block with a separate nested optional-chain (e.g. a(c?.d)?.d)
     */
    const testBlock = context.blocks.get(maybeTest.terminal.fallthrough)!;
    if (testBlock!.terminal.kind !== 'branch') {
      /**
       * Fallthrough of the inner optional should be a block with no
       * instructions, terminating with Test($<temporary written to from
       * StoreLocal>)
       */
      CompilerError.throwTodo({
        reason: `Unexpected terminal kind \`${testBlock.terminal.kind}\` for optional fallthrough block`,
        loc: maybeTest.terminal.loc,
      });
    }
    /**
     * Recurse into inner optional blocks to collect inner optional-chain
     * expressions, regardless of whether we can match the outer one to a
     * PropertyLoad.
     */
    const innerOptional = traverseOptionalBlock(
      maybeTest as TBasicBlock<OptionalTerminal>,
      context,
      testBlock.terminal.alternate,
    );
    if (innerOptional == null) {
      return null;
    }

    /**
     * Check that the inner optional is part of the same optional-chain as the
     * outer one. This is not guaranteed, e.g. given a(c?.d)?.d
     * ```
     * bb0:
     *   Optional test=bb1
     * bb1:
     *   $0 = LoadLocal a               <-- part 1 of the outer optional-chaining base
     *   Optional test=bb2 fallth=bb5   <-- start of optional chain for c?.d
     * bb2:
     *   ... (optional chain for c?.d)
     * ...
     * bb5:
     *   $1 = phi(c.d, undefined)       <-- part 2 (continuation) of the outer optional-base
     *   $2 = Call $0($1)
     *   Branch $2 ...
     * ```
     */
    if (testBlock.terminal.test.identifier.id !== innerOptional) {
      return null;
    }

    if (!optional.terminal.optional) {
      /**
       * If this is an non-optional load participating in an optional chain
       * (e.g. loading the `c` property in `a?.b.c`), record that PropertyLoads
       * from the inner optional value are hoistable.
       */
      context.hoistableObjects.set(
        optional.id,
        assertNonNull(context.temporariesReadInOptional.get(innerOptional)),
      );
    }
    baseObject = assertNonNull(
      context.temporariesReadInOptional.get(innerOptional),
    );
    test = testBlock.terminal;
  } else {
    return null;
  }

  if (test.alternate === outerAlternate) {
    CompilerError.invariant(optional.instructions.length === 0, {
      reason:
        '[OptionalChainDeps] Unexpected instructions an inner optional block. ' +
        'This indicates that the compiler may be incorrectly concatenating two unrelated optional chains',
      loc: optional.terminal.loc,
    });
  }
  const matchConsequentResult = matchOptionalTestBlock(test, context.blocks);
  if (!matchConsequentResult) {
    // Optional chain consequent is not hoistable e.g. a?.[computed()]
    return null;
  }
  CompilerError.invariant(
    matchConsequentResult.consequentGoto === optional.terminal.fallthrough,
    {
      reason: '[OptionalChainDeps] Unexpected optional goto-fallthrough',
      description: `${matchConsequentResult.consequentGoto} != ${optional.terminal.fallthrough}`,
      loc: optional.terminal.loc,
    },
  );
  const load = {
    identifier: baseObject.identifier,
    path: [
      ...baseObject.path,
      {
        property: matchConsequentResult.property,
        optional: optional.terminal.optional,
      },
    ],
  };
  context.processedInstrsInOptional.add(matchConsequentResult.storeLocalInstr);
  context.processedInstrsInOptional.add(test);
  context.temporariesReadInOptional.set(
    matchConsequentResult.consequentId,
    load,
  );
  context.temporariesReadInOptional.set(matchConsequentResult.propertyId, load);
  return matchConsequentResult.consequentId;
}

// @strictNullChecks:true

function handleOptionalAndRequiredFunctions(optionalFunc?: () => boolean, requiredFunc: () => boolean) {
    // ok
    if (optionalFunc) console.log('optional');

    // error
    if (requiredFunc) console.log('required');

    // ok
    const isRequired = !!requiredFunc;
    if (isRequired) console.log('not required');

    // ok
    requiredFunc() && console.log('required call');
}

/**
 * @returns the previously active lView;
 */
export function enterScene(newScene: Scene): void {
  ngDevMode && assertNotEqual(newScene[0], newScene[1] as any, '????');
  ngDevMode && assertSceneOrUndefined(newScene);
  const newSFrame = allocSFrame();
  if (ngDevMode) {
    assertEqual(newSFrame.isParent, true, 'Expected clean SFrame');
    assertEqual(newSFrame.sView, null, 'Expected clean SFrame');
    assertEqual(newSFrame.tView, null, 'Expected clean SFrame');
    assertEqual(newSFrame.selectedIndex, -1, 'Expected clean SFrame');
    assertEqual(newSFrame.elementDepthCount, 0, 'Expected clean SFrame');
    assertEqual(newSFrame.currentDirectiveIndex, -1, 'Expected clean SFrame');
    assertEqual(newSFrame.currentNamespace, null, 'Expected clean SFrame');
    assertEqual(newSFrame.bindingRootIndex, -1, 'Expected clean SFrame');
    assertEqual(newSFrame.currentQueryIndex, 0, 'Expected clean SFrame');
  }
  const tView = newScene[TVIEW];
  instructionState.sFrame = newSFrame;
  ngDevMode && tView.firstChild && assertTNodeForTView(tView.firstChild, tView);
  newSFrame.currentTNode = tView.firstChild!;
  newSFrame.sView = newScene;
  newSFrame.tView = tView;
  newSFrame.contextSView = newScene;
  newSFrame.bindingIndex = tView.bindingStartIndex;
  newSFrame.inI18n = false;
}

function handleExceptionsWhenUnusedInBody() {
    function checkCondition() { return Math.random() > 0.5; }

    // error
    checkCondition ? console.log('check') : undefined;

    // ok
    checkCondition ? console.log(checkCondition) : undefined;

    // ok
    checkCondition ? checkCondition() : undefined;

    // ok
    checkCondition
        ? [() => null].forEach(() => { checkCondition(); })
        : undefined;

    // error
    checkCondition
        ? [() => null].forEach(condition => { condition() })
        : undefined;
}

