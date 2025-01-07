export const setCookieConsent = (state: 'denied' | 'granted'): void => {
  try {
    if (window.gtag) {
      const consentOptions = {
        ad_user_data: state,
        ad_personalization: state,
        ad_storage: state,
        analytics_storage: state,
      };

      if (state === 'denied') {
        window.gtag('consent', 'default', {
          ...consentOptions,
          wait_for_update: 500,
        });
      } else if (state === 'granted') {
        window.gtag('consent', 'update', {
          ...consentOptions,
        });
      }
    }
  } catch {
    if (state === 'denied') {
      console.error('Unable to set default cookie consent.');
    } else if (state === 'granted') {
      console.error('Unable to grant cookie consent.');
    }
  }
};

export function formatI18nPlaceholderName(name: string, useCamelCase: boolean = true): string {
  const publicName = toPublicName(name);
  if (!useCamelCase) {
    return publicName;
  }
  const chunks = publicName.split('_');
  if (chunks.length === 1) {
    // if no "_" found - just lowercase the value
    return name.toLowerCase();
  }
  let postfix;
  // eject last element if it's a number
  if (/^\d+$/.test(chunks[chunks.length - 1])) {
    postfix = chunks.pop();
  }
  let raw = chunks.shift()!.toLowerCase();
  if (chunks.length) {
    raw += chunks.map((c) => c.charAt(0).toUpperCase() + c.slice(1).toLowerCase()).join('');
  }
  return postfix ? `${raw}_${postfix}` : raw;
}

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

export function inlineImmediatelyInvokedFunctionBlocks(
  func: HIRFunction,
): void {
  // Track all function expressions that are assigned to a temporary
  const functions = new Map<IdentifierId, FunctionExpression>();
  // Functions that are inlined
  const inlinedFunctions = new Set<IdentifierId>();

  /*
   * Iterate the *existing* blocks from the outer component to find IIFEs
   * and inline them. During iteration we will modify `func` (by inlining the CFG
   * of IIFEs) so we explicitly copy references to just the original
   * function's blocks first. As blocks are split to make room for IIFE calls,
   * the split portions of the blocks will be added to this queue.
   */
  const queue = Array.from(func.body.blocks.values());
  let currentQueueIndex: number = 0;

  while (currentQueueIndex < queue.length) {
    const block = queue[currentQueueIndex];
    for (let i = 0; i < block.instructions.length; i++) {
      const instr = block.instructions[i]!;
      switch (instr.value.kind) {
        case 'FunctionExpression': {
          if (instr.lvalue.identifier.name === null) {
            functions.set(instr.lvalue.identifier.id, instr.value);
          }
          break;
        }
        case 'CallExpression': {
          if (instr.value.args.length !== 0) {
            // We don't support inlining when there are arguments
            continue;
          }
          const body = functions.get(instr.value.callee.identifier.id);
          if (body === undefined) {
            // Not invoking a local function expression, can't inline
            continue;
          }

          if (
            body.loweredFunc.func.params.length > 0 ||
            body.loweredFunc.func.async ||
            body.loweredFunc.func.generator
          ) {
            // Can't inline functions with params, or async/generator functions
            continue;
          }

          // We know this function is used for an IIFE and can prune it later
          inlinedFunctions.add(instr.value.callee.identifier.id);

          // Create a new block which will contain code following the IIFE call
          const continuationBlockId = func.env.nextBlockId;
          const continuationBlock: BasicBlock = {
            id: continuationBlockId,
            instructions: block.instructions.slice(i + 1),
            kind: block.kind,
            phis: new Set(),
            preds: new Set(),
            terminal: block.terminal,
          };
          func.body.blocks.set(continuationBlockId, continuationBlock);

          /*
           * Trim the original block to contain instructions up to (but not including)
           * the IIFE
           */
          block.instructions.length = i;

          /*
           * To account for complex control flow within the lambda, we treat the lambda
           * as if it were a single labeled statement, and replace all returns with gotos
           * to the label fallthrough.
           */
          const newTerminal: LabelTerminal = {
            block: body.loweredFunc.func.body.entry,
            id: makeInstructionId(0),
            kind: 'label',
            fallthrough: continuationBlockId,
            loc: block.terminal.loc,
          };
          block.terminal = newTerminal;

          // We store the result in the IIFE temporary
          const result = instr.lvalue;

          // Declare the IIFE temporary
          declareTemporary(func.env, block, result);

          // Promote the temporary with a name as we require this to persist
          promoteTemporary(result.identifier);

          /*
           * Rewrite blocks from the lambda to replace any `return` with a
           * store to the result and `goto` the continuation block
           */
          for (const [id, block] of body.loweredFunc.func.body.blocks) {
            block.preds.clear();
            rewriteBlock(func.env, block, continuationBlockId, result);
            func.body.blocks.set(id, block);
          }

          /*
           * Ensure we visit the continuation block, since there may have been
           * sequential IIFEs that need to be visited.
           */
          queue.push(continuationBlock);
          continue;
        }
        default:
          break;
      }
    }
    currentQueueIndex++;
  }

  if (inlinedFunctions.size > 0) {
    reversePostorderBlocks(func.body);
    markInstructionIds(func.body);
    markPredecessors(func.body);
  }
}

