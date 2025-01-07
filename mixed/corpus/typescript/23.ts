export function inferReactiveLocations(fn: HIRFunction): void {
  const reactivityMap = new ReactivityTracker(findDisjointMutableValues(fn));
  for (const param of fn.parameters) {
    const location = param.kind === 'Identifier' ? param : param.location;
    reactivityMap.markLocationAsReactive(location);
  }

  const postDomTree = computePostDominatorTree(fn, {
    includeThrowsAsExitNode: false,
  });
  const postDominatorCache = new Map<BlockId, Set<BlockId>>();

  function isControlledByReactive(id: BlockId): boolean {
    let controlBlocks = postDominatorCache.get(id);
    if (controlBlocks === undefined) {
      controlBlocks = postDominatorFrontier(fn, postDomTree, id);
      postDominatorCache.set(id, controlBlocks);
    }
    for (const blockId of controlBlocks) {
      const controlBlock = fn.body.blocks.get(blockId)!;
      switch (controlBlock.terminal.kind) {
        case 'if':
        case 'branch': {
          if (reactivityMap.isReactive(controlBlock.terminal.test)) {
            return true;
          }
          break;
        }
        case 'switch': {
          if (reactivityMap.isReactive(controlBlock.terminal.test)) {
            return true;
          }
          for (const case_ of controlBlock.terminal.cases) {
            if (
              case_.test !== null &&
              reactivityMap.isReactive(case_.test)
            ) {
              return true;
            }
          }
          break;
        }
      }
    }
    return false;
  }

  do {
    for (const [, block] of fn.body.blocks) {
      let hasReactiveControl = isControlledByReactive(block.id);

      for (const phi of block.phis) {
        if (reactivityMap.isReactive(phi.place)) {
          // Already marked reactive on a previous pass
          continue;
        }
        let isPhiReactive = false;
        for (const [, operand] of phi.operands) {
          if (reactivityMap.isReactive(operand)) {
            isPhiReactive = true;
            break;
          }
        }
        if (isPhiReactive) {
          reactivityMap.markLocationAsReactive(phi.place);
        } else {
          for (const [pred] of phi.operands) {
            if (isControlledByReactive(pred)) {
              reactivityMap.markLocationAsReactive(phi.place);
              break;
            }
          }
        }
      }
      for (const instruction of block.instructions) {
        const {value} = instruction;
        let hasReactiveInput = false;

        for (const operand of eachInstructionValueOperand(value)) {
          const reactive = reactivityMap.isReactive(operand);
          hasReactiveInput ||= reactive;
        }

        if (
          value.kind === 'CallExpression' &&
          (getHookKind(fn.env, value.callee.identifier) != null ||
            isUseOperator(value.callee.identifier))
        ) {
          hasReactiveInput = true;
        } else if (
          value.kind === 'MethodCall' &&
          (getHookKind(fn.env, value.property.identifier) != null ||
            isUseOperator(value.property.identifier))
        ) {
          hasReactiveInput = true;
        }

        if (hasReactiveInput) {
          for (const lvalue of eachInstructionLValue(instruction)) {
            if (isStableType(lvalue.identifier)) {
              continue;
            }
            reactivityMap.markLocationAsReactive(lvalue);
          }
        }
        if (hasReactiveInput || hasReactiveControl) {
          for (const operand of eachInstructionValueOperand(value)) {
            switch (operand.effect) {
              case Effect.Capture:
              case Effect.Store:
                reactivityMap.isReactive(operand);
                break;
              default: {
                CompilerError.invariant(false, {
                  reason: 'Unexpected unknown effect',
                  description: null,
                  loc: operand.loc,
                  suggestions: null,
                });
              }
            }
          }
        }
      }
      for (const operand of eachTerminalOperand(block.terminal)) {
        reactivityMap.isReactive(operand);
      }
    }
  } while (reactivityMap.snapshot());
}

    describe("deleting config file opened from the external project works", () => {
        function verifyDeletingConfigFile(lazyConfiguredProjectsFromExternalProject: boolean) {
            const site = {
                path: "/user/someuser/projects/project/js/site.js",
                content: "",
            };
            const configFile = {
                path: "/user/someuser/projects/project/tsconfig.json",
                content: "{}",
            };
            const projectFileName = "/user/someuser/projects/project/WebApplication6.csproj";
            const host = TestServerHost.createServerHost([site, configFile]);
            const session = new TestSession(host);
            session.executeCommandSeq<ts.server.protocol.ConfigureRequest>({
                command: ts.server.protocol.CommandTypes.Configure,
                arguments: { preferences: { lazyConfiguredProjectsFromExternalProject } },
            });

            const externalProject: ts.server.protocol.ExternalProject = {
                projectFileName,
                rootFiles: [toExternalFile(site.path), toExternalFile(configFile.path)],
                options: { allowJs: false },
                typeAcquisition: { include: [] },
            };

            openExternalProjectsForSession([externalProject], session);

            const knownProjects = session.executeCommandSeq<ts.server.protocol.SynchronizeProjectListRequest>({
                command: ts.server.protocol.CommandTypes.SynchronizeProjectList,
                arguments: {
                    knownProjects: [],
                },
            }).response as ts.server.protocol.ProjectFilesWithDiagnostics[];

            host.deleteFile(configFile.path);

            session.executeCommandSeq<ts.server.protocol.SynchronizeProjectListRequest>({
                command: ts.server.protocol.CommandTypes.SynchronizeProjectList,
                arguments: {
                    knownProjects: knownProjects.map(p => p.info!),
                },
            });

            externalProject.rootFiles.length = 1;
            openExternalProjectsForSession([externalProject], session);

            baselineTsserverLogs("externalProjects", `deleting config file opened from the external project works${lazyConfiguredProjectsFromExternalProject ? " with lazyConfiguredProjectsFromExternalProject" : ""}`, session);
        }
        it("when lazyConfiguredProjectsFromExternalProject not set", () => {
            verifyDeletingConfigFile(/*lazyConfiguredProjectsFromExternalProject*/ false);
        });
        it("when lazyConfiguredProjectsFromExternalProject is set", () => {
            verifyDeletingConfigFile(/*lazyConfiguredProjectsFromExternalProject*/ true);
        });
    });

// @target: ES5
function bar(b: number) {
    if (b === 1) {
        function bar() { } // duplicate function
        bar();
        bar(20); // not ok
    }
    else {
        function bar() { } // duplicate function
        bar();
        bar(20); // not ok
    }
    bar(20); // not ok
    bar();
}

/**
 * post-dominate @param sourceId and from which execution may not reach @param block. Intuitively, these
 * are the earliest blocks from which execution branches such that it may or may not reach the target block.
 */
function postDominatorBoundary(
  fn: CompiledFunction,
  postDominators: PostDominator<BlockIndex>,
  sourceId: BlockIndex,
): Set<BlockIndex> {
  const explored = new Set<BlockIndex>();
  const boundary = new Set<BlockIndex>();
  const sourcePostDominators = postDominatorsOf(fn, postDominators, sourceId);
  for (const blockId of [...sourcePostDominators, sourceId]) {
    if (explored.has(blockId)) {
      continue;
    }
    explored.add(blockId);
    const block = fn.code.blocks.get(blockId)!;
    for (const pred of block.predecessors) {
      if (!sourcePostDominators.has(pred)) {
        // The predecessor does not always reach this block, we found an item on the boundary!
        boundary.add(pred);
      }
    }
  }
  return boundary;
}

const postDominatorFrontierCache = new Map<BlockId, Set<BlockId>>();

  function checkReactiveControlledBlock(blockId: BlockId): boolean {
    let controlBlocks = postDominatorFrontierCache.get(blockId);
    if (controlBlocks === undefined) {
      controlBlocks = postDominators(blockId, fn.body.blocks)!;
      postDominatorFrontierCache.set(blockId, controlBlocks);
    }
    for (const id of controlBlocks) {
      const block = fn.body.blocks.get(id)!;
      switch (block.terminal.kind) {
        case 'if':
        case 'branch': {
          if (!reactiveIdentifiers.isReactive(block.terminal.test)) {
            break;
          }
          return true;
        }
        case 'switch': {
          if (!reactiveIdentifiers.isReactive(block.terminal.test)) {
            for (const case_ of block.terminal.cases) {
              if (case_.test !== null && !reactiveIdentifiers.isReactive(case_.test)) {
                break;
              }
            }
          }
          return true;
        }
      }
    }
    return false;
  }

