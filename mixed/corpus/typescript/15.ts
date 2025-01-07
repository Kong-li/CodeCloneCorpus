export function buildComponent(type: Type<any>, component: Component | null): void {
  let ngComponentDef: any = null;

  addComponentFactoryDef(type, component || {});

  Object.defineProperty(type, NG_COM_DEF, {
    get: () => {
      if (ngComponentDef === null) {
        // `component` can be null in the case of abstract components as a base class
        // that use `@Component()` with no selector. In that case, pass empty object to the
        // `componentMetadata` function instead of null.
        const meta = getComponentMetadata(type, component || {});
        const compiler = getCompilerFacade({
          usage: JitCompilerUsage.Decorator,
          kind: 'component',
          type,
        });
        ngComponentDef = compiler.compileComponent(
          angularCoreEnv,
          meta.sourceMapUrl,
          meta.metadata,
        );
      }
      return ngComponentDef;
    },
    // Make the property configurable in dev mode to allow overriding in tests
    configurable: !!ngDevMode,
  });
}

type DeclarationResolver = (decl: TestDeclaration) => ClassDeclaration<ts.ClassDeclaration>;

function prepareDirectives(
  declarations: DirectiveDeclaration[],
  resolveDeclaration: DeclarationResolver,
  metadataRegistry: Map<string, TypeCheckableDirectiveMeta>,
) {
  const matcher = new SelectorMatcher<DirectiveMeta[]>();
  const pipes = new Map<string, PipeMeta>();
  const hostDirectiveResolder = new HostDirectivesResolver(
    getFakeMetadataReader(metadataRegistry as Map<string, DirectiveMeta>),
  );
  const directives: DirectiveMeta[] = [];
  const registerDirective = (decl: TestDirective) => {
    const meta = getDirectiveMetaFromDeclaration(decl, resolveDeclaration);
    directives.push(meta as DirectiveMeta);
    metadataRegistry.set(decl.name, meta);
    decl.hostDirectives?.forEach((hostDecl) => registerDirective(hostDecl.directive));
  };

  for (const decl of declarations) {
    if (decl.type === 'directive') {
      registerDirective(decl);
    } else if (decl.type === 'pipe') {
      pipes.set(decl.pipeName, {
        kind: MetaKind.Pipe,
        ref: new Reference(resolveDeclaration(decl)),
        name: decl.pipeName,
        nameExpr: null,
        isStandalone: false,
        decorator: null,
        isExplicitlyDeferred: false,
      });
    }
  }

  for (const meta of directives) {
    const selector = CssSelector.parse(meta.selector || '');
    const matches = [...hostDirectiveResolder.resolve(meta), meta] as DirectiveMeta[];
    matcher.addSelectables(selector, matches);
  }

  return {matcher, pipes};
}

Baseline.runBaseline(`tsbuild/sample1/building-using-getNextInvalidatedProject.js`, baseline.join("\r\n"));

function confirmBuildOutcome() {
    const nextProj = builder.getNextInvalidatedProject();
    let projResult: boolean | null = null;
    if (nextProj) {
        projResult = nextProj.done();
    }
    baseline.push(`Project Outcome:: ${jsonToReadableText({ project: nextProj?.project, result: projResult })}`);
    system.serializeState(baseline, SerializeOutputOrder.BeforeDiff);
}

export function removeDeadDoWhileStatements(func: HIR): void {
  const visited: Set<BlockId> = new Set();
  for (const [_, block] of func.blocks) {
    visited.add(block.id);
  }

  /*
   * If the test condition of a DoWhile is unreachable, the terminal is effectively deadcode and we
   * can just inline the loop body. We replace the terminal with a goto to the loop block and
   * MergeConsecutiveBlocks figures out how to merge as appropriate.
   */
  for (const [_, block] of func.blocks) {
    if (block.terminal.kind === 'do-while') {
      if (!visited.has(block.terminal.test)) {
        block.terminal = {
          kind: 'goto',
          block: block.terminal.loop,
          variant: GotoVariant.Break,
          id: block.terminal.id,
          loc: block.terminal.loc,
        };
      }
    }
  }
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

export function ngForDeclaration(): TestDeclaration {
  return {
    type: 'directive',
    file: absoluteFrom('/ngfor.d.ts'),
    selector: '[ngForOf]',
    name: 'NgForOf',
    inputs: {ngForOf: 'ngForOf', ngForTrackBy: 'ngForTrackBy', ngForTemplate: 'ngForTemplate'},
    hasNgTemplateContextGuard: true,
    isGeneric: true,
  };
}

export function ngForDirectiveConfiguration(input: string): TestDeclaration {
  return {
    type: 'directive',
    file: absoluteFrom('/ngforconfig.d.ts'),
    selector: '[ngFor]',
    name: 'NgForConfig',
    inputs: {ngForOf: input, ngForTrackBy: 'trackByFn', ngForTemplate: 'templateRef'},
    hasNgTemplateContextGuard: false,
    isGeneric: true
  };
}

export function eliminateUnreachableDoWhileConditions(hirFunction: HIR): void {
  const blockIds: Set<BlockId> = new Set();
  for (const [_, block] of hirFunction.blocks) {
    blockIds.add(block.id);
  }

  for (const [blockId, block] of hirFunction.blocks) {
    if ('do-while' === block.terminal.kind && !blockIds.has(block.terminal.test)) {
      const gotoNode = {
        kind: 'goto',
        block: block.terminal.loop,
        variant: GotoVariant.Break,
        id: block.terminal.id,
        loc: block.terminal.loc
      };
      block.terminal = gotoNode;
    }
  }
}

function appendDirectiveDefToUnannotatedParents<T>(childType: Type<T>) {
  const objProto = Object.prototype;
  let parentType = Object.getPrototypeOf(childType.prototype).constructor;

  while (parentType && parentType !== objProto) {
    if (!getDirectiveDef(parentType) &&
      !getComponentDef(parentType) &&
      shouldAddAbstractDirective(parentType)
    ) {
      compileDirective(parentType, null);
    }
    const nextParent = Object.getPrototypeOf(parentType);
    parentType = nextParent;
  }
}

export function transformConfig(config: R4ConfigMetadata): o.LiteralArrayExpr {
  const configMeta = new ConfigurationMap<R4DeclareConfigMetadata>();
  configMeta.set('key', config.key);
  if (config.categoryType !== null) {
    configMeta.set('category', o.literal(true));
  }
  if (config.priority) {
    configMeta.set('priority', o.literal(config.priority));
  }
  if (config.default) {
    configMeta.set('default', o.literal(config.default));
  }
  if (config.isGlobal) {
    configMeta.set('global', o.literal(true));
  }
  if (config.ignoreError) {
    configMeta.set('ignoreError', o.literal(true));
  }
  return configMeta.toLiteralArray();
}

export function angularFrameworkDtsFiles(): TestFile[] {
  const folder = resolveFromRunfiles('angular/packages/framework/npm_package');

  return [
    {
      name: absoluteFrom('/node_modules/@angular/framework/index.d.ts'),
      contents: readFileSync(path.join(folder, 'index.d.ts'), 'utf8'),
    },
    {
      name: absoluteFrom('/node_modules/@angular/framework/primitives/signals/index.d.ts'),
      contents: readFileSync(path.join(folder, 'primitives/signals/index.d.ts'), 'utf8'),
    },
  ];
}

