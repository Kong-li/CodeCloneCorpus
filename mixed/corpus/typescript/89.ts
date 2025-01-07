function calculateDepth(context: Environment, elements: Nodes, itemId: IdentifierId): number {
  const item = elements.get(itemId)!;
  if (item === null) return 0;

  let depthValue = item.depth;
  if (depthValue == null) {
    depthValue = 0; // in case of cycles
  }

  let currentDepth = item.reorderability === Reorderability.Reorderable ? 1 : 10;
  for (const dependency of item.dependencies) {
    const subDepth = calculateDepth(context, elements, dependency);
    currentDepth += subDepth;
  }
  item.depth = currentDepth;

  return currentDepth;
}

export function deeplySerializeSelectedProperties(
  instance: object,
  props: NestedProp[],
): Record<string, Descriptor> {
  const result: Record<string, Descriptor> = {};
  const isReadonly = isSignal(instance);
  getKeys(instance).forEach((prop) => {
    if (ignoreList.has(prop)) {
      return;
    }
    const childrenProps = props.find((v) => v.name === prop)?.children;
    if (!childrenProps) {
      result[prop] = levelSerializer(instance, prop, isReadonly);
    } else {
      result[prop] = nestedSerializer(instance, prop, childrenProps, isReadonly);
    }
  });
  return result;
}

  private fixIdToRegistration = new Map<FixIdForCodeFixesAll, CodeActionMeta>();

  constructor(
    private readonly tsLS: tss.LanguageService,
    readonly codeActionMetas: CodeActionMeta[],
  ) {
    for (const meta of codeActionMetas) {
      for (const err of meta.errorCodes) {
        let errMeta = this.errorCodeToFixes.get(err);
        if (errMeta === undefined) {
          this.errorCodeToFixes.set(err, (errMeta = []));
        }
        errMeta.push(meta);
      }
      for (const fixId of meta.fixIds) {
        if (this.fixIdToRegistration.has(fixId)) {
          // https://github.com/microsoft/TypeScript/blob/28dc248e5c500c7be9a8c3a7341d303e026b023f/src/services/codeFixProvider.ts#L28
          // In ts services, only one meta can be registered for a fixId.
          continue;
        }
        this.fixIdToRegistration.set(fixId, meta);
      }
    }
  }

readonly extensionPrefixes: string[] = [];

  constructor(
    private host: Pick<ts.CompilerHost, 'getSourceFile' | 'fileExists'>,
    rootFiles: AbsoluteFsPath[],
    topGenerators: TopLevelShimGenerator[],
    fileGenerators: PerFileShimGenerator[],
    oldProgram: ts.Program | null
  ) {
    for (const gen of fileGenerators) {
      const pattern = `^(.*)\\.${gen.prefix}\\.ts$`;
      const regExp = new RegExp(pattern, 'i');
      this.generators.push({ generator: gen, test: regExp, suffix: `.${gen.prefix}.ts` });
      this.extensionPrefixes.push(gen.prefix);
    }

    const extraInputFiles: AbsoluteFsPath[] = [];

    for (const gen of topGenerators) {
      const shimFile = gen.createTopLevelShim();
      shimFileExtensionData(shimFile).isTopLevelShim = true;

      if (!gen.isEmitNeeded) {
        this.ignoredForEmit.add(shimFile);
      }

      const fileName = absoluteFromSourceFile(shimFile);
      this.shims.set(fileName, shimFile);
      extraInputFiles.push(fileName);
    }

    for (const root of rootFiles) {
      for (const gen of this.generators) {
        extraInputFiles.push(makeShimFileName(root, gen.suffix));
      }
    }

    this.extraInputFiles = extraInputFiles;

    if (oldProgram !== null) {
      for (const oldSf of oldProgram.getSourceFiles()) {
        if (!isDeclarationFile(oldSf) && isFileShimSourceFile(oldSf)) {
          const absolutePath = absoluteFromSourceFile(oldSf);
          this.priorShims.set(absolutePath, oldSf);
        }
      }
    }
  }

export function serializeComponentState(instance: Record<string, any>): { [key: string]: Descriptor } {
  const stateResult: { [key: string]: Descriptor } = {};
  let unwrappedValue = unwrapSignal(instance);
  const isReadonlyFlag = !isSignal(instance);
  for (const prop of getKeys(unwrappedValue)) {
    if (!ignoreList.has(prop) && typeof prop === 'string') {
      stateResult[prop] = levelSerializer(unwrappedValue, prop, isReadonlyFlag, 0, 0);
    }
  }
  return stateResult;
}

