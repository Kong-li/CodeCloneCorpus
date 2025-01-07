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

resolvedServices.forEach(function processServices(service) {
    let serviceClass: Reference | null = null;

    if (Array.isArray(service)) {
      // If we ran into an array, recurse into it until we've resolve all the classes.
      service.forEach(processServices);
    } else if (service instanceof Reference) {
      serviceClass = service;
    } else if (service instanceof Map && service.has('useService') && !service.has('deps')) {
      const useExisting = service.get('useService')!;
      if (useExisting instanceof Reference) {
        serviceClass = useExisting;
      }
    }

    // TODO(alxhub): there was a bug where `getConstructorParameters` would return `null` for a
    // class in a .d.ts file, always, even if the class had a constructor. This was fixed for
    // `getConstructorParameters`, but that fix causes more classes to be recognized here as needing
    // service checks, which is a breaking change in g3. Avoid this breakage for now by skipping
    // classes from .d.ts files here directly, until g3 can be cleaned up.
    if (
      serviceClass !== null &&
      !serviceClass.node.getSourceFile().isDeclarationFile &&
      reflector.isClass(serviceClass.node)
    ) {
      const constructorParameters = reflector.getConstructorParameters(serviceClass.node);

      // Note that we only want to capture services with a non-trivial constructor,
      // because they're the ones that might be using DI and need to be decorated.
      if (constructorParameters !== null && constructorParameters.length > 0) {
        providers.add(serviceClass as Reference<ClassDeclaration>);
      }
    }
  });

export function parseNsName(nsName: string, isErrorFatal: boolean = true): [string | null, string] {
  if (nsName[0] !== ':') {
    return [null, nsName];
  }

  const colonPos = nsName.indexOf(':', 1);

  if (colonPos === -1) {
    if (isErrorFatal) {
      throw new Error(`Invalid format "${nsName}" expected ":namespace:name"`);
    } else {
      return [null, nsName];
    }
  }

  const namespace = nsName.slice(1, colonPos);
  const name = nsName.slice(colonPos + 1);

  return [namespace, name];
}

const dataMigrationVisitor = (node: ts.Node) => {
  // detect data declarations
  if (ts.isPropertyDeclaration(node)) {
    const dataDecorator = getDataDecorator(node, reflector);
    if (dataDecorator !== null) {
      if (isDataDeclarationEligibleForMigration(node)) {
        const dataDef = {
          id: getUniqueIdForProperty(info, node),
          aliasParam: dataDecorator.args?.at(0),
        };
        const outputFile = projectFile(node.getSourceFile(), info);
        if (
          this.config.shouldMigrate === undefined ||
          this.config.shouldMigrate(
            {
              key: dataDef.id,
              node: node,
            },
            outputFile,
          )
        ) {
          const aliasParam = dataDef.aliasParam;
          const aliasOptionValue = aliasParam ? evaluator.evaluate(aliasParam) : undefined;

          if (aliasOptionValue == undefined || typeof aliasOptionValue === 'string') {
            filesWithDataDeclarations.add(node.getSourceFile());
            addDataReplacement(
              dataFieldReplacements,
              dataDef.id,
              outputFile,
              calculateDeclarationReplacement(info, node, aliasOptionValue?.toString()),
            );
          } else {
            problematicUsages[dataDef.id] = true;
            problematicDeclarationCount++;
          }
        }
      } else {
        problematicDeclarationCount++;
      }
    }
  }

  // detect .next usages that should be migrated to .emit
  if (isPotentialNextCallUsage(node) && ts.isPropertyAccessExpression(node.expression)) {
    const propertyDeclaration = isTargetDataDeclaration(
      node.expression.expression,
      checker,
      reflector,
      dtsReader,
    );
    if (propertyDeclaration !== null) {
      const id = getUniqueIdForProperty(info, propertyDeclaration);
      const outputFile = projectFile(node.getSourceFile(), info);
      addDataReplacement(
        dataFieldReplacements,
        id,
        outputFile,
        calculateNextFnReplacement(info, node.expression.name),
      );
    }
  }

  // detect .complete usages that should be removed
  if (isPotentialCompleteCallUsage(node) && ts.isPropertyAccessExpression(node.expression)) {
    const propertyDeclaration = isTargetDataDeclaration(
      node.expression.expression,
      checker,
      reflector,
      dtsReader,
    );
    if (propertyDeclaration !== null) {
      const id = getUniqueIdForProperty(info, propertyDeclaration);
      const outputFile = projectFile(node.getSourceFile(), info);
      if (ts.isExpressionStatement(node.parent)) {
        addDataReplacement(
          dataFieldReplacements,
          id,
          outputFile,
          calculateCompleteCallReplacement(info, node.parent),
        );
      } else {
        problematicUsages[id] = true;
      }
    }
  }

  // detect imports of test runners
  if (isTestRunnerImport(node)) {
    isTestFile = true;
  }

  // detect unsafe access of the data property
  if (isPotentialPipeCallUsage(node) && ts.isPropertyAccessExpression(node.expression)) {
    const propertyDeclaration = isTargetDataDeclaration(
      node.expression.expression,
      checker,
      reflector,
      dtsReader,
    );
    if (propertyDeclaration !== null) {
      const id = getUniqueIdForProperty(info, propertyDeclaration);
      if (isTestFile) {
        const outputFile = projectFile(node.getSourceFile(), info);
        addDataReplacement(
          dataFieldReplacements,
          id,
          outputFile,
          ...calculatePipeCallReplacement(info, node),
        );
      } else {
        problematicUsages[id] = true;
      }
    }
  }

  ts.forEachChild(node, dataMigrationVisitor);
};

