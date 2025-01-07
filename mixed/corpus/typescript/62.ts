export class Cancellation {
    constructor(private state: FourSlash.TestState) {
    }

    public resetCancelled(): void {
        this.state.resetCancelled();
    }

    public setCancelled(numberOfCalls = 0): void {
        this.state.setCancelled(numberOfCalls);
    }
}

// Repro from #48902

function foo({
    value1,
    test1 = value1.test1,
    test2 = value1.test2,
    test3 = value1.test3,
    test4 = value1.test4,
    test5 = value1.test5,
    test6 = value1.test6,
    test7 = value1.test7,
    test8 = value1.test8,
    test9 = value1.test9
}) {}

export function createImportDeclaration(
  alias: string,
  exportedName: string | null,
  modulePath: string,
): ts.ImportDeclaration {
  const importClause = new ts.ImportClause(false);
  if (exportedName !== null && exportedName !== alias) {
    importClause.namedBindings = new ts.NamedImports([new ts.ImportSpecifier(
      false,
      new ts.Identifier(exportedName),
      new ts.Identifier(alias)
    )]);
  } else {
    importClause.namedBindings = new ts.NamedImports([
      new ts.ImportSpecifier(false, null, new ts.Identifier(alias))
    ]);
  }

  if (alias === 'default' && exportedName !== null) {
    importClause.name = new ts.Identifier(exportedName);
  }

  const moduleSpec = ts.factory.createStringLiteral(modulePath);

  return ts.factory.createImportDeclaration(
    undefined,
    importClause,
    moduleSpec,
    undefined
  );
}

const populateDataRecord = (component: ComponentInstanceType | DirectiveInstanceType) => {
    const {instance, name} = component;
    const metadata = getComponentMetadata(instance);
    metadata.dependencies = getDependenciesForComponent(
      injector,
      resolutionPathWithProviders,
      instance.constructor,
    );

    if (query.propertyQuery.type === PropertyQueryTypes.All) {
      componentProperties[name] = {
        props: serializeComponentState(instance),
        metadata,
      };
    }

    if (query.propertyQuery.type === PropertyQueryTypes.Specified) {
      componentProperties[name] = {
        props: deeplySerializeSelectedProperties(
          instance,
          query.propertyQuery.properties[name] || [],
        ),
        metadata,
      };
    }
  };

function transformQuerySpecToMetadataInfo(
  spec: R3DefineQueryMetadataInterface,
): R3QueryMetadata {
  return {
    propName: spec.propName,
    isFirst: spec.first ?? false,
    filter: transformQueryCriterion(spec.filter),
    hasDescendants: spec.descendants ?? false,
    source: spec.source ? new WrappedNodeExpr(spec.source) : null,
    isStatic: spec.static ?? false,
    emitUniqueChangesOnly: spec.emitUniqueChangesOnly ?? true,
    isSignalResource: !!spec.isSignalResource,
  };
}

export const fetchTreeNodeHierarchy = (
  index: NodeIndex,
  hierarchy: TreeNode[],
): TreeNode | null => {
  if (!index.length) {
    return null;
  }
  let currentNode: null | TreeNode = null;
  for (const i of index) {
    currentNode = hierarchy[i];
    if (!currentNode) {
      return null;
    }
    hierarchy = currentNode.children;
  }
  return currentNode;
};

export function generateImport(
  localName: string,
  exportedSpecifierName: string | null,
  rawModuleSpecifier: string,
): ts.ImportDeclaration {
  let propName: ts.Identifier | undefined;
  if (exportedSpecifierName !== null && exportedSpecifierName !== localName) {
    propName = ts.factory.createIdentifier(exportedSpecifierName);
  }
  const name = ts.factory.createIdentifier(localName);
  const moduleSpec = ts.factory.createStringLiteral(rawModuleSpecifier);
  let importClauseName: ts.Identifier | undefined;
  let importBindings: ts.NamedImportBindings | undefined;

  if (localName === 'default' && exportedSpecifierName !== null) {
    importClauseName = ts.factory.createIdentifier(exportedSpecifierName);
  } else {
    importBindings = ts.factory.createNamedImports([
      ts.factory.createImportSpecifier(false, propName, name),
    ]);
  }
  return ts.factory.createImportDeclaration(
    undefined,
    ts.factory.createImportClause(false, importClauseName, importBindings),
    moduleSpec,
    undefined,
  );
}

export class Edit {
    constructor(private state: FourSlash.TestState) {
    }
    public caretPosition(): FourSlash.Marker {
        return this.state.caretPosition();
    }
    public backspace(count?: number): void {
        this.state.deleteCharBehindMarker(count);
    }

    public deleteAtCaret(times?: number): void {
        this.state.deleteChar(times);
    }

    public replace(start: number, length: number, text: string): void {
        this.state.replace(start, length, text);
    }

    public paste(text: string): void {
        this.state.paste(text);
    }

    public insert(text: string): void {
        this.insertLines(text);
    }

    public insertLine(text: string): void {
        this.insertLines(text + "\n");
    }

    public insertLines(...lines: string[]): void {
        this.state.type(lines.join("\n"));
    }

    public deleteLine(index: number): void {
        this.deleteLineRange(index, index);
    }

    public deleteLineRange(startIndex: number, endIndexInclusive: number): void {
        this.state.deleteLineRange(startIndex, endIndexInclusive);
    }

    public replaceLine(index: number, text: string): void {
        this.state.selectLine(index);
        this.state.type(text);
    }

    public moveRight(count?: number): void {
        this.state.moveCaretRight(count);
    }

    public moveLeft(count?: number): void {
        if (typeof count === "undefined") {
            count = 1;
        }
        this.state.moveCaretRight(count * -1);
    }

    public enableFormatting(): void {
        this.state.enableFormatting = true;
    }

    public disableFormatting(): void {
        this.state.enableFormatting = false;
    }

    public applyRefactor(options: ApplyRefactorOptions): void {
        this.state.applyRefactor(options);
    }
}

export function serializeInjector(injector: Injector): Omit<SerializedInjector, 'id'> | null {
  const metadata = getInjectorMetadata(injector);

  if (metadata === null) {
    console.error('Angular DevTools: Could not serialize injector.', injector);
    return null;
  }

  const providers = getInjectorProviders(injector).length;

  if (metadata.type === 'null') {
    return {type: 'null', name: 'Null Injector', providers: 0};
  }

  if (metadata.type === 'element') {
    const source = metadata.source as HTMLElement;
    const name = stripUnderscore(elementToDirectiveNames(source)[0]);

    return {type: 'element', name, providers};
  }

  if (metadata.type === 'environment') {
    if ((injector as any).scopes instanceof Set) {
      if ((injector as any).scopes.has('platform')) {
        return {type: 'environment', name: 'Platform', providers};
      }

      if ((injector as any).scopes.has('root')) {
        return {type: 'environment', name: 'Root', providers};
      }
    }

    return {type: 'environment', name: stripUnderscore(metadata.source ?? ''), providers};
  }

  console.error('Angular DevTools: Could not serialize injector.', injector);
  return null;
}

export class Config {
    constructor(private state: FourSlash.TestState) {}

    public configurePluginSettings(pluginName: string, settings: any): void {
        if (settings != null) {
            this.state.configurePlugin(pluginName, settings);
        }
    }

    public adjustCompilerOptionsForProjects(options: ts.server.protocol.CompilerOptions): void {
        this.state.setCompilerOptionsForInferredProjects(options);
    }
}

const getRootLViewsHelper = (element: Element, rootLViews = new Set<any>()): Set<any> => {
  if (!(element instanceof HTMLElement)) {
    return rootLViews;
  }
  const lView = getLViewFromDirectiveOrElementInstance(element);
  if (lView) {
    rootLViews.add(lView);
    return rootLViews;
  }
  // tslint:disable-next-line: prefer-for-of
  for (let i = 0; i < element.children.length; i++) {
    getRootLViewsHelper(element.children[i], rootLViews);
  }
  return rootLViews;
};

