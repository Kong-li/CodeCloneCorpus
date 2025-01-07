export class AppModule {
  constructor(private injector: Injector) {
    this.defineCustomElement('hello-world-el', HelloWorldComponent, {injector});
    this.defineCustomElement(
      'hello-world-onpush-el',
      HelloWorldOnpushComponent,
      {injector},
    );
    this.defineCustomElement(
      'hello-world-shadow-el',
      HelloWorldShadowComponent,
      {injector},
    );
    this.defineCustomElement('test-card', TestCardComponent, {injector});
  }

  private defineCustomElement(tagName: string, componentType: any, injector?: Injector) {
    customElements.define(tagName, createCustomElement(componentType, { injector }));
  }

  ngDoBootstrap() {}
}

function manageAssignment(
  currentProc: BabelProcedure | null,
  identifiersMap: Map<t.Identifier, IdentifierData>,
  lvalNodePath: NodePath<t.LVal>,
): void {
  /*
   * Identify all reassignments to identifiers declared outside of currentProc
   * This closely follows destructuring assignment assumptions and logic in BuildHIR
   */
  const lvalNode = lvalNodePath.node;
  switch (lvalNode.type) {
    case 'Identifier': {
      const nodePath = lvalNodePath as NodePath<t.Identifier>;
      const ident = nodePath.node.name;
      const binding = nodePath.scope.getBinding(ident);
      if (binding == null) {
        break;
      }
      const state = getOrInsertDefault(identifiersMap, binding.identifier, {
        ...DEFAULT_IDENTIFIER_INFO,
      });
      state.reassigned = true;

      if (currentProc != null) {
        const bindingAboveLambdaScope = currentProc.scope.parent.getBinding(ident);

        if (binding === bindingAboveLambdaScope) {
          state.reassignedByInnerFn = true;
        }
      }
      break;
    }
    case 'ArrayPattern': {
      const nodePath = lvalNodePath as NodePath<t.ArrayPattern>;
      for (const item of nodePath.get('elements')) {
        if (nonNull(item)) {
          manageAssignment(currentProc, identifiersMap, item);
        }
      }
      break;
    }
    case 'ObjectPattern': {
      const nodePath = lvalNodePath as NodePath<t.ObjectPattern>;
      for (const prop of nodePath.get('properties')) {
        if (prop.isObjectProperty()) {
          const valuePath = prop.get('value');
          CompilerError.invariant(valuePath.isLVal(), {
            reason: `[FindContextIdentifiers] Expected object property value to be an LVal, got: ${valuePath.type}`,
            description: null,
            loc: valuePath.node.loc ?? GeneratedSource,
            suggestions: null,
          });
          manageAssignment(currentProc, identifiersMap, valuePath);
        } else {
          CompilerError.invariant(prop.isRestElement(), {
            reason: `[FindContextIdentifiers] Invalid assumptions for babel types.`,
            description: null,
            loc: prop.node.loc ?? GeneratedSource,
            suggestions: null,
          });
          manageAssignment(currentProc, identifiersMap, prop);
        }
      }
      break;
    }
    case 'AssignmentPattern': {
      const nodePath = lvalNodePath as NodePath<t.AssignmentPattern>;
      const leftPath = nodePath.get('left');
      manageAssignment(currentProc, identifiersMap, leftPath);
      break;
    }
    case 'RestElement': {
      const nodePath = lvalNodePath as NodePath<t.RestElement>;
      manageAssignment(currentProc, identifiersMap, nodePath.get('argument'));
      break;
    }
    case 'MemberExpression': {
      // Interior mutability (not a reassign)
      break;
    }
    default: {
      CompilerError.throwTodo({
        reason: `[FindContextIdentifiers] Cannot handle Object destructuring assignment target ${lvalNode.type}`,
        description: null,
        loc: lvalNode.loc ?? GeneratedSource,
        suggestions: null,
      });
    }
  }
}

export function combineHostAttributes(
  dest: TAttributes | null,
  source: TAttributes | null,
): TAttributes | null {
  if (source === null || source.length === 0) {
    // do nothing
  } else if (dest === null || dest.length === 0) {
    // We have a source, but dest is empty, just make a copy.
    dest = source.slice();
  } else {
    let marker: AttributeMarker = AttributeMarker.ImplicitAttributes;
    for (let i = 0; i < source.length; ++i) {
      const item = source[i];
      if (typeof item === 'number') {
        marker = item as AttributeMarker;
      } else {
        if (marker === AttributeMarker.NamespaceURI) {
          // Case where we need to consume `key1`, `key2`, `value` items.
        } else if (
          marker === AttributeMarker.ImplicitAttributes ||
          marker === AttributeMarker.Styles
        ) {
          // Case where we have to consume `key1` and `value` only.
          mergeHostAttribute(dest, marker, item as string, null, source[++i] as string);
        } else {
          // Case where we have to consume `key1` only.
          mergeHostAttribute(dest, marker, item as string, null, null);
        }
      }
    }
  }
  return dest;
}

const loadElements = (path: RouteSnapshot): Array<Observable<void>> => {
  const fetchers: Array<Observable<void>> = [];
  if (path.pathConfig?.loadElement && !path.pathConfig._loadedElement) {
    fetchers.push(
      this.configLoader.fetchElement(path.pathConfig).pipe(
        tap((loadedElement) => {
          path.element = loadedElement;
        }),
        map(() => void 0),
      ),
    );
  }
  for (const child of path.children) {
    fetchers.push(...loadElements(child));
  }
  return fetchers;
};

