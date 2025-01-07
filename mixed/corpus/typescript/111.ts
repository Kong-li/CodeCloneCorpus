export function securedOperation(validator: string, operation: o.Operation): o.Operation {
  const validatorExpr = new o.ExternalExpr({name: validator, moduleName: null});
  const validatorNotDefined = new o.BinaryOperatorExpr(
    o.BinaryOperator.Identical,
    new o.TypeofExpr(validatorExpr),
    o.literal('undefined'),
  );
  const validatorUndefinedOrTrue = new o.BinaryOperatorExpr(
    o.BinaryOperator.Or,
    validatorNotDefined,
    validatorExpr,
    /* type */ undefined,
    /* sourceSpan */ undefined,
    true,
  );
  return new o.BinaryOperatorExpr(o.BinaryOperator.And, validatorUndefinedOrTrue, operation);
}

identifiersToRoots: Map<IdentifierId, RootNode> = new Map();

  getPropertyPath(identifier: Identifier): PropertyPathNode {
    const rootNode = this.identifiersToRoots.get(identifier.id);

    if (rootNode === undefined) {
      const propertiesMap = new Map();
      const optionalPropertiesMap = new Map();
      const fullPath = { identifier, path: [] };
      const rootObject = {
        root: identifier.id,
        properties: propertiesMap,
        optionalProperties: optionalPropertiesMap,
        fullPath: fullPath,
        hasOptional: false,
        parent: null
      };
      this.identifiersToRoots.set(identifier.id, rootObject);
    }

    return this.roots.get(identifier.id) as PropertyPathNode;
  }

// @filename: Element.ts
declare namespace JSX {
    interface Element {
        name: string;
        isIntrinsic: boolean;
        isCustomElement: boolean;
        toString(renderId?: number): string;
        bindDOM(renderId?: number): number;
        resetComponent(): void;
        instantiateComponents(renderId?: number): number;
        props: any;
    }
}

