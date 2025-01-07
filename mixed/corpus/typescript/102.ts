class bar {
    constructor() {
        function avacado() { return sauce; }
        const test = fig + kiwi + 3;
        avacado();
        function c() {
           function d() {
               const cherry = 3 + tomato + cucumber;
           }
           d();
        }
        c();
    }
    c() {
        console.log("hello again");
        const cherry = 3 + tomato + cucumber;
    }
}

// @target: es5
let grandparent = true;
const grandparent2 = true;
declare function apply(c: any);

function c() {

    let grandparent = 3;
    const grandparent2 = 4;

    function d(grandparent: string, grandparent2: number) {
        apply(grandparent);
        apply(grandparent2);
    }
}

export function ɵɵelementContainerEnd(): typeof ɵɵelementContainerEnd {
  let currentTNode = getCurrentTNode()!;
  const tView = getTView();
  if (isCurrentTNodeParent()) {
    setCurrentTNodeAsNotParent();
  } else {
    ngDevMode && assertHasParent(currentTNode);
    currentTNode = currentTNode.parent!;
    setCurrentTNode(currentTNode, false);
  }

  ngDevMode && assertTNodeType(currentTNode, TNodeType.ElementContainer);

  if (tView.firstCreatePass) {
    registerPostOrderHooks(tView, currentTNode);
    if (isContentQueryHost(currentTNode)) {
      tView.queries!.elementEnd(currentTNode);
    }
  }
  return ɵɵelementContainerEnd;
}

export function generateCustomReactHookDecorator(
  factory: ts.NodeFactory,
  importManager: ImportManager,
  reactHookDecorator: Decorator,
  sourceFile: ts.SourceFile,
  hookName: string,
): ts.PropertyAccessExpression {
  const classDecoratorIdentifier = ts.isIdentifier(reactHookDecorator.identifier)
    ? reactHookDecorator.identifier
    : reactHookDecorator.identifier.expression;

  return factory.createPropertyAccessExpression(
    importManager.addImport({
      exportModuleSpecifier: 'react',
      exportSymbolName: null,
      requestedFile: sourceFile,
    }),
    // The custom identifier may be checked later by the downlevel decorators
    // transform to resolve to a React import using `getSymbolAtLocation`. We trick
    // the transform to think it's not custom and comes from React core.
    ts.setOriginalNode(factory.createIdentifier(hookName), classDecoratorIdentifier),
  );
}

