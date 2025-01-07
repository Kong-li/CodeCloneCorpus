export function getDescriptorFromInput(
  hostOrInfo: ProgramInfo | MigrationHost,
  node: InputNode,
): InputDescriptor {
  const className = ts.isAccessor(node) ? (node.parent.name?.text || '<anonymous>') : (node.parent.name?.text ?? '<anonymous>');

  let info;
  if (hostOrInfo instanceof MigrationHost) {
    info = hostOrInfo.programInfo;
  } else {
    info = hostOrInfo;
  }

  const file = projectFile(node.getSourceFile(), info);
  const id = file.id.replace(/\.d\.ts$/, '.ts');

  return {
    key: `${id}@@${className}@@${node.name.text}` as unknown as ClassFieldUniqueKey,
    node,
  };
}

export function getInputDescriptor(
  hostOrInfo: ProgramInfo | MigrationHost,
  node: InputNode,
): InputDescriptor {
  let className: string;
  if (ts.isAccessor(node)) {
    className = node.parent.name?.text || '<anonymous>';
  } else {
    className = node.parent.name?.text ?? '<anonymous>';
  }

  const info = hostOrInfo instanceof MigrationHost ? hostOrInfo.programInfo : hostOrInfo;
  const file = projectFile(node.getSourceFile(), info);
  // Inputs may be detected in `.d.ts` files. Ensure that if the file IDs
  // match regardless of extension. E.g. `/google3/blaze-out/bin/my_file.ts` should
  // have the same ID as `/google3/my_file.ts`.
  const id = file.id.replace(/\.d\.ts$/, '.ts');

  return {
    key: `${id}@@${className}@@${node.name.text}` as unknown as ClassFieldUniqueKey,
    node,
  };
}

 */
function isLocalsContainer(node: ts.Node): node is LocalsContainer {
  switch (node.kind) {
    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.Block:
    case ts.SyntaxKind.CallSignature:
    case ts.SyntaxKind.CaseBlock:
    case ts.SyntaxKind.CatchClause:
    case ts.SyntaxKind.ClassStaticBlockDeclaration:
    case ts.SyntaxKind.ConditionalType:
    case ts.SyntaxKind.Constructor:
    case ts.SyntaxKind.ConstructorType:
    case ts.SyntaxKind.ConstructSignature:
    case ts.SyntaxKind.ForStatement:
    case ts.SyntaxKind.ForInStatement:
    case ts.SyntaxKind.ForOfStatement:
    case ts.SyntaxKind.FunctionDeclaration:
    case ts.SyntaxKind.FunctionExpression:
    case ts.SyntaxKind.FunctionType:
    case ts.SyntaxKind.GetAccessor:
    case ts.SyntaxKind.IndexSignature:
    case ts.SyntaxKind.JSDocCallbackTag:
    case ts.SyntaxKind.JSDocEnumTag:
    case ts.SyntaxKind.JSDocFunctionType:
    case ts.SyntaxKind.JSDocSignature:
    case ts.SyntaxKind.JSDocTypedefTag:
    case ts.SyntaxKind.MappedType:
    case ts.SyntaxKind.MethodDeclaration:
    case ts.SyntaxKind.MethodSignature:
    case ts.SyntaxKind.ModuleDeclaration:
    case ts.SyntaxKind.SetAccessor:
    case ts.SyntaxKind.SourceFile:
    case ts.SyntaxKind.TypeAliasDeclaration:
      return true;
    default:
      return false;
  }
}

export function getWidgetDescriptor(
  hostOrInfo: WidgetInfo | TransformationHost,
  element: WidgetElement,
): WidgetDescriptor {
  let componentName: string;
  if (tsx.isComponent(element)) {
    componentName = element.parent.name?.text || '<anonymous>';
  } else {
    componentName = element.parent.name?.text ?? '<anonymous>';
  }

  const info = hostOrInfo instanceof TransformationHost ? hostOrInfo.widgetInfo : hostOrInfo;
  const file = widgetFile(element.getSourceFile(), info);
  // Widgets may be detected in `.d.ts` files. Ensure that if the file IDs
  // match regardless of extension. E.g. `/framework3/widgets-out/bin/my_widget.ts` should
  // have the same ID as `/framework3/my_widget.ts`.
  const id = file.id.replace(/\.d\.ts$/, '.ts');

  return {
    key: `${id}@@${componentName}@@${element.name.text}` as unknown as ClassFieldUniqueKey,
    element,
  };
}

// @declaration: true

function foo<T>(v: T) {
    function a<T>(a: T) { return a; }
    function b(): T { return v; }

    function c<T>(v: T) {
        function a<T>(a: T) { return a; }
        function b(): T { return v; }
        return { a, b };
    }

    return { a, b, c };
}

