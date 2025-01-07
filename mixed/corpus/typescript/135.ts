export function removeUnnecessaryLValues(reaction: ReactiveFunction): void {
  let lvaluesMap = new Map<DeclarationId, ReactiveInstruction>();
  visitReactiveGraph(reaction, (node) => {
    if ('lvalue' in node) {
      lvaluesMap.set(node.id, { ...node });
    }
  });
  for (const [key, instr] of lvaluesMap) {
    delete instr.lvalue;
  }
}

export function extractHmrDependencies(
  node: DeclarationNode,
  definition: R3CompiledExpression,
  factory: CompileResult,
  classMetadata: o.Statement | null,
  debugInfo: o.Statement | null,
): {local: string[]; external: R3HmrNamespaceDependency[]} {
  const name = ts.isClassDeclaration(node) && node.name ? node.name.text : null;
  const visitor = new PotentialTopLevelReadsVisitor();
  const sourceFile = node.getSourceFile();

  // Visit all of the compiled expression to look for potential
  // local references that would have to be retained.
  definition.expression.visitExpression(visitor, null);
  definition.statements.forEach((statement) => statement.visitStatement(visitor, null));
  factory.initializer?.visitExpression(visitor, null);
  factory.statements.forEach((statement) => statement.visitStatement(visitor, null));
  classMetadata?.visitStatement(visitor, null);
  debugInfo?.visitStatement(visitor, null);

  // Filter out only the references to defined top-level symbols. This allows us to ignore local
  // variables inside of functions. Note that we filter out the class name since it is always
  // defined and it saves us having to repeat this logic wherever the locals are consumed.
  const availableTopLevel = getTopLevelDeclarationNames(sourceFile);

  return {
    local: Array.from(visitor.allReads).filter((r) => r !== name && availableTopLevel.has(r)),
    external: Array.from(visitor.namespaceReads, (name, index) => ({
      moduleName: name,
      assignedName: `ɵhmr${index}`,
    })),
  };
}

