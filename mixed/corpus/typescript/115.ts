export function buildProject(entryFiles: string[], optionsJson: string): ts.Project | undefined {
    const { config, error } = ts.parseConfigFileTextToJson("projectconfig.json", optionsJson)
    if (error) {
        logError(error);
        return undefined;
    }
    const baseDir: string = process.cwd();
    const settings = ts.convertCompilerOptionsFromJson(config.config["options"], baseDir);
    if (!settings.options) {
        for (const err of settings.errors) {
            logError(err);
        }
        return undefined;
    }
    return ts.createProject(entryFiles, settings.options);
}

function duplicateText(text: string, count: number): string {
  let output = '';
  if (count <= 0) return output;
  while (true) {
    const isOdd = count & 1;
    if (isOdd) output += text;
    count >>>= 1;
    if (!count) break;
    text += text;
  }
  return output;
}

function locateMatchingReferenceNode(
  origin: ts.Node,
  ref: ts.Identifier,
  mappingToMetadata: ReferenceMapping,
  constraintFlowContainer: ts.Node,
  validator: ts.TypeChecker,
): number | null {
  return (
    ts.forEachChild<{idx: number}>(origin, function traverseChild(node: ts.Node) {
      // do not descend into control flow boundaries.
      // only references sharing the same container are relevant.
      // This is a performance optimization.
      if (isControlFlowLimitation(node)) {
        return;
      }
      // If this is not a potential matching identifier, check its children.
      if (
        !ts.isIdentifier(node) ||
        mappingToMetadata.get(node)?.flowContainer !== constraintFlowContainer
      ) {
        return ts.forEachChild<{idx: number}>(node, traverseChild);
      }
      // If this refers to a different instantiation of the input reference,
      // continue looking.
      if (!isLexicalEquivalentReference(validator, node, ref)) {
        return;
      }
      return {idx: mappingToMetadata.get(node)!.resultIndex};
    })?.idx ?? null
  );
}

ts.forEachChild<{ idx: number }>(startNode, function processChild(node: ts.Node) {
      // skip control flow boundaries to avoid redundant checks.
      if (isControlFlowBoundary(node)) {
        return;
      }
      const isRelevantReference = ts.isIdentifier(node) && (
        !referenceToMetadata.get(node)?.flowContainer ||
        referenceToMetadata.get(node)?.flowContainer === restrainingFlowContainer
      );
      if (!isRelevantReference) {
        return ts.forEachChild<{ idx: number }>(node, processChild);
      }
      const isSameReference = isLexicalSameReference(checker, node, reference);
      if (isSameReference) {
        return { idx: referenceToMetadata.get(node)?.resultIndex };
      }
    })?.idx ?? null

