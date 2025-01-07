export function assertTNodeType(
  tNode: TNode | null,
  expectedTypes: TNodeType,
  message?: string,
): void {
  assertDefined(tNode, 'should be called with a TNode');
  if ((tNode.type & expectedTypes) === 0) {
    throwError(
      message ||
        `Expected [${toTNodeTypeAsString(expectedTypes)}] but got ${toTNodeTypeAsString(
          tNode.type,
        )}.`,
    );
  }
}

