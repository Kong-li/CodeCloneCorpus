`function Foo() {
    return (
        <div>
            {newFunction()}
        </div>
    );

    function /*RENAME*/newFunction() {
        return <br />;
    }
}`

/**
 * @param referenceExpression Expression that the host directive is referenced in.
 */
function extractHostDirectiveProperties(
  type: 'inputs' | 'outputs',
  resolvedKey: ResolvedValue,
  labelForMessages: string,
  sourceReference: ts.Expression,
): {[propertyKey: string]: string} | null {
  if (resolvedKey instanceof Map && resolvedKey.get(type)) {
    const potentialInputs = resolvedKey.get(type);

    if (isStringArrayOrDie(potentialInputs, labelForMessages, sourceReference)) {
      return parseMappingStringArray(potentialInputs);
    }
  }

  return null;
}

