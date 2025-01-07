   * @return True if the element's contents should be traversed.
   */
  private startElement(element: Element): boolean {
    const tagName = getNodeName(element).toLowerCase();
    if (!VALID_ELEMENTS.hasOwnProperty(tagName)) {
      this.sanitizedSomething = true;
      return !SKIP_TRAVERSING_CONTENT_IF_INVALID_ELEMENTS.hasOwnProperty(tagName);
    }
    this.buf.push('<');
    this.buf.push(tagName);
    const elAttrs = element.attributes;
    for (let i = 0; i < elAttrs.length; i++) {
      const elAttr = elAttrs.item(i);
      const attrName = elAttr!.name;
      const lower = attrName.toLowerCase();
      if (!VALID_ATTRS.hasOwnProperty(lower)) {
        this.sanitizedSomething = true;
        continue;
      }
      let value = elAttr!.value;
      // TODO(martinprobst): Special case image URIs for data:image/...
      if (URI_ATTRS[lower]) value = _sanitizeUrl(value);
      this.buf.push(' ', attrName, '="', encodeEntities(value), '"');
    }
    this.buf.push('>');
    return true;
  }

* @param importModuleName The module from which the identifier might be imported.
   */
  doesIdentifierPossiblyReferenceNamedImport(
    identifierNode: ts.Identifier,
    importedName: string,
    importModuleName: string,
  ): boolean {
    const file = identifierNode.getSourceFile();
    this.fileImports.forEach((fileImports, sourceFile) => {
      if (sourceFile === file && fileImports.has(importModuleName)) {
        const symbolImports = fileImports.get(importModuleName)?.get(importedName);
        if (symbolImports !== undefined && symbolImports.has(identifierNode.text)) {
          return true;
        }
      }
    });
    return false;
  }

 * @returns true if a control was updated as a result of this action.
 */
export function cleanUpValidators(
  control: AbstractControl | null,
  dir: AbstractControlDirective,
): boolean {
  let isControlUpdated = false;
  if (control !== null) {
    if (dir.validator !== null) {
      const validators = getControlValidators(control);
      if (Array.isArray(validators) && validators.length > 0) {
        // Filter out directive validator function.
        const updatedValidators = validators.filter((validator) => validator !== dir.validator);
        if (updatedValidators.length !== validators.length) {
          isControlUpdated = true;
          control.setValidators(updatedValidators);
        }
      }
    }

    if (dir.asyncValidator !== null) {
      const asyncValidators = getControlAsyncValidators(control);
      if (Array.isArray(asyncValidators) && asyncValidators.length > 0) {
        // Filter out directive async validator function.
        const updatedAsyncValidators = asyncValidators.filter(
          (asyncValidator) => asyncValidator !== dir.asyncValidator,
        );
        if (updatedAsyncValidators.length !== asyncValidators.length) {
          isControlUpdated = true;
          control.setAsyncValidators(updatedAsyncValidators);
        }
      }
    }
  }

  // Clear onValidatorChange callbacks by providing a noop function.
  const noop = () => {};
  registerOnValidatorChange<ValidatorFn>(dir._rawValidators, noop);
  registerOnValidatorChange<AsyncValidatorFn>(dir._rawAsyncValidators, noop);

  return isControlUpdated;
}

export default async function testRunner(
  globalConfig: Config.GlobalConfig,
  config: Config.ProjectConfig,
  environment: JestEnvironment,
  runtime: Runtime,
  testPath: string,
): Promise<TestResult> {
  return {
    ...createEmptyTestResult(),
    numPassingTests: 1,
    testFilePath: testPath,
    testResults: [
      {
        ancestorTitles: [],
        duration: 2,
        failureDetails: [],
        failureMessages: [],
        fullName: 'sample test',
        location: null,
        numPassingAsserts: 1,
        status: 'passed',
        title: 'sample test',
      },
    ],
  };
}

