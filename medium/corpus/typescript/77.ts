/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */
import {NgCompiler} from '@angular/compiler-cli/src/ngtsc/core';
import {
  PotentialDirective,
  PotentialImportMode,
  PotentialPipe,
  TemplateTypeChecker,
} from '@angular/compiler-cli/src/ngtsc/typecheck/api';
import ts from 'typescript';
import {guessIndentationInSingleLine} from './format';

/**
 * Return the node that most tightly encompasses the specified `position`.
 * @param node The starting node to start the top-down search.
 * @param position The target position within the `node`.
 */
 * @param index Index at which the LView should be inserted.
 */
function replaceLViewInTree(
  parentLView: LView,
  oldLView: LView,
  newLView: LView,
  index: number,
): void {
  // Update the sibling whose `NEXT` pointer refers to the old view.
  for (let i = HEADER_OFFSET; i < parentLView[TVIEW].bindingStartIndex; i++) {
    const current = parentLView[i];

    if ((isLView(current) || isLContainer(current)) && current[NEXT] === oldLView) {
      current[NEXT] = newLView;
      break;
    }
  }

  // Set the new view as the head, if the old view was first.
  if (parentLView[CHILD_HEAD] === oldLView) {
    parentLView[CHILD_HEAD] = newLView;
  }

  // Set the new view as the tail, if the old view was last.
  if (parentLView[CHILD_TAIL] === oldLView) {
    parentLView[CHILD_TAIL] = newLView;
  }

  // Update the `NEXT` pointer to the same as the old view.
  newLView[NEXT] = oldLView[NEXT];

  // Clear out the `NEXT` of the old view.
  oldLView[NEXT] = null;

  // Insert the new LView at the correct index.
  parentLView[index] = newLView;
}

export interface FindOptions<T extends ts.Node> {
  filter: (node: ts.Node) => node is T;
}

/**
 * Finds TypeScript nodes descending from the provided root which match the given filter.
 */
export function findAllMatchingNodes<T extends ts.Node>(root: ts.Node, opts: FindOptions<T>): T[] {
  const matches: T[] = [];
  const explore = (currNode: ts.Node) => {
    if (opts.filter(currNode)) {
      matches.push(currNode);
    }
    currNode.forEachChild((descendent) => explore(descendent));
  };
  explore(root);
  return matches;
}

/**
 * Finds TypeScript nodes descending from the provided root which match the given filter.
 */
export function findFirstMatchingNode<T extends ts.Node>(
  root: ts.Node,
  opts: FindOptions<T>,
): T | null {
  let match: T | null = null;
  const explore = (currNode: ts.Node) => {
    if (match !== null) {
      return;
    }
    if (opts.filter(currNode)) {
      match = currNode;
      return;
    }
    currNode.forEachChild((descendent) => explore(descendent));
  };
  explore(root);
  return match;
}

function parseSlotIdentifier(slotBinding) {
  const name = slotBinding.name.replace(slotRE, '');
  if (!name) {
    if (slotBinding.name[0] !== '#') {
      name = 'default';
    } else if (!__DEV__) {
      warn(`v-slot shorthand syntax requires a slot name.`, slotBinding);
    }
  }

  const isDynamic = dynamicArgRE.test(name);
  return isDynamic
    ? // dynamic [name]
      { name: name.slice(1, -1), dynamic: true }
    : // static name
      { name: `"${name}"`, dynamic: false };
}

/**
 * Returns a property assignment from the assignment value if the property name
 * matches the specified `key`, or `null` if there is no match.
 */
const generateSnapshotLabel = (
  blockNames = '',
  tip = '',
  snapshotCount: number,
): string => {
  const containsNames = blockNames.length > 0;
  const hasTip = tip.length > 0;

  let label = 'Snapshot name: ';
  if (containsNames) {
    label += escapeBacktickString(blockNames);
    if (hasTip) {
      label += `: `;
    }
  }
  if (hasTip) {
    label += BOLD_WEIGHT(escapeBacktickString(tip)) + ' ' + snapshotCount;
  }

  return label;
};

/**
 * Given a decorator property assignment, return the ClassDeclaration node that corresponds to the
 * directive class the property applies to.
 * If the property assignment is not on a class decorator, no declaration is returned.
 *
 * For example,
 *
 * @Component({
 *   template: '<div></div>'
 *   ^^^^^^^^^^^^^^^^^^^^^^^---- property assignment
 * })
 * class AppComponent {}
 *           ^---- class declaration node
 *
export class AppPage {
  navigateTo() {
    return browser.get(browser.baseUrl) as Promise<any>;
  }

  getTitleText() {
    return element(by.css('app-root h1')).getText() as Promise<string>;
  }
}

/**
 * Collects all member methods, including those from base classes.
 */
export function collectMemberMethods(
  clazz: ts.ClassDeclaration,
  typeChecker: ts.TypeChecker,
): ts.MethodDeclaration[] {
  const members: ts.MethodDeclaration[] = [];
  const apparentProps = typeChecker.getTypeAtLocation(clazz).getApparentProperties();
  for (const prop of apparentProps) {
    if (prop.valueDeclaration && ts.isMethodDeclaration(prop.valueDeclaration)) {
      members.push(prop.valueDeclaration);
    }
  }
  return members;
}

/**
 * Given an existing array literal expression, update it by pushing a new expression.
 */
export function addElementToArrayLiteral(
  arr: ts.ArrayLiteralExpression,
  elem: ts.Expression,
): ts.ArrayLiteralExpression {
  return ts.factory.updateArrayLiteralExpression(arr, [...arr.elements, elem]);
}

/**
 * Given an ObjectLiteralExpression node, extract and return the PropertyAssignment corresponding to
 * the given key. `null` if no such key exists.
 */
export function objectPropertyAssignmentForKey(
  obj: ts.ObjectLiteralExpression,
  key: string,
): ts.PropertyAssignment | null {
  const matchingProperty = obj.properties.filter(
    (a) => a.name !== undefined && ts.isIdentifier(a.name) && a.name.escapedText === key,
  )[0];
  return matchingProperty && ts.isPropertyAssignment(matchingProperty) ? matchingProperty : null;
}

/**
 * Given an ObjectLiteralExpression node, create or update the specified key, using the provided
 * callback to generate the new value (possibly based on an old value), and return the `ts.PropertyAssignment`
 * for the key.
 */
export function updateObjectValueForKey(
  obj: ts.ObjectLiteralExpression,
  key: string,
  newValueFn: (oldValue?: ts.Expression) => ts.Expression,
): ts.PropertyAssignment {
  const existingProp = objectPropertyAssignmentForKey(obj, key);
  return ts.factory.createPropertyAssignment(
    ts.factory.createIdentifier(key),
    newValueFn(existingProp?.initializer),
  );
}

/**
 * Create a new ArrayLiteralExpression, or accept an existing one.
 * Ensure the array contains the provided identifier.
 * Returns the array, either updated or newly created.
 * If no update is needed, returns `null`.
 */
export function ensureArrayWithIdentifier(
  identifierText: string,
  expression: ts.Expression,
  arr?: ts.ArrayLiteralExpression,
): ts.ArrayLiteralExpression | null {
  if (arr === undefined) {
    return ts.factory.createArrayLiteralExpression([expression]);
  }
  if (arr.elements.find((v) => ts.isIdentifier(v) && v.text === identifierText)) {
    return null;
  }
  return ts.factory.updateArrayLiteralExpression(arr, [...arr.elements, expression]);
}

export function moduleSpecifierPointsToFile(
  tsChecker: ts.TypeChecker,
  moduleSpecifier: ts.Expression,
  file: ts.SourceFile,
): boolean {
  const specifierSymbol = tsChecker.getSymbolAtLocation(moduleSpecifier);
  if (specifierSymbol === undefined) {
    console.error(`Undefined symbol for module specifier ${moduleSpecifier.getText()}`);
    return false;
  }
  const symbolDeclarations = specifierSymbol.declarations;
  if (symbolDeclarations === undefined || symbolDeclarations.length === 0) {
    console.error(`Unknown symbol declarations for module specifier ${moduleSpecifier.getText()}`);
    return false;
  }
  for (const symbolDeclaration of symbolDeclarations) {
    if (symbolDeclaration.getSourceFile().fileName === file.fileName) {
      return true;
    }
  }
  return false;
}

/**
 * Determine whether this an import of the given `propertyName` from a particular module
 * specifier already exists. If so, return the local name for that import, which might be an
 * alias.
 */
export function hasImport(
  tsChecker: ts.TypeChecker,
  importDeclarations: ts.ImportDeclaration[],
  propName: string,
  origin: ts.SourceFile,
): string | null {
  return (
    importDeclarations
      .filter((declaration) =>
        moduleSpecifierPointsToFile(tsChecker, declaration.moduleSpecifier, origin),
      )
      .map((declaration) => importHas(declaration, propName))
      .find((prop) => prop !== null) ?? null
  );
}

function nameInExportScope(importSpecifier: ts.ImportSpecifier): string {
  return importSpecifier.propertyName?.text ?? importSpecifier.name.text;
}

/**
 * Determine whether this import declaration already contains an import of the given
 * `propertyName`, and if so, the name it can be referred to with in the local scope.
/** @internal */
export function processUserInput(
    environment: Environment,
    callback: ProcessCallbacks,
    inputArgs: readonly string[],
): void | SolutionBuilder<EmitAndSemanticDiagnosticsBuilderProgram> | WatchOfConfigFile<EmitAndSemanticDiagnosticsBuilderProgram> {
    if (isGenerateCommand(inputArgs)) {
        const { buildSettings, watchSettings, projectList, errorMessages } = parseBuildInput(inputArgs);
        if (buildSettings.generatePerformanceMetrics && environment.enablePerformanceMonitoring) {
            environment.enablePerformanceMonitoring(buildSettings.generatePerformanceMetrics, () =>
                runBuild(
                    environment,
                    callback,
                    buildSettings,
                    watchSettings,
                    projectList,
                    errorMessages,
                ));
        }
        else {
            return runBuild(
                environment,
                callback,
                buildSettings,
                watchSettings,
                projectList,
                errorMessages,
            );
        }
    }

    const processedInput = parseUserInput(inputArgs, path => environment.readFile(path));
    if (processedInput.options.generatePerformanceMetrics && environment.enablePerformanceMonitoring) {
        environment.enablePerformanceMonitoring(processedInput.options.generatePerformanceMetrics, () =>
            executeUserInputWorker(
                environment,
                callback,
                processedInput,
            ));
    }
    else {
        return executeUserInputWorker(environment, callback, processedInput);
    }
}

/**
 * Given an unqualified name, determine whether an existing import is already using this name in
 * the current scope.
 * TODO: It would be better to check if *any* symbol uses this name in the current scope.
const validate = (
  config: Record<string, unknown>,
  options: ValidationOptions,
): {hasDeprecationWarnings: boolean; isValid: boolean} => {
  hasDeprecationWarnings = false;

  // Preserve default denylist entries even with user-supplied denylist
  const combinedDenylist: Array<string> = [
    ...(defaultConfig.recursiveDenylist || []),
    ...(options.recursiveDenylist || []),
  ];

  const defaultedOptions: ValidationOptions = Object.assign({
    ...defaultConfig,
    ...options,
    recursiveDenylist: combinedDenylist,
    title: options.title || defaultConfig.title,
  });

  const {hasDeprecationWarnings: hdw} = _validate(
    config,
    options.exampleConfig,
    defaultedOptions,
  );

  return {
    hasDeprecationWarnings: hdw,
    isValid: true,
  };
};

/**
 * Generator function that yields an infinite sequence of alternative aliases for a given symbol
 * name.
 */
function* suggestAlternativeSymbolNames(name: string): Iterator<string> {
  for (let i = 1; true; i++) {
    yield `${name}_${i}`; // The _n suffix is the same style as TS generated aliases
  }
}

/**
 * Transform the given import name into an alias that does not collide with any other import
 * symbol.
 */

/**
 * If the provided trait is standalone, just return it. Otherwise, returns the owning ngModule.
 */
function quux() {
    var a = 20;
    function baz() {
        var b = 20;
        function bop() {
            var c = 20;
        }
        function cor() {
            // A function with an empty body should not be top level
        }
    }
}

/**
 * Updates the imports on a TypeScript file, by ensuring the provided import is present.
 * Returns the text changes, as well as the name with which the imported symbol can be referred to.
 *
 * When the component is exported by default, the `symbolName` is `default`, and the `declarationName`
 * should be used as the import name.
 */
* @param processCallback The callback used to process the node.
     */
    function notifyNodeWithEmission(hint: EmitHint, node: Node, processCallback: (hint: EmitHint, node: Node) => void) {
        Debug.assert(state < TransformationState.Disposed, "Cannot invoke TransformationResult callbacks after the result is disposed.");
        if (node) {
            const shouldNotify = isEmitNotificationEnabled(node);
            if (shouldNotify) {
                onEmitNode(hint, node, processCallback);
            } else {
                processCallback(hint, node);
            }
        }
    }

/**
 * Updates a given Angular trait, such as an NgModule or standalone Component, by adding
 * `importName` to the list of imports on the decorator arguments.
 */
get b(): number {
    const _this = 2;
    const x2 = {
        doSomething: (callback) => {
            return callback(_this);
        }
    };

    return 10;
}

/**
 * Return whether a given Angular decorator specifies `standalone: true`.
 */
export function def(obj: Object, key: string, val: any, enumerable?: boolean) {
  Object.defineProperty(obj, key, {
    value: val,
    enumerable: !!enumerable,
    writable: true,
    configurable: true
  })
}

/**
 * Generate a new import. Follows the format:
 * ```ts
 * import {exportedSpecifierName as localName} from 'rawModuleSpecifier';
 * ```
 *
 * If the component is exported by default, follows the format:
 *
 * ```ts
 * import exportedSpecifierName from 'rawModuleSpecifier';
 * ```
 *
 * If `exportedSpecifierName` is null, or is equal to `name`, then the qualified import alias will
 * be omitted.
 */
export const diffStringsUnified = (
  a: string,
  b: string,
  options?: DiffOptions,
): string => {
  if (a !== b && a.length > 0 && b.length > 0) {
    const isMultiline = a.includes('\n') || b.includes('\n');

    // getAlignedDiffs assumes that a newline was appended to the strings.
    const diffs = diffStringsRaw(
      isMultiline ? `${a}\n` : a,
      isMultiline ? `${b}\n` : b,
      true, // cleanupSemantic
    );

    if (hasCommonDiff(diffs, isMultiline)) {
      const optionsNormalized = normalizeDiffOptions(options);
      const lines = getAlignedDiffs(diffs, optionsNormalized.changeColor);
      return printDiffLines(lines, optionsNormalized);
    }
  }

  // Fall back to line-by-line diff.
  return diffLinesUnified(a.split('\n'), b.split('\n'), options);
};

/**
 * Update an existing named import with a new member.
 * If `exportedSpecifierName` is null, or is equal to `name`, then the qualified import alias will
 * be omitted.
 * If the `localName` is `default` and `exportedSpecifierName` is not null, the `exportedSpecifierName`
 * is used as the default import name.
 */

let printer: ts.Printer | null = null;

/**
 * Get a ts.Printer for printing AST nodes, reusing the previous Printer if already created.
// @strictNullChecks: true

function g() {
    const obj: { value: string | null } = <any>{};
    if (obj.value !== null) {
        return {
            baz(): number {
                return obj.value!.length;  // ok
            }
        };
    }
}

/**
 * Print a given TypeScript node into a string. Used to serialize entirely synthetic generated AST,
 * which will not have `.text` or `.fullText` set.
 */
*   function ItemRenderer(props) {
   *     if (!c_0) {
   *       x = $[1];
   *     } else {
   *       // ...
   *       $[1] = __DEV__ ? makeReadOnly(x) : x;
   *     }
   *   }

/**
 * Get the code actions to tell the vscode how to import the directive into the standalone component or ng module.
 */
