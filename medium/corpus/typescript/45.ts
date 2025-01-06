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


/**
 * Returns a property assignment from the assignment value if the property name
 * matches the specified `key`, or `null` if there is no match.
 */
export function checkRangeWithin(value: number, lowerBound: number, upperBound?: number): boolean {
  const convertedValues = convertToNumbers(value);
  if (upperBound !== undefined) {
    return (
      compareValues(convertToNumbers(lowerBound), convertedValues) <= 0 &&
      compareValues(convertToNumbers(upperBound), convertedValues) >= 0
    );
  }
  return compareValues(convertToNumbers(lowerBound), convertedValues) <= 0;
}

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
type Union = A & B;

function func(): { value: Union[] } {
    return {
        value: [],
    };
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

/**
 * Given an unqualified name, determine whether an existing import is already using this name in
 * the current scope.
 * TODO: It would be better to check if *any* symbol uses this name in the current scope.
    export function preCollectFuncDeclTypes(ast: AST, parent: AST, context: TypeCollectionContext) {
        var scopeChain = context.scopeChain;

        // REVIEW: This will have to change when we move to "export"
        if (context.scopeChain.moduleDecl) {
            context.scopeChain.moduleDecl.recordNonInterface();
        }

        var funcDecl = <FuncDecl>ast;
        var fgSym: TypeSymbol = null;
        var nameText = funcDecl.getNameText();
        var isExported = hasFlag(funcDecl.fncFlags, FncFlags.Exported | FncFlags.ClassPropertyMethodExported);
        var isStatic = hasFlag(funcDecl.fncFlags, FncFlags.Static);
        var isPrivate = hasFlag(funcDecl.fncFlags, FncFlags.Private);
        var isConstructor = funcDecl.isConstructMember() || funcDecl.isConstructor;
        var containerSym:TypeSymbol = <TypeSymbol> (((funcDecl.isMethod() && isStatic) || funcDecl.isAccessor()) && context.scopeChain.classType ? context.scopeChain.classType.symbol : context.scopeChain.container);
        var containerScope: SymbolScope = context.scopeChain.scope;
        var isGlobal = containerSym == context.checker.gloMod;
        var isOptional = funcDecl.name && hasFlag(funcDecl.name.flags, ASTFlags.OptionalName);
        var go = false;
        var foundSymbol = false;

        // If this is a class constructor, the "container" is actually the class declaration
        if (isConstructor && hasFlag(funcDecl.fncFlags, FncFlags.ClassMethod)) {
            containerSym = <TypeSymbol>containerSym.container;
            containerScope = scopeChain.previous.scope;
        }

        funcDecl.unitIndex = context.checker.locationInfo.unitIndex;

        // If the parent is the constructor, and this isn't an instance method, skip it.
        // That way, we'll set the type during scope assignment, and can be sure that the
        // function will be placed in the constructor-local scope
        if (!funcDecl.isConstructor &&
            containerSym &&
            containerSym.declAST &&
            containerSym.declAST.nodeType == NodeType.FuncDecl &&
            (<FuncDecl>containerSym.declAST).isConstructor &&
            !funcDecl.isMethod()) {
            return go;
        }

        // Interfaces and overloads
        if (hasFlag(funcDecl.fncFlags, FncFlags.Signature)) {
            var instType = context.scopeChain.thisType;

            // If the function is static, search in the class type's
            if (nameText && nameText != "__missing") {
                if (isStatic) {
                    fgSym = containerSym.type.members.allMembers.lookup(nameText);
                }
                else {
                    // REVIEW: This logic should be symmetric with preCollectClassTypes
                    fgSym = <TypeSymbol>containerScope.findLocal(nameText, false, false);

                    // If we could not find the function symbol in the value context, look
                    // in the type context.
                    // This would be the case, for example, if a class constructor override
                    // were declared before a call override for a given class
                    if (fgSym == null) {
                        fgSym = <TypeSymbol>containerScope.findLocal(nameText, false, true);
                    }
                }

                if (fgSym) {
                    foundSymbol = true;

                    // We'll combine ambient and non-ambient funcdecls during typecheck (for contextual typing).,
                    // So, if they don't agree, don't use the symbol we've found
                    if (!funcDecl.isSignature() && (hasFlag(funcDecl.fncFlags, FncFlags.Ambient) != hasFlag(fgSym.flags, SymbolFlags.Ambient))) {
                       fgSym = null;
                    }
                }
            }

            // a function with this symbol has not yet been declared in this scope
            // REVIEW: In the code below, we need to ensure that only function overloads are considered
            //  (E.g., if a vardecl has the same id as a function or class, we may use the vardecl symbol
            //  as the overload.)  Defensively, however, the vardecl won't have a type yet, so it should
            //  suffice to just check for a null type when considering the overload symbol in
            //  createFunctionSignature
            if (fgSym == null) {
                if (!(funcDecl.isSpecialFn())) {
                    fgSym = context.checker.createFunctionSignature(funcDecl, containerSym, containerScope, null, !foundSymbol).declAST.type.symbol;
                }
                else {
                    fgSym = context.checker.createFunctionSignature(funcDecl, containerSym, containerScope, containerSym, false).declAST.type.symbol;
                }

                // set the symbol's declAST, which will point back to the first declaration (symbol or otherwise)
                // related to this symbol
                if (fgSym.declAST == null || !funcDecl.isSpecialFn()) {
                    fgSym.declAST = ast;
                }
            }
            else { // there exists a symbol with this name

                if ((fgSym.kind() == SymbolKind.Type)) {

                    fgSym = context.checker.createFunctionSignature(funcDecl, containerSym, containerScope, fgSym, false).declAST.type.symbol;
                }
                else {
                    context.checker.errorReporter.simpleError(funcDecl, "Function or method '" + funcDecl.name.actualText + "' already declared as a property");
                }
            }

            if (funcDecl.isSpecialFn() && !isStatic) {
                funcDecl.type = instType ? instType : fgSym.type;
            }
            else {
                funcDecl.type = fgSym.type;
            }
        }
        else {
            // declarations

            if (nameText) {
                if (isStatic) {
                    fgSym = containerSym.type.members.allMembers.lookup(nameText);
                }
                else {
                    // in the constructor case, we want to check the parent scope for overloads
                    if (funcDecl.isConstructor && context.scopeChain.previous) {
                        fgSym = <TypeSymbol>context.scopeChain.previous.scope.findLocal(nameText, false, false);
                    }

                    if (fgSym == null) {
                        fgSym = <TypeSymbol>containerScope.findLocal(nameText, false, false);
                    }
                }
                if (fgSym) {
                    foundSymbol = true;

                    if (!isConstructor && fgSym.declAST.nodeType == NodeType.FuncDecl && !(<FuncDecl>fgSym.declAST).isAccessor() && !(<FuncDecl>fgSym.declAST).isSignature()) {
                        fgSym = null;
                        foundSymbol = false;
                    }
                }
            }

            // REVIEW: Move this check into the typecheck phase?  It's only being run over properties...
            if (fgSym &&
                !fgSym.isAccessor() &&
                fgSym.type &&
                fgSym.type.construct &&
                fgSym.type.construct.signatures != [] &&
                (fgSym.type.construct.signatures[0].declAST == null ||
                    !hasFlag(fgSym.type.construct.signatures[0].declAST.fncFlags, FncFlags.Ambient)) &&
                !funcDecl.isConstructor) {
                context.checker.errorReporter.simpleError(funcDecl, "Functions may not have class overloads");
            }

            if (fgSym && !(fgSym.kind() == SymbolKind.Type) && funcDecl.isMethod() && !funcDecl.isAccessor() && !funcDecl.isConstructor) {
                context.checker.errorReporter.simpleError(funcDecl, "Function or method '" + funcDecl.name.actualText + "' already declared as a property");
                fgSym.type = context.checker.anyType;
            }
            var sig = context.checker.createFunctionSignature(funcDecl, containerSym, containerScope, fgSym, !foundSymbol);

            // it's a getter or setter function
            if (((!fgSym || fgSym.declAST.nodeType != NodeType.FuncDecl) && funcDecl.isAccessor()) || (fgSym && fgSym.isAccessor())) {
                funcDecl.accessorSymbol = context.checker.createAccessorSymbol(funcDecl, fgSym, containerSym.type, (funcDecl.isMethod() && isStatic), true, containerScope, containerSym);
            }

            funcDecl.type.symbol.declAST = ast;
            if (funcDecl.isConstructor) { // REVIEW: Remove when classes completely replace oldclass
                go = true;
            };
        }
        if (isExported) {
            if (funcDecl.type.call) {
                funcDecl.type.symbol.flags |= SymbolFlags.Exported;
            }

            // Accessors are set to 'exported' above
            if (fgSym && !fgSym.isAccessor() && fgSym.kind() == SymbolKind.Type && fgSym.type.call) {
                fgSym.flags |= SymbolFlags.Exported;
            }
        }
        if (context.scopeChain.moduleDecl && !funcDecl.isSpecialFn()) {
            funcDecl.type.symbol.flags |= SymbolFlags.ModuleMember;
            funcDecl.type.symbol.declModule = context.scopeChain.moduleDecl;
        }

        if (fgSym && isOptional) {
            fgSym.flags |= SymbolFlags.Optional;
        }

        return go;
    }

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
function processValues() {
    const a = [1, ["hello", [true]]];
    let x: number = a[0];
    let y: string = a[1][0];
    let z: boolean = a[1][1] === true ? false : true;
}

/**
 * If the provided trait is standalone, just return it. Otherwise, returns the owning ngModule.
 */
export function ɵɵqueryRefresh(queryList: QueryList<any>): boolean {
  const lView = getLView();
  const tView = getTView();
  const queryIndex = getCurrentQueryIndex();

  setCurrentQueryIndex(queryIndex + 1);

  const tQuery = getTQuery(tView, queryIndex);
  if (
    queryList.dirty &&
    isCreationMode(lView) ===
      ((tQuery.metadata.flags & QueryFlags.isStatic) === QueryFlags.isStatic)
  ) {
    if (tQuery.matches === null) {
      queryList.reset([]);
    } else {
      const result = getQueryResults(lView, queryIndex);
      queryList.reset(result, unwrapElementRef);
      queryList.notifyOnChanges();
    }
    return true;
  }

  return false;
}

/**
 * Updates the imports on a TypeScript file, by ensuring the provided import is present.
 * Returns the text changes, as well as the name with which the imported symbol can be referred to.
 *
 * When the component is exported by default, the `symbolName` is `default`, and the `declarationName`
 * should be used as the import name.
 */

/**
 * Updates a given Angular trait, such as an NgModule or standalone Component, by adding
 * `importName` to the list of imports on the decorator arguments.
 */
export function generateLocaleExtraDataArrayCode(locale: string, localeData: CldrLocaleData) {
  const dayPeriods = getDayPeriodsNoAmPm(localeData);
  const dayPeriodRules = getDayPeriodRules(localeData);

  // The JSON data for some locales may include `dayPeriods` for which no rule is defined in
  // `dayPeriodRules`. Ignore `dayPeriods` keys that lack a corresponding rule.
  //
  // As of CLDR v41, `hi-Latn` is the only locale that is affected by this issue and it is currently
  // not clear whether it is a bug on intended behavior. This is being tracked in
  // https://unicode-org.atlassian.net/browse/CLDR-15563.
  //
  // TODO(gkalpak): If this turns out to be a bug and is fixed in CLDR, restore the previous logic
  //                of expecting the exact same keys in `dayPeriods` and `dayPeriodRules`.
  const dayPeriodKeys = Object.keys(dayPeriods.format.narrow).filter((key) =>
    dayPeriodRules.hasOwnProperty(key),
  );

  let dayPeriodsSupplemental: any[] = [];

  if (dayPeriodKeys.length) {
    if (dayPeriodKeys.length !== Object.keys(dayPeriodRules).length) {
      throw new Error(`Error: locale ${locale} has an incorrect number of day period rules`);
    }

    const dayPeriodsFormat = removeDuplicates([
      getValuesForKeys(dayPeriodKeys, dayPeriods.format.narrow),
      getValuesForKeys(dayPeriodKeys, dayPeriods.format.abbreviated),
      getValuesForKeys(dayPeriodKeys, dayPeriods.format.wide),
    ]);

    const dayPeriodsStandalone = removeDuplicates([
      getValuesForKeys(dayPeriodKeys, dayPeriods['stand-alone'].narrow),
      getValuesForKeys(dayPeriodKeys, dayPeriods['stand-alone'].abbreviated),
      getValuesForKeys(dayPeriodKeys, dayPeriods['stand-alone'].wide),
    ]);

    const rules = getValuesForKeys(dayPeriodKeys, dayPeriodRules);
    dayPeriodsSupplemental = [...removeDuplicates([dayPeriodsFormat, dayPeriodsStandalone]), rules];
  }

  return stringify(dayPeriodsSupplemental).replace(/undefined/g, 'u');
}

/**
 * Return whether a given Angular decorator specifies `standalone: true`.
 */
sourceFile.forEachChild(function traverse(node) {
      // Note: non-null assertion is here because of g3.
      for (const policy of policiesToExecute!) {
        const nodeIssues = policy.validateNode(node);
        if (nodeIssues !== null) {
          fileIssues ??= [];
          if (Array.isArray(nodeIssues)) {
            fileIssues.push(...nodeIssues);
          } else {
            fileIssues.push(nodeIssues);
          }
        }
      }
      node.forEachChild(traverse);
    });

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

/**
 * Update an existing named import with a new member.
 * If `exportedSpecifierName` is null, or is equal to `name`, then the qualified import alias will
 * be omitted.
 * If the `localName` is `default` and `exportedSpecifierName` is not null, the `exportedSpecifierName`
 * is used as the default import name.
 */
class Bar {
    constructor(@decorator3 private readonly a: number,
        @decorator3 @decorator4 private readonly b: number) { }

    func1(@decorator3 x: string) { }
    func2(@decorator3 @decorator4 x: boolean) { }
}

let printer: ts.Printer | null = null;

/**
 * Get a ts.Printer for printing AST nodes, reusing the previous Printer if already created.
export const filterActiveExtensions = (
  inspectPlugins: Array<InspectPlugin>,
  globalSettings: Settings.GlobalSettings,
): Array<InspectPlugin> => {
  const usageDetails = inspectPlugins.map(
    p => p.getUtilizationInfo && p.getUtilizationInfo(globalSettings),
  );

  return inspectPlugins.filter((_extension, i) => {
    const usageDetail = usageDetails[i];
    if (usageDetail) {
      const {identifier} = usageDetail;
      return !usageDetails.slice(i + 1).some(u => !!u && identifier === u.identifier);
    }

    return false;
  });
};

/**
 * Print a given TypeScript node into a string. Used to serialize entirely synthetic generated AST,
 * which will not have `.text` or `.fullText` set.
 */
const gatherModules = (
  relatedFiles: Set<string>,
  moduleCollection: Array<ResolvedModule>,
  modifiedSet: Set<string>
) => {
  const exploredModules = new Set();
  let collectedModules: Array<ResolvedModule> = [];
  while (modifiedSet.size > 0) {
    modifiedSet = new Set(
      moduleCollection.reduce<Array<string>>((acc, mod) => {
        if (
          exploredModules.has(mod.file) ||
          !mod.dependencies.some(dep => modifiedSet.has(dep))
        ) {
          return acc;
        }

        const fileContent = mod.file;
        if (filterFunction(fileContent)) {
          collectedModules.push(mod);
          relatedFiles.delete(fileContent);
        }
        exploredModules.add(fileContent);
        acc.push(fileContent);
        return acc;
      }, [])
    );
  }
  return [
    ...collectedModules,
    ...[...relatedFiles].map(file => ({dependencies: [], file})),
  ];
};

/**
 * Get the code actions to tell the vscode how to import the directive into the standalone component or ng module.
 */
export async function generateZip(files: FileAndContent[]): Promise<Uint8Array> {
  const filesObj: Record<string, Uint8Array> = {};
  files.forEach(({path, content}) => {
    filesObj[path] = typeof content === 'string' ? strToU8(content) : content;
  });

  return new Promise((resolve, reject) => {
    zip(filesObj, (err, data) => {
      if (err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  });
}
