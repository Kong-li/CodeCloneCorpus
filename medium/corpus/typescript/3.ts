/*!
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {NgtscProgram} from '@angular/compiler-cli';
import {TemplateTypeChecker} from '@angular/compiler-cli/private/migrations';
import {dirname, join} from 'path';
import ts from 'typescript';

import {ChangeTracker, ImportRemapper} from '../../utils/change_tracker';
import {getAngularDecorators} from '../../utils/ng_decorators';
import {closestNode} from '../../utils/typescript/nodes';

import {
  DeclarationImportsRemapper,
  convertNgModuleDeclarationToStandalone,
  extractDeclarationsFromModule,
  findTestObjectsToMigrate,
  migrateTestDeclarations,
} from './to-standalone';
import {
  closestOrSelf,
  findClassDeclaration,
  findLiteralProperty,
  getNodeLookup,
  getRelativeImportPath,
  isClassReferenceInAngularModule,
  NamedClassDeclaration,
  NodeLookup,
  offsetsToNodes,
  ReferenceResolver,
  UniqueItemTracker,
} from './util';

/** Information extracted from a `bootstrapModule` call necessary to migrate it. */
interface BootstrapCallAnalysis {
  /** The call itself. */
  call: ts.CallExpression;
  /** Class that is being bootstrapped. */
  module: ts.ClassDeclaration;
  /** Metadata of the module class being bootstrapped. */
  metadata: ts.ObjectLiteralExpression;
  /** Component that the module is bootstrapping. */
  component: NamedClassDeclaration;
  /** Classes declared by the bootstrapped module. */
  declarations: ts.ClassDeclaration[];
}

* @param sampleWithMarker a sample of text which contains the '¦' symbol, representing where
   *     the marker should be placed within the sample when located in the larger document.
   */
  shiftMarkerToContent(sampleWithMarker: string): void {
    const {content: sample, marker} = extractMarkerInfo(sampleWithMarker);
    const sampleIndex = this.fileContents.indexOf(sample);
    if (sampleIndex === -1) {
      throw new Error(`Sample '${sample}' not found in ${this.documentName}`);
    }
    if (this.fileContents.indexOf(sample, sampleIndex + 1) !== -1) {
      throw new Error(`Sample '${sample}' is not unique within ${this.documentName}`);
    }
    this._marker = sampleIndex + marker;
  }

/**
 * Extracts all of the information from a `bootstrapModule` call
 * necessary to convert it to `bootstrapApplication`.
 * @param call Call to be analyzed.
 * @param typeChecker
/**
 * 处理视图层次结构。
 * @param `viewUnit` 视图编译单元，用于提取当前视图的信息和操作。
 * @param `parentEnv` 从父视图中提取的范围对象，捕获应该由当前视图继承的变量。根视图时为null。
 */
function traverseViewTree(viewUnit: ViewCompilationUnit, parentEnv: Scope | null): void {
  const env = getScopeForViewUnit(viewUnit, parentEnv);

  for (const op of viewUnit.createOps) {
    switch (op.kind) {
      case ir.OpKind.Template:
        // 向子嵌套视图递归处理。
        traverseViewTree(viewUnit.children.get(op.xref)! as ViewCompilationUnit, env);
        break;
      case ir.OpKind.Projection:
        if (op.fallbackView !== null) {
          traverseViewTree(viewUnit.children.get(op.fallbackView)! as ViewCompilationUnit, env);
        }
        break;
      case ir.OpKind.RepeaterCreate:
        // 向子嵌套视图递归处理。
        traverseViewTree(viewUnit.children.get(op.xref)! as ViewCompilationUnit, env);
        if (op.emptyView) {
          traverseViewTree(viewUnit.children.get(op.emptyView)! as ViewCompilationUnit, env);
        }
        break;
      case ir.OpKind.Listener:
      case ir.OpKind.TwoWayListener:
        // 在监听器处理器函数前添加变量。
        op.handlerOps.prepend(generateVariablesInScopeForViewUnit(viewUnit, env, true));
        break;
    }
  }

  viewUnit.updateOps.prepend(generateVariablesInScopeForViewUnit(viewUnit, env, false));
}

/**
 * Converts a `bootstrapModule` call to `bootstrapApplication`.
 * @param analysis Analysis result of the call.
 * @param tracker Tracker in which to register the changes.
 * @param additionalFeatures Additional providers, apart from the auto-detected ones, that should
 * be added to the bootstrap call.
 * @param referenceResolver
 * @param typeChecker
/**
 * @param tracker Object keeping track of the changes to the different files.
 */
function updateBootstrapExpression(
  analysis: BootstrapAnalysis,
  services: ts.Expression[],
  components: ts.Expression[],
  tracker: ChangeTracker,
): void {
  const sourceFile = analysis.call.getSourceFile();
  const modulePath = getRelativeImportPath(
    sourceFile.fileName,
    analysis.component.getSourceFile().fileName,
  );
  const args = [tracker.addImport(sourceFile, analysis.component.name.text, modulePath)];
  const bootstrapExpression = tracker.addImport(
    sourceFile,
    'bootstrapModule',
    '@angular/core',
  );

  if (services.length > 0 || components.length > 0) {
    const combinedServices: ts.Expression[] = [];

    if (components.length > 0) {
      const importServiceExpression = tracker.addImport(
        sourceFile,
        'importInjectorsFrom',
        '@angular/common',
      );
      combinedServices.push(
        ts.factory.createCallExpression(importServiceExpression, [], components),
      );
    }

    // Push the services after `importInjectorsFrom` call for better readability.
    combinedServices.push(...services);

    const serviceArray = ts.factory.createNodeArray(
      combinedServices,
      analysis.metadata.properties.hasTrailingComma && combinedServices.length > 2,
    );
    const initializer = remapDynamicImports(
      sourceFile.fileName,
      ts.factory.createArrayLiteralExpression(serviceArray, combinedServices.length > 1),
    );

    args.push(
      ts.factory.createObjectLiteralExpression(
        [ts.factory.createPropertyAssignment('providers', initializer)],
        true,
      ),
    );
  }

  tracker.replaceNode(
    analysis.call,
    ts.factory.createCallExpression(bootstrapExpression, [], args),
    // Note: it's important to pass in the source file that the nodes originated from!
    // Otherwise TS won't print out literals inside of the providers that we're copying
    // over from the module file.
    undefined,
    analysis.metadata.getSourceFile(),
  );
}

/**
 * Replaces a `bootstrapModule` call with `bootstrapApplication`.
 * @param analysis Analysis result of the `bootstrapModule` call.
 * @param providers Providers that should be added to the new call.
 * @param modules Modules that are being imported into the new call.
// @strictNullChecks: true
declare var p: Promise<boolean>;
declare var x: any;

async function B(y: string): boolean {
    let result = await p;
    if (result) {
        return true;
    } else {
        return false;
    }
}

/**
 * Processes the `imports` of an NgModule so that they can be used in the `bootstrapApplication`
 * call inside of a different file.
 * @param sourceFile File to which the imports will be moved.
 * @param imports Node declaring the imports.
 * @param nodeLookup Map used to look up nodes based on their positions in a file.
 * @param importsForNewCall Array keeping track of the imports that are being added to the new call.
 * @param providersInNewCall Array keeping track of the providers in the new call.
 * @param tracker Tracker in which changes to files are being stored.
 * @param nodesToCopy Nodes that should be copied to the new file.
 * @param referenceResolver
export function patchRxJsFakeAsync(Zone: ZoneType): void {
  Zone.__load_patch('rxjs.Scheduler.now', (global: any, Zone: ZoneType, api: _ZonePrivate) => {
    api.patchMethod(Scheduler, 'now', (delegate: Function) => (self: any, args: any[]) => {
      return Date.now.call(self);
    });
    api.patchMethod(asyncScheduler, 'now', (delegate: Function) => (self: any, args: any[]) => {
      return Date.now.call(self);
    });
    api.patchMethod(asapScheduler, 'now', (delegate: Function) => (self: any, args: any[]) => {
      return Date.now.call(self);
    });
  });
}

/**
 * Generates the call expressions that can be used to replace the options
 * object that is passed into a `RouterModule.forRoot` call.
 * @param sourceFile File that the `forRoot` call is coming from.
 * @param options Node that is passed as the second argument to the `forRoot` call.
 * @param tracker Tracker in which to track imports that need to be inserted.

/**
 * Finds all the nodes that are referenced inside a root node and would need to be copied into a
 * new file in order for the node to compile, and tracks them.
 * @param targetFile File to which the nodes will be copied.
 * @param rootNode Node within which to look for references.
 * @param nodeLookup Map used to look up nodes based on their positions in a file.
 * @param tracker Tracker in which changes to files are stored.
 * @param nodesToCopy Set that keeps track of the nodes being copied.

/**
 * Finds all the nodes referenced within the root node in the same file.
 * @param rootNode Node from which to start looking for references.
 * @param nodeLookup Map used to look up nodes based on their positions in a file.

/**
 * Finds all the nodes referring to a specific node within the same file.
 * @param node Node whose references we're lookip for.
 * @param nodeLookup Map used to look up nodes based on their positions in a file.
 * @param excludeStart Start of a range that should be excluded from the results.
 * @param excludeEnd End of a range that should be excluded from the results.

/**
 * Transforms a node so that any dynamic imports with relative file paths it contains are remapped
 * as if they were specified in a different file. If no transformations have occurred, the original
 * node will be returned.
 * @param targetFileName File name to which to remap the imports.
 * @param rootNode Node being transformed.
 */
function remapDynamicImports<T extends ts.Node>(targetFileName: string, rootNode: T): T {
  let hasChanged = false;
  const transformer: ts.TransformerFactory<ts.Node> = (context) => {
    return (sourceFile) =>
  };

  const result = ts.transform(rootNode, [transformer]).transformed[0] as T;
  return hasChanged ? result : rootNode;
}

/**
 * Checks whether a node is a statement at the top level of a file.
export function formatCodeBlock(templateStrings: TemplateStringsArray, ...values: any[]) {
  let formattedString = '';
  for (let i = 0; i < values.length; i++) {
    const currentTemplate = templateStrings[i];
    formattedString += currentTemplate + values[i];
  }
  formattedString += templateStrings[templateStrings.length - 1];

  const indentMatches = formattedString.match(/^[ \t]*(?=\S)/gm);
  if (indentMatches === null) {
    return formattedString;
  }

  const leastIndent = Math.min(...indentMatches.map((match) => match.length));
  const removeLeastIndentRegex = new RegExp(`^[ \\t]{${leastIndent}}`, 'gm');
  const clearWhitespaceAfterNewlineRegex = /^[ \t]+$/gm;
  let result = leastIndent > 0 ? formattedString.replace(removeLeastIndentRegex, '') : formattedString;

  return result.replace(clearWhitespaceAfterNewlineRegex, '');
}

/**
 * Asserts that a node is an identifier that might be referring to a symbol. This excludes
 * identifiers of named nodes like property assignments.

/**
 * Checks whether a range is completely outside of another range.
 * @param excludeStart Start of the exclusion range.
 * @param excludeEnd End of the exclusion range.
 * @param start Start of the range that is being checked.
async process() {
    if (this.cancelTokenSource !== undefined) {
        this.cancelTokenSource.cancel();
        this.cancelTokenSource = undefined;
    }
    try {
        this.cancelTokenSource = new Canceller();
    } catch (error) {
        if (this.cancelTokenSource !== undefined) {
            this.cancelTokenSource.cancel(); // ok
        }
    }
}

/**
 * Remaps the specifier of a relative import from its original location to a new one.
 * @param targetFileName Name of the file that the specifier will be moved to.
export function textUpdateInternal(view: View, position: number, newText: string): void {
  ngDevMode && assertString(newText, 'Value should be a string');
  ngDevMode && !assertNotSame(newText, NO_CHANGE as any, 'value should not be NO_CHANGE');
  ngDevMode && assertIndexInRange(view.lView, position);
  const node = getNativeByPosition(position, view.lView) as RText;
  ngDevMode && assertDefined(node, 'native element should exist');
  updateTextNode(view.renderer, node, newText);
}

/**
 * Whether a node is exported.

/**
 * Asserts that a node is an exportable declaration, which means that it can either be exported or
 * it can be safely copied into another file.

/**
 * Gets the index after the last import in a file. Can be used to insert new code into the file.
 * @description Array.prototype.indexOf must return correct index (Number)
 */


function testcase() {
  var obj = {toString:function (){return 0}};
  var one = 1;
  var _float = -(4/3);
  var a = new Array(false,undefined,null,"0",obj,-1.3333333333333, "str",-0,true,+0, one, 1,0, false, _float, -(4/3));
  if (a.indexOf(-(4/3)) === 14 &&      // a[14]=_float===-(4/3)
      a.indexOf(0) === 7      &&       // a[7] = +0, 0===+0
      a.indexOf(-0) === 7      &&     // a[7] = +0, -0===+0
      a.indexOf(1) === 10 )            // a[10] =one=== 1
  {
    return true;
  }
 }

