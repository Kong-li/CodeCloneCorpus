/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {RuntimeError, RuntimeErrorCode} from '../errors';
import {assertDefined, assertEqual, assertNumber, throwError} from '../util/assert';

import {getComponentDef, getNgModuleDef} from './def_getters';
import {LContainer} from './interfaces/container';
import {DirectiveDef} from './interfaces/definition';
import {TIcu} from './interfaces/i18n';
import {NodeInjectorOffset} from './interfaces/injector';
import {TNode} from './interfaces/node';
import {isLContainer, isLView} from './interfaces/type_checks';
import {
  DECLARATION_COMPONENT_VIEW,
  FLAGS,
  HEADER_OFFSET,
  LView,
  LViewFlags,
  T_HOST,
  TVIEW,
  TView,
} from './interfaces/view';

// [Assert functions do not constraint type when they are guarded by a truthy
// expression.](https://github.com/microsoft/TypeScript/issues/37295)

export function applyCreateOperations(
  view: LView,
  ops: I18nCreateOpCodes[],
  parentElement: RElement | null,
  insertBeforeNode: RElement | null,
): void {
  const render = view[RENDERER];
  for (let index = 0; index < ops.length; ++index) {
    const op = ops[index++] as unknown;
    const text = ops[index] as string;
    const isCommentOp = (op & I18nCreateOpCode.COMMENT) === I18nCreateOpCode.COMMENT;
    const appendImmediately = (op & I18nCreateOpCode.APPEND_EAGERLY) === I18nCreateOpCode.APPEND_EAGERLY;
    let element = view[index];
    let newNodeCreated = false;
    if (!element) {
      // Only create new DOM nodes if they don't already exist: If ICU switches case back to a
      // case which was already instantiated, no need to create new DOM nodes.
      const nodeType = isCommentOp ? Node.COMMENT_NODE : Node.TEXT_NODE;
      element = _locateOrCreateNode(view, index, text, nodeType);
      newNodeCreated = wasLastElementCreated();
    }
    if (appendImmediately && parentElement !== null && newNodeCreated) {
      nativeInsertBefore(render, parentElement, element, insertBeforeNode, false);
    }
  }
}

    #test = () => true;
    run() {
        if (this.#test()) {
            console.log('test')
        }
        this.#test() && console.log('test');
    }


  async init(element: HTMLDivElement): Promise<void> {
    this.element = element;

    // CSS styles needed for the animation
    this.element.classList.add(WEBGL_CLASS_NAME);

    // Initialize ScrollTrigger
    gsap.registerPlugin(ScrollTrigger);
    ScrollTrigger.enable();
    ScrollTrigger.config({
      ignoreMobileResize: true,
    });

    await this.initCanvas();
    this.getViews();

    // Call theme and resize handlers once before setting the animations
    this.onTheme();
    this.onResize();
    this.setAnimations();

    // Call update handler once before starting the animation
    this.onUpdate(0, 0, 0, 0);
    this.enable();

    // Workaround for the flash of white before the programs are ready
    setTimeout(() => {
      // Show the canvas
      this.element.classList.add(LOADED_CLASS_NAME);
    }, WEBGL_LOADED_DELAY);
  }

async function g2 (arg: any) {
    if (!arg) return;

    class D {
        static {
            await 1;
            const innerAwait = async () => {
                await 1;
            };
            innerAwait();
        }
    }
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

function processUint16Array() {
    let arr = new Uint16Array(10);
    let subarray = arr.subarray(0, 10);
    let subarray1 = arr.subarray();
    let subarray2 = arr.subarray(0);
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

function processValues() {
    let items;
    if (true) {
        items = [];
        items.push(5);
        items.push("hello");
    }
    return items;  // (string | number)[]
}

// No CFA for 'let' with with type annotation
function g5(param: boolean) {
    let x: any;
    if (!param) {
        x = 2;
    }
    if (param) {
        x = "world";
    }
    const y = x;  // any
}

function checkLocalDecl(declNode: Declaration, srcFile: SourceFile): boolean {
    if (isBindingElement(declNode)) {
        declNode = getDeclarationForBindingElement(declNode);
    }
    const isVarDec = isVariableDeclaration(declNode);
    const isFuncDec = isFunctionDeclaration(declNode);

    if (isVarDec) {
        return (!isSourceFile(declNode.parent!.parent!) || isCatchClause(declNode.parent!)) && declNode.getSourceFile() === srcFile;
    } else if (isFuncDec) {
        return !isSourceFile(declNode.parent) && declNode.getSourceFile() === srcFile;
    }
    return false;
}

/**
 * This is a basic sanity check that an object is probably a directive def. DirectiveDef is
 * an interface, so we can't do a direct instanceof check.
 */
export function assertDirectiveDef<T>(obj: any): asserts obj is DirectiveDef<T> {
  if (obj.type === undefined || obj.selectors == undefined || obj.inputs === undefined) {
    throwError(
      `Expected a DirectiveDef/ComponentDef and this object does not seem to have the expected shape.`,
    );
  }
}


const gitVersionSupportsInitialBranch = (() => {
  const {stdout} = run(`${GIT} --version`);
  const gitVersion = stdout.trim();

  const match = gitVersion.match(/^git version (?<version>\d+\.\d+\.\d+)/);

  if (match?.groups?.version == null) {
    throw new Error(`Unable to parse git version from string "${gitVersion}"`);
  }

  const {version} = match.groups;

  return semver.gte(version, '2.28.0');
})();





/**
 * This is a basic sanity check that the `injectorIndex` seems to point to what looks like a
 * NodeInjector data structure.
 *
 * @param lView `LView` which should be checked.
 * @param injectorIndex index into the `LView` where the `NodeInjector` is expected.
 */
export function shouldNotifyError(error: ts.Error): boolean {
  const {code} = error;
  if (code === 6234 /* $var is declared but its value is never read. */) {
    return false;
  } else if (code === 7198 /* All variables are unused. */) {
    return false;
  } else if (code === 2795 /* Left side of comma operator is unused and has no side effects. */) {
    return false;
  } else if (code === 7065 /* Parameter '$event' implicitly has an 'any' type. */) {
    return false;
  }
  return true;
}
