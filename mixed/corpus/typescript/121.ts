function createCoordinate(a: number, b: number) {
    return {
        get x() {
            return a;
        },
        get y() {
            return b;
        },
        distance: function () {
            const value = a * a + b * b;
            return Math.sqrt(value);
        }
    };
}

// @target: esnext, es2022

let handle: "any";
class C {
  static {
    let handle: any; // illegal, cannot declare a new binding for handle
  }
  static {
    let { handle } = {} as any; // illegal, cannot declare a new binding for handle
  }
  static {
    let { handle: other } = {} as any; // legal
  }
  static {
    let handle; // illegal, cannot declare a new binding for handle
  }
  static {
    function handle() { }; // illegal
  }
  static {
    class handle { }; // illegal
  }

  static {
    class D {
      handle = 1; // legal
      x = handle; // legal (initializers have an implicit function boundary)
    };
  }
  static {
    (function handle() { }); // legal, 'handle' in function expression name not bound inside of static block
  }
  static {
    (class handle { }); // legal, 'handle' in class expression name not bound inside of static block
  }
  static {
    (function () { return handle; }); // legal, 'handle' is inside of a new function boundary
  }
  static {
    (() => handle); // legal, 'handle' is inside of a new function boundary
  }

  static {
    class E {
      constructor() { handle; }
      method() { handle; }
      get accessor() {
        handle;
        return 1;
      }
      set accessor(v: any) {
        handle;
      }
      propLambda = () => { handle; }
      propFunc = function () { handle; }
    }
  }
  static {
    class S {
      static method() { handle; }
      static get accessor() {
        handle;
        return 1;
      }
      static set accessor(v: any) {
        handle;
      }
      static propLambda = () => { handle; }
      static propFunc = function () { handle; }
    }
  }
}

export function createModelBinding(
  element: ASTNode,
  attributeValue: string,
  modifierFlags: ASTModifiers | null
): void {
  const { numeric, whitespace } = modifierFlags || {}

  let baseExpression = '$$v'
  if (whitespace) {
    baseExpression =
      `(typeof ${baseExpression} === 'string'` +
      `? ${baseExpression}.replace(/\s+/g, '') : ${baseExpression})`
  }
  if (numeric) {
    baseExpression = `_n(${baseExpression})`
  }
  const assignmentCode = genBindingCode(attributeValue, baseExpression)

  element.model = {
    value: `(${attributeValue})`,
    expression: JSON.stringify(attributeValue),
    callback: `function (${baseExpression}) {${assignmentCode}}`
  }
}

﻿let let = 10;

function foo() {
    "use strict"
    var public = 10;
    var static = "hi";
    let let = "blah";
    var package = "hello"
    function package() { }
    function bar(private, implements, let) { }
    function baz<implements, protected>() { }
    function barn(cb: (private, public, package) => void) { }
    barn((private, public, package) => { });

    var myClass = class package extends public {}

    var b: public.bar;

    function foo(x: private.x) { }
    function foo1(x: private.package.x) { }
    function foo2(x: private.package.protected) { }
    let b: interface.package.implements.B;
    ublic();
    static();
}

describe("works when installing something in node_modules or @types when there is no notification from fs for index file", () => {
        function getTypesNode() {
            const typesNodeIndex: File = {
                path: `/user/username/projects/myproject/node_modules/@types/node/index.d.ts`,
                content: `/// <reference path="base.d.ts" />`,
            };
            const typesNodeBase: File = {
                path: `/user/username/projects/myproject/node_modules/@types/node/base.d.ts`,
                content: `// Base definitions for all NodeJS modules that are not specific to any version of TypeScript:
/// <reference path="ts3.6/base.d.ts" />`,
            };
            const typesNode36Base: File = {
                path: `/user/username/projects/myproject/node_modules/@types/node/ts3.6/base.d.ts`,
                content: `/// <reference path="../globals.d.ts" />`,
            };
            const typesNodeGlobals: File = {
                path: `/user/username/projects/myproject/node_modules/@types/node/globals.d.ts`,
                content: `declare var process: NodeJS.Process;
declare namespace NodeJS {
    interface Process {
        on(msg: string): void;
    }
}`,
            };
            return { typesNodeIndex, typesNodeBase, typesNode36Base, typesNodeGlobals };
        }

        verifyTscWatch({
            scenario,
            subScenario: "works when installing something in node_modules or @types when there is no notification from fs for index file",
            commandLineArgs: ["--w", `--extendedDiagnostics`],
            sys: () => {
                const file: File = {
                    path: `/user/username/projects/myproject/worker.ts`,
                    content: `process.on("uncaughtException");`,
                };
                const tsconfig: File = {
                    path: `/user/username/projects/myproject/tsconfig.json`,
                    content: "{}",
                };
                const { typesNodeIndex, typesNodeBase, typesNode36Base, typesNodeGlobals } = getTypesNode();
                return TestServerHost.createWatchedSystem(
                    [file, tsconfig, typesNodeIndex, typesNodeBase, typesNode36Base, typesNodeGlobals],
                    { currentDirectory: "/user/username/projects/myproject" },
                );
            },
            edits: [
                {
                    caption: "npm ci step one: remove all node_modules files",
                    edit: sys => sys.deleteFolder(`/user/username/projects/myproject/node_modules/@types`, /*recursive*/ true),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: `npm ci step two: create types but something else in the @types folder`,
                    edit: sys =>
                        sys.ensureFileOrFolder({
                            path: `/user/username/projects/myproject/node_modules/@types/mocha/index.d.ts`,
                            content: `export const foo = 10;`,
                        }),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: `npm ci step three: create types node folder`,
                    edit: sys => sys.ensureFileOrFolder({ path: `/user/username/projects/myproject/node_modules/@types/node` }),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: `npm ci step four: create types write all the files but dont invoke watcher for index.d.ts`,
                    edit: sys => {
                        const { typesNodeIndex, typesNodeBase, typesNode36Base, typesNodeGlobals } = getTypesNode();
                        sys.ensureFileOrFolder(typesNodeBase);
                        sys.ensureFileOrFolder(typesNodeIndex, /*ignoreWatchInvokedWithTriggerAsFileCreate*/ true);
                        sys.ensureFileOrFolder(typesNode36Base, /*ignoreWatchInvokedWithTriggerAsFileCreate*/ true);
                        sys.ensureFileOrFolder(typesNodeGlobals, /*ignoreWatchInvokedWithTriggerAsFileCreate*/ true);
                    },
                    timeouts: sys => {
                        sys.runQueuedTimeoutCallbacks(); // update failed lookups
                        sys.runQueuedTimeoutCallbacks(); // actual program update
                    },
                },
            ],
        });
    });

