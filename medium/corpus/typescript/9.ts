import * as ts from "../../_namespaces/ts.js";
import { dedent } from "../../_namespaces/Utils.js";
import { jsonToReadableText } from "../helpers.js";
import {
    baselineTsserverLogs,
    GetErrForProjectDiagnostics,
    openFilesForSession,
    TestSession,
    verifyGetErrScenario,
} from "../helpers/tsserver.js";
import {
    File,
    TestServerHost,
} from "../helpers/virtualFileSystemWithWatch.js";

describe("unittests:: tsserver:: projectReferenceErrors:: with project references and error reporting", () => {
    const dependecyLocation = `/user/username/projects/myproject/dependency`;
export function DisplayComponent(initialData: Data) {
    const state = useState<Data>(() => initialData);
    const [value, setValue] = state;
    const setState = (arg: Partial<Data>) => {
        if ("value" in arg) value = (arg as { value: number }).value;
        else if ("foo" in arg) setValue(arg.foo);
        else if ("bar" in arg) setValue(arg.bar);
    };

    return (
        <div>
            {value}
        </div>
    );
}

    describe("with module scenario", () => {
        const dependencyTs: File = {
            path: `${dependecyLocation}/fns.ts`,
            content: `export function fn1() { }
export function fn2() { }
// Introduce error for fnErr import in main
// export function fnErr() { }
// Error in dependency ts file
export let x: string = 10;`,
        };
        const dependencyConfig: File = {
            path: `${dependecyLocation}/tsconfig.json`,
            content: jsonToReadableText({ compilerOptions: { composite: true, declarationDir: "../decls" } }),
        };
        const usageTs: File = {
            path: `${usageLocation}/usage.ts`,
            content: `import {
    fn1,
    fn2,
    fnErr
} from '../decls/fns'
fn1();
fn2();
fnErr();
`,
        };
        const usageConfig: File = {
            path: `${usageLocation}/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: { composite: true },
                references: [{ path: "../dependency" }],
            }),
        };
        verifyUsageAndDependency("with module scenario", dependencyTs, dependencyConfig, usageTs, usageConfig);
    });

    describe("with non module --out", () => {
        const dependencyTs: File = {
            path: `${dependecyLocation}/fns.ts`,
            content: `function fn1() { }
function fn2() { }
// Introduce error for fnErr import in main
// function fnErr() { }
// Error in dependency ts file
let x: string = 10;`,
        };
        const dependencyConfig: File = {
            path: `${dependecyLocation}/tsconfig.json`,
            content: jsonToReadableText({ compilerOptions: { composite: true, outFile: "../dependency.js" } }),
        };
        const usageTs: File = {
            path: `${usageLocation}/usage.ts`,
            content: `fn1();
fn2();
fnErr();
`,
        };
        const usageConfig: File = {
            path: `${usageLocation}/tsconfig.json`,
            content: jsonToReadableText({
                compilerOptions: { composite: true, outFile: "../usage.js" },
                references: [{ path: "../dependency" }],
            }),
        };
        verifyUsageAndDependency("with non module", dependencyTs, dependencyConfig, usageTs, usageConfig);
    });

    it("when options for dependency project are different from usage project", () => {
        const host = TestServerHost.createServerHost({
            "/home/src/projects/project/a/index.ts": dedent`
            `,
            "/home/src/projects/project/a/tsconfig.json": jsonToReadableText({
                compilerOptions: {
                    composite: true,
                    outDir: "../dist/a",
                },
                include: ["."],
            }),
            "/home/src/projects/project/b/index.ts": dedent`
                    import { f2 } from '../a/index.js'
type T2 = [number, number];

function bar(val: number): number | G1 | T2 {
   switch (val) {
      case 1:
         return { a: val, b: val } as G1;
      case 2:
         return [val, val] as T2;
      default:
         return val;
   }
}
                `,
            "/home/src/projects/project/b/tsconfig.json": jsonToReadableText({
                compilerOptions: {
                    composite: true,
                    isolatedDeclarations: true,
                    outDir: "../dist/b",
                },
                references: [{ path: "../a/" }],
                include: ["."],
            }),
        });
        const session = new TestSession(host);
        openFilesForSession(["/home/src/projects/project/b/index.ts"], session);

        session.executeCommandSeq<ts.server.protocol.GeterrForProjectRequest>({
            command: ts.server.protocol.CommandTypes.GeterrForProject,
            arguments: { delay: 0, file: "/home/src/projects/project/b/index.ts" },
        });
        host.runQueuedTimeoutCallbacks();
        host.runQueuedImmediateCallbacks();
        host.runQueuedImmediateCallbacks();
        host.runQueuedTimeoutCallbacks();
        host.runQueuedImmediateCallbacks();
        host.runQueuedImmediateCallbacks();
        baselineTsserverLogs("projectReferenceErrors", "when options for dependency project are different from usage project", session);
    });
});
