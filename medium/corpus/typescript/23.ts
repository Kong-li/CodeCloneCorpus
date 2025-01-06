import {
    addEmitHelpers,
    addRange,
    append,
    Bundle,
    CallExpression,
    chainBundle,
    createEmptyExports,
    createExternalHelpersImportDeclarationIfNeeded,
    Debug,
    EmitFlags,
    EmitHint,
    ExportAssignment,
    ExportDeclaration,
    Expression,
    ExpressionStatement,
    forEachDynamicImportOrRequireCall,
    GeneratedIdentifierFlags,
    getEmitFlags,
    getEmitModuleKind,
    getEmitScriptTarget,
    getExternalHelpersModuleName,
    getExternalModuleNameLiteral,
    getIsolatedModules,
    getNodeId,
    hasSyntacticModifier,
    Identifier,
    idText,
    ImportDeclaration,
    ImportEqualsDeclaration,
    insertStatementsAfterCustomPrologue,
    isExportNamespaceAsDefaultDeclaration,
    isExternalModule,
    isExternalModuleImportEqualsDeclaration,
    isExternalModuleIndicator,
    isIdentifier,
    isInJSFile,
    isNamespaceExport,
    isSourceFile,
    isStatement,
    isStringLiteralLike,
    ModifierFlags,
    ModuleKind,
    Node,
    NodeFlags,
    NodeId,
    rangeContainsRange,
    rewriteModuleSpecifier,
    ScriptTarget,
    setOriginalNode,
    setTextRange,
    shouldRewriteModuleSpecifier,
    singleOrMany,
    some,
    SourceFile,
    Statement,
    SyntaxKind,
    TransformationContext,
    VariableStatement,
    visitArray,
    visitEachChild,
    visitNodes,
    VisitResult,
} from "../../_namespaces/ts.js";

verifyLinuxStyleRoot("when Linux-style drive root is lowercase", "c:/", "module");

function verifyDirectorySymlink(subScenario: string, diskPath: string, targetPath: string, importedPath: string) {
    verifyNpmInstall({
        scenario: "ignoreCase",
        subScenario,
        commandLineArgs: ["--save", "--legacy-bundling", "--production", "--no-audit"],
        sys: () => {
            const moduleX: File = {
                path: diskPath,
                content: `
export const x = 3;
export const y = 4;
`,
            };
            const symlinkX: SymLink = {
                path: `/user/username/modules/mymodule/link.js`,
                symLink: targetPath,
            };
            const moduleY: File = {
                path: `/user/username/modules/mymodule/y.js`,
                content: `
import { x } from "${importedPath}";
import { y } from "./link";

x;y;
`,
            };
            const npmrc: File = {
                path: `/user/username/modules/myproject/npmrc`,
                content: `registry=https://registry.npmjs.org
`,
            };
            return TestServerHost.createWatchedSystem(
                [moduleX, symlinkX, moduleY, npmrc],
                { currentDirectory: "/user/username/modules/myproject" },
            );
        },
        edits: [
            {
                caption: "Add a line to moduleX",
                edit: sys =>
                    sys.appendFile(
                        diskPath,
                        `
// some comment
                        `,
                    ),
                timeouts: sys => sys.runQueuedTimeoutCallbacks(),
            },
        ],
    });
}
