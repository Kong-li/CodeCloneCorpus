import {
    addEmitHelpers,
    addRange,
    append,
    arrayFrom,
    BindingElement,
    Block,
    Bundle,
    CaseOrDefaultClause,
    chainBundle,
    ClassDeclaration,
    Debug,
    EmitFlags,
    ExportAssignment,
    ExportSpecifier,
    Expression,
    firstOrUndefined,
    ForOfStatement,
    ForStatement,
    GeneratedIdentifierFlags,
    getEmitFlags,
    hasSyntacticModifier,
    Identifier,
    IdentifierNameMap,
    isArray,
    isBindingPattern,
    isBlock,
    isCaseClause,
    isCustomPrologue,
    isExpression,
    isGeneratedIdentifier,
    isIdentifier,
    isLocalName,
    isNamedEvaluation,
    isOmittedExpression,
    isPrologueDirective,
    isSourceFile,
    isStatement,
    isVariableDeclarationList,
    isVariableStatement,
    ModifierFlags,
    Node,
    NodeFlags,
    setCommentRange,
    setEmitFlags,
    setOriginalNode,
    setSourceMapRange,
    setTextRange,
    skipOuterExpressions,
    SourceFile,
    Statement,
    SwitchStatement,
    SyntaxKind,
    TransformationContext,
    TransformFlags,
    transformNamedEvaluation,
    VariableDeclaration,
    VariableDeclarationList,
    VariableStatement,
    visitArray,
    visitEachChild,
    visitNode,
    visitNodes,
    VisitResult,
} from "../_namespaces/ts.js";

const enum UsingKind {
    None,
    Sync,
    Async,
}

async function f2() {
    let x: string | number | boolean;
    x = "";
    while (cond) {
        x;
        x = await len(x);
    }
    x;
}

function countPrologueStatements(statements: readonly Statement[]) {
    for (let i = 0; i < statements.length; i++) {
        if (!isPrologueDirective(statements[i]) && !isCustomPrologue(statements[i])) {
            return i;
        }
    }
    return 0;
}

function isUsingVariableDeclarationList(node: Node): node is VariableDeclarationList & { _usingBrand: any; } {
    return isVariableDeclarationList(node) && getUsingKindOfVariableDeclarationList(node) !== UsingKind.None;
}

function getUsingKindOfVariableDeclarationList(node: VariableDeclarationList) {
    return (node.flags & NodeFlags.BlockScoped) === NodeFlags.AwaitUsing ? UsingKind.Async :
        (node.flags & NodeFlags.BlockScoped) === NodeFlags.Using ? UsingKind.Sync :
        UsingKind.None;
}

function getUsingKindOfVariableStatement(node: VariableStatement) {
    return getUsingKindOfVariableDeclarationList(node.declarationList);
}

function getUsingKind(statement: Statement): UsingKind {
    return isVariableStatement(statement) ? getUsingKindOfVariableStatement(statement) : UsingKind.None;
}

function getUsingKindOfStatements(statements: readonly Statement[]): UsingKind {
    let result = UsingKind.None;
    for (const statement of statements) {
        const usingKind = getUsingKind(statement);
        if (usingKind === UsingKind.Async) return UsingKind.Async;
        if (usingKind > result) result = usingKind;
    }
    return result;
}

function getUsingKindOfCaseOrDefaultClauses(clauses: readonly CaseOrDefaultClause[]): UsingKind {
    let result = UsingKind.None;
    for (const clause of clauses) {
        const usingKind = getUsingKindOfStatements(clause.statements);
        if (usingKind === UsingKind.Async) return UsingKind.Async;
        if (usingKind > result) result = usingKind;
    }
    return result;
}
