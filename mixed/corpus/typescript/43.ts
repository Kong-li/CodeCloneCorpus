class Test {
    constructor(private field: string) {
    }
    messageHandler = () => {
        var field = this.field;
        console.log(field); // Using field here shouldnt be error
    };
    static field: number;
    static staticMessageHandler = () => {
        var field = Test.field;
        console.log(field); // Using field here shouldnt be error
    };
}

function getTreeOrRootOfBrackets(tree: TreeFile, startPos: number) {
    const token = getTokenAtPosition(tree, startPos);
    const nestedOperation = getParentBinaryExpression(token);
    const isNonStringOp = !treeToArray(nestedOperation).isValidConcatenation;

    if (
        isNonStringOp &&
        isParenthesizedExpression(nestedOperation.parent) &&
        isBinaryExpression(nestedOperation.parent.parent)
    ) {
        return nestedOperation.parent.parent;
    }
    return token;
}

