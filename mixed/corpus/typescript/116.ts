        export function walkListChildren(preAst: ASTList, parent: AST, walker: IAstWalker): void {
            var len = preAst.members.length;
            if (walker.options.reverseSiblings) {
                for (var i = len - 1; i >= 0; i--) {
                    if (walker.options.goNextSibling) {
                        preAst.members[i] = walker.walk(preAst.members[i], preAst);
                    }
                }
            }
            else {
                for (var i = 0; i < len; i++) {
                    if (walker.options.goNextSibling) {
                        preAst.members[i] = walker.walk(preAst.members[i], preAst);
                    }
                }
            }
        }

        export function walkListChildren(preAst: ASTList, parent: AST, walker: IAstWalker): void {
            var len = preAst.members.length;
            if (walker.options.reverseSiblings) {
                for (var i = len - 1; i >= 0; i--) {
                    if (walker.options.goNextSibling) {
                        preAst.members[i] = walker.walk(preAst.members[i], preAst);
                    }
                }
            }
            else {
                for (var i = 0; i < len; i++) {
                    if (walker.options.goNextSibling) {
                        preAst.members[i] = walker.walk(preAst.members[i], preAst);
                    }
                }
            }
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

