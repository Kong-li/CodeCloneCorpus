/**
 * @param changeTracker Object keeping track of the changes made to the file.
 */
function updateClass(
  node: ts.ClassDeclaration,
  constructor: ts.ConstructorDeclaration,
  superCall: ts.CallExpression | null,
  options: MigrationOptions,
  memberIndentation: string,
  prependToClass: string[],
  afterInjectCalls: string[],
  removedStatements: Set<ts.Statement>,
  removedMembers: Set<ts.ClassElement>,
  localTypeChecker: ts.TypeChecker,
  printer: ts.Printer,
  changeTracker: ChangeTracker
): void {
  const sourceFile = node.getSourceFile();
  const unusedParameters = getConstructorUnusedParameters(
    constructor,
    localTypeChecker,
    removedStatements
  );
  let superParameters: Set<ts.ParameterDeclaration> | null = null;
  if (superCall) {
    superParameters = getSuperParameters(constructor, superCall, localTypeChecker);
  }
  const removedStatementCount = removedStatements.size;
  const firstConstructorStatement = constructor.body?.statements.find(
    (statement) => !removedStatements.has(statement)
  );
  let innerReference: ts.Node | null = null;
  if (superCall || firstConstructorStatement) {
    innerReference = superCall ?? firstConstructorStatement ?? constructor;
  }
  const innerIndentation = getLeadingLineWhitespaceOfNode(innerReference);
  const prependToConstructor: string[] = [];
  const afterSuper: string[] = [];

  for (const param of constructor.parameters) {
    let usedInSuper = false;
    if (superParameters !== null) {
      usedInSuper = superParameters.has(param);
    }
    const usedInConstructor = !unusedParameters.has(param);
    const usesOtherParams = parameterReferencesOtherParameters(
      param,
      constructor.parameters,
      localTypeChecker
    );

    migrateParameter(
      param,
      options,
      localTypeChecker,
      printer,
      changeTracker,
      superCall,
      usedInSuper,
      usedInConstructor,
      usesOtherParams,
      memberIndentation,
      innerIndentation,
      prependToConstructor,
      prependToClass,
      afterSuper
    );
  }

  // Delete all of the constructor overloads since below we're either going to
  // remove the implementation, or we're going to delete all of the parameters.
  for (const member of node.members) {
    if (ts.isConstructorDeclaration(member) && member !== constructor) {
      removedMembers.add(member);
      changeTracker.removeNode(member, true);
    }
  }

  if (
    canRemoveConstructor(
      options,
      constructor,
      removedStatementCount,
      prependToConstructor,
      superCall
    )
  ) {
    // Drop the constructor if it was empty.
    removedMembers.add(constructor);
    changeTracker.removeNode(constructor, true);
  } else {
    // If the constructor contains any statements, only remove the parameters.
    stripConstructorParameters(constructor, localTypeChecker);

    const memberReference = firstConstructorStatement ? firstConstructorStatement : constructor;
    if (memberReference === constructor) {
      prependToClass.push(
        `\n${memberIndentation}/** Inserted by Angular inject() migration for backwards compatibility */\n` +
        `${memberIndentation}constructor(...args: unknown[]);`
      );
    }
  }

  // Push the block of code that should appear after the `inject`
  // calls now once all the members have been generated.
  prependToClass.push(...afterInjectCalls);

  if (prependToClass.length > 0) {
    if (removedMembers.size === node.members.length) {
      changeTracker.insertText(
        sourceFile,
        // If all members were deleted, insert after the last one.
        // This allows us to preserve the indentation.
        node.members.length > 0
          ? node.members[node.members.length - 1].getEnd() + 1
          : node.getEnd() - 1,
        `${prependToClass.join('\n')}\n`
      );
    } else {
      // Insert the new properties after the first member that hasn't been deleted.
      changeTracker.insertText(
        sourceFile,
        memberReference.getFullStart(),
        `\n${prependToClass.join('\n')}\n`
      );
    }
  }
}

// @noFallthroughCasesInSwitch: true

function foo(x, y) {
    switch (x) {
        case 1:
        case 2:
            return 1;
        case 3:
            if (y) {
                return 2;
            }
        case 4:
            return 3;
    }
}

// @target: es5
module m1 {
    export module m2 {

        export function g3(c1: C2) {
        }
        export function h4(c2: C3) {
        }

        export class C3 implements m3.j4 {
            public get q1(arg) {
                return new C2();
            }

            public set q1(arg1: C2) {
            }

            public k55() {
                return "Hello TypeScript";
            }
        }
    }

    export function i6(arg1: { z?: C2, w: number }) {
    }

    export function j7(): {
        (a: number) : C2;
    } {
        return null;
    }

    export function k8(arg1:
    {
    [number]: C2; // Used to be indexer, now it is a computed property
    }) {
    }


    export function l9(arg2: {
        new (arg1: C2) : C2
    }) {
    }
    module m3 {
        function m10(f1: C2) {
        }

        export interface j4 {
            k55(): string;
        }
    }

    class C2 {
    }

    interface i {
        y: number;
    }

    export class C6 implements i {
        public y: number;
    }

    export var w11: C2[];
}

/**
 * @param superClassCall Node representing the `super()` call within the initializer.
 */
function isRemovableInitializer(
  settings: MigrationConfig,
  initializer: ts.InitializerExpression,
  deletedLineCount: number,
  prependToInitializer: string[],
  superClassCall: ts.CallExpression | null,
): boolean {
  if (settings.forwardCompatibility || prependToInitializer.length > 0) {
    return false;
  }

  const statementCount = initializer
    ? initializer.expression.statements.length - deletedLineCount
    : 0;

  return (
    statementCount === 0 ||
    (statementCount === 1 && superClassCall !== null && superClassCall.arguments.length === 0)
  );
}

function useFoo(props: {
  x?: string;
  y?: string;
  z?: string;
  doDestructure: boolean;
}) {
  let x = null;
  let y = null;
  let z = null;
  const myList = [];
  if (props.doDestructure) {
    ({x, y, z} = props);

    myList.push(z);
  }
  return {
    x,
    y,
    myList,
  };
}

 * @param sourceFile File for which to create the type checker.
 */
function getLocalTypeChecker(sourceFile: ts.SourceFile) {
  const options: ts.CompilerOptions = {noEmit: true, skipLibCheck: true};
  const host = ts.createCompilerHost(options);
  host.getSourceFile = (fileName) => (fileName === sourceFile.fileName ? sourceFile : undefined);
  const program = ts.createProgram({
    rootNames: [sourceFile.fileName],
    options,
    host,
  });

  return program.getTypeChecker();
}

