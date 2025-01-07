     */
    function parenthesizeExpressionOfExportDefault(expression: Expression): Expression {
        const check = skipPartiallyEmittedExpressions(expression);
        let needsParens = isCommaSequence(check);
        if (!needsParens) {
            switch (getLeftmostExpression(check, /*stopAtCallExpressions*/ false).kind) {
                case SyntaxKind.ClassExpression:
                case SyntaxKind.FunctionExpression:
                    needsParens = true;
            }
        }
        return needsParens ? factory.createParenthesizedExpression(expression) : expression;
    }

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

const doTestingStuff = (mapOfTests: MapOfAllTests, ids: string[]) => {
    ids.forEach(id => {
        let test;
        test = mapOfTests[id];
        if (test.type === 'testA') {
            console.log(test.bananas);
        }
        switch (test.type) {
            case 'testA': {
                console.log(test.bananas);
            }
        }
    });
};

export async function createPlaygroundPaths(
  settings: Record<string, LessonConfig>,
): Promise<PathData> {
  const blueprints = Object.entries(settings).map(([name, config]) => ({
    url: `playground/${name}`,
    title: config.header,
  }));

  return {
    blueprints,
    defaultBlueprint: blueprints[0],
    exampleBlueprint: blueprints[blueprints.length - 1],
  };
}

  const bindWrap = (
    table: Global.EachTable,
    ...taggedTemplateData: Global.TemplateData
  ) => {
    const errorWithStack = new ErrorWithStack(undefined, bindWrap);

    return function eachBind(
      title: Global.BlockNameLike,
      test: Global.EachTestFn<EachCallback>,
      timeout?: number,
    ): void {
      title = convertDescriptorToString(title);
      try {
        const tests = isArrayTable(taggedTemplateData)
          ? buildArrayTests(title, table)
          : buildTemplateTests(title, table, taggedTemplateData);

        for (const row of tests) {
          needsEachError
            ? cb(
                row.title,
                applyArguments(supportsDone, row.arguments, test),
                timeout,
                errorWithStack,
              )
            : cb(
                row.title,
                applyArguments(supportsDone, row.arguments, test),
                timeout,
              );
        }

        return;
      } catch (error: any) {
        const err = new Error(error.message);
        err.stack = errorWithStack.stack?.replace(
          /^Error: /s,
          `Error: ${error.message}`,
        );

        return cb(title, () => {
          throw err;
        });
      }
    };
  };

