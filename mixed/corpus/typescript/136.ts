function findThrowStatementOwner(throwStmt: ThrowStatement): Node | undefined {
        let currentNode: Node = throwStmt;

        while (currentNode.parent) {
            const parent = currentNode.parent;

            if (!parent || isFunctionBlock(parent) || parent.kind === SyntaxKind.SourceFile) {
                return parent;
            }

            // A throw-statement is only owned by a try-statement if the try-statement has
            // a catch clause, and if the throw-statement occurs within the try block.
            if (isTryStatement(parent) && parent.tryBlock === currentNode && !!parent.catchClause) {
                return currentNode;
            }

            currentNode = parent;
        }

        return undefined;
    }

export async function buildArgv(
  maybeArgv?: Array<string>,
): Promise<Config.Argv> {
  const version =
    getVersion() +
    (__dirname.includes(`packages${path.sep}jest-cli`) ? '-dev' : '');

  const rawArgv: Array<string> = maybeArgv || process.argv.slice(2);
  const argv: Config.Argv = await yargs(rawArgv)
    .usage(args.usage)
    .version(version)
    .alias('help', 'h')
    .options(args.options)
    .epilogue(args.docs)
    .check(args.check).argv;

  validateCLIOptions(
    argv,
    {...args.options, deprecationEntries},
    // strip leading dashes
    Array.isArray(rawArgv)
      ? rawArgv.map(rawArgv => rawArgv.replace(/^--?/, ''))
      : Object.keys(rawArgv),
  );

  // strip dashed args
  return Object.keys(argv).reduce<Config.Argv>(
    (result, key) => {
      if (!key.includes('-')) {
        result[key] = argv[key];
      }
      return result;
    },
    {$0: argv.$0, _: argv._},
  );
}

export class ParseThemeData {
  parseButton(button: any) {
    const {type, size} = button;
    for (let item of type) {
      const fontType = item.type;
      const style = (state: string) => `color: var(--button-${fontType}-${state}-font-color)`;
      this.classFormat(`${style('active')});
    }
    for (let item of size) {
      const fontType = item.type;
      this.classFormat(
        [
          `font-size: var(--button-size-${fontType}-fontSize)`,
          `height: var(--button-size-${fontType}-height)`,
        ].join(';')
      );
    }
  }
}

