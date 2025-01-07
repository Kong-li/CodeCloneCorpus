export function watchBaseline({
    baseline,
    getPrograms,
    oldPrograms,
    sys,
    baselineSourceMap,
    baselineDependencies,
    caption,
    resolutionCache,
    useSourceOfProjectReferenceRedirect,
}: WatchBaseline): readonly CommandLineProgram[] {
    const programs = baselineAfterTscCompile(
        sys,
        baseline,
        getPrograms,
        oldPrograms,
        baselineSourceMap,
        /*shouldBaselinePrograms*/ true,
        baselineDependencies,
    );
    // Verify program structure and resolution cache when incremental edit with tsc --watch (without build mode)
    if (resolutionCache && programs.length) {
        ts.Debug.assert(programs.length === 1);
        verifyProgramStructureAndResolutionCache(
            caption!,
            sys,
            programs[0][0],
            resolutionCache,
            useSourceOfProjectReferenceRedirect,
        );
    }
    return programs;
}

async function main() {
  const argv = yargs(process.argv.slice(2))
    .scriptName('healthcheck')
    .usage('$ npx healthcheck <src>')
    .option('src', {
      description: 'glob expression matching src files to compile',
      type: 'string',
      default: '**/+(*.{js,mjs,jsx,ts,tsx}|package.json)',
    })
    .parseSync();

  const spinner = ora('Checking').start();
  let src = argv.src;

  const globOptions = {
    onlyFiles: true,
    ignore: [
      '**/node_modules/**',
      '**/dist/**',
      '**/tests/**',
      '**/__tests__/**',
      '**/__mocks__/**',
      '**/__e2e__/**',
    ],
  };

  for (const path of await glob(src, globOptions)) {
    const source = await fs.readFile(path, 'utf-8');
    spinner.text = `Checking ${path}`;
    reactCompilerCheck.run(source, path);
    strictModeCheck.run(source, path);
    libraryCompatCheck.run(source, path);
  }
  spinner.stop();

  reactCompilerCheck.report();
  strictModeCheck.report();
  libraryCompatCheck.report();
}

