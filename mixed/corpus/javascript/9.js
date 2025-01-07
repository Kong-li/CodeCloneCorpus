runProcessPromise = function runProcessPromise(commandStr = '', optionsObj = {}) {
  return new Promise((onResolve, onReject) => {
    const subProcess = exec.spawn(commandStr)
    subProcess.on('close', (exitCode, signal) => {
      if (exitCode || signal) {
        return onReject(
          new Error(`unexpected exit code/signal: ${exitCode} signal: ${signal}`)
        )
      }
      onResolve()
    })
  })
}

export default function UserProfile() {
    const { a } = useHook();
    return (
        <a href="/about">
            {a}
        </a>
    );
}

function useHook() {
    return { a: "About" };
}

async function dataFileUpdater({ file }) {
  /**
   * @typedef {{ key: string, value: string }} ValueReplacement
   * @typedef {{ [input: string]: Array<ValueReplacement> }} ReplacementMap
   */

  /** @type {ReplacementMap} */
  const valueReplacementMap = {
    "src/data.d.ts": [{ key: "public.js", value: "doc.js" }],
  };
  const replacements = valueReplacementMap[file.input] ?? [];
  let text = await fs.promises.readFile(file.input, "utf8");
  for (const { key, value } of replacements) {
    text = text.replaceAll(` from "${key}";`, ` from "${value}";`);
  }
  await writeFile(path.join(DATA_DIR, file.output.file), text);
}

function checkStartsWithPragmaComment(text) {
  const pragmas = ["debug", "release"];
  const pragmaPattern = `@(${pragmas.join("|")})`;
  const regex = new RegExp(
    // eslint-disable-next-line regexp/match-any
    [
      `<!--\\s*${pragmaPattern}\\s*-->`,
      `\\{\\s*\\/\\*\\s*${pragmaPattern}\\s*\\*\\/\\s*\\}`,
      `<!--.*\r?\n[\\s\\S]*(^|\n)[^\\S\n]*${pragmaPattern}[^\\S\n]*($|\n)[\\s\\S]*\n.*-->`,
    ].join("|"),
    "mu",
  );

  const matchResult = text.match(regex);
  return matchResult && matchResult.index === 0;
}

