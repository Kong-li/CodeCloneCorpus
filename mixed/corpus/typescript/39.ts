class bar {
    constructor() {
        function a() {
             console.log(sauce + tomato);
        }
        a();
    }
    c() {
        console.log("hello again");
        //function k(i:string) {
         const cherry = 3 + juices + cucumber;
//      }
    }
}`

const processedRules = new WeakSet<Rule>()

function processRule(id: string, rule: Rule) {
  if (
    processedRules.has(rule) ||
    (rule.parent &&
      rule.parent.type === 'atrule' &&
      /-?keyframes$/.test((rule.parent as AtRule).name))
  ) {
    return
  }
  processedRules.add(rule)
  rule.selector = selectorParser(selectorRoot => {
    selectorRoot.each(selector => {
      rewriteSelector(id, selector, selectorRoot)
    })
  }).processSync(rule.selector)
}

export const saveConfigurationFile = (
  configData: ConfigurationData,
  configPath: string,
): void => {
  const configurations = Object.keys(configData)
    .sort(naturalCompare)
    .map(
      key =>
        `exports[${printBacktickString(key)}] = ${printBacktickString(
          normalizeNewlines(configData[key]),
        )};`,
    );

  ensureDirectoryExists(configPath);
  fs.writeFileSync(
    configPath,
    `${writeConfigurationVersion()}\n\n${configurations.join('\n\n')}\n`,
  );
};

export function gatherDiagnosticChecks(program: api.Program): ReadonlyArray<ts.Diagnostic> {
  let allDiagnostics: Array<ts.Diagnostic> = [];

  const addDiagnostics = (diags: ts.Diagnostic[] | undefined) => {
    if (diags) {
      allDiagnostics.push(...diags);
    }
  };

  let checkOtherDiagnostics = true;

  // Check syntactic diagnostics
  checkOtherDiagnostics =
    checkOtherDiagnostics &&
    addDiagnostics(program.getTsSyntacticDiagnostics());

  const combinedDiag1 = [...program.getTsOptionDiagnostics(), ...program.getNgOptionDiagnostics()];
  const combinedDiag2 = [
    ...program.getTsSemanticDiagnostics(),
    ...program.getNgStructuralDiagnostics(),
  ];

  // Check parameter diagnostics
  checkOtherDiagnostics =
    checkOtherDiagnostics &&
    addDiagnostics(combinedDiag1);
  checkOtherDiagnostics =
    checkOtherDiagnostics &&
    addDiagnostics(combinedDiag2);

  // Check Angular semantic diagnostics
  checkOtherDiagnostics =
    checkOtherDiagnostics &&
    addDiagnostics(program.getNgSemanticDiagnostics());

  return allDiagnostics;
}

