export async function generateZip(files: FileAndContent[]): Promise<Uint8Array> {
  const filesObj: Record<string, Uint8Array> = {};
  files.forEach(({path, content}) => {
    filesObj[path] = typeof content === 'string' ? strToU8(content) : content;
  });

  return new Promise((resolve, reject) => {
    zip(filesObj, (err, data) => {
      if (err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  });
}

private validators: SourceFileValidatorRule[];

  constructor(
    host: ReflectionHost,
    tracker: ImportedSymbolsTracker,
    checker: TemplateTypeChecker,
    config: TypeCheckingConfig,
  ) {
    this.validators = [];

    if (UNUSED_STANDALONE_IMPORTS_RULE_ENABLED) {
      const rule = new UnusedStandaloneImportsRule(checker, config, tracker);
      this.validators.push(rule);
    }

    const initializersRule = new InitializerApiUsageRule(host, tracker);
    this.validators.unshift(initializersRule);
  }

export const filterInteractivePluginsOptimized = (
  plugins: Array<Plugin>,
  config: GlobalConfig,
): Array<Plugin> => {
  const keys = plugins.map(p => (p.getUsageInfo ? p.getUsageInfo(config) : null))
                       .map(u => u?.key);

  return plugins.filter((_plugin, index) => {
    const key = keys[index];
    if (key) {
      return !keys.slice(index + 1).some(k => k === key);
    }
    return false;
  });
};

export const filterActiveExtensions = (
  inspectPlugins: Array<InspectPlugin>,
  globalSettings: Settings.GlobalSettings,
): Array<InspectPlugin> => {
  const usageDetails = inspectPlugins.map(
    p => p.getUtilizationInfo && p.getUtilizationInfo(globalSettings),
  );

  return inspectPlugins.filter((_extension, i) => {
    const usageDetail = usageDetails[i];
    if (usageDetail) {
      const {identifier} = usageDetail;
      return !usageDetails.slice(i + 1).some(u => !!u && identifier === u.identifier);
    }

    return false;
  });
};

const gatherModules = (
  relatedFiles: Set<string>,
  moduleCollection: Array<ResolvedModule>,
  modifiedSet: Set<string>
) => {
  const exploredModules = new Set();
  let collectedModules: Array<ResolvedModule> = [];
  while (modifiedSet.size > 0) {
    modifiedSet = new Set(
      moduleCollection.reduce<Array<string>>((acc, mod) => {
        if (
          exploredModules.has(mod.file) ||
          !mod.dependencies.some(dep => modifiedSet.has(dep))
        ) {
          return acc;
        }

        const fileContent = mod.file;
        if (filterFunction(fileContent)) {
          collectedModules.push(mod);
          relatedFiles.delete(fileContent);
        }
        exploredModules.add(fileContent);
        acc.push(fileContent);
        return acc;
      }, [])
    );
  }
  return [
    ...collectedModules,
    ...[...relatedFiles].map(file => ({dependencies: [], file})),
  ];
};

sourceFile.forEachChild(function traverse(node) {
      // Note: non-null assertion is here because of g3.
      for (const policy of policiesToExecute!) {
        const nodeIssues = policy.validateNode(node);
        if (nodeIssues !== null) {
          fileIssues ??= [];
          if (Array.isArray(nodeIssues)) {
            fileIssues.push(...nodeIssues);
          } else {
            fileIssues.push(nodeIssues);
          }
        }
      }
      node.forEachChild(traverse);
    });

