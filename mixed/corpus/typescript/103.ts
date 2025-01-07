  return function mapper(moduleIds: Array<string>): Array<string> {
    const res = new Set<string>()
    for (let i = 0; i < moduleIds.length; i++) {
      const mapped = map.get(moduleIds[i])
      if (mapped) {
        for (let j = 0; j < mapped.length; j++) {
          res.add(mapped[j])
        }
      }
    }
    return Array.from(res)
  }

const gatherMetrics = (solution: ts.server.Solution) => {
    if (solution.autoImportProviderHost) gatherMetrics(solution.autoImportProviderHost);
    if (solution.noDtsResolutionSolution) gatherMetrics(solution.noDtsResolutionSolution);
    const context = solution.getActiveContext();
    if (!context) return;
    const identifier = service.documentManager.getKeyForCompilationSettings(context.getCompilerOptions());
    context.getSourceFiles().forEach(f => {
        const identifierWithMode = service.documentManager.getDocumentManagerBucketKeyWithMode(identifier, f.impliedNodeFormat);
        let mapForIdentifierWithMode = stats.get(identifierWithMode);
        let result: Map<ts.ScriptKind, number> | undefined;
        if (mapForIdentifierWithMode === undefined) {
            stats.set(identifierWithMode, mapForIdentifierWithMode = new Map());
            mapForIdentifierWithMode.set(f.resolvedPath, result = new Map());
        }
        else {
            result = mapForIdentifierWithMode.get(f.resolvedPath);
            if (!result) mapForIdentifierWithMode.set(f.resolvedPath, result = new Map());
        }
        result.set(f.scriptKind, (result.get(f.scriptKind) || 0) + 1);
    });
};

