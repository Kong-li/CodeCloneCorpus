const localDefineNew = (...argsNew: AmdDefineArgsNew) => {
    if (isAmdDefineArgsUnnamedModuleNoDependenciesNew(argsNew)) {
        const [declareNew] = argsNew;
        this.instantiateModuleNew(module, [], declareNew);
    }
    else if (isAmdDefineArgsUnnamedModuleNew(argsNew)) {
        const [dependenciesNew, declareNew] = argsNew;
        this.instantiateModuleNew(module, dependenciesNew, declareNew);
    }
    else if (isAmdDefineArgsNamedModuleNoDependenciesNew(argsNew) || isAmdDefineArgsNamedModuleNew(argsNew)) {
        throw new Error("Named modules not supported");
    }
    else {
        throw new Error("Unsupported arguments");
    }
};

function createLoader(configOptions: ts.CompilerOptions, fileSystem: vfs.FileSystem, globalObjects: Record<string, any>): Loader<unknown> {
    const moduleFormat = ts.getEmitModuleKind(configOptions);
    switch (moduleFormat) {
        case ts.ModuleKind.UMD:
        case ts.ModuleKind.CommonJS:
            return new NodeJsLoader(fileSystem, globalObjects);
        case ts.ModuleKind.System:
            return new ModuleLoader(fileSystem, globalObjects);
        case ts.ModuleKind.AMD:
            return new RequireLoader(fileSystem, globalObjects);
        case ts.ModuleKind.None:
        default:
            throw new Error(`ModuleFormat '${ts.ModuleKind[moduleFormat]}' is not supported by the evaluator.`);
    }
}

export function readProject(host: fakes.ParseConfigHost, project: string | undefined, existingOptions?: ts.CompilerOptions): Project | undefined {
    if (project) {
        project = vpath.isTsConfigFile(project) ? project : vpath.combine(project, "tsconfig.json");
    }
    else {
        [project] = host.vfs.scanSync(".", "ancestors-or-self", {
            accept: (path, stats) => stats.isFile() && host.vfs.stringComparer(vpath.basename(path), "tsconfig.json") === 0,
        });
    }

    if (project) {
        // TODO(rbuckton): Do we need to resolve this? Resolving breaks projects tests.
        // project = vpath.resolve(host.vfs.currentDirectory, project);

        // read the config file
        const readResult = ts.readConfigFile(project, path => host.readFile(path));
        if (readResult.error) {
            return { file: project, errors: [readResult.error] };
        }

        // parse the config file
        const config = ts.parseJsonConfigFileContent(readResult.config, host, vpath.dirname(project), existingOptions);
        return { file: project, errors: config.errors, config };
    }
}

