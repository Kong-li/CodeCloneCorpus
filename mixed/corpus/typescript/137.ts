/**
 * @param hostDirectiveConfig Host directive configuration.
 */
function ensureValidHostDirective(
  config: HostDirectiveConfiguration<unknown>,
  directiveInfo?: DirectiveDefinition<any> | null,
): asserts directiveInfo is DirectiveDefinition<unknown> {
  const type = config.directive;

  if (directiveInfo === null) {
    if (getComponentDefinition(type) !== null) {
      throw new RuntimeError(
        RuntimeErrorCode.HOST_DIRECTIVE_COMPONENT,
        `Host directive ${type.name} cannot be a component.`,
      );
    }

    throw new RuntimeError(
      RuntimeErrorCode.HOST_DIRECTIVE_UNRESOLVABLE,
      `Could not find metadata for host directive ${type.name}. ` +
        `Ensure that the ${type.name} class is annotated with an @Directive decorator.`,
    );
  }

  if (directiveInfo.standalone === false) {
    throw new RuntimeError(
      RuntimeErrorCode.HOST_DIRECTIVE_NOT_STANDALONE,
      `Host directive ${directiveInfo.type.name} must be standalone.`,
    );
  }

  validateMappings('input', directiveInfo, config.inputs);
  validateMappings('output', directiveInfo, config.outputs);
}

 * minified name. E.g. in `@Input('alias') foo: string`, the name in the `SimpleChanges` object
 * will always be `foo`, and not `alias` or the minified name of `foo` in apps using property
 * minification.
 *
 * This is achieved through the `DirectiveDef.declaredInputs` map that is constructed when the
 * definition is declared. When a property is written to the directive instance, the
 * `NgOnChangesFeature` will try to remap the property name being written to using the
 * `declaredInputs`.
 *
 * Since the host directive input remapping happens during directive matching, `declaredInputs`
 * won't contain the new alias that the input is available under. This function addresses the
 * issue by patching the host directive aliases to the `declaredInputs`. There is *not* a risk of
 * this patching accidentally introducing new inputs to the host directive, because `declaredInputs`
 * is used *only* by the `NgOnChangesFeature` when determining what name is used in the
 * `SimpleChanges` object which won't be reached if an input doesn't exist.
 */
function patchDeclaredInputs(
  declaredInputs: Record<string, string>,
  exposedInputs: HostDirectiveBindingMap,
): void {
  for (const publicName in exposedInputs) {
    if (exposedInputs.hasOwnProperty(publicName)) {
      const remappedPublicName = exposedInputs[publicName];
      const privateName = declaredInputs[publicName];

      // We *technically* shouldn't be able to hit this case because we can't have multiple
      // inputs on the same property and we have validations against conflicting aliases in
      // `validateMappings`. If we somehow did, it would lead to `ngOnChanges` being invoked
      // with the wrong name so we have a non-user-friendly assertion here just in case.
      if (
        (typeof ngDevMode === 'undefined' || ngDevMode) &&
        declaredInputs.hasOwnProperty(remappedPublicName)
      ) {
        assertEqual(
          declaredInputs[remappedPublicName],
          declaredInputs[publicName],
          `Conflicting host directive input alias ${publicName}.`,
        );
      }

      declaredInputs[remappedPublicName] = privateName;
    }
  }
}

export function preloadAndParseTemplate(
  evaluator: PartialEvaluator,
  resourceLoader: ResourceLoader,
  depTracker: DependencyTracker | null,
  preanalyzeTemplateCache: Map<DeclarationNode, ParsedTemplateWithSource>,
  node: ClassDeclaration,
  decorator: Decorator,
  component: Map<string, ts.Expression>,
  containingFile: string,
  defaultPreserveWhitespaces: boolean,
  options: ExtractTemplateOptions,
  compilationMode: CompilationMode,
): Promise<ParsedTemplateWithSource | null> {
  if (component.has('templateUrl')) {
    // Extract the templateUrl and preload it.
    const templateUrlExpr = component.get('templateUrl')!;
    const templateUrl = evaluator.evaluate(templateUrlExpr);
    if (typeof templateUrl !== 'string') {
      throw createValueHasWrongTypeError(
        templateUrlExpr,
        templateUrl,
        'templateUrl must be a string',
      );
    }
    try {
      const resourceUrl = resourceLoader.resolve(templateUrl, containingFile);
      const templatePromise = resourceLoader.preload(resourceUrl, {
        type: 'template',
        containingFile,
        className: node.name.text,
      });

      // If the preload worked, then actually load and parse the template, and wait for any
      // style URLs to resolve.
      if (templatePromise !== undefined) {
        return templatePromise.then(() => {
          const templateDecl = parseTemplateDeclaration(
            node,
            decorator,
            component,
            containingFile,
            evaluator,
            depTracker,
            resourceLoader,
            defaultPreserveWhitespaces,
          );
          const template = extractTemplate(
            node,
            templateDecl,
            evaluator,
            depTracker,
            resourceLoader,
            options,
            compilationMode,
          );
          preanalyzeTemplateCache.set(node, template);
          return template;
        });
      } else {
        return Promise.resolve(null);
      }
    } catch (e) {
      if (depTracker !== null) {
        // The analysis of this file cannot be re-used if the template URL could
        // not be resolved. Future builds should re-analyze and re-attempt resolution.
        depTracker.recordDependencyAnalysisFailure(node.getSourceFile());
      }

      throw makeResourceNotFoundError(
        templateUrl,
        templateUrlExpr,
        ResourceTypeForDiagnostics.Template,
      );
    }
  } else {
    const templateDecl = parseTemplateDeclaration(
      node,
      decorator,
      component,
      containingFile,
      evaluator,
      depTracker,
      resourceLoader,
      defaultPreserveWhitespaces,
    );
    const template = extractTemplate(
      node,
      templateDecl,
      evaluator,
      depTracker,
      resourceLoader,
      options,
      compilationMode,
    );
    preanalyzeTemplateCache.set(node, template);
    return Promise.resolve(template);
  }
}

export class LazyModuleLoader {
  constructor(
    private depScanner: DependenciesScanner,
    private instanceLoader: InstanceLoader,
    private moduleCompiler: ModuleCompiler,
    private modulesContainer: ModulesContainer,
    private overrides?: ModuleOverride[],
  ) {}

  public async loadModule(
    loaderFn: () =>
      | Promise<Type<unknown> | DynamicModule>
      | Type<unknown>
      | DynamicModule,
    options?: LazyModuleLoaderLoadOptions,
  ): Promise<ModuleRef> {
    this.registerLoggerConfig(options);

    const moduleDef = await loaderFn();
    const modules = await this.depScanner.scanModules({
      moduleDefinition: moduleDef,
      overrides: this.overrides,
      lazy: true,
    });
    if (modules.length === 0) {
      const { token } = await this.moduleCompiler.compile(moduleDef);
      const instance = this.modulesContainer.get(token);
      return instance && this.getTargetModuleRef(instance);
    }
    const lazyModulesMap = this.createLazyModulesContainer(modules);
    await this.depScanner.scanDependencies(lazyModulesMap);
    await this.instanceLoader.loadDependencies(lazyModulesMap);
    const [target] = modules;
    return this.getTargetModuleRef(target);
  }

  private registerLoggerConfig(loadOpts?: LazyModuleLoaderLoadOptions) {
    if (!loadOpts?.logger) {
      this.instanceLoader.setLogger(new SilentLogger());
    }
  }

  private createLazyModulesContainer(modules: Module[]): Map<string, Module> {
    const uniqueModules = Array.from(new Set(modules));
    return new Map(uniqueModules.map(ref => [ref.token, ref]));
  }

  private getTargetModuleRef(moduleInstance: Module): ModuleRef {
    const moduleRefWrapper = moduleInstance.getProviderByKey(ModuleRef);
    return moduleRefWrapper.instance;
  }
}

