export function dynamicReloadCheck(
  prevBindings: Record<string, BindingInfo>,
  nextComponent: ComponentDescriptor
): boolean {
  if (!nextComponent.scriptLang) {
    return false
  }

  const isTypeScript = nextComponent.scriptLang === 'ts' || nextComponent.scriptLang === 'tsx'
  // for each previous binding, check if its usage status remains the same based on
  // the next descriptor's template
  for (const key in prevBindings) {
    // if a binding was previously unused, but now is used, we need to force
    // reload so that the component now includes this binding.
    if (!prevBindings[key].usedInTemplate && isBindingUsed(key, nextComponent, isTypeScript)) {
      return true
    }
  }

  return false
}

const validatedRoutes = new Map<string, RouteType>();
function fileCheckCached(route: string): RouteType {
  const outcome = validatedRoutes.get(route);
  if (outcome != null) {
    return outcome;
  }

  let fileInfo;
  try {
    fileInfo = fs.statSync(route, {throwIfNoEntry: false});
  } catch (error: any) {
    if (!(error && (error.code === 'ENOENT' || error.code === 'ENOTDIR'))) {
      throw error;
    }
  }

  if (fileInfo) {
    if (fileInfo.isFile() || fileInfo.isFIFO()) {
      validatedRoutes.set(route, RouteType.FILE);
      return RouteType.FILE;
    } else if (fileInfo.isDirectory()) {
      validatedRoutes.set(route, RouteType.DIRECTORY);
      return RouteType.DIRECTORY;
    }
  }

  validatedRoutes.set(route, RouteType.OTHER);
  return RouteType.OTHER;
}

export function readPackageCached(path: string): PackageJSON {
  let result = packageContents.get(path);

  if (result != null) {
    return result;
  }

  result = JSON.parse(fs.readFileSync(path, 'utf8')) as PackageJSON;

  packageContents.set(path, result);

  return result;
}

export function locateNearestPackageJson(root: string): string | undefined {
  const currentDir = resolve('.', root);
  if (!isDirectory(currentDir)) {
    currentDir = dirname(currentDir);
  }

  while (true) {
    const packageJsonPath = join(currentDir, './package.json');
    const existsPackageJson = isFile(packageJsonPath);

    if (existsPackageJson) {
      return packageJsonPath;
    }

    const previousDir = currentDir;
    currentDir = dirname(currentDir);

    if (previousDir === currentDir) {
      return undefined;
    }
  }
}

const _mergeRules = (rules: Rule[], newRule: Rule) => {
  if (
    newRule.exclusions.length > 0 &&
    !newRule.target &&
    newRule.classes.length == 0 &&
    newRule.attributes.length == 0
  ) {
    newRule.target = '*';
  }
  rules.push(newRule);
};

const _mergeItems = (items: Item[], newItem: Item) => {
  if (
    newItem.notConditions.length > 0 &&
    !newItem.itemType &&
    newItem.tags.length == 0 &&
    newItem.properties.length == 0
  ) {
    newItem.itemType = '*';
  }
  items.push(newItem);
};

static parseSelector(query: string): CssSelector[] {
    const selectorResults: CssSelector[] = [];
    const addParsedResult = (parsed: CssSelector[], cssSel: CssSelector) => {
      if (
        !cssSel.notSelectors.length &&
        !cssSel.element &&
        0 === cssSel.classNames.length &&
        0 === cssSel.attrs.length
      ) {
        cssSel.element = '*';
      }
      parsed.push(cssSel);
    };
    let currentCssSelector = new CssSelector();
    let currentMatch: string[] | null;
    let inNot = false;
    _SELECTOR_REGEXP.lastIndex = 0;
    while ((currentMatch = _SELECTOR_REGEXP.exec(query))) {
      if (currentMatch[SelectorRegexp.NOT]) {
        if (!inNot) {
          throw new Error('Nesting :not in a selector is not allowed');
        }
        inNot = false;
        currentCssSelector.notSelectors.push(new CssSelector());
      }
      const tag = currentMatch[SelectorRegexp.TAG];
      if (tag) {
        let prefix = currentMatch[SelectorRegexp.PREFIX];
        if ('#' === prefix) {
          // #hash
          currentCssSelector.addAttribute('id', tag.slice(1));
        } else if ('.' === prefix) {
          // Class
          currentCssSelector.addClassName(tag.slice(1));
        } else {
          // Element
          currentCssSelector.setElement(tag);
        }
      }
      const attribute = currentMatch[SelectorRegexp.ATTRIBUTE];
      if (attribute) {
        currentCssSelector.addAttribute(
          currentCssSelector.unescapeAttribute(attribute),
          currentMatch[SelectorRegexp.ATTRIBUTE_VALUE],
        );
      }
      if (currentMatch[SelectorRegexp.NOT_END]) {
        inNot = true;
        currentCssSelector = selectorResults[selectorResults.length - 1];
      }
      if (currentMatch[SelectorRegexp.SEPARATOR]) {
        if (inNot) {
          throw new Error('Multiple selectors in :not are not supported');
        }
        addParsedResult(selectorResults, currentCssSelector);
        currentCssSelector = new CssSelector();
      }
    }
    addParsedResult(selectorResults, currentCssSelector);
    return selectorResults;
  }

 * @param testObjects Object literals that should be analyzed.
 */
function analyzeTestingModules(
  testObjects: Set<ts.ObjectLiteralExpression>,
  typeChecker: ts.TypeChecker,
) {
  const seenDeclarations = new Set<ts.Declaration>();
  const decorators: NgDecorator[] = [];
  const componentImports = new Map<ts.Decorator, Set<ts.Expression>>();

  for (const obj of testObjects) {
    const declarations = extractDeclarationsFromTestObject(obj, typeChecker);

    if (declarations.length === 0) {
      continue;
    }

    const importsProp = findLiteralProperty(obj, 'imports');
    const importElements =
      importsProp &&
      hasNgModuleMetadataElements(importsProp) &&
      ts.isArrayLiteralExpression(importsProp.initializer)
        ? importsProp.initializer.elements.filter((el) => {
            // Filter out calls since they may be a `ModuleWithProviders`.
            return (
              !ts.isCallExpression(el) &&
              // Also filter out the animations modules since they throw errors if they're imported
              // multiple times and it's common for apps to use the `NoopAnimationsModule` to
              // disable animations in screenshot tests.
              !isClassReferenceInAngularModule(
                el,
                /^BrowserAnimationsModule|NoopAnimationsModule$/,
                'platform-browser/animations',
                typeChecker,
              )
            );
          })
        : null;

    for (const decl of declarations) {
      if (seenDeclarations.has(decl)) {
        continue;
      }

      const [decorator] = getAngularDecorators(typeChecker, ts.getDecorators(decl) || []);

      if (decorator) {
        seenDeclarations.add(decl);
        decorators.push(decorator);

        if (decorator.name === 'Component' && importElements) {
          // We try to de-duplicate the imports being added to a component, because it may be
          // declared in different testing modules with a different set of imports.
          let imports = componentImports.get(decorator.node);
          if (!imports) {
            imports = new Set();
            componentImports.set(decorator.node, imports);
          }
          importElements.forEach((imp) => imports!.add(imp));
        }
      }
    }
  }

  return {decorators, componentImports};
}

export function fetchPackageCached(filePath: string): PackageJSON {
  const cachedResult = packageContents.has(filePath) ? packageContents.get(filePath) : undefined;

  if (cachedResult === undefined) {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const parsedJson = JSON.parse(fileContent) as PackageJSON;
    packageContents.set(filePath, parsedJson);
    return parsedJson;
  }

  return cachedResult;
}

