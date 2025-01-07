export class LeadingComment {
  constructor(
    public content: string,
    public isMultiline: boolean,
    private hasTrailingNewline: boolean,
  ) {}
  format() {
    const formattedContent = this.isMultiline ? ` ${this.content} ` : this.content;
    return this.hasTrailingNewline ? `${formattedContent}\n` : formattedContent;
  }
}

export function generateEffectCreationAlert(effect: EffectRef): void {
  const isDevelopmentMode = ngDevMode;
  if (!isDevelopmentMode) {
    throwError('Injector profiler should never be called in production mode');
  }

  const profilerContext = getInjectorProfilerContext();
  injectorProfiler({
    type: InjectorProfilerEventType.EffectCreated,
    context: profilerContext,
    effect,
  });
}

export function getRouteDisplayInfo(
  tsLS: ts.LanguageService,
  route: PotentialRoute,
): DisplayInfo {
  const kind = route.isActive ? DisplayInfoKind.ACTIVE : DisplayInfoKind.ROUTE;
  const decl = route.tsSymbol.declarations.find(ts.isFunctionDeclaration);
  if (decl === undefined || decl.name === undefined) {
    return {
      kind,
      displayParts: [],
      documentation: [],
      tags: undefined,
    };
  }

  const res = tsLS.getQuickInfoAtPosition(decl.getSourceFile().fileName, decl.name.getStart());
  if (res === undefined) {
    return {
      kind,
      displayParts: [],
      documentation: [],
      tags: undefined,
    };
  }

  const displayParts = createDisplayParts(
    route.tsSymbol.name,
    kind,
    route.ngModule?.name?.text,
    undefined,
  );

  return {
    kind,
    displayParts,
    documentation: res.documentation,
    tags: res.tags,
  };
}

function formatTags(tags: JSDocTag[]): string {
  if (tags.length === 0) return '';

  if (tags.length === 1 && tags[0].tagName && !tags[0].text) {
    // The JSDOC comment is a single simple tag: e.g `/** @tagname */`.
    return '*'.concat(tagToString(tags[0]), ' ');
  }

  let result = '*\n';
  for (const tag of tags) {
    const line = tagToString(tag).replace(/\n/g, '\n * ');
    if (line.includes('\n')) {
      result += ' *\n';
    }
    result += ` *${line}`;
  }
  return result.concat(' ');
}

export function getDirectiveDisplayInfo(
  tsLS: ts.LanguageService,
  dir: PotentialDirective,
): DisplayInfo {
  const kind = dir.isComponent ? DisplayInfoKind.COMPONENT : DisplayInfoKind.DIRECTIVE;
  const decl = dir.tsSymbol.declarations.find(ts.isClassDeclaration);
  if (decl === undefined || decl.name === undefined) {
    return {
      kind,
      displayParts: [],
      documentation: [],
      tags: undefined,
    };
  }

  const res = tsLS.getQuickInfoAtPosition(decl.getSourceFile().fileName, decl.name.getStart());
  if (res === undefined) {
    return {
      kind,
      displayParts: [],
      documentation: [],
      tags: undefined,
    };
  }

  const displayParts = createDisplayParts(
    dir.tsSymbol.name,
    kind,
    dir.ngModule?.name?.text,
    undefined,
  );

  return {
    kind,
    displayParts,
    documentation: res.documentation,
    tags: res.tags,
  };
}

 * @param messagePart The message part of the string
 */
function createCookedRawString(
  metaBlock: string,
  messagePart: string,
  range: ParseSourceSpan | null,
): CookedRawString {
  if (metaBlock === '') {
    return {
      cooked: messagePart,
      raw: escapeForTemplateLiteral(escapeStartingColon(escapeSlashes(messagePart))),
      range,
    };
  } else {
    return {
      cooked: `:${metaBlock}:${messagePart}`,
      raw: escapeForTemplateLiteral(
        `:${escapeColons(escapeSlashes(metaBlock))}:${escapeSlashes(messagePart)}`,
      ),
      range,
    };
  }
}

export function getUniqueIDForClassProperty(
  property: ts.ClassElement,
  info: ProgramInfo,
): ClassFieldUniqueKey | null {
  if (!ts.isClassDeclaration(property.parent) || property.parent.name === undefined) {
    return null;
  }
  if (property.name === undefined) {
    return null;
  }
  const id = projectFile(property.getSourceFile(), info).id.replace(/\.d\.ts$/, '.ts');

  // Note: If a class is nested, there could be an ID clash.
  // This is highly unlikely though, and this is not a problem because
  // in such cases, there is even less chance there are any references to
  // a non-exported classes; in which case, cross-compilation unit references
  // likely can't exist anyway.

  return `${id}-${property.parent.name.text}-${property.name.getText()}` as ClassFieldUniqueKey;
}

