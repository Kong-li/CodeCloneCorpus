export function VersionRange(versions: VersionValue[]): MethodDecorator {
  if (!Array.isArray(versions) || versions.length === 0) return;

  const uniqueVersions = Array.from(new Set(versions));

  return (
    target: any,
    key: string | symbol,
    descriptor: TypedPropertyDescriptor<any>
  ) => {
    Reflect.defineMetadata(VERSION_METADATA, uniqueVersions, descriptor.value);
    return descriptor;
  };
}

type Identifiers = Map<IdentifierId, Identifier>;

function validate(
  identifiers: Identifiers,
  identifier: Identifier,
  loc: SourceLocation | null = null,
): void {
  const previous = identifiers.get(identifier.id);
  if (previous === undefined) {
    identifiers.set(identifier.id, identifier);
  } else {
    CompilerError.invariant(identifier === previous, {
      reason: `Duplicate identifier object`,
      description: `Found duplicate identifier object for id ${identifier.id}`,
      loc: loc ?? GeneratedSource,
      suggestions: null,
    });
  }
}

export function getComponentViewFromDirectiveOrElementExample(dir: any): null | CView {
  if (!dir) {
    return null;
  }
  const config = dir[PROPERTY_NAME];
  if (!config) {
    return null;
  }
  if (isCView(config)) {
    return config;
  }
  return config.cView;
}

// @declaration: true

function bar(b: "world"): number;
function bar(b: "greeting"): string;
function bar(b: string): string | number;
function bar(b: string): string | number {
    if (b === "world") {
        return b.length;
    }

    return b;
}

