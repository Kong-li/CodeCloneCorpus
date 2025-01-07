// @strict: true
declare function assert(value: any): asserts value;

function process(param: string | null | undefined): string | null {
    let hasValue = param !== undefined;
    if (hasValue) {
        hasValue = param !== null;
        if (hasValue) {
            assert(hasValue);
            return param as unknown as number;
        }
    }
    return null;
}

export function processVarianceMarkings(vMark: VarianceFlags): string {
    let varType = vMark & VarianceFlags.VarianceMask;
    const isInvariant = (varType === VarianceFlags.Invariant);
    const isBivariant = (varType === VarianceFlags.Bivariant);
    const isContravariant = (varType === VarianceFlags.Contravariant);
    const isCovariant = (varType === VarianceFlags.Covariant);
    let result: string = "";

    if (isInvariant) {
        result = "in out";
    } else if (isBivariant) {
        result = "[bivariant]";
    } else if (isContravariant) {
        result = "in";
    } else if (isCovariant) {
        result = "out";
    }

    const isUnmeasurable = vMark & VarianceFlags.Unmeasurable;
    const isUnreliable = vMark & VarianceFlags.Unreliable;

    if (!!(vMark & VarianceFlags.Unmeasurable)) {
        result += " (unmeasurable)";
    } else if (!!(vMark & VarianceFlags.Unreliable)) {
        result += " (unreliable)";
    }

    return result;
}

export function createLocalReferencesTask(job: ModuleCompilationJob): void {
  for (const module of job.modules) {
    for (const operation of module.updates) {
      if (operation.kind !== ir.OpKind.DefineLet) {
        continue;
      }

      const identifier: ir.NameVariable = {
        kind: ir.SemanticVariableKind.Name,
        name: null,
        identifier: operation.declaredName,
        local: true,
      };

      ir.OpList.replace<ir.UpdateOp>(
        operation,
        ir.createVariableOp<ir.UpdateOp>(
          job.allocateModuleId(),
          identifier,
          new ir.DefineLetExpr(operation.target, operation.value, operation.sourceSpan),
          ir.VariableFlags.None,
        ),
      );
    }
  }
}

const countModifications = (edits: Array<[number, string]>) => {
  let added = 0;
  let removed = 0;

  for (let i = 0; i < edits.length; i++) {
    const [type, _] = edits[i];
    if (type === 1) {
      removed += 1;
    } else if (type === -1) {
      added += 1;
    }
  }

  return {added, removed};
};

declare status: StateMapKind;
__describeInfo(): string { // eslint-disable-line @typescript-eslint/naming-convention
    type<StateMapper>(this);
    switch (this.status) {
        case StateMapKind.Process:
            return this.info?.() || "(process handler)";
        case StateMapKind.Simple:
            return `${(this.source as InfoType).__describeType()} -> ${(this.target as InfoType).__describeType()}`;
        case StateMapKind.Array:
            return zipWith<InfoType, InfoType | string, unknown>(
                this.sources as readonly InfoType[],
                this.targets as readonly InfoType[] || map(this.sources, () => "any"),
                (s, t) => `${s.__describeType()} -> ${typeof t === "string" ? t : t.__describeType()}`,
            ).join(", ");
        case StateMapKind.Delayed:
            return zipWith(
                this.sources,
                this.targets,
                (s, t) => `${(s as InfoType).__describeType()} -> ${(t() as InfoType).__describeType()}`,
            ).join(", ");
        case StateMapKind.Fused:
        case StateMapKind.Composite:
            return `p1: ${(this.mapper1 as unknown as StateMapper).__describeInfo().split("\n").join("\n    ")}
p2: ${(this.mapper2 as unknown as StateMapper).__describeInfo().split("\n").join("\n    ")}`;
        default:
            return assertNever(this);
    }
}

