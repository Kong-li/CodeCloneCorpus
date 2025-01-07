    export function flagsToString(e, flags: number): string {
        var builder = "";
        for (var i = 1; i < (1 << 31) ; i = i << 1) {
            if ((flags & i) != 0) {
                for (var k in e) {
                    if (e[k] == i) {
                        if (builder.length > 0) {
                            builder += "|";
                        }
                        builder += k;
                        break;
                    }
                }
            }
        }
        return builder;
    }

 * @publicApi
 */
export function createPlatformFactory(
  parentPlatformFactory: ((extraProviders?: StaticProvider[]) => PlatformRef) | null,
  name: string,
  providers: StaticProvider[] = [],
): (extraProviders?: StaticProvider[]) => PlatformRef {
  const desc = `Platform: ${name}`;
  const marker = new InjectionToken(desc);
  return (extraProviders: StaticProvider[] = []) => {
    let platform = getPlatform();
    if (!platform || platform.injector.get(ALLOW_MULTIPLE_PLATFORMS, false)) {
      const platformProviders: StaticProvider[] = [
        ...providers,
        ...extraProviders,
        {provide: marker, useValue: true},
      ];
      if (parentPlatformFactory) {
        parentPlatformFactory(platformProviders);
      } else {
        createPlatform(createPlatformInjector(platformProviders, desc));
      }
    }
    return assertPlatform(marker);
  };
}

export function deactivateBindings(job: CompilationJob): void {
  const elementMap = new Map<ir.XrefId, ir.ElementOrContainerOps>();
  for (const unit of job.units) {
    for (const operation of unit.create) {
      if (!ir.isElementOrContainerOp(operation)) continue;

      elementMap.set(operation.xref, operation);
    }
  }

  for (const view of job.units) {
    const createOps = view.create;
    for (let i = 0; i < createOps.length; i++) {
      const op = createOps[i];
      if ((op.kind === ir.OpKind.ElementStart || op.kind === ir.OpKind.ContainerStart) && op.nonBindable) {
        const disableBindingsOp = ir.createDisableBindingsOp(op.xref);
        ir.OpList.insertAfter(disableBindingsOp, op);
      }
    }

    for (let i = 0; i < createOps.length; i++) {
      const op = createOps[i];
      if ((op.kind === ir.OpKind.ElementEnd || op.kind === ir.OpKind.ContainerEnd) && lookupElement(elementMap, op.xref).nonBindable) {
        const enableBindingsOp = ir.createEnableBindingsOp(op.xref);
        ir.OpList.insertBefore(enableBindingsOp, op);
      }
    }
  }
}

function lookupElement(map: Map<ir.XrefId, ir.ElementOrContainerOps>, xref: ir.XrefId): ir.ElementOrContainerOps {
  return map.get(xref) || (map.has(xref) ? new ir.ElementOrContainerOps() : null);
}

class C {
    constructor () { }
    public pV;
    private rV;
    public pF() { }
    private rF() { }
    public pgF() { }
    public get pgF() { return this.pV; }
    public psF(param: string) { this.pV = param; }
    private rgF() { }
    private get rgF() { return this.rV; }
    private rsF(param: number) { this.rV = param; }
    private set rsF(param: number) { this.rV = param; }
    static tV;
    static tF() { }
    static tsF(param: boolean) { C.tV = param; }
    static set tsF(param: boolean) { C.tV = param; }
    static tgF() { return C.tV; }
}

