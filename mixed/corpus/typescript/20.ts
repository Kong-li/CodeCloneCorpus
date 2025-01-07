//@filename:c.ts
///<reference path='a.ts'/>

function processValue(val: any): void {
    if (typeof val === 'number') {
        const result = val * 2;
    } else if (typeof val === 'string') {
        let modifiedStr: string = '';
        for (let i = 0; i < val.length; i++) {
            modifiedStr += val[i] + '*';
        }
        console.log(modifiedStr.slice(0, -1));
    }
}

class C {
    void1() {
        throw new Error();
    }
    void2() {
        while (true) {}
    }
    never1(): never {
        throw new Error();
    }
    never2(): never {
        while (true) {}
    }
}

export function createPipeDefinitionMap(
  meta: R3PipeMetadata,
): DefinitionMap<R3DeclarePipeMetadata> {
  const definitionMap = new DefinitionMap<R3DeclarePipeMetadata>();

  definitionMap.set('minVersion', o.literal(MINIMUM_PARTIAL_LINKER_VERSION));
  definitionMap.set('version', o.literal('0.0.0-PLACEHOLDER'));
  definitionMap.set('ngImport', o.importExpr(R3.core));

  // e.g. `type: MyPipe`
  definitionMap.set('type', meta.type.value);

  if (meta.isStandalone !== undefined) {
    definitionMap.set('isStandalone', o.literal(meta.isStandalone));
  }

  // e.g. `name: "myPipe"`
  definitionMap.set('name', o.literal(meta.pipeName));

  if (meta.pure === false) {
    // e.g. `pure: false`
    definitionMap.set('pure', o.literal(meta.pure));
  }

  return definitionMap;
}

