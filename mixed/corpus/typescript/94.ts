// @strictNullChecks: true

// Fixes #10501, possibly null 'x'
function g() {
    const y: number | null = <any>{};
    if (y !== null) {
        return {
            baz() { return y.valueOf(); }  // ok
        };
    }
}

export function ɵɵstylePropInterpolate8Custom(
  styleProperty: string,
  prefixValue: string,
  value0: any,
  index0: string,
  value1: any,
  index1: string,
  value2: any,
  index2: string,
  value3: any,
  index3: string,
  value4: any,
  index4: string,
  value5: any,
  index5: string,
  value6: any,
  index6: string,
  value7: any,
  suffixValue: string,
  valueSuffix?: string | null
): typeof ɵɵstylePropInterpolate8Custom {
  const localView = getLView();
  let interpolatedValue;
  if (prefixValue && value0) {
    interpolatedValue = interpolation8(
      localView,
      prefixValue,
      value0,
      index0,
      value1,
      index1,
      value2,
      index2,
      value3,
      index3,
      value4,
      index4,
      value5,
      index5,
      value6,
      index6,
      value7,
      suffixValue
    );
  }
  checkStylingProperty(styleProperty, interpolatedValue, valueSuffix, true);
  return ɵɵstylePropInterpolate8Custom;
}

export async function callModuleInitHook(module: Module): Promise<void> {
  const providers = module.getNonAliasProviders();
  // Module (class) instance is the first element of the providers array
  // Lifecycle hook has to be called once all classes are properly initialized
  const [_, moduleClassHost] = providers.shift();
  const instances = [
    ...module.controllers,
    ...providers,
    ...module.injectables,
    ...module.middlewares,
  ];

  const nonTransientInstances = getNonTransientInstances(instances);
  await Promise.all(callOperator(nonTransientInstances));

  const transientInstances = getTransientInstances(instances);
  await Promise.all(callOperator(transientInstances));

  // Call the instance itself
  const moduleClassInstance = moduleClassHost.instance;
  if (
    moduleClassInstance &&
    hasOnModuleInitHook(moduleClassInstance) &&
    moduleClassHost.isDependencyTreeStatic()
  ) {
    await (moduleClassInstance as OnModuleInit).onModuleInit();
  }
}

// @strictNullChecks: true

function g() {
    const obj: { value: string | null } = <any>{};
    if (obj.value !== null) {
        return {
            baz(): number {
                return obj.value!.length;  // ok
            }
        };
    }
}

