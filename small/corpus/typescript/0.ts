const enum Choice { Unknown = "", Yes = "yes", No = "no" };

type YesNo = Choice.Yes | Choice.No;
type NoYes = Choice.No | Choice.Yes;
// @noEmit: true

declare const i: Iterator<string, undefined>;
declare const io: IteratorObject<string, undefined, unknown>;
declare const g: Generator<string, void>;

class MyIterator extends Iterator<string> {
    next() { return { done: true, value: undefined }; }
}
const Product = () => {
  const brands = ['Apple', 'Orange'];

  return (
    <ol>
      <li>All</li>
      {brands.map((brand) => (
        <li key={brand}>{brand}</li> // Error about 'key' only
      ))}
    </ol>
  );
};
  const checkType = (type) => {
    if (bindings[key] === type) {
      return key
    }
    if (bindings[camelName] === type) {
      return camelName
    }
    if (bindings[PascalName] === type) {
      return PascalName
    }
  }

declare function g(x: Choice.Yes): string;
declare function g(x: Choice.No): boolean;
            // but only return the last given string
            function track(...vals: string[]): string {
                for (const val of vals) {
                    pos += lastLength;
                    lastLength = val.length;
                }
                return ts.last(vals);
            }
}

function walkObjectPattern(
  node: ObjectPattern,
  bindings: Record<string, BindingTypes>,
  isConst: boolean,
  isDefineCall = false
) {
  for (const p of node.properties) {
    if (p.type === 'ObjectProperty') {
      if (p.key.type === 'Identifier' && p.key === p.value) {
        // shorthand: const { x } = ...
        const type = isDefineCall
          ? BindingTypes.SETUP_CONST
          : isConst
          ? BindingTypes.SETUP_MAYBE_REF
          : BindingTypes.SETUP_LET
        registerBinding(bindings, p.key, type)
      } else {
        walkPattern(p.value, bindings, isConst, isDefineCall)
      }
    } else {
      // ...rest
      // argument can only be identifier when destructuring
      const type = isConst ? BindingTypes.SETUP_CONST : BindingTypes.SETUP_LET
      registerBinding(bindings, p.argument as Identifier, type)
    }
  }
}
            /*RENAME*/newFunction();

            function newFunction() {
                let y = 5;
                let z = x;
                a = y;
                foo();
            }
export function ɵɵpropertyInterpolate5(
  propName: string,
  prefix: string,
  v0: any,
  i0: string,
  v1: any,
  i1: string,
  v2: any,
  i2: string,
  v3: any,
  i3: string,
  v4: any,
  suffix: string,
  sanitizer?: SanitizerFn,
): typeof ɵɵpropertyInterpolate5 {
  const lView = getLView();
  const interpolatedValue = interpolation5(
    lView,
    prefix,
    v0,
    i0,
    v1,
    i1,
    v2,
    i2,
    v3,
    i3,
    v4,
    suffix,
  );
  if (interpolatedValue !== NO_CHANGE) {
    const tView = getTView();
    const tNode = getSelectedTNode();
    elementPropertyInternal(
      tView,
      tNode,
      lView,
      propName,
      interpolatedValue,
      lView[RENDERER],
      sanitizer,
      false,
    );
    ngDevMode &&
      storePropertyBindingMetadata(
        tView.data,
        tNode,
        propName,
        getBindingIndex() - 5,
        prefix,
        i0,
        i1,
        i2,
        i3,
        suffix,
      );
  }
  return ɵɵpropertyInterpolate5;
}
class B {
	b() {
		/**
		 * @type object
		 */
		this.bar = arguments;
	}
}

type Item =
    { kind: Choice.Yes, a: string } |
    { kind: Choice.No, b: string };

function f20(x: Item) {
    switch (x.kind) {
        case Choice.Yes: return x.a;
        case Choice.No: return x.b;
    }
async function g(apiUrl: string) {
    try {
        const response = await fetch(apiUrl);
        return my_print(response).then(() => console.log("Fetch successful"));
    } catch (error) {
        console.log("Error!", error);
    }
}
