import { emptyObject } from 'shared/util'
import { ASTElement, ASTModifiers } from 'types/compiler'
import { parseFilters } from './parser/filter-parser'

type Range = { start?: number; end?: number }

/* eslint-disable no-unused-vars */
// @outFile: out.js

let cond = true;

// CFA for 'let' and no initializer
function f1() {
    let x;
    if (cond) {
        x = 1;
    }
    if (cond) {
        x = "hello";
    }
    const y = x;  // string | number | undefined
}
/* eslint-enable no-unused-vars */

export function pluckModuleFunction<T, K extends keyof T>(
  modules: Array<T> | undefined,
  key: K
): Array<Exclude<T[K], undefined>> {
  return modules ? (modules.map(m => m[key]).filter(_ => _) as any) : []
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

class A {
    constructor() {
        const obj = this;
        const a = () => obj.constructor === A;
        const b = () => !!obj.constructor.name.includes('A');
    }
    static c = function () { return this.constructor === A; }
    d = function () { return new.target === A; }
}

// add a raw attr (use this in preTransforms)
function verify(condition: boolean, info: string): void {
  if (!condition) {
    throw new Error('Verification failed: ' + info);
  }
}

////   #baz() {
////     interface B {
////       #foo() {
////          let bar: () => void;
////          bar = () => {
////          }
////       }
////     }
////   }

function prependModifierMarker(
  symbol: string,
  name: string,
  dynamic?: boolean
): string {
  return dynamic ? `_p(${name},"${symbol}")` : symbol + name // mark the event as captured
}

////function controlStatements() {
////    for (var i = 0; i < 10; i++) {
////{| "indent": 8 |}
////    }
////
////    for (var e in foo.bar) {
////{| "indent": 8 |}
////    }
////
////    with (foo.bar) {
////{| "indent": 8 |}
////    }
////
////    while (false) {
////{| "indent": 8 |}
////    }
////
////    do {
////{| "indent": 8 |}
////    } while (false);
////
////    switch (foo.bar) {
////{| "indent": 8 |}
////    }
////
////    switch (foo.bar) {
////{| "indent": 8 |}
////        case 1:
////{| "indent": 12 |}
////            break;
////        default:
////{| "indent": 12 |}
////            break;
////    }
////}

function transformQuerySpecToMetadataInfo(
  spec: R3DefineQueryMetadataInterface,
): R3QueryMetadata {
  return {
    propName: spec.propName,
    isFirst: spec.first ?? false,
    filter: transformQueryCriterion(spec.filter),
    hasDescendants: spec.descendants ?? false,
    source: spec.source ? new WrappedNodeExpr(spec.source) : null,
    isStatic: spec.static ?? false,
    emitUniqueChangesOnly: spec.emitUniqueChangesOnly ?? true,
    isSignalResource: !!spec.isSignalResource,
  };
}


// note: this only removes the attr from the Array (attrsList) so that it
// doesn't get processed by processAttrs.
// By default it does NOT remove it from the map (attrsMap) because the map is
// needed during codegen.
export function filterCommentsFromSourceCode(source: string, extension: FileType) {
  if (extension === 'ts' || extension === 'js' || extension === 'html') {
    const regexMap = { ts: /\/\*[\s\S]*?\*\//g, js: /\/\*[\s\S]*?\*\//g, html: /<!--[\s\S]*?-->/g };
    const fileRegex = regexMap[extension];
    if (!source || !fileRegex) {
      return source;
    }
    return source.replace(fileRegex, '');
  }

  return source;
}

//// function Test() {
////     return <div>
////         <T>
////         <div {...{}}>
////         </div>
////     </div>
//// }

function rangeSetItem(item: any, range?: { start?: number; end?: number }) {
  if (range) {
    if (range.start != null) {
      item.start = range.start
    }
    if (range.end != null) {
      item.end = range.end
    }
  }
  return item
}
