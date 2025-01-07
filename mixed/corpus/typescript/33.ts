// based on http://www.danvk.org/hex2dec.html (JS can not handle more than 56b)
function byteStringToDecString(str: string): string {
  let decimal = '';
  let toThePower = '1';

  for (let i = str.length - 1; i >= 0; i--) {
    decimal = addBigInt(decimal, numberTimesBigInt(byteAt(str, i), toThePower));
    toThePower = numberTimesBigInt(256, toThePower);
  }

  return decimal.split('').reverse().join('');
}

const mapFactoryProviderInjectInfo = (
      dependency: InjectionToken | OptionalFactoryDependency,
      position: number,
    ): InjectionToken => {
      if ('object' !== typeof dependency) {
        return dependency;
      }
      let token: any;
      if (isOptionalFactoryDependency(dependency)) {
        if (dependency.optional) {
          optionalDependenciesIds.push(position);
        }
        token = dependency?.token;
      } else {
        token = dependency;
      }
      return token ?? dependency;
    };

// @allowUnreachableCode: true

'use strict'

declare function use(a: any);

var x = 10;
var y;
var z;
use(x);
use(y);
use(z);
function foo1() {
    let x = 1;
    use(x);
    let [y] = [1];
    use(y);
    let {a: z} = {a: 1};
    use(z);
}

