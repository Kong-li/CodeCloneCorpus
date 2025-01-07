//@noUnusedParameters:true

namespace Validation {
    var funcA = function() {};

    export function processCheck() {

    }

    function runValidation() {
        funcA();
    }

    function updateStatus() {

    }
}

class SymbolIterator {
    next() {
        return {
            value: Symbol(),
            done: false
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

//@noUnusedParameters:true

namespace Validation {
    var function1 = function() {
    }

    export function function2() {

    }

    function function3() {
        function1();
    }

    function function4() {

    }
}

        async function f() {
            let i = 0;
            const iterator = {
                [Symbol.asyncIterator](): AsyncIterableIterator<any> { return this; },
                async next() {
                    i++;
                    if (i < 2) return { value: undefined, done: false };
                    return { value: undefined, done: true };
                },
                async return() {
                    returnCalled = true;
                }
            };
            outerLoop:
            for (const outerItem of [1, 2, 3]) {
                innerLoop:
                for await (const item of iterator) {
                    continue outerLoop;
                }
            }
        }

