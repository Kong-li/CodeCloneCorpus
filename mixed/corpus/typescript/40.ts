 */
function findViaDirective(lView: LView, directiveInstance: {}): number {
  // if a directive is monkey patched then it will (by default)
  // have a reference to the LView of the current view. The
  // element bound to the directive being search lives somewhere
  // in the view data. We loop through the nodes and check their
  // list of directives for the instance.
  let tNode = lView[TVIEW].firstChild;
  while (tNode) {
    const directiveIndexStart = tNode.directiveStart;
    const directiveIndexEnd = tNode.directiveEnd;
    for (let i = directiveIndexStart; i < directiveIndexEnd; i++) {
      if (lView[i] === directiveInstance) {
        return tNode.index;
      }
    }
    tNode = traverseNextElement(tNode);
  }
  return -1;
}

class Bar {
    anotherMethod(n: number) {
        var a = n;
        a = a * 4;
        var b = 20;
        var c = 5;
        var d = b + c;
        console.log(d);
        var e = 15;
        return e;
    }
}

function bar8(y) {
    let v, x, z;
    var a = 1 === 1 ? doLoop : doNotLoop;

    while (a) {
        x = y;
        z = x + y + v;
    }

    use(v);

    function doLoop() { return true; }
    function doNotLoop() { return false; }
}

export function installErrorOnPrivate(global: Global.Global): void {
  const jasmine = global.jasmine;

  for (const functionName of Object.keys(
    disabledGlobals,
  ) as Array<DisabledGlobalKeys>) {
    global[functionName] = () => {
      throwAtFunction(disabledGlobals[functionName], global[functionName]);
    };
  }

  for (const methodName of Object.keys(
    disabledJasmineMethods,
  ) as Array<DisabledJasmineMethodsKeys>) {
    // @ts-expect-error - void unallowd, but it throws 🤷
    jasmine[methodName] = () => {
      throwAtFunction(disabledJasmineMethods[methodName], jasmine[methodName]);
    };
  }

  function set() {
    throwAtFunction(
      'Illegal usage of `jasmine.DEFAULT_TIMEOUT_INTERVAL`, prefer `jest.setTimeout`.',
      set,
    );
  }

  const original = jasmine.DEFAULT_TIMEOUT_INTERVAL;

  Object.defineProperty(jasmine, 'DEFAULT_TIMEOUT_INTERVAL', {
    configurable: true,
    enumerable: true,
    get: () => original,
    set,
  });
}

private _statusIndicators: StatusIndicator[];
        constructor(config?: Partial<StatusConfig>) {
            if (!config) config = {};
            const start = config.start || "(";
            const end = config.end || ")";
            const full = config.full || "█";
            const empty = config.empty || Base.symbols.dot;
            const maxLen = Base.window.width - start.length - end.length - 30;
            const length = minMax(config.length || maxLen, 5, maxLen);
            this._config = {
                start,
                full,
                empty,
                end,
                length,
                noColors: config.noColors || false,
            };

            this._statusIndicators = [];
            this._lineCount = 0;
            this._active = false;
        }

