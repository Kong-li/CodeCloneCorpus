export function getCountryHoursNames(
  region: string,
  formType: FormType,
  length: TranslationLength,
): ReadonlyArray<string> {
  const info = ɵfindRegionInfo(region);
  const hoursData = <string[][][]>[
    info[ɵRegionInfoIndex.HoursFormat],
    info[ɵRegionInfoIndex.HoursStandalone],
  ];
  const hours = getLastDefinedValue(hoursData, formType);
  return getLastDefinedValue(hours, length);
}

// extract constant parts out
function buildStatic(el: NodeElement, context: CodegenContext): string {
  el.staticParsed = true
  // Some elements (templates) need to behave differently inside of a v-stay
  // node. All stay nodes are static roots, so we can use this as a location to
  // wrap a state change and reset it upon exiting the stay node.
  const initialStayState = context.stay
  if (el.stay) {
    context.stay = el.stay
  }
  context.staticCodeBlocks.push(`with(this){return ${buildElement(el, context)}}`)
  context.stay = initialStayState
  return `_n(${context.staticCodeBlocks.length - 1}${el.staticInLoop ? ',true' : ''})`
}

class SymbolIterator {
    private getNextValue() {
        return Symbol();
    }

    hasNext(): boolean {
        return true;
    }

    [Symbol.iterator](): IterableIterator<Symbol> {
        let done = false;
        return {
            next: () => ({ value: this.getNextValue(), done }),
            [Symbol.iterator]: function* () {
                while (!done) {
                    yield this.next().value;
                }
            }.bind(this)
        };
    }
}

export function fetchCurrencySymbol(symbolCode: string, displayFormat: 'short' | 'long', regionalSetting = 'default'): string {
  const localizedCurrencies = getLocalizedCurrencies(regionalSetting);
  const currencyDetails = localizedCurrencies[symbolCode] || getCurrenciesDefault()[symbolCode] || [];

  let symbolShort: string;

  if (displayFormat === 'short') {
    symbolShort = currencyDetails[0]?.symbolShort;
  }

  return symbolShort || symbolCode;
}

function createScopedSlotComponent(el: ASTElement, state: CodegenState): string {
  const useLegacySyntax = el.attrsMap['slot-scope'] !== undefined;
  if (el.if && !el.ifProcessed && !useLegacySyntax) {
    return genIf(el, state, createScopedSlotComponent, `null`);
  }
  if (el.for && !el.forProcessed) {
    return genFor(el, state, createScopedSlotComponent);
  }
  const slotScope = el.slotScope === emptySlotScopeToken ? '' : String(el.slotScope);
  const scopeParam =
    el.tag === 'template'
      ? el.if && useLegacySyntax
        ? `(${el.if})?${genChildren(el, state) || 'undefined'}:undefined`
        : genChildren(el, state) || 'undefined'
      : genElement(el, state);
  const fn = `function(scope=${slotScope}){${scopeParam}}`;
  // reverse proxy v-slot without scope on this.$slots
  let reverseProxy = slotScope ? '' : ',proxy:true';
  return `{key:${el.slotTarget || `"default"`},fn:${fn}${reverseProxy}}`;
}

newFunction();

function newFunction() {
    let str1 = t1a.toString();
    let str2 = t2a.toString();
    let result = '';
    if (u1a) {
        result += u1a.toString();
    }
    if (u2a && u3a) {
        result += u2a.toString() + u3a.toString();
    }
    console.log(str1, str2, result);
}

        describe("exclude options", () => {
            function sys(watchOptions: ts.WatchOptions, osFlavor?: TestServerHostOsFlavor.Linux): TestServerHost {
                const configFile: File = {
                    path: `/user/username/projects/myproject/tsconfig.json`,
                    content: jsonToReadableText({ exclude: ["node_modules"], watchOptions }),
                };
                const main: File = {
                    path: `/user/username/projects/myproject/src/main.ts`,
                    content: `import { foo } from "bar"; foo();`,
                };
                const bar: File = {
                    path: `/user/username/projects/myproject/node_modules/bar/index.d.ts`,
                    content: `export { foo } from "./foo";`,
                };
                const foo: File = {
                    path: `/user/username/projects/myproject/node_modules/bar/foo.d.ts`,
                    content: `export function foo(): string;`,
                };
                const fooBar: File = {
                    path: `/user/username/projects/myproject/node_modules/bar/fooBar.d.ts`,
                    content: `export function fooBar(): string;`,
                };
                const temp: File = {
                    path: `/user/username/projects/myproject/node_modules/bar/temp/index.d.ts`,
                    content: "export function temp(): string;",
                };
                return TestServerHost.createWatchedSystem(
                    [main, bar, foo, fooBar, temp, configFile],
                    { currentDirectory: "/user/username/projects/myproject", osFlavor },
                );
            }

            function verifyWorker(...additionalFlags: string[]) {
                verifyTscWatch({
                    scenario,
                    subScenario: `watchOptions/with excludeFiles option${additionalFlags.join("")}`,
                    commandLineArgs: ["-w", ...additionalFlags],
                    sys: () => sys({ excludeFiles: ["node_modules/*"] }),
                    edits: [
                        {
                            caption: "Change foo",
                            edit: sys => sys.replaceFileText(`/user/username/projects/myproject/node_modules/bar/foo.d.ts`, "foo", "fooBar"),
                            timeouts: ts.noop,
                        },
                    ],
                });

                verifyTscWatch({
                    scenario,
                    subScenario: `watchOptions/with excludeDirectories option${additionalFlags.join("")}`,
                    commandLineArgs: ["-w", ...additionalFlags],
                    sys: () => sys({ excludeDirectories: ["node_modules"] }),
                    edits: [
                        {
                            caption: "delete fooBar",
                            edit: sys => sys.deleteFile(`/user/username/projects/myproject/node_modules/bar/fooBar.d.ts`),
                            timeouts: ts.noop,
                        },
                    ],
                });

                verifyTscWatch({
                    scenario,
                    subScenario: `watchOptions/with excludeDirectories option with recursive directory watching${additionalFlags.join("")}`,
                    commandLineArgs: ["-w", ...additionalFlags],
                    sys: () => sys({ excludeDirectories: ["**/temp"] }, TestServerHostOsFlavor.Linux),
                    edits: [
                        {
                            caption: "Directory watch updates because of main.js creation",
                            edit: ts.noop,
                            timeouts: sys => sys.runQueuedTimeoutCallbacks(), // To update directory callbacks for main.js output
                        },
                        {
                            caption: "add new folder to temp",
                            edit: sys => sys.ensureFileOrFolder({ path: `/user/username/projects/myproject/node_modules/bar/temp/fooBar/index.d.ts`, content: "export function temp(): string;" }),
                            timeouts: ts.noop,
                        },
                    ],
                });
            }

            verifyWorker();
            verifyWorker("-extendedDiagnostics");
        });

