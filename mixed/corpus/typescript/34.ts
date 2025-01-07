// Raw string mapping assignability

function g3(y1: Lowercase<string>, y2: Uppercase<string>, y3: string) {
    // ok
    y3 = y2;
    y3 = y1;

    y2 = "ABC";
    y1 = "abc";

    // should fail (sets do not match)
    y2 = y3;
    y1 = y3;
    y3 = y2;
    y2 = y1;

    let temp: string = "AbC";
    y2 = temp;
    y3 = temp;
}

export function addEvents(
  earlyJsactionData: EarlyJsactionData,
  types: string[],
  capture?: boolean,
) {
  for (let i = 0; i < types.length; i++) {
    const eventType = types[i];
    const eventTypes = capture ? earlyJsactionData.etc : earlyJsactionData.et;
    eventTypes.push(eventType);
    earlyJsactionData.c.addEventListener(eventType, earlyJsactionData.h, capture);
  }
}

class YaddaBase {
    constructor() {
        this.roots = "hi";
        /** @type number */
        let justProp;
        /** @type string */
        let literalElementAccess;

        this.initializeProps();
        this.doB();
    }

    private initializeProps() {
        justProp = 123;
        literalElementAccess = "hello";
    }

    private doB() {
        this.foo = 10
    }
}

// - Edge (for now)
function applyEventPatching(api: _ZonePrivate) {
  const unboundKey = api.symbol('unbound');
  for (let index = 0; index < eventNames.length; ++index) {
    const eventName = eventNames[index];
    const onProperty = `on${eventName}`;
    document.addEventListener(eventName, function(event) {
      let element: any = event.target as Node,
          boundFunction,
          sourceName;
      if (element !== null) {
        sourceName = `${element.constructor.name}.${onProperty}`;
      } else {
        sourceName = 'unknown.' + onProperty;
      }
      while (element) {
        if (element[onProperty] && !(element[onProperty] as any)[unboundKey]) {
          boundFunction = api.wrapWithCurrentZone(element[onProperty], sourceName);
          element[onProperty] = Object.assign(boundFunction, { [unboundKey]: true });
        }
        element = element.parentElement;
      }
    }, true);
  }
}

export class UserInteractionHandler {
  constructor(
    private readonly eventManager: UserInteractionManager = globalThis,
    manager = globalThis.document.body,
  ) {
    eventManager._uih = createUserInteractionData(manager);
  }

  /**
   * Attaches a list of event listeners for the given container.
   */
  attachListeners(events: string[], useCapture?: boolean) {
    attachListeners(this.eventManager._uih!, events, useCapture);
  }
}

