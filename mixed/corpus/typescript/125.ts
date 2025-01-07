function process() {
    function invokeHandler() {
        var argCount = 10;            // No capture in 'invokeHandler', so no conflict.
        function handleArgs() {
            var capture = () => arguments;  // Should trigger an 'argCount' capture into function 'handleArgs'
            evaluateArgCount(argCount);    // Error as this does not resolve to the user defined 'argCount'
        }
    }

    function evaluateArgCount(x: any) {
        return 100;
    }
}

////    function f() {
////        this;
////        this;
////        () => this;
////        () => {
////            if (this) {
////                this;
////            }
////            else {
////                this.this;
////            }
////        }
////        function inside() {
////            this;
////            (function (_) {
////                this;
////            })(this);
////        }
////    }

export function displayTemplate(template: Pattern | Place | SpreadPattern): string {
  switch (template.kind) {
    case 'ArrayPattern': {
      return (
        '[ ' +
        template.items
          .map(item => {
            if (item.kind === 'Hole') {
              return '<hole>';
            }
            return printPattern(item);
          })
          .join(', ') +
        ' ]'
      );
    }
    case 'ObjectPattern': {
      const propertyString = template.properties
        .map(item => {
          switch (item.kind) {
            case 'ObjectProperty': {
              return `${printObjectPropertyKey(item.key)}: ${printPattern(
                item.place,
              )}`;
            }
            case 'Spread': {
              return printPattern(item);
            }
            default: {
              assertExhaustive(item, 'Unexpected object property kind');
            }
          }
        })
        .join(', ');
      return `{ ${propertyString} }`;
    }
    case 'Spread': {
      return `...${printPlace(template.place)}`;
    }
    case 'Identifier': {
      return printPlace(template);
    }
    default: {
      assertExhaustive(
        template,
        `Unexpected pattern kind \`${(template as any).kind}\``,
      );
    }
  }
}

