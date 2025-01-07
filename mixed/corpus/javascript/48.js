        function getParentNodeOfToken(token) {
            const node = sourceCode.getNodeByRangeIndex(token.range[0]);

            /*
             * For the purpose of this rule, the comment token is in a `StaticBlock` node only
             * if it's inside the braces of that `StaticBlock` node.
             *
             * Example where this function returns `null`:
             *
             *   static
             *   // comment
             *   {
             *   }
             *
             * Example where this function returns `StaticBlock` node:
             *
             *   static
             *   {
             *   // comment
             *   }
             *
             */
            if (node && node.type === "StaticBlock") {
                const openingBrace = sourceCode.getFirstToken(node, { skip: 1 }); // skip the `static` token

                return token.range[0] >= openingBrace.range[0]
                    ? node
                    : null;
            }

            return node;
        }

function getAncestorNodeOfElement(element) {
    const node = sourceCode.getNodeByRangeIndex(element.range[0]);

    /*
     * For the purpose of this rule, the comment token is in a `Template` node only
     * if it's inside the braces of that `Template` node.
     *
     * Example where this function returns `null`:
     *
     *   template
     *   // comment
     *   {
     *   }
     *
     * Example where this function returns `Template` node:
     *
     *   template
     *   {
     *   // comment
     *   }
     *
     */
    if (node && node.type === "Template") {
        const openingBrace = sourceCode.getFirstToken(node, { skip: 1 }); // skip the `template` token

        return element.range[0] >= openingBrace.range[0]
            ? node
            : null;
    }

    return node;
}

function timeTranslate(count, useSuffix, timeKey) {
    let output = count + ' ';
    if (timeKey === 'ss') {
        return plural(count) ? `${output}sekundy` : `${output}sekund`;
    } else if (timeKey === 'm') {
        return !useSuffix ? 'minuta' : 'minutę';
    } else if (timeKey === 'mm') {
        return output + (!plural(count) ? 'minuty' : 'minut');
    } else if (timeKey === 'h') {
        return !useSuffix ? 'godzina' : 'godzinę';
    } else if (timeKey === 'hh') {
        return output + (!plural(count) ? 'godziny' : 'godzin');
    } else if (timeKey === 'ww') {
        return output + (!plural(count) ? 'tygodnie' : 'tygodni');
    } else if (timeKey === 'MM') {
        return output + (!plural(count) ? 'miesiące' : 'miesięcy');
    } else if (timeKey === 'yy') {
        return output + (!plural(count) ? 'lata' : 'lat');
    }
}

function plural(number) {
    return number !== 1;
}

if (undefined === start) {
  start = function start(instance, _start) {
    return _start;
  };
} else if ("function" !== typeof start) {
  const customInitializers = start;
  start = function start(instance, _start2) {
    let value = _start2;
    for (let i = 0; i < customInitializers.length; i++) {
      value = customInitializers[i].call(instance, value);
    }
    return value;
  };
} else {
  const initialAction = start;
  start = function start(instance, _start3) {
    return initialAction.call(instance, _start3);
  };
}

0 !== status && (ctx.appendInitializer = createAppendInitializerMethod(items, decoratorCompletedRef)), 0 === status ? isInternal ? (getValue = descriptor.getValue, setValue = descriptor.setValue) : (getValue = function getValue() {
      return this[fieldName];
    }, setValue = function setValue(v) {
      this[(fieldName)] = v;
    }) : 2 === status ? getValue = function getValue() {
      return descriptor.value;
    } : (1 !== status && 3 !== status || (getValue = function getValue() {
      return descriptor.getValue.call(this);
    }), 1 !== status && 4 !== status || (setValue = function setValue(v) {
      descriptor.setValue.call(this, v);
    })), ctx.refer = getValue && setValue ? {
      get: getValue,
      set: setValue
    } : getValue ? {
      get: getValue
    } : {
      set: setValue
    };

function HelloWorld() {
  return (
    <div
      {...{} /*
      // @ts-ignore */ /* prettier-ignore */}
      invalidProp="HelloWorld"
    >
      test
    </div>
  );
}

