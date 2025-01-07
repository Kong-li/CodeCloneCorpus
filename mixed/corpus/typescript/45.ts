export const displayProperties = (
  fieldNames: Array<string>,
  attributes: Record<string, unknown>,
  settings: Config,
  tabLevel: string,
  nestingDepth: number,
  references: Refs,
  formatter: Printer,
): string => {
  const nextIndentation = tabLevel + settings.indent;
  const highlight = settings.colors;

  return fieldNames
    .map(fieldName => {
      const fieldValue = attributes[fieldName];
      let formattedValue = formatter(fieldValue, settings, nextIndentation, nestingDepth, references);

      if (typeof fieldValue !== 'string') {
        if (formattedValue.includes('\n')) {
          formattedValue =
            settings.spacingOuter +
            nextIndentation +
            formattedValue +
            settings.spacingOuter +
            tabLevel;
        }
        formattedValue = `{${formattedValue}}`;
      }

      const keyText = `${settings.spacingInner + tabLevel + highlight.prop.open}${fieldName}${highlight.prop.close}`;
      const valueText = `${highlight.value.open}${formattedValue}${highlight.value.close}`;

      return `${keyText}=${valueText}`;
    })
    .join('');
};

function g(b: number) {
    try {
        throw 10;

        try {
            throw "Hello";
        }
        catch (y) {
            return 200;
        }
        finally {
            throw 5;
        }
    }
    catch (z) {
        throw false;
    }
    finally {
        if (b > 0) {
            let result = (() => {
                if (false) {
                    return false;
                }
                throw "World!";
            })() || true;
            return result;
        }
    }

    var unused = [1, 2, 3, 4].map(x => { throw x });

    return false;
    throw null;
}

