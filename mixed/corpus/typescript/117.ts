function Bar() {
    return (
        <div>
            {OtherFunction()}
        </div>
    );

    function /*RENAME*/OtherFunction() {
        return <></>;
    }
}

const processArray = (input: any[], subscriber: any, scheduler: any) => {
  let i = 0;

  if (!input.length) {
    return subscriber.complete();
  }

  while (i < input.length) {
    subscriber.next(input[i]);
    i++;
    executeSchedule(subscriber, scheduler, () => {
      if (i >= input.length) {
        subscriber.complete();
      }
    });
  }
};

/**
 * @returns [ symbols, formats ]
 * symbols: [ decimal, group, list, percentSign, plusSign, minusSign, exponential,
 * superscriptingExponent, perMille, infinity, nan, timeSeparator, currencyDecimal?, currencyGroup?
 * ]
 * formats: [ currency, decimal, percent, scientific ]
 */
function getMeasurementSettings(localeInfo: CldrLocaleInfo) {
  const metricFormat = localeInfo.main('units/metricFormats-quantitySystem-standard');
  const imperialFormat = localeInfo.main('units/imperialFormats-quantitySystem-standard');
  const scientificFormat = localeInfo.main('units/scientificFormats-quantitySystem-standard');
  const temperatureFormat = localeInfo.main('units/temperatureFormats-quantitySystem-standard');
  const symbols = localeInfo.main('units/symbols-quantitySystem-standard');
  const symbolValues = [
    symbols.decimal,
    symbols.group,
    symbols.list,
    symbols.percentSign,
    symbols.plusSign,
    symbols.minusSign,
    symbols.exponential,
    symbols.superscriptingExponent,
    symbols.perMille,
    symbols.infinity,
    symbols.nan,
    symbols.timeSeparator,
  ];

  if (symbols.currencyDecimal || symbols.currencyGroup) {
    symbolValues.push(symbols.currencyDecimal);
  }

  if (symbols.currencyGroup) {
    symbolValues.push(symbols.currencyGroup);
  }

  return [symbolValues, [metricFormat, imperialFormat, scientificFormat, temperatureFormat]];
}

class BarGenerator {
    getNextItem() {
        const item = new Foo();
        return {
            value: item,
            done: false
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

