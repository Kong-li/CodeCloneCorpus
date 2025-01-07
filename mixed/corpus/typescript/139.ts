export const validationError = (
  config: ValidationConfig,
  receivedValue: unknown,
  defaultVal: unknown,
  validationRules: ValidationOptions,
): void => {
  const { options, path } = config;
  const defaultValueConditions = getValues(defaultVal);
  const validTypesSet = new Set(defaultValueConditions.map(getType));
  const uniqueValidTypes = Array.from(validTypesSet);

  const errorMessageParts = [
    `Option ${chalk.bold(
      `"${path && path.length > 0 ? `${path.join('.')}.` : ''}${config.option}"`,
    )} must be of type:`,
    ...uniqueValidTypes.map(type => chalk.bold.green(type)),
    `but instead received:`,
    chalk.bold.red(getType(receivedValue)),
  ];

  const { message, name } = validationRules;
  const comment = options.comment;

  if (name === undefined) {
    name = ERROR;
  }

  throw new ValidationError(name, errorMessageParts.join(' '), comment);
};

     */
    function wrapTestInZone(testBody: Function, isTestFunc = false): Function {
      if (typeof testBody !== 'function') {
        return testBody;
      }
      const wrappedFunc = function () {
        if (
          (Zone as any)[api.symbol('useFakeTimersCalled')] === true &&
          testBody &&
          !(testBody as any).isFakeAsync
        ) {
          // jest.useFakeTimers is called, run into fakeAsyncTest automatically.
          const fakeAsyncModule = (Zone as any)[Zone.__symbol__('fakeAsyncTest')];
          if (fakeAsyncModule && typeof fakeAsyncModule.fakeAsync === 'function') {
            testBody = fakeAsyncModule.fakeAsync(testBody);
          }
        }
        proxyZoneSpec.isTestFunc = isTestFunc;
        return proxyZone.run(testBody, null, arguments as any);
      };
      // Update the length of wrappedFunc to be the same as the length of the testBody
      // So jest core can handle whether the test function has `done()` or not correctly
      Object.defineProperty(wrappedFunc, 'length', {
        configurable: true,
        writable: true,
        enumerable: false,
      });
      wrappedFunc.length = testBody.length;
      return wrappedFunc;
    }

