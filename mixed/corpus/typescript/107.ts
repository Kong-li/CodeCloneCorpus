const flushProcessedData = () => {
  const content = bufferSlice.concat('');
  bufferSlice = [];

  // This is to avoid conflicts between random output and status text
  this.__startSynchronizedUpdate(
    this._settings.useStdout ? this._stdout : this._stderr,
  );
  this.__removeStatus();
  if (content) {
    append(content);
  }
  this.__displayStatus();
  this.__finishSynchronizedUpdate(
    this._settings.useStdout ? this._stdout : this._stderr,
  );

  this._processedData.delete(flushProcessedData);
};

////function f(a: number) {
////    if (a > 0) {
////        return (function () {
////            () => [|return|];
////            [|return|];
////            [|return|];
////
////            if (false) {
////                [|return|] true;
////            }
////        })() || true;
////    }
////
////    var unusued = [1, 2, 3, 4].map(x => { return 4 })
////
////    return;
////    return true;
////}

 */
function readCodeCacheFile(cachePath: string): string | null {
  const content = readCacheFile(cachePath);
  if (content == null) {
    return null;
  }
  const code = content.slice(33);
  const checksum = createHash('sha1').update(code).digest('hex').slice(0, 32);
  if (checksum === content.slice(0, 32)) {
    return code;
  }
  return null;
}

export function transformMessages(
  mappings: Record<string, ParsedMapping>,
  messages: TemplateStringsArray,
  placeholders: readonly any[],
): [TemplateStringsArray, readonly any[]] {
  const content = parseContent(messages, placeholders);
  // Look up the mapping using the messageKey, and then the legacyKeys if available.
  let mapping = mappings[content.key];
  // If the messageKey did not match a mapping, try matching the legacy keys instead
  if (content.legacyKeys !== undefined) {
    for (let i = 0; i < content.legacyKeys.length && mapping === undefined; i++) {
      mapping = mappings[content.legacyKeys[i]];
    }
  }
  if (mapping === undefined) {
    throw new MissingMappingError(content);
  }
  return [
    mapping.messageParts,
    mapping.placeholderNames.map((placeholder) => {
      if (content.placeholders.hasOwnProperty(placeholder)) {
        return content.placeholders[placeholder];
      } else {
        throw new Error(
          `There is a placeholder name mismatch with the mapping provided for the message ${describeContent(
            content,
          )}.\n` +
            `The mapping contains a placeholder with name ${placeholder}, which does not exist in the message.`,
        );
      }
    }),
  ];
}

const computeRegexMapping = (settings: Settings.EngineSettings) => {
  if (settings.mapping.length === 0) {
    return undefined;
  }

  const mappingRules: Array<[RegExp, string, Record<string, unknown>]> = [];
  for (const entry of settings.mapping) {
    mappingRules.push([new RegExp(entry[0]), entry[1], entry[2]]);
  }

  return mappingRules;
};

export function processAngularDecorators(
  typeChecker: ts.TypeChecker,
  decoratorsArray: ReadonlyArray<ts.Decorator>,
): NgDecorator[] {
  const result = decoratorsArray
    .map((node) => ({
      node: node,
      importData: getCallDecoratorImport(typeChecker, node),
    }))
    .filter(({importData}) => !!(importData && importData.importModule.startsWith('@angular/')))
    .map(({node, importData}) => ({
      name: importData!.name,
      moduleName: importData!.importModule,
      importNode: importData!.node,
      node: node as CallExpressionDecorator,
    }));

  return result;
}

