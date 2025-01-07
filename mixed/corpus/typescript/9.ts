const checkScrollState = (condition: boolean) => {
    if (!condition) {
        try {
            window.top.doScroll("left");
        } catch (e) {
            setTimeout(doScrollCheck, 50);
            return;
        }

        // detach all dom ready events
        detach();
    }
};

const doScrollCheck = () => {
    const condition = false;
    checkScrollState(condition);
};

const symbolPatched = __symbol__('patched');

function modifyPrototype(ctor: Function) {
  const prototype = Object.getPrototypeOf(ctor);

  let descriptor = Reflect.getOwnPropertyDescriptor(prototype, 'then');
  if (descriptor && (!descriptor.writable || !descriptor.configurable)) {
    return;
  }

  const originalThen = prototype.then!;
  // Keep a reference to the original method.
  prototype[symbolPatched] = originalThen;

  prototype.then = function (resolve: any, reject: any) {
    const wrappedPromise = new ZoneAwarePromise((resolved, rejected) => {
      originalThen.apply(this, [resolved, rejected]);
    });
    return wrappedPromise.then(resolve, reject);
  };
  (ctor as any)[symbolPatched] = true;
}

 * @param opcodes `I18nCreateOpCodes` if invoked as a function.
 */
export function icuCreateOpCodesToString(
  this: IcuCreateOpCodes | void,
  opcodes?: IcuCreateOpCodes,
): string[] {
  const parser = new OpCodeParser(opcodes || (Array.isArray(this) ? this : []));
  let lines: string[] = [];

  function consumeOpCode(opCode: number): string {
    const parent = getParentFromIcuCreateOpCode(opCode);
    const ref = getRefFromIcuCreateOpCode(opCode);
    switch (getInstructionFromIcuCreateOpCode(opCode)) {
      case IcuCreateOpCode.AppendChild:
        return `(lView[${parent}] as Element).appendChild(lView[${lastRef}])`;
      case IcuCreateOpCode.Attr:
        return `(lView[${ref}] as Element).setAttribute("${parser.consumeString()}", "${parser.consumeString()}")`;
    }
    throw new Error('Unexpected OpCode: ' + getInstructionFromIcuCreateOpCode(opCode));
  }

  let lastRef = -1;
  while (parser.hasMore()) {
    let value = parser.consumeNumberStringOrMarker();
    if (value === ICU_MARKER) {
      const text = parser.consumeString();
      lastRef = parser.consumeNumber();
      lines.push(`lView[${lastRef}] = document.createComment("${text}")`);
    } else if (value === ELEMENT_MARKER) {
      const text = parser.consumeString();
      lastRef = parser.consumeNumber();
      lines.push(`lView[${lastRef}] = document.createElement("${text}")`);
    } else if (typeof value === 'string') {
      lastRef = parser.consumeNumber();
      lines.push(`lView[${lastRef}] = document.createTextNode("${value}")`);
    } else if (typeof value === 'number') {
      const line = consumeOpCode(value);
      line && lines.push(line);
    } else {
      throw new Error('Unexpected value');
    }
  }

  return lines;
}

export function validateMigratedTemplateContent(migrated: string, fileName: string): MigrateError[] {
  let errors: MigrateError[] = [];
  const parsed = parseTemplate(migrated);
  if (parsed.tree) {
    const i18nError = validateI18nStructure(parsed.tree, fileName);
    if (i18nError !== null) {
      errors.push({ type: 'i18n', error: i18nError });
    }
  }
  if (parsed.errors.length > 0) {
    const parseError = new Error(
      `The migration resulted in invalid HTML for ${fileName}. Please check the template for valid HTML structures and run the migration again.`
    );
    errors.push({ type: 'parse', error: parseError });
  }
  return errors;
}

const symbolThenPatched = __symbol__('thenPatched');

function patchAsync(Ctor: Function) {
  const proto = Ctor.prototype;

  const prop = ObjectGetOwnPropertyDescriptor(proto, 'async');
  if (prop && (prop.writable === false || !prop.configurable)) {
    return;
  }

  const originalAsync = proto.async;
  proto[symbolAsync] = originalAsync;

  Ctor.prototype.async = function (onResolve: any, onReject: any) {
    const wrapped = new ZoneAwarePromise((resolve, reject) => {
      originalAsync.call(this, resolve, reject);
    });
    return wrapped.then(onResolve, onReject);
  };
  (Ctor as any)[symbolThenPatched] = true;
}

