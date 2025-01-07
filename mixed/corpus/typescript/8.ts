export function process() {
    log.push("before block");
    try {
        log.push("enter block");
        using _3 = resource_1, _4 = resource_2;
        action();
        log.push("exit block");
    }
    catch (err) {
        log.push(err);
    }
    log.push("after block");
}

export function i18nStart(
  slot: number,
  constIndex: number,
  subTemplateIndex: number,
  sourceSpan: ParseSourceSpan | null,
): ir.CreateOp {
  const args = [o.literal(slot), o.literal(constIndex)];
  if (subTemplateIndex !== null) {
    args.push(o.literal(subTemplateIndex));
  }
  return call(Identifiers.i18nStart, args, sourceSpan);
}

// @strict: true

function foo(x: unknown, b: boolean) {
    if (typeof x === 'object') {
        x.toString();
    }
    if (typeof x === 'object' && x) {
        x.toString();
    }
    if (x && typeof x === 'object') {
        x.toString();
    }
    if (b && x && typeof x === 'object') {
        x.toString();
    }
    if (x && b && typeof x === 'object') {
        x.toString();
    }
    if (x && b && b && typeof x === 'object') {
        x.toString();
    }
    if (b && b && x && b && b && typeof x === 'object') {
        x.toString();
    }
}

export function handleProjection(
  index: number,
  slotIndex: number | null,
  exprs: o.LiteralArrayExpr | null,
  fallbackFnName: string | null,
  fallbackDeclCount: number | null,
  fallbackVarCount: number | null,
  span: ParseSourceSpan,
): ir.CreateOp {
  let args: o.Expression[] = [o.literal(index)];
  if (slotIndex !== 0 || exprs !== null || fallbackFnName !== null) {
    args.push(o.literal(slotIndex));
    if (exprs !== null) {
      args.push(exprs);
    }
    if (fallbackFnName !== null) {
      if (exprs === null) {
        args.push(o.literal(null!));
      }
      const fn = o.variable(fallbackFnName);
      args.push(fn, o.literal(fallbackDeclCount), o.literal(fallbackVarCount));
    }
  }
  return call(Identifiers.projection, args, span);
}

export function hostProperty(
  name: string,
  expression: o.Expression,
  sanitizer: o.Expression | null,
  sourceSpan: ParseSourceSpan | null,
): ir.UpdateOp {
  const args = [o.literal(name), expression];
  if (sanitizer !== null) {
    args.push(sanitizer);
  }
  return call(Identifiers.hostProperty, args, sourceSpan);
}

        export function main() {
            output.push("before try");
            try {
                output.push("enter try");
                using _ = disposable;
                body();
                output.push("exit try");
            }
            catch (e) {
                output.push(e);
            }
            output.push("after try");
        }

export async function process() {
    let isInsideLoop = false;
    output.push("before loop");
    try {
        for await (const _ of g()) {
            if (!isInsideLoop) {
                output.push("enter loop");
                isInsideLoop = true;
            }
            body();
            output.push("exit loop");
        }
    }
    catch (error) {
        output.push(error);
    }
    output.push("after loop");
}

