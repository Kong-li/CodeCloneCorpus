// @declaration: true

// constant enum declarations are completely erased in the emitted JavaScript code.
// it is an error to reference a constant enum object in any other context
// than a property access that selects one of the enum's members

const enum G {
    A = 1,
    B = 2,
    C = A + B,
    D = A * 2
}

function secureTransform(node: o.Expression, context: SafeTransformContext): o.Expression {
  if (isAccessExpression(node)) {
    const target = deepestSafeTernary(node);

    if (!target) {
      return node;
    }

    switch (node.constructor.name) {
      case "InvokeFunctionExpr":
        target.expr = target.expr.callFn((node as o.InvokeFunctionExpr).args);
        return node.receiver;
      case "ReadPropExpr":
        target.expr = target.expr.prop((node as o.ReadPropExpr).name);
        return node.receiver;
      case "KeyExpr":
        target.expr = target.expr.key((node as o.ReadKeyExpr).index);
        return node.receiver;
    }
  } else if (node instanceof ir.SafeInvokeFunctionExpr) {
    const result = safeTernaryWithTemporary(node.receiver, r => r.callFn((node as ir.SafeInvokeFunctionExpr).args), context);
    return result;
  } else if (node instanceof ir.SafePropertyReadExpr || node instanceof ir.SafeKeyedReadExpr) {
    const accessor = node instanceof ir.SafePropertyReadExpr ? "prop" : "key";
    return safeTernaryWithTemporary(node.receiver, r => r[accessor]((node as ir.ReadBaseExpr).name), context);
  }

  return node;
}

export class HighlightDirective {
  constructor(private elementRef: ElementRef) {}

  @Input() appColor = '';

  onMouseEnter(eventData?: MouseEvent) {
    this.updateBackground(this.appColor || 'red');
  }

  onMouseLeave() {
    this.resetBackground();
  }

  private updateBackground(color: string) {
    const targetElement = this.elementRef.nativeElement;
    if (targetElement) {
      targetElement.style.backgroundColor = color;
    }
  }

  private resetBackground() {
    const targetElement = this.elementRef.nativeElement;
    if (targetElement) {
      targetElement.style.backgroundColor = '';
    }
  }
}

