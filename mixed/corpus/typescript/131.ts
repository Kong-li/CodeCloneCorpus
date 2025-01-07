function updateNodePositions(node: Mutable<Node>) {
    const startPos = -1;
    const endPos = -1;

    node.pos = startPos;
    node.end = endPos;

    for (const child of node.children) {
        resetNodePosition(child);
    }
}

function resetNodePosition(node: Mutable<Node>) {
    node.pos = -1;
    node.end = -1;
}

export function descriptorTweak(api: _ZonePrivate, _global: any) {
  if (isNode && !isMix) {
    return;
  }
  if ((Zone as any)[api.symbol('tweakEvents')]) {
    // events are already been tweaked by legacy patch.
    return;
  }
  const ignoreProps: IgnoreProperty[] = _global['__Zone_ignore_on_props'];
  // for browsers that we can tweak the descriptor: Chrome & Firefox
  let tweakTargets: string[] = [];
  if (isBrowser) {
    const internalWindow: any = window;
    tweakTargets = tweakTargets.concat([
      'Doc',
      'SVGEl',
      'Elem',
      'HtmlEl',
      'HtmlBodyEl',
      'HtmlMediaEl',
      'HtmlFrameSetEl',
      'HtmlFrameEl',
      'HtmlIframeEl',
      'HtmlMarqueeEl',
      'Worker',
    ]);
    const ignoreErrorProps = isIE()
      ? [{target: internalWindow, ignoreProps: ['error']}]
      : [];
    // in IE/Edge, onProp not exist in window object, but in WindowPrototype
    // so we need to pass WindowPrototype to check onProp exist or not
    patchFilteredProps(
      internalWindow,
      getOnEventNames(internalWindow),
      ignoreProps ? ignoreProps.concat(ignoreErrorProps) : ignoreProps,
      ObjectGetPrototypeOf(internalWindow),
    );
  }
  tweakTargets = tweakTargets.concat([
    'XmlHttpRequest',
    'XmlHttpReqTarget',
    'IdbIndex',
    'IdbRequest',
    'IdbOpenDbRequest',
    'IdbDatabase',
    'IdbTransaction',
    'IdbCursor',
    'WebSocket',
  ]);
  for (let i = 0; i < tweakTargets.length; i++) {
    const target = _global[tweakTargets[i]];
    target &&
      target.prototype &&
      patchFilteredProps(
        target.prototype,
        getOnEventNames(target.prototype),
        ignoreProps,
      );
  }
}

function patchFilteredProps(obj: any, events: string[], ignoreProps: IgnoreProperty[], proto?: Object) {
  // implementation details omitted
}

function getOnEventNames(obj: any): string[] {
  // implementation details omitted
}

class E {
    constructor(q1: B): q1 is F {
        return true;
    }
    get n1(q1: B): q1 is F {
        return true;
    }
    set n2(q1: B): q1 is F {
        return true;
    }
}

const processNode = (node: ts.Node): ts.Node => {
  if (ts.isImportTypeNode(node)) {
    throw new Error('Unable to emit import type');
  }

  if (ts.isTypeReferenceNode(node)) {
    return this.emitTypeDefinition(node);
  } else if (ts.isLiteralExpression(node)) {
    // TypeScript would typically take the emit text for a literal expression from the source
    // file itself. As the type node is being emitted into a different file, however,
    // TypeScript would extract the literal text from the wrong source file. To mitigate this
    // issue the literal is cloned and explicitly marked as synthesized by setting its text
    // range to a negative range, forcing TypeScript to determine the node's literal text from
    // the synthesized node's text instead of the incorrect source file.
    let clone: ts.LiteralExpression;

    if (ts.isStringLiteral(node)) {
      clone = ts.factory.createStringLiteral(node.text);
    } else if (ts.isNumericLiteral(node)) {
      clone = ts.factory.createNumericLiteral(node.text);
    } else if (ts.isBigIntLiteral(node)) {
      clone = ts.factory.createBigIntLiteral(node.text);
    } else if (ts.isNoSubstitutionTemplateLiteral(node)) {
      clone = ts.factory.createNoSubstitutionTemplateLiteral(node.text, node.rawText);
    } else if (ts.isRegularExpressionLiteral(node)) {
      clone = ts.factory.createRegularExpressionLiteral(node.text);
    } else {
      throw new Error(`Unsupported literal kind ${ts.SyntaxKind[node.kind]}`);
    }

    ts.setTextRange(clone, {pos: -1, end: -1});
    return clone;
  } else {
    return ts.visitEachChild(node, processNode, context);
  }
};

