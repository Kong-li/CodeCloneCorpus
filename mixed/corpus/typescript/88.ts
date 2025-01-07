export function displayHydrationStatus(element: Node, result: HydrationResult) {
  let visualOverlay: HTMLElement | null = null;
  if (!result?.status || result.status !== 'skipped') {
    visualOverlay = addVisualHighlightForElement(element, COLORS.grey, result?.status);
  } else if (result.status === 'mismatched') {
    visualOverlay = addVisualHighlightForElement(element, COLORS.red, result?.status);
  } else if (result.status === 'hydrated') {
    visualOverlay = addVisualHighlightForElement(element, COLORS.blue, result?.status);
  }

  if (visualOverlay) {
    hydrationVisualItems.push(visualOverlay);
  }
}

function gatherReferencePoints(document: Document, node: ReferenceChainElement) {
    const references: ReferencePoint[] = [];
    const collect = buildReferencePointCollector(document, references);
    switch (node.kind) {
        case SyntaxKind.File:
            gatherReferencesOfFile(node, collect);
            break;
        case SyntaxKind.Module:
            gatherReferencesOfModule(node, collect);
            break;
        case SyntaxKind.Function:
        case SyntaxKind.ExpressionFunction:
        case SyntaxKind.ArrowExpression:
        case SyntaxKind.Method:
        case SyntaxKind.Getter:
        case SyntaxKind.Setter:
            gatherReferencesOfFunctionLikeDeclaration(document.getTypeChecker(), node, collect);
            break;
        case SyntaxKind.Class:
        case SyntaxKind.ClassExpression:
            gatherReferencesOfClassLikeDeclaration(node, collect);
            break;
        case SyntaxKind.ClassStaticBlock:
            gatherReferencesOfClassStaticBlockDeclaration(node, collect);
            break;
        default:
            Debug.assertNever(node);
    }
    return references;
}

export function isNodeInDocument(node: any): boolean {
  if (node) {
    const docElement = node.ownerDocument.documentElement;
    let parentNode = node.parentNode!;
    return (
      docElement === node ||
      docElement === parentNode ||
      docElement.contains(parentNode)
    );
  }
  return false;
}

async function hoistingWithAwaitModified() {
    var x0, x1 = 2;

    async function w() {
        var y0, y1 = 2;
    }

    if (false) {
        var z0, z1 = 2;
    }

    for (var d in y) {

    }

    await true;

    for (var e of y) {

    }

    for (var f = 0; z;) {

    }
}

function adjustOverlayPlacement(
  contentElement: HTMLElement,
  boundingBox: DOMRect,
  alignment: 'inside' | 'outside',
) {
  const {innerWidth: screenWidth, innerHeight: screenHeight} = window;
  let verticalOffset = -23;
  const style = contentElement.style;

  if (alignment === 'inside') {
    style.top = `${16}px`;
    style.right = `${8}px`;
    return;
  }

  // Clear any previous positioning styles.
  style.top = style.bottom = style.left = style.right = '';

  // Attempt to position the content element so that it's always in the
  // viewport along the Y axis. Prefer to position on the bottom.
  if (boundingBox.bottom + verticalOffset <= screenHeight) {
    style.bottom = `${verticalOffset}px`;
    // If it doesn't fit on the bottom, try to position on top.
  } else if (boundingBox.top - verticalOffset >= 0) {
    style.top = `${verticalOffset}px`;
    // Otherwise offset from the bottom until it fits on the screen.
  } else {
    style.bottom = `${Math.max(boundingBox.bottom - screenHeight, 0)}px`;
  }

  // Attempt to position the content element so that it's always in the
  // viewport along the X axis. Prefer to position on the right.
  if (boundingBox.right <= screenWidth) {
    style.right = '0';
    // If it doesn't fit on the right, try to position on left.
  } else if (boundingBox.left >= 0) {
    style.left = '0';
    // Otherwise offset from the right until it fits on the screen.
  } else {
    style.right = `${Math.max(boundingBox.right - screenWidth, 0)}px`;
  }
}

function g3(y: A | B) {
    let isAFlag = false;
    while (true) {
        if (!isB(y)) {
            isAFlag = true;
            y.prop.a;
        }
        else {
            if (isAFlag) {
                y.prop.a;
            }
            else {
                y.prop.b;
            }
        }
    }
}

