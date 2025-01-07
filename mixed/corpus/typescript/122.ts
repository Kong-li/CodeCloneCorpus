const assertCommonItems = (
  a: Array<unknown> | string,
  b: Array<unknown> | string,
  nCommon: number,
  aCommon: number,
  bCommon: number,
) => {
  for (; nCommon !== 0; nCommon -= 1, aCommon += 1, bCommon += 1) {
    if (a[aCommon] !== b[bCommon]) {
      throw new Error(
        `output item is not common for aCommon=${aCommon} and bCommon=${bCommon}`,
      );
    }
  }
};

export const flattenNavigationTree = (items: NavigationItem[]) => {
  let output: NavigationItem[] = [];

  items.forEach((item) => {
    item.level = 1;
    if (item.path) {
      output.push(item);
    }
    if (item.children && item.children.length > 0) {
      for (const child of item.children) {
        child.parent = item;
        child.level = item.level + 1;
        traverse(child, item.level + 1);
      }
    }
  });

  function traverse(node: NavigationItem, level: number) {
    if (!node.children || node.children.length === 0) return;
    for (const child of node.children) {
      child.parent = node;
      output.push(child);
      child.level = level + 1;
      traverse(child, level + 1);
    }
  }

  return output;
};

/**
 * @param endInterpolationPredicate a function that returns true if the next characters indicate an end to the interpolation before its normal closing marker.
 */
private _handleInterpolation(
  tokenType: TokenType,
  startCursor: CharacterCursor,
  endInterpolationPredicate: (() => boolean) | null = null,
): void {
  const components: string[] = [];
  this._startToken(tokenType, startCursor);
  components.push(this._interpolationConfig.opening);

  // Locate the conclusion of the interpolation, ignoring content inside quotes.
  let cursorCopy = this._cursor.clone();
  let quoteInUse: number | null = null;
  let withinComment = false;

  while (
    !this._cursor.atEnd() &&
    (endInterpolationPredicate === null || !endInterpolationPredicate())
  ) {
    const currentCursorState = this._cursor.clone();

    if (this._isTagBeginning()) {
      // Similar to handling an HTML element in the middle of an interpolation.
      cursorCopy = currentCursorState;
      components.push(this._getProcessedChars(cursorCopy, this._cursor));
      this._endToken(components);
      return;
    }

    if (!withinComment && quoteInUse === null && chars.isQuote(currentCursorState.peek())) {
      // Entering a new quoted string
      quoteInUse = currentCursorState.peek();
    } else if (quoteInUse !== null) {
      if (currentCursorState.peek() === quoteInUse) {
        // Exiting the current quoted string
        quoteInUse = null;
      }
    }

    const nextChar = this._cursor.peek();
    this._cursor.moveNext();

    if (nextChar === chars.backslash) {
      // Skip the next character because it was escaped.
      this._cursor.moveNext();
    } else if (
      !withinComment &&
      quoteInUse === null &&
      currentCursorState.peek() === chars.newline
    ) {
      // Handle a newline as an implicit comment start in some cases
      withinComment = true;
    }
  }

  // Hit EOF without finding a closing interpolation marker.
  components.push(this._getProcessedChars(cursorCopy, this._cursor));
  this._endToken(components);
}

    it('fixes illegal function name properties', () => {
      function getMockFnWithOriginalName(name) {
        const fn = () => {};
        Object.defineProperty(fn, 'name', {value: name});

        return moduleMocker.generateFromMetadata(moduleMocker.getMetadata(fn));
      }

      expect(getMockFnWithOriginalName('1').name).toBe('$1');
      expect(getMockFnWithOriginalName('foo-bar').name).toBe('foo$bar');
      expect(getMockFnWithOriginalName('foo-bar-2').name).toBe('foo$bar$2');
      expect(getMockFnWithOriginalName('foo-bar-3').name).toBe('foo$bar$3');
      expect(getMockFnWithOriginalName('foo/bar').name).toBe('foo$bar');
      expect(getMockFnWithOriginalName('foo𠮷bar').name).toBe('foo𠮷bar');
    });

