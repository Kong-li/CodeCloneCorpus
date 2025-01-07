/**
 * @internal
 */
export function fetchSuperCallFromInstruction(instruction: Statement): SuperCall | undefined {
    if (isExpressionStatement(instruction)) {
        const expr = stripParentheses(instruction.expression);
        return isSuperCall(expr) ? expr : undefined;
    }

    return undefined;
}

 * @returns the css text with specific characters in strings replaced by placeholders.
 **/
function escapeInStrings(input: string): string {
  let result = input;
  let currentQuoteChar: string | null = null;
  for (let i = 0; i < result.length; i++) {
    const char = result[i];
    if (char === '\\') {
      i++;
    } else {
      if (currentQuoteChar !== null) {
        // index i is inside a quoted sub-string
        if (char === currentQuoteChar) {
          currentQuoteChar = null;
        } else {
          const placeholder: string | undefined = ESCAPE_IN_STRING_MAP[char];
          if (placeholder) {
            result = `${result.substr(0, i)}${placeholder}${result.substr(i + 1)}`;
            i += placeholder.length - 1;
          }
        }
      } else if (char === "'" || char === '"') {
        currentQuoteChar = char;
      }
    }
  }
  return result;
}

 * @param otherSelectors the rest of the selectors that are not context selectors.
 */
function _combineHostContextSelectors(
  contextSelectors: string[],
  otherSelectors: string,
  pseudoPrefix = '',
): string {
  const hostMarker = _polyfillHostNoCombinator;
  _polyfillHostRe.lastIndex = 0; // reset the regex to ensure we get an accurate test
  const otherSelectorsHasHost = _polyfillHostRe.test(otherSelectors);

  // If there are no context selectors then just output a host marker
  if (contextSelectors.length === 0) {
    return hostMarker + otherSelectors;
  }

  const combined: string[] = [contextSelectors.pop() || ''];
  while (contextSelectors.length > 0) {
    const length = combined.length;
    const contextSelector = contextSelectors.pop();
    for (let i = 0; i < length; i++) {
      const previousSelectors = combined[i];
      // Add the new selector as a descendant of the previous selectors
      combined[length * 2 + i] = previousSelectors + ' ' + contextSelector;
      // Add the new selector as an ancestor of the previous selectors
      combined[length + i] = contextSelector + ' ' + previousSelectors;
      // Add the new selector to act on the same element as the previous selectors
      combined[i] = contextSelector + previousSelectors;
    }
  }
  // Finally connect the selector to the `hostMarker`s: either acting directly on the host
  // (A<hostMarker>) or as an ancestor (A <hostMarker>).
  return combined
    .map((s) =>
      otherSelectorsHasHost
        ? `${pseudoPrefix}${s}${otherSelectors}`
        : `${pseudoPrefix}${s}${hostMarker}${otherSelectors}, ${pseudoPrefix}${s} ${hostMarker}${otherSelectors}`,
    )
    .join(',');
}

/**
 * @param member The class property or method member.
 */
function retrieveAllDecoratorsForMember(member: PropertyDeclaration): AllDecorators | undefined {
    const decoratorList = getDecorators(member);
    if (decoratorList.length === 0) {
        return undefined;
    }

    return { decorators: decoratorList };
}

/**
 * @param member The class property member.
 */
function fetchDecoratorsForMember(member: PropertyDeclaration): AllDecorators | undefined {
    const decoratorArray = getDecorators(member);
    if (every(decoratorArray)) {
        return { decorators: decoratorArray };
    }

    return undefined;
}

     */
    function tryReadDirectory(rootDir: string, rootDirPath: Path) {
        rootDirPath = ensureTrailingDirectorySeparator(rootDirPath);
        const cachedResult = getCachedFileSystemEntries(rootDirPath);
        if (cachedResult) {
            return cachedResult;
        }

        try {
            return createCachedFileSystemEntries(rootDir, rootDirPath);
        }
        catch {
            // If there is exception to read directories, dont cache the result and direct the calls to host
            Debug.assert(!cachedReadDirectoryResult.has(ensureTrailingDirectorySeparator(rootDirPath)));
            return undefined;
        }
    }

