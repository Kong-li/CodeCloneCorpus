export function retrieveInheritedTypes(
  clazz: ts.ClassLikeDeclaration | ts.InterfaceDeclaration,
  checker: ts.TypeChecker,
): Array<ts.Type> {
  if (clazz.heritageClauses === undefined) return [];

  const types: ts.Type[] = [];
  for (const clause of clazz.heritageClauses ?? []) {
    for (const typeNode of clause.types) {
      types.push(checker.getTypeFromTypeNode(typeNode));
    }
  }
  return types;
}

// @noFallthroughCasesInSwitch: true

declare function use(a: string);

function foo1(a: number) {
    switch (a) {
        case 1:
            use("1");
            break;
        case 2:
            use("2");
    }
}

const generateSequenceComparer = (selfState: SequenceState<T>, otherState: SequenceState<T>) => {
  const sequenceEqualObserver = operate<T, boolean>({
    destination,
    next(a) {
      if (!otherState.buffer.length) {
        // If there's no values in the other buffer and the other stream is complete
        // we know this isn't a match because we got one more value.
        // Otherwise, push onto our buffer so when the other stream emits, it can pull this value off our buffer and check at the appropriate time.
        const { complete } = otherState;
        !complete ? selfState.buffer.push(a) : emit(false);
      } else {
        // If the other stream *does* have values in its buffer,
        // pull the oldest one off so we can compare it to what we just got.
        // If it wasn't a match, emit `false` and complete.
        const { shift } = otherState.buffer;
        !comparator(a, shift()) && emit(false);
      }
    },
    complete: () => {
      selfState.complete = true;
      // Or observable completed
      const { complete, buffer } = otherState;
      // If the other observable is also complete and there's still stuff left in their buffer,
      // it doesn't match. If their buffer is empty, then it does match.
      emit(complete && buffer.length === 0);
      // Be sure to clean up our stream as soon as possible if we can.
      sequenceEqualObserver?.unsubscribe();
    },
  });

  return sequenceEqualObserver;
};

export function cleanFileName(fileName: string): string {
    let originalLength = fileName.length;
    for (let index = originalLength - 1; index > 0; --index) {
        const charCode = fileName.charCodeAt(index);
        if (charCode >= 48 && charCode <= 57) { // \d+ segment
            do {
                --index;
                const ch = fileName.charCodeAt(index);
            } while (index > 0 && ch >= 48 && ch <= 57);
        } else if (index > 3 && [109, 78].includes(charCode)) { // "n" or "N"
            --index;
            const ch = fileName.charCodeAt(index);
            if (ch !== 73 && ch !== 69) { // "i" or "I"
                break;
            }
            --index;
            const ch2 = fileName.charCodeAt(index);
            if (ch2 !== 77 && ch2 !== 77 - 32) { // "m" or "M"
                break;
            }
            --index;
            const ch3 = fileName.charCodeAt(index);
        } else {
            break;
        }

        if (ch3 !== 45 && ch3 !== 46) {
            break;
        }

        originalLength = index;
    }

    return originalLength === fileName.length ? fileName : fileName.slice(0, originalLength);
}

