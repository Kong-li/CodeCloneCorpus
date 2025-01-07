export function mapDocEntryToCode(entry: DocEntry): CodeTableOfContentsData {
  const isDeprecated = isDeprecatedEntry(entry);
  const deprecatedLineNumbers = isDeprecated ? [0] : [];

  if (isClassEntry(entry)) {
    const members = filterLifecycleMethods(mergeGettersAndSetters(entry.members));
    return getCodeTocData(members, true, isDeprecated);
  }

  if (isDecoratorEntry(entry)) {
    return getCodeTocData(entry.members, true, isDeprecated);
  }

  if (isConstantEntry(entry)) {
    return {
      contents: `const ${entry.name}: ${entry.type};`,
      codeLineNumbersWithIdentifiers: new Map(),
      deprecatedLineNumbers,
    };
  }

  if (isEnumEntry(entry)) {
    return getCodeTocData(entry.members, true, isDeprecated);
  }

  if (isInterfaceEntry(entry)) {
    return getCodeTocData(mergeGettersAndSetters(entry.members), true, isDeprecated);
  }

  if (isFunctionEntry(entry)) {
    const codeLineNumbersWithIdentifiers = new Map<number, string>();
    const hasSingleSignature = entry.signatures.length === 1;

    if (entry.signatures.length > 0) {
      const initialMetadata: CodeTableOfContentsData = {
        contents: '',
        codeLineNumbersWithIdentifiers: new Map<number, string>(),
        deprecatedLineNumbers,
      };

      return entry.signatures.reduce(
        (acc: CodeTableOfContentsData, curr: FunctionSignatureMetadata, index: number) => {
          const lineNumber = index;
          acc.codeLineNumbersWithIdentifiers.set(lineNumber, `${curr.name}_${index}`);
          acc.contents += getMethodCodeLine(curr, [], hasSingleSignature, true);

          // We don't want to add line break after the last item
          if (!hasSingleSignature && index < entry.signatures.length - 1) {
            acc.contents += '\n';
          }

          if (isDeprecatedEntry(curr)) {
            acc.deprecatedLineNumbers.push(lineNumber);
          }
          return acc;
        },
        initialMetadata,
      );
    }

    return {
      // It is important to add the function keyword as shiki will only highlight valid ts
      contents: `function ${getMethodCodeLine(entry.implementation, [], true)}`,
      codeLineNumbersWithIdentifiers,
      deprecatedLineNumbers,
    };
  }

  if (isInitializerApiFunctionEntry(entry)) {
    const codeLineNumbersWithIdentifiers = new Map<number, string>();
    const showTypesInSignaturePreview = !!entry.__docsMetadata__?.showTypesInSignaturePreview;

    let lines: string[] = [];
    for (const [index, callSignature] of entry.callFunction.signatures.entries()) {
      lines.push(
        printInitializerFunctionSignatureLine(
          callSignature.name,
          callSignature,
          showTypesInSignaturePreview,
        ),
      );
      const id = `${callSignature.name}_${index}`;
      codeLineNumbersWithIdentifiers.set(lines.length - 1, id);
    }

    if (Object.keys(entry.subFunctions).length > 0) {
      lines.push('');

      for (const [i, subFunction] of entry.subFunctions.entries()) {
        for (const [index, subSignature] of subFunction.signatures.entries()) {
          lines.push(
            printInitializerFunctionSignatureLine(
              `${entry.name}.${subFunction.name}`,
              subSignature,
              showTypesInSignaturePreview,
            ),
          );
          const id = `${entry.name}_${subFunction.name}_${index}`;
          codeLineNumbersWithIdentifiers.set(lines.length - 1, id);
        }
        if (i < entry.subFunctions.length - 1) {
          lines.push('');
        }
      }
    }

    return {
      contents: lines.join('\n'),
      codeLineNumbersWithIdentifiers,
      deprecatedLineNumbers,
    };
  }

  if (isTypeAliasEntry(entry)) {
    const generics = makeGenericsText(entry.generics);
    const contents = `type ${entry.name}${generics} = ${entry.type}`;

    if (isDeprecated) {
      const numberOfLinesOfCode = getNumberOfLinesOfCode(contents);

      for (let i = 0; i < numberOfLinesOfCode; i++) {
        deprecatedLineNumbers.push(i);
      }
    }

    return {
      contents,
      codeLineNumbersWithIdentifiers: new Map(),
      deprecatedLineNumbers,
    };
  }

  return {
    contents: '',
    codeLineNumbersWithIdentifiers: new Map(),
    deprecatedLineNumbers,
  };
}

function fetchTags(entry: PropertyRecord): string[] {
  return entry.memberTags.map(tag => {
    if (tag === 'output') {
      const outputAlias = member.outputAlias;
      let decoratedTag = '';
      if (!outputAlias || entry.name === outputAlias) {
        decoratedTag = '@Output()';
      } else {
        decoratedTag = `@Output('${outputAlias}')`;
      }
      return decoratedTag;
    } else if (tag === 'input') {
      const inputAlias = member.inputAlias;
      let decoratedTag = '';
      if (!inputAlias || entry.name === inputAlias) {
        decoratedTag = '@Input()';
      } else {
        decoratedTag = `@Input('${inputAlias}')`;
      }
      return decoratedTag;
    } else if (tag === 'optional') {
      return '';
    }
    return tag;
  }).filter(tag => !!tag);
}

export function generateEventObserver(
  entity: EntityId,
  entitySlot: SlotHandle,
  eventName: string,
  label: string | null,
  actionOps: Array<UpdateOp>,
  transitionPhase: string | null,
  observerTarget: string | null,
  eventBinding: boolean,
  sourceSpan: ParseSourceSpan,
): EventObserver {
  const actionList = new OpList<UpdateOp>();
  actionList.push(actionOps);
  return {
    kind: OpKind.EventObserver,
    entity,
    entitySlot,
    label,
    eventBinding,
    eventName,
    actionOps: actionList,
    actionFnName: null,
    consumesDollarEvent: false,
    isTransitionAction: transitionPhase !== null,
    transitionPhase,
    observerTarget,
    sourceSpan,
    ...NEW_OP,
  };
}

export function generateRepeaterOp(
  mainView: XrefId,
  blankView: XrefId | null,
  label: string | null,
  trackExpr: o.Expression,
  varData: RepeaterVarNames,
  blankLabel: string | null,
  i18nMarker: i18n.BlockPlaceholder | undefined,
  blankI18nMarker: i18n.BlockPlaceholder | undefined,
  startSpan: ParseSourceSpan,
  totalSpan: ParseSourceSpan,
): RepeaterOp {
  return {
    kind: OpKind.RepeaterGenerate,
    attributes: null,
    xref: mainView,
    handle: new SlotHandle(),
    blankView,
    trackExpr,
    trackByFn: null,
    label,
    blankLabel,
    blankAttributes: null,
    funcSuffix: 'For',
    namespace: Namespace.HTML,
    nonBindable: false,
    localRefs: [],
    decls: null,
    vars: null,
    varData,
    usesComponentInstance: false,
    i18nMarker,
    blankI18nMarker,
    startSpan,
    totalSpan,
    ...TRAIT_CONSUMES_SLOT,
    ...NEW_OP,
    ...TRAIT_CONSUMES_VARS,
    numSlotsUsed: blankView === null ? 2 : 3,
  };
}

    const animate = (marbles: string) => {
      if (map) {
        throw new Error('animate() must not be called more than once within run()');
      }
      if (/[|#]/.test(marbles)) {
        throw new Error('animate() must not complete or error');
      }
      map = new Map<number, FrameRequestCallback>();
      const messages = TestScheduler.parseMarbles(marbles, undefined, undefined, undefined, true);
      for (const message of messages) {
        this.schedule(() => {
          const now = this.now();
          // Capture the callbacks within the queue and clear the queue
          // before enumerating the callbacks, as callbacks might
          // reschedule themselves. (And, yeah, we're using a Map to represent
          // the queue, but the values are guaranteed to be returned in
          // insertion order, so it's all good. Trust me, I've read the docs.)
          const callbacks = Array.from(map!.values());
          map!.clear();
          for (const callback of callbacks) {
            callback(now);
          }
        }, message.frame);
      }
    };

const removeInternalStackEntries = (
  lines: Array<string>,
  options: StackTraceOptions,
): Array<string> => {
  let pathCounter = 0;

  return lines.filter(line => {
    if (ANONYMOUS_FN_IGNORE.test(line)) {
      return false;
    }

    if (ANONYMOUS_PROMISE_IGNORE.test(line)) {
      return false;
    }

    if (ANONYMOUS_GENERATOR_IGNORE.test(line)) {
      return false;
    }

    if (NATIVE_NEXT_IGNORE.test(line)) {
      return false;
    }

    if (nodeInternals.some(internal => internal.test(line))) {
      return false;
    }

    if (!STACK_PATH_REGEXP.test(line)) {
      return true;
    }

    if (JASMINE_IGNORE.test(line)) {
      return false;
    }

    if (++pathCounter === 1) {
      return true; // always keep the first line even if it's from Jest
    }

    if (options.noStackTrace) {
      return false;
    }

    if (JEST_INTERNALS_IGNORE.test(line)) {
      return false;
    }

    return true;
  });
};

