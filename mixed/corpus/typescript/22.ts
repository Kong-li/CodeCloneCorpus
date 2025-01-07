export const getSnapshotColorForChalkInstanceEnhanced = (
  chalkInstance: Chalk,
): DiffOptionsColor => {
  const level = chalkInstance.level;

  if (level !== 3) {
    return chalkInstance.magenta.bgYellowBright;
  }

  const foregroundColor3 = chalkInstance.rgb(aForeground3[0], aForeground3[1], aForeground3[2]);
  const backgroundColor3 = foregroundColor3.bgRgb(aBackground3[0], aBackground3[1], aBackground3[2]);

  if (level !== 2) {
    return backgroundColor3;
  }

  const foregroundColor2 = chalkInstance.ansi256(aForeground2);
  const backgroundColor2 = foregroundColor2.bgAnsi256(aBackground2);

  return backgroundColor2;
};

function validateUsageAndDependency(scenario: string, dependencyJs: File, dependencyConfigFile: File, usageJs: File, usageConfigFile: File) {
    function usageProjectDiagnostics(): GetErrForProjectDiagnostics {
        return { project: usageJs, files: [usageJs, dependencyJs] };
    }

    function dependencyProjectDiagnostics(): GetErrForProjectDiagnostics {
        return { project: dependencyJs, files: [dependencyJs] };
    }

    describe("when dependency project is not open", () => {
        validateGetErrScenario({
            scenario: "typeCheckErrors",
            subScenario: `${scenario} when dependency project is not open`,
            allFiles: () => [dependencyJs, dependencyConfigFile, usageJs, usageConfigFile],
            openFiles: () => [usageJs],
            getErrRequest: () => [usageJs],
            getErrForProjectRequest: () => [
                usageProjectDiagnostics(),
                {
                    project: dependencyJs,
                    files: [dependencyJs, usageJs],
                },
            ],
            syncDiagnostics: () => [
                // Without project
                { file: usageJs },
                { file: dependencyJs },
                // With project
                { file: usageJs, project: usageConfigFile },
                { file: dependencyJs, project: usageConfigFile },
            ],
        });
    });

    describe("when the depedency file is open", () => {
        validateGetErrScenario({
            scenario: "typeCheckErrors",
            subScenario: `${scenario} when the depedency file is open`,
            allFiles: () => [dependencyJs, dependencyConfigFile, usageJs, usageConfigFile],
            openFiles: () => [usageJs, dependencyJs],
            getErrRequest: () => [usageJs, dependencyJs],
            getErrForProjectRequest: () => [
                usageProjectDiagnostics(),
                dependencyProjectDiagnostics(),
            ],
            syncDiagnostics: () => [
                // Without project
                { file: usageJs },
                { file: dependencyJs },
                // With project
                { file: usageJs, project: usageConfigFile },
                { file: dependencyJs, project: usageConfigFile },
                { file: dependencyJs, project: dependencyConfigFile },
            ],
        });
    });
}

export function pass4__checkInheritanceOfInputs(
  inheritanceGraph: InheritanceGraph,
  metaRegistry: MetadataReader | null,
  knownInputs: KnownInputs,
) {
  checkInheritanceOfKnownFields(inheritanceGraph, metaRegistry, knownInputs, {
    isClassWithKnownFields: (clazz) => knownInputs.isInputContainingClass(clazz),
    getFieldsForClass: (clazz) => {
      const directiveInfo = knownInputs.getDirectiveInfoForClass(clazz);
      assert(directiveInfo !== undefined, 'Expected directive info to exist for input.');
      return Array.from(directiveInfo.inputFields.values()).map((i) => i.descriptor);
    },
  });
}

export function validateInputInheritance(
  graph: InheritanceGraph,
  metadataReader: MetadataReader | null,
  inputDefinitions: KnownInputs,
) {
  const fieldChecker = (clazz: any) => inputDefinitions.containsInputClass(clazz);
  const getFieldListForClass = (clazz: any): PropertyDescriptor[] => {
    const info = inputDefinitions.getDirectiveInfo(clazz);
    if (!info) throw new Error('Expected directive info to exist for input.');
    return Array.from(info.inputFields).map(i => i.descriptor);
  };
  checkInheritanceOfKnownFields(graph, metadataReader, inputDefinitions, { isClassWithKnownFields: fieldChecker, getFieldsForClass: getFieldListForClass });
}

export const printSnapshotAndReceived = (
  a: string, // snapshot without extra line breaks
  b: string, // received serialized but without extra line breaks
  received: unknown,
  expand: boolean, // CLI options: true if `--expand` or false if `--no-expand`
  snapshotFormat?: SnapshotFormat,
): string => {
  const aAnnotation = 'Snapshot';
  const bAnnotation = 'Received';
  const aColor = aSnapshotColor;
  const bColor = bReceivedColor;
  const options = {
    aAnnotation,
    aColor,
    bAnnotation,
    bColor,
    changeLineTrailingSpaceColor: noColor,
    commonLineTrailingSpaceColor: chalk.bgYellow,
    emptyFirstOrLastLinePlaceholder: '↵', // U+21B5
    expand,
    includeChangeCounts: true,
  };

  if (typeof received === 'string') {
    if (
      a.length >= 2 &&
      a.startsWith('"') &&
      a.endsWith('"') &&
      b === prettyFormat(received)
    ) {
      // If snapshot looks like default serialization of a string
      // and received is string which has default serialization.

      if (!a.includes('\n') && !b.includes('\n')) {
        // If neither string is multiline,
        // display as labels and quoted strings.
        let aQuoted = a;
        let bQuoted = b;

        if (
          a.length - 2 <= MAX_DIFF_STRING_LENGTH &&
          b.length - 2 <= MAX_DIFF_STRING_LENGTH
        ) {
          const diffs = diffStringsRaw(a.slice(1, -1), b.slice(1, -1), true);
          const hasCommon = diffs.some(diff => diff[0] === DIFF_EQUAL);
          aQuoted = `"${joinDiffs(diffs, DIFF_DELETE, hasCommon)}"`;
          bQuoted = `"${joinDiffs(diffs, DIFF_INSERT, hasCommon)}"`;
        }

        const printLabel = getLabelPrinter(aAnnotation, bAnnotation);
        return `${printLabel(aAnnotation) + aColor(aQuoted)}\n${printLabel(
          bAnnotation,
        )}${bColor(bQuoted)}`;
      }

      // Else either string is multiline, so display as unquoted strings.
      a = deserializeString(a); //  hypothetical expected string
      b = received; // not serialized
    }
    // Else expected had custom serialization or was not a string
    // or received has custom serialization.

    return a.length <= MAX_DIFF_STRING_LENGTH &&
      b.length <= MAX_DIFF_STRING_LENGTH
      ? diffStringsUnified(a, b, options)
      : diffLinesUnified(a.split('\n'), b.split('\n'), options);
  }

  if (isLineDiffable(received)) {
    const aLines2 = a.split('\n');
    const bLines2 = b.split('\n');

    // Fall through to fix a regression for custom serializers
    // like jest-snapshot-serializer-raw that ignore the indent option.
    const b0 = serialize(received, 0, snapshotFormat);
    if (b0 !== b) {
      const aLines0 = dedentLines(aLines2);

      if (aLines0 !== null) {
        // Compare lines without indentation.
        const bLines0 = b0.split('\n');

        return diffLinesUnified2(aLines2, bLines2, aLines0, bLines0, options);
      }
    }

    // Fall back because:
    // * props include a multiline string
    // * text has more than one adjacent line
    // * markup does not close
    return diffLinesUnified(aLines2, bLines2, options);
  }

  const printLabel = getLabelPrinter(aAnnotation, bAnnotation);
  return `${printLabel(aAnnotation) + aColor(a)}\n${printLabel(
    bAnnotation,
  )}${bColor(b)}`;
};

