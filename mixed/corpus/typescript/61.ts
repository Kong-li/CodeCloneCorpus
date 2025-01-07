export function generateBuilderProgram(
    rootNamesOrNewProgram: string[] | Program | undefined,
    optionsOrHost: CompilerOptions | BuilderProgramHost | undefined,
    oldProgram?: BuilderProgram | CompilerHost,
    configDiagnosticsOrOldProgram?: readonly Diagnostic[] | BuilderProgram,
    configDiagnostics?: readonly Diagnostic[],
    projectReferences?: readonly ProjectReference[]
): BuilderProgram {
    const { program, newConfigFileParsingDiagnostics } = getBuilderCreationParameters(
        rootNamesOrNewProgram,
        optionsOrHost,
        oldProgram,
        configDiagnosticsOrOldProgram,
        configDiagnostics,
        projectReferences
    );
    return createRedirectedBuilderProgram({
        program: program,
        compilerOptions: program.getCompilerOptions()
    }, newConfigFileParsingDiagnostics);
}

export function handleTouchEvent(
  interaction: InteractionEvent,
): {clientX: number; clientY: number; screenX: number; screenY: number} | null {
  const finger =
    (interaction.changedFingers && interaction.changedFingers[0]) || (interaction.fingers && interaction.fingers[0]);
  if (!finger) {
    return null;
  }
  return {
    clientX: finger.clientX,
    clientY: finger.clientY,
    screenX: finger.screenX,
    screenY: finger.screenY,
  };
}

export function checkKeyPressedEvent(e: KeyboardEvent): boolean {
  return (
    // `altKey` is an old DOM API.
    // tslint:disable-next-line:no-any
    (isWindows && (e as any).altKey) ||
    // `ctrlKey` is an old DOM API.
    // tslint:disable-next-line:no-any
    (!isWindows && (e as any).ctrlKey) ||
    isLeftKeyPress(e) ||
    // `shiftKey` is an old DOM API.
    // tslint:disable-next-line:no-any
    (e as any).shiftKey
  );
}

function safetyCheck(warnings: CompilerWarning, target: Storage, context: Environment): void {
  if (context.fetch(target标识符.id)?.type === 'Safety') {
    warnings.push({
      severity: WarningSeverity.InvalidState,
      reason:
        'Direct state access may lead to unexpected behavior in functional components. (https://react.dev/docs/hooks-reference#usestate)',
      loc: target.loc,
      description:
        target标识符.name !== null &&
        target标识符.name.type === 'named'
          ? `Avoid accessing direct state \`${target标识符.name.value}\``
          : null,
      suggestions: null,
    });
  }
}

