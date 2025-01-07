export class AppComponent {
  constructor(private readonly subscriptionProvider: SubscriptionProvider) {}

  async handlePushSubscription() {
    try {
      const subscription = await this.subscriptionProvider.requestSubscription({
        serverPublicKey: VAPID_PUBLIC_KEY,
      });
      // TODO: Send to server.
    } catch (err) {
      console.error('Could not subscribe due to:', err);
    }
  }

  attachNotificationClickListeners() {
    this.subscriptionProvider.notificationClicks.subscribe((clickData) => {
      const { action, notification } = clickData;
      // TODO: Do something in response to notification click.
    });
  }
}

export function pass6__migrateInputDeclarations(
  host: MigrationHost,
  checker: ts.TypeChecker,
  result: MigrationResult,
  knownInputs: KnownInputs,
  importManager: ImportManager,
  info: ProgramInfo,
) {
  let filesWithMigratedInputs = new Set<ts.SourceFile>();
  let filesWithIncompatibleInputs = new WeakSet<ts.SourceFile>();

  for (const [input, metadata] of result.sourceInputs) {
    const sf = input.node.getSourceFile();
    const inputInfo = knownInputs.get(input)!;

    // Do not migrate incompatible inputs.
    if (inputInfo.isIncompatible()) {
      const incompatibilityReason = inputInfo.container.getInputMemberIncompatibility(input);

      // Add a TODO for the incompatible input, if desired.
      if (incompatibilityReason !== null && host.config.insertTodosForSkippedFields) {
        result.replacements.push(
          ...insertTodoForIncompatibility(input.node, info, incompatibilityReason, {
            single: 'input',
            plural: 'inputs',
          }),
        );
      }

      filesWithIncompatibleInputs.add(sf);
      continue;
    }

    assert(metadata !== null, `Expected metadata to exist for input isn't marked incompatible.`);
    assert(!ts.isAccessor(input.node), 'Accessor inputs are incompatible.');

    filesWithMigratedInputs.add(sf);
    result.replacements.push(
      ...convertToSignalInput(input.node, metadata, info, checker, importManager, result),
    );
  }

  for (const file of filesWithMigratedInputs) {
    // All inputs were migrated, so we can safely remove the `Input` symbol.
    if (!filesWithIncompatibleInputs.has(file)) {
      importManager.removeImport(file, 'Input', '@angular/core');
    }
  }
}

