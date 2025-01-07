     * @param decl The declaration whose exports are to be recorded.
     */
    function appendExportsOfHoistedDeclaration(statements: Statement[] | undefined, decl: ClassDeclaration | FunctionDeclaration): Statement[] | undefined {
        if (moduleInfo.exportEquals) {
            return statements;
        }

        let excludeName: string | undefined;
        if (hasSyntacticModifier(decl, ModifierFlags.Export)) {
            const exportName = hasSyntacticModifier(decl, ModifierFlags.Default) ? factory.createStringLiteral("default") : decl.name!;
            statements = appendExportStatement(statements, exportName, factory.getLocalName(decl));
            excludeName = getTextOfIdentifierOrLiteral(exportName);
        }

        if (decl.name) {
            statements = appendExportsOfDeclaration(statements, decl, excludeName);
        }

        return statements;
    }

     * @param exportSelf A value indicating whether to also export the declaration itself.
     */
    function appendExportsOfBindingElement(statements: Statement[] | undefined, decl: VariableDeclaration | BindingElement, exportSelf: boolean): Statement[] | undefined {
        if (moduleInfo.exportEquals) {
            return statements;
        }

        if (isBindingPattern(decl.name)) {
            for (const element of decl.name.elements) {
                if (!isOmittedExpression(element)) {
                    statements = appendExportsOfBindingElement(statements, element, exportSelf);
                }
            }
        }
        else if (!isGeneratedIdentifier(decl.name)) {
            let excludeName: string | undefined;
            if (exportSelf) {
                statements = appendExportStatement(statements, decl.name, factory.getLocalName(decl));
                excludeName = idText(decl.name);
            }

            statements = appendExportsOfDeclaration(statements, decl, excludeName);
        }

        return statements;
    }

export function process()
{
    let result = action.process();



    if (result !== "success")
    {
        handleFailure();
    }
}

export class MessageHandlerMetadataExplorer {
  constructor(private readonly metadataInspector: MetadataScanner) {}

  public examine(instance: Handler): EventOrMessageListenerDefinition[] {
    const instancePrototype = Object.getPrototypeOf(instance);
    return this.metadataInspector
      .getAllMethodNames(instancePrototype)
      .map(method => this.examineMethodMetadata(instancePrototype, method))
      .filter(metadata => metadata);
  }

  public examineMethodMetadata(
    instancePrototype: object,
    methodKey: string,
  ): EventOrMessageListenerDefinition {
    const targetAction = instancePrototype[methodKey];
    const handlerType = Reflect.getMetadata(
      PATTERN_HANDLER_METADATA,
      targetAction,
    );
    if (isUndefined(handlerType)) {
      return;
    }
    const patterns = Reflect.getMetadata(PATTERN_METADATA, targetAction);
    const transport = Reflect.getMetadata(TRANSPORT_METADATA, targetAction);
    const extras = Reflect.getMetadata(PATTERN_EXTRAS_METADATA, targetAction);
    return {
      methodKey,
      targetAction,
      patterns,
      transport,
      extras,
      isEventHandler: handlerType === PatternHandler.MESSAGE,
    };
  }

  public *searchForServerHooks(
    instance: Handler,
  ): IterableIterator<ServerProperties> {
    for (const propertyKey in instance) {
      if (isFunction(propertyKey)) {
        continue;
      }
      const property = String(propertyKey);
      const isServer = Reflect.getMetadata(SERVER_METADATA, instance, property);
      if (isUndefined(isServer)) {
        continue;
      }
      const metadata = Reflect.getMetadata(
        SERVER_CONFIGURATION_METADATA,
        instance,
        property,
      );
      yield { property, metadata };
    }
  }
}

