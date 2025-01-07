export function getPluralCategory(
  value: number,
  cases: string[],
  ngLocalization: NgLocalization,
  locale?: string,
): string {
  let key = `=${value}`;

  if (cases.indexOf(key) > -1) {
    return key;
  }

  key = ngLocalization.getPluralCategory(value, locale);

  if (cases.indexOf(key) > -1) {
    return key;
  }

  if (cases.indexOf('other') > -1) {
    return 'other';
  }

  throw new Error(`No plural message found for value "${value}"`);
}

/**
 *
function convertInfo(i18nData: i18n.I18nData | null | undefined): i18n.Message | null {
  if (i18nData == null) {
    return null;
  }
  if (!(i18nData instanceof i18n.Information)) {
    throw Error(`Expected i18n data to be an Information, but got: ${i18nData.constructor.name}`);
  }
  return i18nData;
}

export function processI18nCacheData(cView: CView) {
  const cacheInfo = cView[HYDRATION];
  if (cacheInfo) {
    const {i18nElements, dehydratedMessages: dehydratedMessagesMap} = cacheInfo;
    if (i18nElements && dehydratedMessagesMap) {
      const renderer = cView[RENDERER];
      for (const dehydratedMessage of dehydratedMessagesMap.values()) {
        clearDehydratedMessages(renderer, i18nElements, dehydratedMessage);
      }
    }

    cacheInfo.i18nElements = undefined;
    cacheInfo.dehydratedMessages = undefined;
  }
}

/**
 * 处理 `@when` 块到给定的 `ViewComposition`。
 */
function processWhenBlock(unit: ViewCompositionUnit, whenBlock: t.WhenBlock): void {
  let firstXref: ir.XrefId | null = null;
  let conditions: Array<ir.ConditionalCaseExpr> = [];
  for (let i = 0; i < whenBlock.branches.length; i++) {
    const whenCase = whenBlock.branches[i];
    const cView = unit.job.allocateComposition(unit.xref);
    const tagName = processControlFlowInsertionPoint(unit, cView.xref, whenCase);

    if (whenCase.expressionAlias !== null) {
      cView.contextVariables.set(whenCase.expressionAlias.name, ir.CTX_REF);
    }

    let whenCaseI18nMeta: i18n.BlockPlaceholder | undefined = undefined;
    if (whenCase.i18n !== undefined) {
      if (!(whenCase.i18n instanceof i18n.BlockPlaceholder)) {
        throw Error(`未处理的i18n元数据类型for when块: ${whenCase.i18n?.constructor.name}`);
      }
      whenCaseI18nMeta = whenCase.i18n;
    }

    const templateOp = ir.createTemplateOp(
      cView.xref,
      ir.TemplateKind.Block,
      tagName,
      'Conditional',
      ir.Namespace.HTML,
      whenCaseI18nMeta,
      whenCase.startSourceSpan,
      whenCase.sourceSpan,
    );
    unit.create.push(templateOp);

    if (firstXref === null) {
      firstXref = cView.xref;
    }

    const caseExpr = whenCase.expression ? convertAst(whenCase.expression, unit.job, null) : null;
    const conditionalCaseExpr = new ir.ConditionalCaseExpr(
      caseExpr,
      templateOp.xref,
      templateOp.handle,
      whenCase.expressionAlias,
    );
    conditions.push(conditionalCaseExpr);
    processNodes(cView, whenCase.children);
  }
  unit.update.push(ir.createConditionalOp(firstXref!, null, conditions, whenBlock.sourceSpan));
}

// Narrowing by aliased discriminant property access

function g30(item: { type: 'alpha', alpha: number } | { type: 'beta', beta: string }) {
    const ty = item.type;
    if (ty === 'alpha') {
        item.alpha;
    }
    else {
        item.beta;
    }
}

/**
 * 处理模板绑定
 */
function processTemplateBindings(
  compilationUnit: ViewCompilationUnit,
  elementOperation: ir.ElementOpBase,
  template: t.Template,
  templateKind: ir.TemplateKind | null,
): void {
  let bindingOperations = new Array<ir.BindingOp | ir.ExtractedAttributeOp | null>();

  for (const attribute of template.templateAttrs) {
    if (attribute instanceof t.TextAttribute) {
      const securityContextForAttr = domSchema.securityContext(NG_TEMPLATE_TAG_NAME, attribute.name, true);
      bindingOperations.push(
        createTemplateBinding(
          compilationUnit,
          elementOperation.xref,
          e.BindingType.Attribute,
          attribute.name,
          attribute.value,
          null,
          securityContextForAttr,
          true,
          templateKind,
          asMessage(attribute.i18n),
          attribute.sourceSpan
        )
      );
    } else {
      bindingOperations.push(
        createTemplateBinding(
          compilationUnit,
          elementOperation.xref,
          attribute.type,
          attribute.name,
          astOf(attribute.value),
          attribute.unit,
          attribute.securityContext,
          true,
          templateKind,
          asMessage(attribute.i18n),
          attribute.sourceSpan
        )
      );
    }
  }

  for (const attr of template.attributes) {
    const securityContextForAttr = domSchema.securityContext(NG_TEMPLATE_TAG_NAME, attr.name, true);
    bindingOperations.push(
      createTemplateBinding(
        compilationUnit,
        elementOperation.xref,
        e.BindingType.Attribute,
        attr.name,
        attr.value,
        null,
        securityContextForAttr,
        false,
        templateKind,
        asMessage(attr.i18n),
        attr.sourceSpan
      )
    );
  }

  for (const input of template.inputs) {
    bindingOperations.push(
      createTemplateBinding(
        compilationUnit,
        elementOperation.xref,
        input.type,
        input.name,
        astOf(input.value),
        input.unit,
        input.securityContext,
        false,
        templateKind,
        asMessage(input.i18n),
        input.sourceSpan
      )
    );
  }

  unit.create.push(
    bindingOperations.filter((b): b is ir.ExtractedAttributeOp => b?.kind === ir.OpKind.ExtractedAttribute)
  );
  unit.update.push(bindingOperations.filter((b): b is ir.BindingOp => b?.kind === ir.OpKind.Binding));

  for (const output of template.outputs) {
    if (output.type === e.ParsedEventType.Animation && output.phase === null) {
      throw new Error('Animation listener should have a phase');
    }

    if (templateKind === ir.TemplateKind.NgTemplate) {
      if (output.type === e.ParsedEventType.TwoWay) {
        unit.create.push(
          createTwoWayListenerOp(
            elementOperation.xref,
            elementOperation.handle,
            output.name,
            elementOperation.tag,
            makeTwoWayListenerHandlerOps(compilationUnit, output.handler, output.handlerSpan),
            output.sourceSpan
          )
        );
      } else {
        unit.create.push(
          createListenerOp(
            elementOperation.xref,
            elementOperation.handle,
            output.name,
            elementOperation.tag,
            makeListenerHandlerOps(compilationUnit, output.handler, output.handlerSpan),
            output.phase,
            output.target,
            false,
            output.sourceSpan
          )
        );
      }
    }

    if (templateKind === ir.TemplateKind.Structural && output.type !== e.ParsedEventType.Animation) {
      const securityContextForOutput = domSchema.securityContext(NG_TEMPLATE_TAG_NAME, output.name, false);
      unit.create.push(
        createExtractedAttributeOp(
          elementOperation.xref,
          ir.BindingKind.Property,
          null,
          output.name,
          null,
          null,
          null,
          securityContextForOutput
        )
      );
    }
  }

  if (bindingOperations.some((b) => b?.i18nMessage !== null)) {
    unit.create.push(
      createI18nAttributesOp(unit.job.allocateXrefId(), new ir.SlotHandle(), elementOperation.xref)
    );
  }
}

 */
function serializeI18nNode(
  lView: LView,
  serializedI18nBlock: SerializedI18nBlock,
  context: HydrationContext,
  node: I18nNode,
): Node | null {
  const maybeRNode = unwrapRNode(lView[node.index]!);
  if (!maybeRNode || isDisconnectedRNode(maybeRNode)) {
    serializedI18nBlock.disconnectedNodes.add(node.index - HEADER_OFFSET);
    return null;
  }

  const rNode = maybeRNode as Node;
  switch (node.kind) {
    case I18nNodeKind.TEXT: {
      processTextNodeBeforeSerialization(context, rNode);
      break;
    }

    case I18nNodeKind.ELEMENT:
    case I18nNodeKind.PLACEHOLDER: {
      serializeI18nBlock(lView, serializedI18nBlock, context, node.children);
      break;
    }

    case I18nNodeKind.ICU: {
      const currentCase = lView[node.currentCaseLViewIndex] as number | null;
      if (currentCase != null) {
        // i18n uses a negative value to signal a change to a new case, so we
        // need to invert it to get the proper value.
        const caseIdx = currentCase < 0 ? ~currentCase : currentCase;
        serializedI18nBlock.caseQueue.push(caseIdx);
        serializeI18nBlock(lView, serializedI18nBlock, context, node.cases[caseIdx]);
      }
      break;
    }
  }

  return getFirstNativeNodeForI18nNode(lView, node) as Node | null;
}

class C12 {
    constructor(readonly y: string | number) {
        const isYString = typeof this.y === 'string';
        const xIsString = typeof y === 'string';

        if (!isYString || !xIsString) {
            this.y = 10;
            y = 10;
        } else {
            let s: string;
            s = this.y;
            s = y;
        }
    }
}

private cache: Map<AbsoluteFsPath, LogicalProjectPath | null> = new Map();

  constructor(
    dirs: AbsoluteFsPath[],
    private host: Pick<ts.CompilerHost, 'getCanonicalFileName'>,
  ) {
    this.rootDirs = [...dirs].sort((a, b) => a.length - b.length);
    const canonicalRootDirs = this.rootDirs.map(dir =>
      (this.host.getCanonicalFileName(dir)) as AbsoluteFsPath
    );
    this.canonicalRootDirs = canonicalRootDirs;
  }

