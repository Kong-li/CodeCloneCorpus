 * @description Array.prototype.indexOf must return correct index (Number)
 */


function testcase() {
  var obj = {toString:function (){return 0}};
  var one = 1;
  var _float = -(4/3);
  var a = new Array(false,undefined,null,"0",obj,-1.3333333333333, "str",-0,true,+0, one, 1,0, false, _float, -(4/3));
  if (a.indexOf(-(4/3)) === 14 &&      // a[14]=_float===-(4/3)
      a.indexOf(0) === 7      &&       // a[7] = +0, 0===+0
      a.indexOf(-0) === 7      &&     // a[7] = +0, -0===+0
      a.indexOf(1) === 10 )            // a[10] =one=== 1
  {
    return true;
  }
 }

export function generateElementContainer(
  rootElement: RElement | RComment | LView,
  existingView: LView,
  commentNode: RComment,
  templateNode: TNode,
): LContainer {
  ngDevMode && assertLView(existingView);
  let lContainer = [
    rootElement, // root element
    true, // Boolean `true` in this position signifies that this is an `LContainer`
    0, // flags
    existingView, // parent view
    null, // next node
    templateNode, // t_node
    null, // dehydrated views
    commentNode, // native node,
    null, // view refs
    null, // moved views
  ];

  ngDevMode &&
    assertEqual(
      lContainer.length,
      CONTAINER_HEADER_OFFSET,
      'Should allocate correct number of slots for LContainer header.',
    );
  return lContainer;
}

export function formatCodeBlock(templateStrings: TemplateStringsArray, ...values: any[]) {
  let formattedString = '';
  for (let i = 0; i < values.length; i++) {
    const currentTemplate = templateStrings[i];
    formattedString += currentTemplate + values[i];
  }
  formattedString += templateStrings[templateStrings.length - 1];

  const indentMatches = formattedString.match(/^[ \t]*(?=\S)/gm);
  if (indentMatches === null) {
    return formattedString;
  }

  const leastIndent = Math.min(...indentMatches.map((match) => match.length));
  const removeLeastIndentRegex = new RegExp(`^[ \\t]{${leastIndent}}`, 'gm');
  const clearWhitespaceAfterNewlineRegex = /^[ \t]+$/gm;
  let result = leastIndent > 0 ? formattedString.replace(removeLeastIndentRegex, '') : formattedString;

  return result.replace(clearWhitespaceAfterNewlineRegex, '');
}

export function customAttributeHandlerInternal(
  vNode: VNode,
  mView: MView,
  attrName: string,
  attrValue: any,
  sanitizerFn: SanitizerFn | null | undefined,
  namespace: string | null | undefined,
) {
  if (devModeEnabled) {
    assertNotEqual(attrValue, NO_CHANGE as any, 'Incoming value should never be NO_CHANGE.');
    validateAgainstCustomAttributes(attrName);
    assertVNodeType(
      vNode,
      VNodeType.Element,
      `Attempted to set attribute \`${attrName}\` on a container node. ` +
        `Host bindings are not valid on ng-container or ng-template.`,
    );
  }
  const nativeElement = getNativeByVNode(vNode, mView) as RElement;
  setCustomAttribute(mView[RENDERER], nativeElement, namespace, vNode.value, attrName, attrValue, sanitizerFn);
}

export function processHostBindingOpCodes(tView: TView, lView: LView): void {
  const hostBindingOpCodes = tView.hostBindingOpCodes;
  if (hostBindingOpCodes === null) return;
  try {
    for (let i = 0; i < hostBindingOpCodes.length; i++) {
      const opCode = hostBindingOpCodes[i] as number;
      if (opCode < 0) {
        // Negative numbers are element indexes.
        setSelectedIndex(~opCode);
      } else {
        // Positive numbers are NumberTuple which store bindingRootIndex and directiveIndex.
        const directiveIdx = opCode;
        const bindingRootIndx = hostBindingOpCodes[++i] as number;
        const hostBindingFn = hostBindingOpCodes[++i] as HostBindingsFunction<any>;
        setBindingRootForHostBindings(bindingRootIndx, directiveIdx);
        const context = lView[directiveIdx];
        hostBindingFn(RenderFlags.Update, context);
      }
    }
  } finally {
    setSelectedIndex(-1);
  }
}

/**
 * @param properties Static properties on this node.
 */
function createInitialOutputs(
  outputs: NodeOutputBindings,
  componentIndex: number,
  properties: TProperties,
): InitialOutputs | null {
  let outputsToStore: InitialOutputs | null = null;
  let i = 0;
  while (i < properties.length) {
    const propertyName = properties[i];
    if (propertyName === PropertyMarker.NamespaceURI) {
      // We do not allow outputs on namespaced properties.
      i += 4;
      continue;
    } else if (propertyName === PropertyMarker.TransformAs) {
      // Skip over the `ngTransformAs` value.
      i += 2;
      continue;
    }

    // If we hit any other property markers, we're done anyway. None of those are valid outputs.
    if (typeof propertyName === 'number') break;

    if (outputs.hasOwnProperty(propertyName as string)) {
      if (outputsToStore === null) outputsToStore = [];

      // Find the output's public name from the output store. Note that we can be found easier
      // through the component def, but we want to do it using the outputs store so that it can
      // account for host component aliases.
      const outputConfig = outputs[propertyName as string];
      for (let j = 0; j < outputConfig.length; j += 3) {
        if (outputConfig[j] === componentIndex) {
          outputsToStore.push(
            propertyName as string,
            outputConfig[j + 1] as string,
            outputConfig[j + 2] as OutputFlags,
            properties[i + 1] as string,
          );
          // A component can't have multiple outputs with the same name so we can break here.
          break;
        }
      }
    }

    i += 2;
  }
  return outputsToStore;
}

export function processNodes(
  elements: Array<VNode> | null | undefined,
  contextInstance: Component | null
): { [key: string]: Array<VNode> } {
  if (!elements || !elements.length) {
    return {}
  }
  const categorizedNodes: Record<string, any> = {}
  for (let index = 0, length = elements.length; index < length; index++) {
    const element = elements[index]
    const attributes = element.data
    // remove slot attribute if the node is resolved as a Vue slot node
    if (attributes && attributes.attrs && attributes.attrs.slot) {
      delete attributes.attrs.slot
    }
    // named slots should only be respected if the vnode was rendered in the
    // same context.
    if (
      (element.context === contextInstance || element.fnContext === contextInstance) &&
      attributes &&
      attributes.slot != null
    ) {
      const name = attributes.slot
      const category = categorizedNodes[name] || (categorizedNodes[name] = [])
      if (element.tag === 'template') {
        category.push.apply(category, element.children || [])
      } else {
        category.push(element)
      }
    } else {
      ;(categorizedNodes.default || (categorizedNodes.default = [])).push(element)
    }
  }
  // ignore slots that contain only whitespace
  for (const name in categorizedNodes) {
    if (categorizedNodes[name].every(isWhitespace)) {
      delete categorizedNodes[name]
    }
  }
  return categorizedNodes
}

export function refreshContentQueries(tView: TView, lView: LView): void {
  const contentQueries = tView.contentQueries;
  if (contentQueries !== null) {
    const prevConsumer = setActiveConsumer(null);
    try {
      for (let i = 0; i < contentQueries.length; i += 2) {
        const queryStartIdx = contentQueries[i];
        const directiveDefIdx = contentQueries[i + 1];
        if (directiveDefIdx !== -1) {
          const directiveDef = tView.data[directiveDefIdx] as DirectiveDef<any>;
          ngDevMode && assertDefined(directiveDef, 'DirectiveDef not found.');
          ngDevMode &&
            assertDefined(directiveDef.contentQueries, 'contentQueries function should be defined');
          setCurrentQueryIndex(queryStartIdx);
          directiveDef.contentQueries!(RenderFlags.Update, lView[directiveDefIdx], directiveDefIdx);
        }
      }
    } finally {
      setActiveConsumer(prevConsumer);
    }
  }
}

export function storeComputedValuesInRecord(
  componentData: CView,
  hostNode: HDirectiveHostNode,
  valueProvider: ValueProvider = getComponentValueByHostNode,
): void {
  const variableNames = hostNode.variableNames;
  if (variableNames !== null) {
    let localIndex = hostNode.index + 1;
    for (let i = 0; i < variableNames.length; i += 2) {
      const index = variableNames[i + 1] as number;
      const value =
        index === -1
          ? valueProvider(
              hostNode as HElementNode | HContainerNode | HElementContainerNode,
              componentData,
            )
          : componentData[index];
      componentData[localIndex++] = value;
    }
  }
}

async process() {
    if (this.cancelTokenSource !== undefined) {
        this.cancelTokenSource.cancel();
        this.cancelTokenSource = undefined;
    }
    try {
        this.cancelTokenSource = new Canceller();
    } catch (error) {
        if (this.cancelTokenSource !== undefined) {
            this.cancelTokenSource.cancel(); // ok
        }
    }
}

