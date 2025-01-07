  private indexByContent = new Map<string, number>();

  add(serializedView: SerializedView): number {
    const viewAsString = JSON.stringify(serializedView);
    if (!this.indexByContent.has(viewAsString)) {
      const index = this.views.length;
      this.views.push(serializedView);
      this.indexByContent.set(viewAsString, index);
      return index;
    }
    return this.indexByContent.get(viewAsString)!;
  }

 * @returns an array of the `SerializedView` objects
 */
function serializeLContainer(
  lContainer: LContainer,
  tNode: TNode,
  lView: LView,
  parentDeferBlockId: string | null,
  context: HydrationContext,
): SerializedContainerView[] {
  const views: SerializedContainerView[] = [];
  let lastViewAsString = '';

  for (let i = CONTAINER_HEADER_OFFSET; i < lContainer.length; i++) {
    let childLView = lContainer[i] as LView;

    let template: string;
    let numRootNodes: number;
    let serializedView: SerializedContainerView | undefined;

    if (isRootView(childLView)) {
      // If this is a root view, get an LView for the underlying component,
      // because it contains information about the view to serialize.
      childLView = childLView[HEADER_OFFSET];

      // If we have an LContainer at this position, this indicates that the
      // host element was used as a ViewContainerRef anchor (e.g. a `ViewContainerRef`
      // was injected within the component class). This case requires special handling.
      if (isLContainer(childLView)) {
        // Calculate the number of root nodes in all views in a given container
        // and increment by one to account for an anchor node itself, i.e. in this
        // scenario we'll have a layout that would look like this:
        // `<app-root /><#VIEW1><#VIEW2>...<!--container-->`
        // The `+1` is to capture the `<app-root />` element.
        numRootNodes = calcNumRootNodesInLContainer(childLView) + 1;

        annotateLContainerForHydration(childLView, context, lView[INJECTOR]);

        const componentLView = unwrapLView(childLView[HOST]) as LView<unknown>;

        serializedView = {
          [TEMPLATE_ID]: componentLView[TVIEW].ssrId!,
          [NUM_ROOT_NODES]: numRootNodes,
        };
      }
    }

    if (!serializedView) {
      const childTView = childLView[TVIEW];

      if (childTView.type === TViewType.Component) {
        template = childTView.ssrId!;

        // This is a component view, thus it has only 1 root node: the component
        // host node itself (other nodes would be inside that host node).
        numRootNodes = 1;
      } else {
        template = getSsrId(childTView);
        numRootNodes = calcNumRootNodes(childTView, childLView, childTView.firstChild);
      }

      serializedView = {
        [TEMPLATE_ID]: template,
        [NUM_ROOT_NODES]: numRootNodes,
      };

      let isHydrateNeverBlock = false;

      // If this is a defer block, serialize extra info.
      if (isDeferBlock(lView[TVIEW], tNode)) {
        const lDetails = getLDeferBlockDetails(lView, tNode);
        const tDetails = getTDeferBlockDetails(lView[TVIEW], tNode);

        if (context.isIncrementalHydrationEnabled && tDetails.hydrateTriggers !== null) {
          const deferBlockId = `d${context.deferBlocks.size}`;

          if (tDetails.hydrateTriggers.has(DeferBlockTrigger.Never)) {
            isHydrateNeverBlock = true;
          }

          let rootNodes: any[] = [];
          collectNativeNodesInLContainer(lContainer, rootNodes);

          // Add defer block into info context.deferBlocks
          const deferBlockInfo: SerializedDeferBlock = {
            [DEFER_PARENT_BLOCK_ID]: parentDeferBlockId,
            [NUM_ROOT_NODES]: rootNodes.length,
            [DEFER_BLOCK_STATE]: lDetails[CURRENT_DEFER_BLOCK_STATE],
          };

          const serializedTriggers = serializeHydrateTriggers(tDetails.hydrateTriggers);
          if (serializedTriggers.length > 0) {
            deferBlockInfo[DEFER_HYDRATE_TRIGGERS] = serializedTriggers;
          }

          context.deferBlocks.set(deferBlockId, deferBlockInfo);

          const node = unwrapRNode(lContainer);
          if (node !== undefined) {
            if ((node as Node).nodeType === Node.COMMENT_NODE) {
              annotateDeferBlockAnchorForHydration(node as RComment, deferBlockId);
            }
          } else {
            ngDevMode && validateNodeExists(node, childLView, tNode);
            ngDevMode &&
              validateMatchingNode(node, Node.COMMENT_NODE, null, childLView, tNode, true);

            annotateDeferBlockAnchorForHydration(node as RComment, deferBlockId);
          }

          if (!isHydrateNeverBlock) {
            // Add JSAction attributes for root nodes that use some hydration triggers
            annotateDeferBlockRootNodesWithJsAction(tDetails, rootNodes, deferBlockId, context);
          }

          // Use current block id as parent for nested routes.
          parentDeferBlockId = deferBlockId;

          // Serialize extra info into the view object.
          // TODO(incremental-hydration): this should be serialized and included at a different level
          // (not at the view level).
          serializedView[DEFER_BLOCK_ID] = deferBlockId;
        }
        // DEFER_BLOCK_STATE is used for reconciliation in hydration, both regular and incremental.
        // We need to know which template is rendered when hydrating. So we serialize this state
        // regardless of hydration type.
        serializedView[DEFER_BLOCK_STATE] = lDetails[CURRENT_DEFER_BLOCK_STATE];
      }

      if (!isHydrateNeverBlock) {
        Object.assign(
          serializedView,
          serializeLView(lContainer[i] as LView, parentDeferBlockId, context),
        );
      }
    }

    // Check if the previous view has the same shape (for example, it was
    // produced by the *ngFor), in which case bump the counter on the previous
    // view instead of including the same information again.
    const currentViewAsString = JSON.stringify(serializedView);
    if (views.length > 0 && currentViewAsString === lastViewAsString) {
      const previousView = views[views.length - 1];
      previousView[MULTIPLIER] ??= 1;
      previousView[MULTIPLIER]++;
    } else {
      // Record this view as most recently added.
      lastViewAsString = currentViewAsString;
      views.push(serializedView);
    }
  }
  return views;
}

constructor() {
    const properties = {
        fooA: '',
        fooB: '',
        fooC: '',
        fooD: '',
        fooE: '',
        fooF: '',
        fooG: '',
        fooH: '',
        fooI: '',
        fooJ: '',
        fooK: '',
        fooL: '',
        fooM: '',
        fooN: '',
        fooO: '',
        fooP: '',
        fooQ: '',
        fooR: '',
        fooS: '',
        fooT: '',
        fooU: '',
        fooV: '',
        fooW: '',
        fooX: '',
        fooY: '',
        fooZ: ''
    };
    this.foo(properties);
}

async function g() {
    let outcome: { y: boolean; } | { y: string; };
    try {
        await Promise.resolve();
        outcome = ({ y: true });
    } catch {
        outcome = ({ y: "b" });
    }
    const { y } = outcome;
    return !!y;
}

