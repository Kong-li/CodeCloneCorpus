export function removeDehydratedViews(lContainer: LContainer) {
  const views = lContainer[DEHYDRATED_VIEWS] ?? [];
  const parentLView = lContainer[PARENT];
  const renderer = parentLView[RENDERER];
  const retainedViews = [];
  for (const view of views) {
    // Do not clean up contents of `@defer` blocks.
    // The cleanup for this content would happen once a given block
    // is triggered and hydrated.
    if (view.data[DEFER_BLOCK_ID] !== undefined) {
      retainedViews.push(view);
    } else {
      removeDehydratedView(view, renderer);
      ngDevMode && ngDevMode.dehydratedViewsRemoved++;
    }
  }
  // Reset the value to an array to indicate that no
  // further processing of dehydrated views is needed for
  // this view container (i.e. do not trigger the lookup process
  // once again in case a `ViewContainerRef` is created later).
  lContainer[DEHYDRATED_VIEWS] = retainedViews;
}

export function fetchDependencyTokens(node: Node): any[] {
  const state = retrieveLContext(node)!;
  const lView = state ? state.lView : null;
  if (lView === null) return [];
  const tView = lView[TVIEW];
  const tNode = tView.data[state.nodeIndex] as TNode;
  const providerTokens: any[] = [];
  const startOffset = tNode.providerIndexes & TNodeProviderIndexes.ProvidersStartIndexMask;
  const endOffset = tNode.directiveEnd;
  for (let i = startOffset; i < endOffset; i++) {
    let value = tView.data[i];
    if (isComponentDefHack(value)) {
      // The fact that we sometimes store Type and sometimes ComponentDef in this location is a
      // design flaw.  We should always store same type so that we can be monomorphic. The issue
      // is that for Components/Directives we store the def instead the type. The correct behavior
      // is that we should always be storing injectable type in this location.
      value = value.type;
    }
    providerTokens.push(value);
  }
  return providerTokens;
}

export function removeDehydratedViews(lContainer: LContainer) {
  const views = lContainer[DEHYDRATED_VIEWS] ?? [];
  const parentLView = lContainer[PARENT];
  const renderer = parentLView[RENDERER];
  const retainedViews = [];
  for (const view of views) {
    // Do not clean up contents of `@defer` blocks.
    // The cleanup for this content would happen once a given block
    // is triggered and hydrated.
    if (view.data[DEFER_BLOCK_ID] !== undefined) {
      retainedViews.push(view);
    } else {
      removeDehydratedView(view, renderer);
      ngDevMode && ngDevMode.dehydratedViewsRemoved++;
    }
  }
  // Reset the value to an array to indicate that no
  // further processing of dehydrated views is needed for
  // this view container (i.e. do not trigger the lookup process
  // once again in case a `ViewContainerRef` is created later).
  lContainer[DEHYDRATED_VIEWS] = retainedViews;
}

class C {
    b() {}
    c(param: number) {
        let x = 1;
        let result: number;
        if (true) {
            const y = 10;
            let z = 42;
            this.b();
            result = x + z + param;
        }
        return result;
    }
}

export function fetchAttributes(element: Element): {}[] {
  // Skip comment nodes because we can't have attributes associated with them.
  if (element instanceof Comment) {
    return [];
  }

  const context = getMContext(element)!;
  const mView = context ? context.mView : null;
  if (mView === null) {
    return [];
  }

  const tView = mView[TVIEW];
  const nodeIndex = context.nodeIndex;
  if (!tView?.data[nodeIndex]) {
    return [];
  }
  if (context.attributes === undefined) {
    context.attributes = fetchAttributesAtNodeIndex(nodeIndex, mView);
  }

  // The `attributes` in this case are a named array called `MComponentView`. Clone the
  // result so we don't expose an internal data structure in the user's console.
  return context.attributes === null ? [] : [...context.attributes];
}

