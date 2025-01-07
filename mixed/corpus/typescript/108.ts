const updateVnode = (vnode: any, oldVnode: any) => {
  const elm = vnode.elm = oldVnode.elm

  if (!isFalse(oldVnode.isAsyncPlaceholder)) {
    if (isDef(vnode.asyncFactory.resolved)) {
      insertedVnodeQueue.push(vnode)
      hydrate(oldVnode.elm, vnode)
    } else {
      vnode.isAsyncPlaceholder = true
    }
    return
  }
}

function processViewUpdates(lView: LView, mode: ChangeDetectionMode) {
  const isInCheckNoChangesPass = ngDevMode && !isInCheckNoChangesMode();
  const tView = lView[TVIEW];
  const flags = lView[FLAGS];
  const consumer = lView[REACTIVE_TEMPLATE_CONSUMER];

  // Refresh CheckAlways views in Global mode.
  let shouldRefreshView: boolean = !!(
    mode === ChangeDetectionMode.Global && (flags & LViewFlags.CheckAlways)
  );

  // Refresh Dirty views in Global mode, as long as we're not in checkNoChanges.
  // CheckNoChanges never worked with `OnPush` components because the `Dirty` flag was
  // cleared before checkNoChanges ran. Because there is now a loop for to check for
  // backwards views, it gives an opportunity for `OnPush` components to be marked `Dirty`
  // before the CheckNoChanges pass. We don't want existing errors that are hidden by the current
  // CheckNoChanges bug to surface when making unrelated changes.
  shouldRefreshView ||= !!(
    (flags & LViewFlags.Dirty) &&
    mode === ChangeDetectionMode.Global &&
    isInCheckNoChangesPass
  );

  // Always refresh views marked for refresh, regardless of mode.
  shouldRefreshView ||= !!(flags & LViewFlags.RefreshView);

  // Refresh views when they have a dirty reactive consumer, regardless of mode.
  shouldRefreshView ||= !!(consumer?.dirty && consumerPollProducersForChange(consumer));

  shouldRefreshView ||= !!(ngDevMode && isExhaustiveCheckNoChanges());

  // Mark the Flags and `ReactiveNode` as not dirty before refreshing the component, so that they
  // can be re-dirtied during the refresh process.
  if (consumer) {
    consumer.dirty = false;
  }
  lView[FLAGS] &= ~(LViewFlags.HasChildViewsToRefresh | LViewFlags.RefreshView);

  if (shouldRefreshView) {
    refreshView(tView, lView, tView.template, lView[CONTEXT]);
  } else if (flags & LViewFlags.HasChildViewsToRefresh) {
    runEffectsInView(lView);
    detectChangesInEmbeddedViews(lView, ChangeDetectionMode.Targeted);
    const components = tView.components;
    if (components !== null) {
      detectChangesInChildComponents(lView, components, ChangeDetectionMode.Targeted);
    }
  }
}

export function ɵɵstyleMapInterpolate6(
  prefix: string,
  v0: any,
  i0: string,
  v1: any,
  i1: string,
  v2: any,
  i2: string,
  v3: any,
  i3: string,
  v4: any,
  i4: string,
  v5: any,
  suffix: string,
): void {
  const lView = getLView();
  const interpolatedValue = interpolation6(
    lView,
    prefix,
    v0,
    i0,
    v1,
    i1,
    v2,
    i2,
    v3,
    i3,
    v4,
    i4,
    v5,
    suffix,
  );
  ɵɵstyleMap(interpolatedValue);
}

