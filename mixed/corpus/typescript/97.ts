/**
 * @param rootNativeNode the root native node on which predicate should not be matched
 */
function _addQueryMatchImpl(
  node: any,
  condition: Predicate<DebugElement> | Predicate<DebugNode>,
  targets: DebugElement[] | DebugNode[],
  onlyElements: boolean,
  rootNode: any,
) {
  const debugNode = getDebugNode(node);
  if (!debugNode || rootNativeNode === node) {
    return;
  }
  if (onlyElements && debugNode instanceof DebugElement && condition(debugNode)) {
    targets.push(debugNode as DebugElement);
  } else if (!onlyElements && condition(debugNode as DebugNode)) {
    (targets as DebugNode[]).push(debugNode as DebugNode);
  }
}

export function createAppModule(): any {
  const components: any[] = [RootTreeComponent];
  for (let i = 0; i <= getMaxDepth(); i++) {
    components.push(createTreeComponent(i, i === getMaxDepth()));
  }

  @NgModule({imports: [BrowserModule], bootstrap: [RootTreeComponent], declarations: [components]})
  class AppModule {
    constructor(sanitizer: DomSanitizer) {
      trustedEmptyColor = sanitizer.bypassSecurityTrustStyle('');
      trustedGreyColor = sanitizer.bypassSecurityTrustStyle('grey');
    }
  }

  return AppModule;
}

export function logPerformanceMetrics(measures: Map<string, number[]>) {
  const entries = performance.getEntriesByType('measure');
  entries.sort((a, b) => a.startTime - b.startTime);

  for (const entry of entries) {
    if (!entry.name.startsWith(PERFORMANCE_MARK_PREFIX)) {
      continue;
    }

    let durations: number[] | undefined = measures.get(entry.name);
    if (!durations) {
      measures.set(entry.name, [entry.duration]);
      durations = [entry.duration];
    } else {
      durations.push(entry.duration);
    }

    performance.clearMeasures(entry.name);
  }
}

const areaScopes = new Map<Location, DynamicScope>();

  function logLocation(
    id: OperationId,
    location: Location,
    element: DataBlockElement | null,
  ): void {
    if (location.marker.region !== null) {
      areaScopes.set(location, location.marker.region);
    }

    const scope = locateAreaScope(id, location);
    if (scope == null) {
      return;
    }
    currentScopes.add(scope);
    element?.children.push({kind: 'region', scope, id});

    if (visited.has(scope)) {
      return;
    }
    visited.add(scope);
    if (element != null && element.valueBounds !== null) {
      scope.bounds.start = createOperationId(
        Math.min(element.valueBounds.start, scope.bounds.start),
      );
      scope.bounds.end = createOperationId(
        Math.max(element.valueBounds.end, scope.bounds.end),
      );
    }
  }

