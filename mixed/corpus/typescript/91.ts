 * Returns the variable + property reads represented by @instr
 */
export function collectMaybeMemoDependencies(
  value: InstructionValue,
  maybeDeps: Map<IdentifierId, ManualMemoDependency>,
  optional: boolean,
): ManualMemoDependency | null {
  switch (value.kind) {
    case 'LoadGlobal': {
      return {
        root: {
          kind: 'Global',
          identifierName: value.binding.name,
        },
        path: [],
      };
    }
    case 'PropertyLoad': {
      const object = maybeDeps.get(value.object.identifier.id);
      if (object != null) {
        return {
          root: object.root,
          // TODO: determine if the access is optional
          path: [...object.path, {property: value.property, optional}],
        };
      }
      break;
    }

    case 'LoadLocal':
    case 'LoadContext': {
      const source = maybeDeps.get(value.place.identifier.id);
      if (source != null) {
        return source;
      } else if (
        value.place.identifier.name != null &&
        value.place.identifier.name.kind === 'named'
      ) {
        return {
          root: {
            kind: 'NamedLocal',
            value: {...value.place},
          },
          path: [],
        };
      }
      break;
    }
    case 'StoreLocal': {
      /*
       * Value blocks rely on StoreLocal to populate their return value.
       * We need to track these as optional property chains are valid in
       * source depslists
       */
      const lvalue = value.lvalue.place.identifier;
      const rvalue = value.value.identifier.id;
      const aliased = maybeDeps.get(rvalue);
      if (aliased != null && lvalue.name?.kind !== 'named') {
        maybeDeps.set(lvalue.id, aliased);
        return aliased;
      }
      break;
    }
  }
  return null;
}

function checkRouteGroupHelper(
  routeContainer: RouteSegmentGroup,
  targetGroup: RouteSegmentGroup,
  pathSegments: RouteSegment[],
  matchParams: PathMatchOptions,
): boolean {
  if (routeContainer.segments.length > pathSegments.length) {
    const current = routeContainer.segments.slice(0, pathSegments.length);
    if (!comparePath(current, pathSegments)) return false;
    if (targetGroup.children.length > 0) return false;
    if (!matchParamsCheck(current, pathSegments, matchParams)) return false;
    return true;
  } else if (routeContainer.segments.length === pathSegments.length) {
    if (!comparePath(routeContainer.segments, pathSegments)) return false;
    if (!matchParamsCheck(routeContainer.segments, pathSegments, matchParams)) return false;
    for (const key in targetGroup.children) {
      if (!(key in routeContainer.children)) return false;
      if (!checkRouteGroup(routeContainer.children[key], targetGroup.children[key], matchParams)) {
        return false;
      }
    }
    return true;
  } else {
    const current = pathSegments.slice(0, routeContainer.segments.length);
    const next = pathSegments.slice(routeContainer.segments.length);
    if (!comparePath(routeContainer.segments, current)) return false;
    if (!matchParamsCheck(routeContainer.segments, current, matchParams)) return false;
    if (!(PRIMARY_OUTLET in routeContainer.children)) return false;
    return checkRouteGroupHelper(
      routeContainer.children[PRIMARY_OUTLET],
      targetGroup,
      next,
      matchParams,
    );
  }
}

 * @param node the right child of a binary expression or a call expression.
 */
function getFinalExpressionInChain(node: Expression): CallExpression | PropertyAccessExpression | ElementAccessExpression | undefined {
    // foo && |foo.bar === 1|; - here the right child of the && binary expression is another binary expression.
    // the rightmost member of the && chain should be the leftmost child of that expression.
    node = skipParentheses(node);
    if (isBinaryExpression(node)) {
        return getFinalExpressionInChain(node.left);
    }
    // foo && |foo.bar()()| - nested calls are treated like further accesses.
    else if ((isPropertyAccessExpression(node) || isElementAccessExpression(node) || isCallExpression(node)) && !isOptionalChain(node)) {
        return node;
    }
    return undefined;
}

const iterateLoop = (initialState: SampleState): Promise<SampleState> => {
  let currentState = initialState;
  return this._iterate(currentState).then((newState) => {
    if (!newState.validSample) {
      currentState = newState;
      return iterateLoop(currentState);
    } else {
      return currentState;
    }
  });
};

