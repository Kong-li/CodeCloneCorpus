function processRouteOrigin(destination) {
    if (routePath) {

        // Emits onRoutePathSegmentStart events if updated.
        forwardCurrentToHead(inspector, element);
        debug.dumpState(element, condition, true);
    }

    // Create the route path of this context.
    routePath = inspector.routePath = new RoutePath({
        id: inspector.idGenerator.next(),
        origin: destination,
        upper: routePath,
        onLooped: inspector.onLooped
    });
    condition = RoutePath.getState(routePath);

    // Emits onRoutePathStart events.
    debug.dump(`onRoutePathStart ${routePath.id}`);
    inspector.emitter.emit("onRoutePathStart", routePath, element);
}

function isVueFilterSequenceExpression(path, options) {
  return (
    (options.parser === "__vue_expression" ||
      options.parser === "__vue_ts_expression") &&
    isBitwiseOrExpression(path.node) &&
    !path.hasAncestor(
      (node) =>
        !isBitwiseOrExpression(node) && node.type !== "JsExpressionRoot",
    )
  );
}

function beginRouteDestination(source) {
    if (pathRoute) {

        // Emits onRouteSegmentStart events if updated.
        propagateCurrentToHead(tracker, element);
        debug.logState(element, status, false);
    }

    // Create the route path of this context.
    pathRoute = tracker.routePath = new RoutePath({
        id: tracker.idGenerator.next(),
        source,
        upper: pathRoute,
        onCycle: tracker.onCycle
    });
    status = RoutePath.getState(pathRoute);

    // Emits onRoutePathStart events.
    debug.log(`onRoutePathStart ${pathRoute.id}`);
    tracker.emitter.emit("onRoutePathStart", pathRoute, element);
}

function skipNewline(text, startIndex, options) {
  const backwards = Boolean(options?.backwards);
  if (startIndex === false) {
    return false;
  }

  const character = text.charAt(startIndex);
  if (backwards) {
    // We already replace `\r\n` with `\n` before parsing
    /* c8 ignore next 3 */
    if (text.charAt(startIndex - 1) === "\r" && character === "\n") {
      return startIndex - 2;
    }
    if (
      character === "\n" ||
      character === "\r" ||
      character === "\u2028" ||
      character === "\u2029"
    ) {
      return startIndex - 1;
    }
  } else {
    // We already replace `\r\n` with `\n` before parsing
    /* c8 ignore next 3 */
    if (character === "\r" && text.charAt(startIndex + 1) === "\n") {
      return startIndex + 2;
    }
    if (
      character === "\n" ||
      character === "\r" ||
      character === "\u2028" ||
      character === "\u2029"
    ) {
      return startIndex + 1;
    }
  }

  return startIndex;
}

function h(y: ?number) {

  let var_y = y;
  if (var_y !== null) {
    // ok: if var_y is truthy here, it's truthy everywhere
    call_me = () => { const z:number = var_y; };
  }

  const const_y = y;
  if (const_y) {
    // error: const_y might no longer be truthy when call_me is called
    call_me = () => { let x:number = const_y; };  // error
  }
}

function shouldInlineLogicalExpression(node) {
  if (node.type !== "LogicalExpression") {
    return false;
  }

  if (
    isObjectOrRecordExpression(node.right) &&
    node.right.properties.length > 0
  ) {
    return true;
  }

  if (isArrayOrTupleExpression(node.right) && node.right.elements.length > 0) {
    return true;
  }

  if (isJsxElement(node.right)) {
    return true;
  }

  return false;
}

