const PLANNED = 'planned';

      function planAction(action: Action) {
        const config = <HTTPConfig>action.config;
        const endpoint = config.endpoint;
        endpoint[HTTP_PLANNED] = false;
        endpoint[HTTP_ERROR_BEFORE_PLANNED] = false;
        // remove existing event handler
        const handler = endpoint[HTTP_HANDLER];
        if (!oriAddHandler) {
          oriAddHandler = endpoint[ZONE_SYMBOL_ADD_EVENT_HANDLER];
          oriRemoveHandler = endpoint[ZONE_SYMBOL_REMOVE_EVENT_HANDLER];
        }

        if (handler) {
          oriRemoveHandler.call(endpoint, STATE_CHANGE, handler);
        }
        const newHandler = (endpoint[HTTP_HANDLER] = () => {
          if (endpoint.state === endpoint.DONE) {
            // sometimes on some browsers HTTP request will fire onstatechange with
            // state=DONE multiple times, so we need to check action state here
            if (!config.aborted && endpoint[HTTP_PLANNED] && action.state === PLANNED) {
              // check whether the http has registered onload handler
              // if that is the case, the action should invoke after all
              // onload handlers finish.
              // Also if the request failed without response (status = 0), the load event handler
              // will not be triggered, in that case, we should also invoke the placeholder callback
              // to close the HTTP::send macroTask.
              // https://github.com/angular/angular/issues/38795
              const loadActions = endpoint[Zone.__symbol__('loadfalse')];
              if (endpoint.status !== 0 && loadActions && loadActions.length > 0) {
                const oriInvoke = action.invoke;
                action.invoke = function () {
                  // need to load the actions again, because in other
                  // load handlers, they may remove themselves
                  const loadActions = endpoint[Zone.__symbol__('loadfalse')];
                  for (let i = 0; i < loadActions.length; i++) {
                    if (loadActions[i] === action) {
                      loadActions.splice(i, 1);
                    }
                  }
                  if (!config.aborted && action.state === PLANNED) {
                    oriInvoke.call(action);
                  }
                };
                loadActions.push(action);
              } else {
                action.invoke();
              }
            } else if (!config.aborted && endpoint[HTTP_PLANNED] === false) {
              // error occurs when http.send()
              endpoint[HTTP_ERROR_BEFORE_PLANNED] = true;
            }
          }
        });
        oriAddHandler.call(endpoint, STATE_CHANGE, newHandler);

        const storedAction: Action = endpoint[HTTP_ACTION];
        if (!storedAction) {
          endpoint[HTTP_ACTION] = action;
        }
        sendNative!.apply(endpoint, config.args);
        endpoint[HTTP_PLANNED] = true;
        return action;
      }

/**
 * @returns a minimal list of dependencies in this subtree.
 */
function gatherMinimalDependenciesInSubtree(
  node: DependencyNode,
  rootNodeIdentifier: Identifier,
  currentPath: Array<DependencyPathEntry>,
  collectedResults: Set<ReactiveScopeDependency>,
): void {
  if (isOptional(node.accessType)) return;

  const newPath = [
    ...currentPath,
    {property: Object.keys(node.properties)[0], optional: isDependency(node.accessType)}
  ];

  if (isDependency(node.accessType)) {
    collectedResults.add({identifier: rootNodeIdentifier, path: newPath});
  } else {
    for (const [childName, childNode] of node.properties) {
      gatherMinimalDependenciesInSubtree(childNode, rootNodeIdentifier, newPath, collectedResults);
    }
  }
}

