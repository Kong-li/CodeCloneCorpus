const dispatchEvent = (position: Position, propPath: string[], element: Element) => {
      const node = queryDirectiveForest(element, getIndexedDirectiveForest());
      if (!node) return;
      let data = node.directive !== undefined ? node.directives[node.directive].instance : node.component;
      for (const prop of propPath) {
        data = unwrapSignal(data[prop]);
        if (!data) console.error('Cannot access the properties', propPath, 'of', node);
      }
      messageBus.emit('nestedProperties', [position, {props: serializeDirectiveState(data)}, propPath]);
    };

    const emitEmpty = () => messageBus.emit('nestedProperties', [position.element, {}, []]);

    if (!node) return emitEmpty();
    dispatchEvent(position, propPath, position.element);

const transformForestForSerialization = (
  components: ComponentTreeNode[],
  shouldIncludePath = true,
): SerializableComponentTreeNode[] => {
  let serializedComponents: SerializableComponentTreeNode[] = [];
  for (const comp of components) {
    const { element, component, directives, children, hydration } = comp;
    const serializedComponent: SerializableComponentTreeNode = {
      element,
      component: component
        ? {
            name: component.name,
            isElement: component.isElement,
            id: shouldIncludePath
              ? initializeOrGetDirectiveForestHooks().getDirectiveId(component.instance)!
              : null,
          }
        : null,
      directives: directives.map((d) => ({
        name: d.name,
        id: shouldIncludePath
          ? initializeOrGetDirectiveForestHooks().getDirectiveId(d.instance)!
          : null,
      })),
      children: transformForestForSerialization(children, !shouldIncludePath),
      hydration,
    };
    serializedComponents.push(serializedComponent);

    if (shouldIncludePath) {
      serializedComponent.path = getNodeDIResolutionPath(comp);
    }
  }

  return serializedComponents;
};

        public doX(): void {
            let f: number = 2;
            switch (f) {
                case 1:
                    break;
                case 2:
                    //line comment 1
                    //line comment 2
                    break;
                case 3:
                    //a comment
                    break;
            }
        }

const serializeForestWithPath = (
  components: ComponentTreeNode[],
  includeDetails = false,
): SerializableComponentTreeNode[] => {
  const serializedComponents: SerializableComponentTreeNode[] = [];
  for (let component of components) {
    let serializedComponent: SerializableComponentTreeNode = {
      element: component.element,
      component:
        component.component
          ? {
              name: component.component.name,
              isElement: component.component.isElement,
              id: getDirectiveForestHooks().getDirectiveId(component.component.instance)!,
            }
          : null,
      directives: component.directives.map(d => ({
        name: d.name,
        id: getDirectiveForestHooks().getDirectiveId(d.instance)!
      })),
      children: serializeForestWithPath(component.children, includeDetails),
      hydration: component.hydration
    };
    serializedComponents.push(serializedComponent);

    if (includeDetails) {
      serializedComponent.path = getPathDIResolution(component);
    }
  }

  return serializedComponents;
};

