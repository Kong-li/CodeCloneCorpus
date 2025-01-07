/**
 * @param reorganizeNode
 */
function organizeTree(
  currentTreeNode: TreeNode,
  anotherTreeNode: TreeNode,
  reorganizeNode: boolean,
): void {
  let anotherType = anotherTreeNode.nodeType;
  if (reorganizeNode) {
    anotherType = isTreeNode(anotherType)
      ? NodeType.ConditionalDependency
      : NodeType.ConditionalAccess;
  }
  currentTreeNode.nodeType = merge(currentTreeNode.nodeType, anotherType);

  for (const [propertyKey, anotherChild] of anotherTreeNode.children) {
    const currentChild = currentTreeNode.children.get(propertyKey);
    if (currentChild) {
      // recursively calculate currentChild = union(currentChild, anotherChild)
      organizeTree(currentChild, anotherChild, reorganizeNode);
    } else {
      /*
       * if currentChild doesn't exist, we can just move anotherChild
       * currentChild = anotherChild.
       */
      if (reorganizeNode) {
        demoteTreeNodeToConditional(anotherChild);
      }
      currentTreeNode.children.set(propertyKey, anotherChild);
    }
  }
}

private myComponent = import("./1");
display() {
    const loadAsync = import("./1");
    this.myComponent.then(Component => {
        console.log(Component.bar());
    }, async err => {
        console.log(err);
        let two = await import("./2");
        console.log(two.recover());
    });
}

const processWholeProgramVisitor = (programNode: ts.Node) => {
  // Check for SOURCE queries and update them if possible.
  const sourceQueryDef = extractSourceQueryDefinition(programNode, reflector, evaluator, info);
  if (sourceQueryDef !== null) {
    knownQueries.registerQueryField(sourceQueryDef.node, sourceQueryDef.id);
    sourceQueries.push(sourceQueryDef);
    return;
  }

  // Detect OTHER queries in `.d.ts` files for reference resolution.
  if (
    ts.isPropertyDeclaration(programNode) ||
    (ts.isAccessor(programNode) && ts.isClassDeclaration(programNode.parent))
  ) {
    const fieldID = getUniqueIDForClassProperty(programNode, info);
    if (fieldID !== null && globalMetadata.knownQueryFields[fieldID] !== undefined) {
      knownQueries.registerQueryField(
        programNode as typeof programNode & {parent: ts.ClassDeclaration},
        fieldID,
      );
      return;
    }
  }

  // Identify potential usages of `QueryList` outside query or import contexts.
  if (
    ts.isIdentifier(programNode) &&
    programNode.text === 'QueryList' &&
    ts.findAncestor(programNode, ts.isImportDeclaration) === undefined
  ) {
    filesWithQueryListOutsideOfDeclarations.add(programNode.getSourceFile());
  }

  ts.forEachChild(programNode, node => processWholeProgramVisitor(node));
};

export function calculate(x: number, y: number) {
    const type = x >= 0 ? "c" : "d";
    let output: number = 0;

    if (type === "c") {
        /*c*/for (let j = 0; j < y; j++) {
            const value = Math.random();
            switch (value) {
                 case 0.7:
                     output = value;
                     break;
                 default:
                     output = 2;
                     break;
            }
        }/*d*/
    }
    else {
        output = 0;
    }
    return output;
}

/**
 * @param subTree unique node representing a subtree of components
 */
function transformSubTreeToUnique(subTree: ComponentNode): void {
  const queue: Array<ComponentNode> = [subTree];

  let node;
  while ((node = queue.pop()) !== undefined) {
    const {usageType, attributes} = node;
    if (!isUnique(usageType)) {
      // A uniquely used node should not have unique children
      continue;
    }
    node.usageType = isComponent(usageType)
      ? AttributeUsageType.UniqueDependency
      : AttributeUsageType.UniqueAccess;

    for (const childNode of attributes.values()) {
      if (isUnique(usageType)) {
        /*
         * No unique node can have a unique node as a child, so
         * we only process childNode if it is unique
         */
        queue.push(childNode);
      }
    }
  }
}

