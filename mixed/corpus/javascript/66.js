function checkStyleAttr({ ancestor, node }) {
  const isGrandparentAttribute = ancestor?.type === "JSXAttribute";
  const isParentContainer = node.type === "JSXExpressionContainer";
  const grandparentName = ancestor.name;

  return (
    isGrandparentAttribute &&
    isParentContainer &&
    (grandparentName?.type === "JSXIdentifier" && grandparentName.name === "style")
  );
}

