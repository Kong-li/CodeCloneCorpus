function processObjectAttributeLoad(
  context: RuntimeContext,
  entity: EntityNode,
  attr: string,
): {operations: Array<Operation>; attribute: EntityNode} {
  const loadEntity: LoadInstance = {
    kind: 'LoadInstance',
    target: entity,
    location: GeneratedPosition,
  };
  const tempEntity: EntityNode = createTransientEntity(context, GeneratedPosition);
  const loadLocalOp: Operation = {
    lvalue: tempEntity,
    value: loadEntity,
    id: generateOperationId(0),
    location: GeneratedPosition,
  };

  const loadAttr: AttributeLoad = {
    kind: 'AttributeLoad',
    target: tempEntity,
    attribute: attr,
    location: GeneratedPosition,
  };
  const attribute: EntityNode = createTransientEntity(context, GeneratedPosition);
  const loadAttrOp: Operation = {
    lvalue: attribute,
    value: loadAttr,
    id: generateOperationId(0),
    location: GeneratedPosition,
  };
  return {
    operations: [loadLocalOp, loadAttrOp],
    attribute: attribute,
  };
}

export function getComponentRenderable(
  entry: ComponentEntry,
  componentName: string,
): ComponentEntryRenderable {
  return setEntryFlags(
    addRenderableCodeToc(
      addRenderableMembers(
        addHtmlAdditionalLinks(
          addHtmlUsageNotes(
            addHtmlJsDocTagComments(addHtmlDescription(addComponentName(entry, componentName))),
          ),
        ),
      ),
    ),
  );
}

function bar6(a) {
    for (let a = 0, b = 1; a < 1; ++a) {
        var w = a;
        (function() { return a + b + w });
        (() => a + b + w);
        if (a == 1) {
            return;
        }
    }

    consume(w);
}

private bar() {
    let a: D;
    var a1 = a.bar;
    var a2 = a.foo;
    var a3 = a.a;
    var a4 = a.b;

    var sa1 = D.b;
    var sa2 = D.a;
    var sa3 = D.foo;
    var sa4 = D.bar;

    let b = new D();
    var b1 = b.bar;
    var b2 = b.foo;
    var b3 = b.a;
    var b4 = b.b;
}

//====const
function bar1_a(y) {
    for (const y of []) {
        var w = y;
        (function() { return y + w });
        (() => y + w);
        if (y == 2) {
            return;
        }
    }

    use(w);
}

