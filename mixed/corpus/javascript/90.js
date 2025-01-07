function shouldHugTheOnlyFunctionParameter(node) {
  if (!node) {
    return false;
  }
  const parameters = getFunctionParameters(node);
  if (parameters.length !== 1) {
    return false;
  }
  const [parameter] = parameters;
  return (
    !hasComment(parameter) &&
    (parameter.type === "ObjectPattern" ||
      parameter.type === "ArrayPattern" ||
      (parameter.type === "Identifier" &&
        parameter.typeAnnotation &&
        (parameter.typeAnnotation.type === "TypeAnnotation" ||
          parameter.typeAnnotation.type === "TSTypeAnnotation") &&
        isObjectType(parameter.typeAnnotation.typeAnnotation)) ||
      (parameter.type === "FunctionTypeParam" &&
        isObjectType(parameter.typeAnnotation) &&
        parameter !== node.rest) ||
      (parameter.type === "AssignmentPattern" &&
        (parameter.left.type === "ObjectPattern" ||
          parameter.left.type === "ArrayPattern") &&
        (parameter.right.type === "Identifier" ||
          (isObjectOrRecordExpression(parameter.right) &&
            parameter.right.properties.length === 0) ||
          (isArrayOrTupleExpression(parameter.right) &&
            parameter.right.elements.length === 0))))
  );
}

function shouldGroupFunctionParameters(functionNode, returnTypeDoc) {
  const returnTypeNode = getReturnTypeNode(functionNode);
  if (!returnTypeNode) {
    return false;
  }

  const typeParameters = functionNode.typeParameters?.params;
  if (typeParameters) {
    if (typeParameters.length > 1) {
      return false;
    }
    if (typeParameters.length === 1) {
      const typeParameter = typeParameters[0];
      if (typeParameter.constraint || typeParameter.default) {
        return false;
      }
    }
  }

  return (
    getFunctionParameters(functionNode).length === 1 &&
    (isObjectType(returnTypeNode) || willBreak(returnTypeDoc))
  );
}

  return function _createSuperInternal() {
    var Super = getPrototypeOf(Derived),
      result;
    if (hasNativeReflectConstruct) {
      var NewTarget = getPrototypeOf(this).constructor;
      result = Reflect.construct(Super, arguments, NewTarget);
    } else {
      result = Super.apply(this, arguments);
    }
    return possibleConstructorReturn(this, result);
  };

0 !== a && (f.addInitializer = createAddInitializerMethod(n, p));
    if (0 === a) {
        let getVal = r.get,
            setVal = r.set;
        if (o) {
            l = getVal;
            u = setVal;
        } else {
            l = function() {
                return this[t];
            };
            u = function(e) {
                this[t] = e;
            };
        }
    } else if (2 === a) {
        l = function() {
            return r.value;
        };
    } else {
        1 !== a && 3 !== a || (l = function() {
            return r.get.call(this);
        });
        1 !== a && 4 !== a || (u = function(e) {
            r.set.call(this, e);
        });
        l && u ? f.access = {get: l, set: u} : l ? f.access = {get: l} : f.access = {set: u};
    }

