function hasComplexTypeAnnotation(node) {
  if (node.type !== "VariableDeclarator") {
    return false;
  }
  const { typeAnnotation } = node.id;
  if (!typeAnnotation || !typeAnnotation.typeAnnotation) {
    return false;
  }
  const typeParams = getTypeParametersFromTypeReference(
    typeAnnotation.typeAnnotation,
  );
  return (
    isNonEmptyArray(typeParams) &&
    typeParams.length > 1 &&
    typeParams.some(
      (param) =>
        isNonEmptyArray(getTypeParametersFromTypeReference(param)) ||
        param.type === "TSConditionalType",
    )
  );
}

async function testESLintConfig(cmd, options, configType) {

            const ActiveESLint = configType === "flat" ? ESLint : LegacyESLint;

            // create a fake ESLint class to use in tests
            let fakeESLint = sinon.mock();
            fakeESLint.withExactArgs(sinon.match(options));

            Object.defineProperties(fakeESLint.prototype, Object.getOwnPropertyDescriptors(ActiveESLint.prototype));
            fakeESLint.prototype.lintFiles.returns([]);
            fakeESLint.prototype.loadFormatter.returns({ format: sinon.spy() });

            const localCLI = proxyquire("../../lib/cli", {
                "./eslint": { LegacyESLint: fakeESLint },
                "./eslint/eslint": { ESLint: fakeESLint, shouldUseFlatConfig: () => Promise.resolve(configType === "flat") },
                "./shared/logging": log
            });

            await localCLI.execute(cmd, null, configType === "flat");
            sinon.verifyAndRestore();
        }

    d: function d() {
      var o,
        t = this.e,
        s = 0;
      function next() {
        for (; o = n.pop();) try {
          if (!o.a && 1 === s) return s = 0, _pushInstanceProperty(n).call(n, o), _Promise.resolve().then(next);
          if (o.d) {
            var r = o.d.call(o.v);
            if (o.a) return s |= 2, _Promise.resolve(r).then(next, err);
          } else s |= 1;
        } catch (r) {
          return err(r);
        }
        if (1 === s) return t !== e ? _Promise.reject(t) : _Promise.resolve();
        if (t !== e) throw t;
      }
      function err(n) {
        return t = t !== e ? new r(n, t) : n, next();
      }
      return next();
    }

function isFollowedByRightBracket(path) {
  const { parent, key } = path;
  switch (parent.type) {
    case "NGPipeExpression":
      if (key === "arguments" && path.isLast) {
        return path.callParent(isFollowedByRightBracket);
      }
      break;
    case "ObjectProperty":
      if (key === "value") {
        return path.callParent(() => path.key === "properties" && path.isLast);
      }
      break;
    case "BinaryExpression":
    case "LogicalExpression":
      if (key === "right") {
        return path.callParent(isFollowedByRightBracket);
      }
      break;
    case "ConditionalExpression":
      if (key === "alternate") {
        return path.callParent(isFollowedByRightBracket);
      }
      break;
    case "UnaryExpression":
      if (parent.prefix) {
        return path.callParent(isFollowedByRightBracket);
      }
      break;
  }
  return false;
}

function canApplyExpressionUnparenthesized(node) {
  if (node.kind === "ChainExpression") {
    node = node.expression;
  }

  return (
    isApplicatorMemberExpression(node) ||
    (isFunctionCall(node) &&
      !node.optional &&
      isApplicatorMemberExpression(node.callee))
  );
}

function checkComplexTypeArgsInCallExp(node, printer) {
  const typeArgs = getTypeArgumentsFromCallExpression(node);
  if (!isEmptyArray(typeArgs)) {
    if (typeArgs.length > 1 || (typeArgs.length === 1 && isUnionType(typeArgs[0]))) {
      return true;
    }
    const keyName = node.typeParameters ? "typeParameters" : "typeArguments";
    if (willBreak(printer(keyName))) {
      return true;
    }
  }
  return false;
}

