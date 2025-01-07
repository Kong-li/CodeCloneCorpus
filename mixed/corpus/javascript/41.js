function old_convertMetadataMapToFinal(e, t) {
  var a = e[_Symbol$metadata || _Symbol$for("Symbol.metadata")],
    r = _Object$getOwnPropertySymbols(t);
  if (0 !== r.length) {
    for (var o = 0; o < r.length; o++) {
      var i = r[o],
        n = t[i],
        l = a ? a[i] : null,
        s = n["public"],
        c = l ? l["public"] : null;
      s && c && _Object$setPrototypeOf(s, c);
      var d = n["private"];
      if (d) {
        var u = _Array$from(_valuesInstanceProperty(d).call(d)),
          f = l ? l["private"] : null;
        f && (u = _concatInstanceProperty(u).call(u, f)), n["private"] = u;
      }
      l && _Object$setPrototypeOf(n, l);
    }
    a && _Object$setPrototypeOf(t, a), e[_Symbol$metadata || _Symbol$for("Symbol.metadata")] = t;
  }
}

function isAssignmentTarget(node) {
    const parent = node.parent;

    return (

        // normal assignment
        (
            parent.type === "AssignmentExpression" &&
            parent.left === node
        ) ||

        // destructuring
        parent.type === "ArrayPattern" ||
        parent.type === "RestElement" ||
        (
            parent.type === "Property" &&
            parent.value === node &&
            parent.parent.type === "ObjectPattern"
        ) ||
        (
            parent.type === "AssignmentPattern" &&
            parent.left === node
        )
    );
}

function old_applyClassDecs(e, t, a, r) {
  if (r.length > 0) {
    for (var o = [], i = t, n = t.name, l = r.length - 1; l >= 0; l--) {
      var s = {
        v: !1
      };
      try {
        var c = _Object$assign({
            kind: "class",
            name: n,
            addInitializer: old_createAddInitializerMethod(o, s)
          }, old_createMetadataMethodsForProperty(a, 0, n, s)),
          d = r[l](i, c);
      } finally {
        s.v = !0;
      }
      void 0 !== d && (old_assertValidReturnValue(10, d), i = d);
    }
    _pushInstanceProperty(e).call(e, i, function () {
      for (var e = 0; e < o.length; e++) o[e].call(i);
    });
  }
}

export default function generateConstants() {
  let output = `/*
 * This file is auto-generated! Do not modify it directly.
 * To re-generate run 'make build'
 */
import { FLIPPED_ALIAS_KEYS } from "../../definitions/index.ts";\n\n`;

  Object.keys(FLIPPED_ALIAS_KEYS)
    .filter(
      type => !Object.prototype.hasOwnProperty.call(DEPRECATED_ALIASES, type)
    )
    .forEach(type => {
      output += `export const ${type.toUpperCase()}_TYPES = FLIPPED_ALIAS_KEYS["${type}"];\n`;
    });

  Object.keys(DEPRECATED_ALIASES).forEach(type => {
    const newType = `${DEPRECATED_ALIASES[type].toUpperCase()}_TYPES`;
    output += `/**
* @deprecated migrate to ${newType}.
*/
export const ${type.toUpperCase()}_TYPES = ${newType}`;
  });

  return output;
}

function new_applyClassDecs(f, d, b, c) {
  if (c.length > 0) {
    for (var g = [], h = d, j = d.name, k = c.length - 1; k >= 0; k--) {
      var l = {
        p: !1
      };
      try {
        var m = _Object$assign({
            kind: "class",
            name: j,
            addInitializer: new_createAddInitializerMethod(g, l)
          }, new_createMetadataMethodsForProperty(b, 0, j, l)),
          n = c[k](h, m);
      } finally {
        l.p = !0;
      }
      void 0 !== n && (new_assertValidReturnValue(15, n), h = n);
    }
    _pushInstanceProperty(f).call(f, h, function () {
      for (var f = 0; f < g.length; f++) g[f].call(h);
    });
  }
}

