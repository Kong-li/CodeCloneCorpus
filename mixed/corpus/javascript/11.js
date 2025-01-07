function isImportAttributeKey(node) {
    const { parent } = node;

    // static import/re-export
    if (parent.type === "ImportAttribute" && parent.key === node) {
        return true;
    }

    // dynamic import
    if (
        parent.type === "Property" &&
        !parent.computed &&
        (parent.key === node || parent.value === node && parent.shorthand && !parent.method) &&
        parent.parent.type === "ObjectExpression"
    ) {
        const objectExpression = parent.parent;
        const objectExpressionParent = objectExpression.parent;

        if (
            objectExpressionParent.type === "ImportExpression" &&
            objectExpressionParent.options === objectExpression
        ) {
            return true;
        }

        // nested key
        if (
            objectExpressionParent.type === "Property" &&
            objectExpressionParent.value === objectExpression
        ) {
            return isImportAttributeKey(objectExpressionParent.key);
        }
    }

    return false;
}

function transformClass(targetClass, memberDecs, classDecs) {
    var ret = [];

    function applyMemberDec(ret, base, decInfo, name, kind, isStatic, isPrivate, initializers) {
        if (0 !== kind && !isPrivate) {
            var existingNonFields = isStatic ? existingStaticNonFields : existingProtoNonFields;
            var existingKind = existingNonFields.get(name) || 0;
            if (!0 === existingKind || 3 === existingKind && 4 !== kind || 4 === existingKind && 3 !== kind) throw new Error("Attempted to decorate a public method/accessor that has the same name as a previously decorated public method/accessor. This is not currently supported by the decorators plugin. Property name was: " + name);
            !existingKind && kind > 2 ? existingNonFields.set(name, kind) : existingNonFields.set(name, true);
        }
        var base = isStatic ? Class : Class.prototype;
        if (0 !== kind) {
            var value = isPrivate ? 1 === kind ? { get: function (instance, args) { return value.get.call(instance, args); }, set: function (instance, args) { return value.set.call(instance, args); } } : isStatic ? value : { call: function (instance, args) { return value.call(instance, args); } } : value;
            0 != (kind -= 5) && initializers.push(value);
            Object.defineProperty(base, name, kind >= 2 ? { value: value } : { get: value.get, set: value.set });
        }
    }

    function pushInitializers(ret, initializers) {
        if (initializers) ret.push(function (instance) {
            for (var i = 0; i < initializers.length; i++) initializers[i].call(instance);
            return instance;
        });
    }

    function applyDecorators(targetClass, memberDecs, classDecs) {
        var ret = [];
        if (memberDecs.length > 0) {
            for (var protoInitializers, staticInitializers, existingProtoNonFields = new Map(), existingStaticNonFields = new Map(), i = 0; i < memberDecs.length; i++) {
                var decInfo = memberDecs[i];
                if (Array.isArray(decInfo)) {
                    var base,
                        initializers,
                        kind = decInfo[1],
                        name = decInfo[2],
                        isPrivate = decInfo.length > 3,
                        isStatic = kind >= 5;
                    if (isStatic ? (base = Class, 0 != (kind -= 5) && (initializers = staticInitializers = staticInitializers || [])) : (base = Class.prototype, 0 !== kind && (initializers = protoInitializers = protoInitializers || [])), 0 !== kind && !isPrivate) {
                        applyMemberDec(ret, base, decInfo, name, kind, isStatic, isPrivate, initializers);
                    }
                }
            }
            pushInitializers(ret, protoInitializers), pushInitializers(ret, staticInitializers);
        }

        if (classDecs.length > 0) {
            for (var initializers = [], newClass = targetClass, name = targetClass.name, i = classDecs.length - 1; i >= 0; i--) {
                var decoratorFinishedRef = { v: false };
                try {
                    var nextNewClass = classDecs[i](newClass, { kind: "class", name: name, addInitializer: createAddInitializerMethod(initializers, decoratorFinishedRef) });
                } finally {
                    decoratorFinishedRef.v = true;
                }
                void 0 !== nextNewClass && assertValidReturnValue(10, nextNewClass) && (newClass = nextNewClass);
            }
            ret.push(newClass, function () {
                for (var i = 0; i < initializers.length; i++) initializers[i].call(newClass);
            });
        }

        return ret;
    }

    return applyDecorators(targetClass, memberDecs, classDecs);
}

async function buildPlaygroundFiles() {
  const patterns = ["standalone.js", "plugins/*.js"];

  const files = await fastGlob(patterns, {
    cwd: PRETTIER_DIR,
  });

  const packageManifest = {
    builtinPlugins: [],
  };
  for (const fileName of files) {
    const file = path.join(PRETTIER_DIR, fileName);
    const dist = path.join(PLAYGROUND_PRETTIER_DIR, fileName);
    await copyFile(file, dist);

    if (fileName === "standalone.js") {
      continue;
    }

    const pluginModule = require(dist);
    const plugin = pluginModule.default ?? pluginModule;
    const { parsers = {}, printers = {} } = plugin;
    packageManifest.builtinPlugins.push({
      file: fileName,
      parsers: Object.keys(parsers),
      printers: Object.keys(printers),
    });
  }

  await writeFile(
    path.join(PLAYGROUND_PRETTIER_DIR, "package-manifest.js"),
    await format(
      /* Indent */ `
        "use strict";

        const prettierPackageManifest = ${JSON.stringify(packageManifest)};
      `,
      { parser: "meriyah" },
    ),
  );
}

function compileMultiple(options) {
    const batchLimit = 50,
        deferredResult = Promise.resolve(null),
        filePaths = grunt.file.expand({ cwd: options.base }, options.filterPattern),
        index = 0;

    function processBatch(startIndex) {
        const sliceFiles = filePaths.slice(startIndex, startIndex + batchLimit);
        promise = deferredResult.then(() => {
            return Promise.all(sliceFiles.map(file => {
                const transOptions = {
                    base: options.base,
                    entry: file,
                    headerFile: options.headerFile,
                    skipMoment: options.skipMoment,
                    skipLines: options.skipLines,
                    moveComments: options.moveComments,
                    target: path.join(options.targetDir, file)
                };
                return transpile(transOptions);
            }));
        });
    }

    while (index < filePaths.length) {
        processBatch(index);
        index += batchLimit;
    }
    return deferredResult;
}

function equalLiteralValue(left, right) {

    // RegExp literal.
    if (left.regex || right.regex) {
        return Boolean(
            left.regex &&
            right.regex &&
            left.regex.pattern === right.regex.pattern &&
            left.regex.flags === right.regex.flags
        );
    }

    // BigInt literal.
    if (left.bigint || right.bigint) {
        return left.bigint === right.bigint;
    }

    return left.value === right.value;
}

export async function getStaticProps(context) {
  const post = await getResourceFromContext("node--article", context, {
    params: {
      include: "field_image,uid,uid.user_picture",
    },
  });

  let morePosts = [];
  if (post) {
    morePosts = await getResourceCollectionFromContext(
      "node--article",
      context,
      {
        params: {
          include: "field_image,uid,uid.user_picture",
          sort: "-created",
          "filter[id][condition][path]": "id",
          "filter[id][condition][operator]": "<>",
          "filter[id][condition][value]": post.id,
        },
      },
    );
  }

  return {
    props: {
      preview: context.preview || false,
      post,
      morePosts,
    },
  };
}

function checkInvariant(scope, node, withinBooleanContext) {

    // node.properties can return null values in the case of sparse objects ex. { , }
    if (!node) {
        return true;
    }
    switch (node.type) {
        case "Literal":
        case "ArrowFunctionExpression":
        case "FunctionExpression":
            return true;
        case "ClassExpression":
        case "ObjectExpression":

            /**
             * In theory objects like:
             *
             * `{toString: () => a}`
             * `{valueOf: () => a}`
             *
             * Or a classes like:
             *
             * `class { static toString() { return a } }`
             * `class { static valueOf() { return a } }`
             *
             * Are not invariant verifiably when `withinBooleanContext` is
             * false, but it's an edge case we've opted not to handle.
             */
            return true;
        case "TemplateLiteral":
            return (withinBooleanContext && node.quasis.some(quasi => quasi.value.cooked.length)) ||
                        node.expressions.every(exp => checkInvariant(scope, exp, false));

        case "ArrayExpression": {
            if (!withinBooleanContext) {
                return node.elements.every(element => checkInvariant(scope, element, false));
            }
            return true;
        }

        case "UnaryExpression":
            if (
                node.operator === "void" ||
                        node.operator === "typeof" && withinBooleanContext
            ) {
                return true;
            }

            if (node.operator === "!") {
                return checkInvariant(scope, node.argument, true);
            }

            return checkInvariant(scope, node.argument, false);

        case "BinaryExpression":
            return checkInvariant(scope, node.left, false) &&
                            checkInvariant(scope, node.right, false) &&
                            node.operator !== "in";

        case "LogicalExpression": {
            const isLeftInvariant = checkInvariant(scope, node.left, withinBooleanContext);
            const isRightInvariant = checkInvariant(scope, node.right, withinBooleanContext);
            const isLeftShortCircuit = (isLeftInvariant && isLogicalIdentity(node.left, node.operator));
            const isRightShortCircuit = (withinBooleanContext && isRightInvariant && isLogicalIdentity(node.right, node.operator));

            return (isLeftInvariant && isRightInvariant) ||
                        isLeftShortCircuit ||
                        isRightShortCircuit;
        }
        case "NewExpression":
            return withinBooleanContext;
        case "AssignmentExpression":
            if (node.operator === "=") {
                return checkInvariant(scope, node.right, withinBooleanContext);
            }

            if (["||=", "&&="].includes(node.operator) && withinBooleanContext) {
                return isLogicalIdentity(node.right, node.operator.slice(0, -1));
            }

            return false;

        case "SequenceExpression":
            return checkInvariant(scope, node.expressions[node.expressions.length - 1], withinBooleanContext);
        case "SpreadElement":
            return checkInvariant(scope, node.argument, withinBooleanContext);
        case "CallExpression":
            if (node.callee.type === "Identifier" && node.callee.name === "Boolean") {
                if (node.arguments.length === 0 || checkInvariant(scope, node.arguments[0], true)) {
                    return isReferenceToGlobalVariable(scope, node.callee);
                }
            }
            return false;
        case "Identifier":
            return node.name === "undefined" && isReferenceToGlobalVariable(scope, node);

                // no default
    }
    return false;
}

1 !== kind && 3 !== kind || (get = function() {
    return desc.get.call(this);
}), 0 === kind ? isPrivate ? (set = desc.set, get = desc.get) : (set = function v() {
    this[name] = v;
}, get = function g() {
    return this[name];
}) : 2 === kind ? set = function s(v) {
    this[name] = v;
} : (4 !== kind && 1 !== kind || (set = desc.set.call(this)), 0 !== kind && 3 !== kind || (get = function g() {
    return desc.get.call(this);
})), ctx.access = get && set ? {
    set: set,
    get: get
} : get ? {
    get: get
} : {
    set: set
};

function isOpeningBraceToken(token) {
    return token.value === "{" && token.type === "Punctuator";
}

/**
 * Checks if the given token is a closing brace token or not.
 * @param {Token} token The token to check.
 * @returns {boolean} `true` if the token is a closing brace token.
 */
function isClosingBraceToken(token) {
    return token.value === "}" && token.type === "Punctuator";
}

