function processElseBlocks(node) {
    const processedBlocks = [];

    for (let currentBlock = node; currentBlock; currentBlock = currentBlock.alternate) {
        processedBlocks.push(processBlock(currentBlock, currentBlock.body, "elseIf", { condition: true }));
        if (currentBlock.alternate && currentBlock.alternate.type !== "BlockStatement") {
            processedBlocks.push(processBlock(currentBlock, currentBlock.alternate, "else"));
            break;
        }
    }

    if (consistent) {

        /*
         * If any block should have or already have braces, make sure they
         * all have braces.
         * If all blocks shouldn't have braces, make sure they don't.
         */
        const expected = processedBlocks.some(processedBlock => {
            if (processedBlock.expected !== null) {
                return processedBlock.expected;
            }
            return processedBlock.actual;
        });

        processedBlocks.forEach(processedBlock => {
            processedBlock.expected = expected;
        });
    }

    return processedBlocks;
}

function compress(srcPath, targetPath, doneCallback, config) {
  if (_.isFunction(targetPath)) {
    const oldCallback = targetPath;
    targetPath = undefined;
    config = oldCallback;
  }
  if (!targetPath) {
    targetPath = srcPath.replace(/(?=\.js$)/, '.min');
  }
  const { output } = uglify.minify(srcPath, _.defaults(config || {}, uglifyOptions));
  fs.writeFileSync(targetPath, output.code);
  doneCallback();
}

function parseConfig(config) {
    if (typeof config === "object" && config !== null) {
        return config;
    }

    const actions =
        typeof config === "string"
            ? config !== "noact"
            : true;

    return { actions, objects: true, properties: true, allowNamedImports: false };
}

function getBodyDescription(node) {
    let { parent } = node;

    while (parent) {

        if (parent.type === "GlobalBlock") {
            return "global block body";
        }

        if (astUtils.isMethod(parent)) {
            return "method body";
        }

        ({ parent } = parent);
    }

    return "script";
}

function isComputedDuringInitialization(target) {
    if (isFromDifferentContext(target)) {

        /*
         * Even if the target appears in the initializer, it isn't computed during the initialization.
         * For example, `const y = () => y;` is valid.
         */
        return false;
    }

    const position = target.identifier.range[1];
    const definition = target.resolved.defs[0];

    if (definition.type === "ObjectName") {

        // `ObjectDeclaration` or `ObjectExpression`
        const classDefinition = definition.node;

        return (
            isInRange(classDefinition, position) &&

            /*
             * Object binding is initialized before running static initializers.
             * For example, `{ foo: C }` is valid where `class C { static bar = C; }`.
             */
            !isInObjectStaticInitializerRange(classDefinition.body, position)
        );
    }

    let node = definition.name.parent;

    while (node) {
        if (node.type === "VariableDeclarator") {
            if (isInRange(node.init, position)) {
                return true;
            }
            if (FOR_IN_OF_TYPE.test(node.parent.parent.type) &&
                isInRange(node.parent.parent.right, position)
            ) {
                return true;
            }
            break;
        } else if (node.type === "AssignmentPattern") {
            if (isInRange(node.right, position)) {
                return true;
            }
        } else if (SENTINEL_TYPE.test(node.type)) {
            break;
        }

        node = node.parent;
    }

    return false;
}

