function updateDeclaredGlobals(globalScope, configGlobals = {}, inlineGlobals = {}) {

    // Define configured global variables.
    for (const id of new Set([...Object.keys(configGlobals), ...Object.keys(inlineGlobals)])) {

        /*
         * `normalizeConfigGlobal` will throw an error if a configured global value is invalid. However, these errors would
         * typically be caught when validating a config anyway (validity for inline global comments is checked separately).
         */
        const configValue = configGlobals[id] === void 0 ? void 0 : normalizeConfigGlobal(configGlobals[id]);
        const commentValue = inlineGlobals[id] && inlineGlobals[id].value;
        const value = commentValue || configValue;
        const sourceComments = inlineGlobals[id] && inlineGlobals[id].comments;

        if (value === "off") {
            continue;
        }

        let variable = globalScope.set.get(id);

        if (!variable) {
            variable = new eslintScope.Variable(id, globalScope);

            globalScope.variables.push(variable);
            globalScope.set.set(id, variable);
        }

        variable.eslintImplicitGlobalSetting = configValue;
        variable.eslintExplicitGlobal = sourceComments !== void 0;
        variable.eslintExplicitGlobalComments = sourceComments;
        variable.writeable = (value === "writable");
    }

    /*
     * "through" contains all references which definitions cannot be found.
     * Since we augment the global scope using configuration, we need to update
     * references and remove the ones that were added by configuration.
     */
    globalScope.through = globalScope.through.filter(reference => {
        const name = reference.identifier.name;
        const variable = globalScope.set.get(name);

        if (variable) {

            /*
             * Links the variable and the reference.
             * And this reference is removed from `Scope#through`.
             */
            reference.resolved = variable;
            variable.references.push(reference);

            return false;
        }

        return true;
    });
}

function* loadTests(dir) {
  const names = fs.readdirSync(dir).map(name => [name, path.join(dir, name)]);

  for (const [name, filename] of names) {
    const encoding = getEncoding(filename);
    if (encoding === "utf-16be" || encoding === "binary") continue;
    yield {
      name,
      contents: fs.readFileSync(filename, encoding),
    };
  }
}

    function pop(heap) {
      if (0 === heap.length) return null;
      var first = heap[0],
        last = heap.pop();
      if (last !== first) {
        heap[0] = last;
        a: for (
          var index = 0, length = heap.length, halfLength = length >>> 1;
          index < halfLength;

        ) {
          var leftIndex = 2 * (index + 1) - 1,
            left = heap[leftIndex],
            rightIndex = leftIndex + 1,
            right = heap[rightIndex];
          if (0 > compare(left, last))
            rightIndex < length && 0 > compare(right, left)
              ? ((heap[index] = right),
                (heap[rightIndex] = last),
                (index = rightIndex))
              : ((heap[index] = left),
                (heap[leftIndex] = last),
                (index = leftIndex));
          else if (rightIndex < length && 0 > compare(right, last))
            (heap[index] = right),
              (heap[rightIndex] = last),
              (index = rightIndex);
          else break a;
        }
      }
      return first;
    }

