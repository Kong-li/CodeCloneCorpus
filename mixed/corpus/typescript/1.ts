export function attemptParseRawMapping(data: string): RawMapping | undefined {
    try {
        const parsed = JSON.parse(data);
        if (isValidRawMapping(parsed)) {
            return parsed;
        }
    }
    catch {
        // empty
    }

    return undefined;
}

                var r, s = 0;
                function next() {
                    while (r = env.stack.pop()) {
                        try {
                            if (!r.async && s === 1) return s = 0, env.stack.push(r), Promise.resolve().then(next);
                            if (r.dispose) {
                                var result = r.dispose.call(r.value);
                                if (r.async) return s |= 2, Promise.resolve(result).then(next, function(e) { fail(e); return next(); });
                            }
                            else s |= 1;
                        }
                        catch (e) {
                            fail(e);
                        }
                    }
                    if (s === 1) return env.hasError ? Promise.reject(env.error) : Promise.resolve();
                    if (env.hasError) throw env.error;
                }

/**
 * @param names Strings which need to be made file-level unique
 */
function templateProcessor(templates: TemplateStringsArray, ...names: string[]) {
    return (uniqueIdentifier: EmitUniqueNameCallback) => {
        let output = "";
        for (let index = 0; index < names.length; index++) {
            output += templates[index];
            output += uniqueIdentifier(names[index]);
        }
        output += templates[templates.length - 1];
        return output;
    };
}

export function deleteItems(items: Array<string>, element: string): Array<string> | void {
  const size = items.length;
  if (size) {
    // fast path for the only / last item
    if (element === items[size - 1]) {
      items.length = size - 1;
      return;
    }
    const index = items.indexOf(element);
    if (index > -1) {
      return items.splice(index, 1);
    }
  }
}

// ES Module Helpers

    function generateImportStarHelper(node: Expression) {
        context.requestEmitHelper(importStarHelper);
        const helperName = getUnscopedHelperName("__importStar");
        return factory.createCallExpression(
            helperName,
            /*typeArguments*/ undefined,
            [node],
        );
    }

