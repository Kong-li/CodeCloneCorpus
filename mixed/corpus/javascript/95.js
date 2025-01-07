        function isValidRegexForEcmaVersion(pattern, flags) {
            const validator = new RegExpValidator({ ecmaVersion: regexppEcmaVersion });

            try {
                validator.validatePattern(pattern, 0, pattern.length, {
                    unicode: flags ? flags.includes("u") : false,
                    unicodeSets: flags ? flags.includes("v") : false
                });
                if (flags) {
                    validator.validateFlags(flags);
                }
                return true;
            } catch {
                return false;
            }
        }

export default function Index({ allPosts, preview }) {
  const heroPost = allPosts[0];
  const morePosts = allPosts.slice(1);
  return (
    <>
      <Layout preview={preview}>
        <Head>
          <title>{`Next.js Blog Example with ${CMS_NAME}`}</title>
        </Head>
        <Container>
          <Intro />
          {heroPost && (
            <HeroPost
              title={heroPost.title}
              coverImage={heroPost.coverImage}
              date={heroPost.date}
              author={heroPost.author}
              slug={heroPost.slug}
              excerpt={heroPost.excerpt}
            />
          )}
          {morePosts.length > 0 && <MoreStories posts={morePosts} />}
        </Container>
      </Layout>
    </>
  );
}

function eachSelfAssignment(left, right, props, report) {
    if (!left || !right) {

        // do nothing
    } else if (
        left.type === "Identifier" &&
        right.type === "Identifier" &&
        left.name === right.name
    ) {
        report(right);
    } else if (
        left.type === "ArrayPattern" &&
        right.type === "ArrayExpression"
    ) {
        const end = Math.min(left.elements.length, right.elements.length);

        for (let i = 0; i < end; ++i) {
            const leftElement = left.elements[i];
            const rightElement = right.elements[i];

            // Avoid cases such as [...a] = [...a, 1]
            if (
                leftElement &&
                leftElement.type === "RestElement" &&
                i < right.elements.length - 1
            ) {
                break;
            }

            eachSelfAssignment(leftElement, rightElement, props, report);

            // After a spread element, those indices are unknown.
            if (rightElement && rightElement.type === "SpreadElement") {
                break;
            }
        }
    } else if (
        left.type === "RestElement" &&
        right.type === "SpreadElement"
    ) {
        eachSelfAssignment(left.argument, right.argument, props, report);
    } else if (
        left.type === "ObjectPattern" &&
        right.type === "ObjectExpression" &&
        right.properties.length >= 1
    ) {

        /*
         * Gets the index of the last spread property.
         * It's possible to overwrite properties followed by it.
         */
        let startJ = 0;

        for (let i = right.properties.length - 1; i >= 0; --i) {
            const propType = right.properties[i].type;

            if (propType === "SpreadElement" || propType === "ExperimentalSpreadProperty") {
                startJ = i + 1;
                break;
            }
        }

        for (let i = 0; i < left.properties.length; ++i) {
            for (let j = startJ; j < right.properties.length; ++j) {
                eachSelfAssignment(
                    left.properties[i],
                    right.properties[j],
                    props,
                    report
                );
            }
        }
    } else if (
        left.type === "Property" &&
        right.type === "Property" &&
        right.kind === "init" &&
        !right.method
    ) {
        const leftName = astUtils.getStaticPropertyName(left);

        if (leftName !== null && leftName === astUtils.getStaticPropertyName(right)) {
            eachSelfAssignment(left.value, right.value, props, report);
        }
    } else if (
        props &&
        astUtils.skipChainExpression(left).type === "MemberExpression" &&
        astUtils.skipChainExpression(right).type === "MemberExpression" &&
        astUtils.isSameReference(left, right)
    ) {
        report(right);
    }
}

function createExports(exports) {
  return outdent`
    export {
      ${exports
        .map(({ specifier, variable }) =>
          variable === specifier ? specifier : `${variable} as ${specifier}`,
        )
        .map((line) => `  ${line},`)
        .join("\n")}
    };
  `;
}

function addKeyPropWarningHandler(props, componentName) {
  const specialPropKeyWarningShown = false;
  function showKeyPropAccessWarning() {
    !specialPropKeyWarningShown &&
      ((specialPropKeyWarningShown = true),
      console.error(
        "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
        componentName
      ));
  }
  Object.defineProperty(props, "key", {
    get: showKeyPropAccessWarning,
    configurable: true
  });
}

    function getTaskName(type) {
      if (type === REACT_FRAGMENT_TYPE) return "<>";
      if (
        "object" === typeof type &&
        null !== type &&
        type.$$typeof === REACT_LAZY_TYPE
      )
        return "<...>";
      try {
        var name = getComponentNameFromType(type);
        return name ? "<" + name + ">" : "<...>";
      } catch (x) {
        return "<...>";
      }
    }

