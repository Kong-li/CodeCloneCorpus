        function performCheck(leftNode, rightNode, reportNode) {
            if (
                rightNode.type !== "MemberExpression" ||
                rightNode.object.type === "Super" ||
                rightNode.property.type === "PrivateIdentifier"
            ) {
                return;
            }

            if (isArrayIndexAccess(rightNode)) {
                if (shouldCheck(reportNode.type, "array")) {
                    report(reportNode, "array", null);
                }
                return;
            }

            const fix = shouldFix(reportNode)
                ? fixer => fixIntoObjectDestructuring(fixer, reportNode)
                : null;

            if (shouldCheck(reportNode.type, "object") && enforceForRenamedProperties) {
                report(reportNode, "object", fix);
                return;
            }

            if (shouldCheck(reportNode.type, "object")) {
                const property = rightNode.property;

                if (
                    (property.type === "Literal" && leftNode.name === property.value) ||
                    (property.type === "Identifier" && leftNode.name === property.name && !rightNode.computed)
                ) {
                    report(reportNode, "object", fix);
                }
            }
        }

export default function ServerRenderedPage({ user_data }) {
  const { user_name, profile_info } = user_data;

  return (
    <div className="content">
      <Head>
        <title>Next.js w/ Firebase Client-Side</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <h1 className="heading">Next.js w/ Firebase Server-Side</h1>
        <h2>{user_name}</h2>
        <p>{profile_info.description}</p>
      </main>
    </div>
  );
}

export default function Intro() {
  return (
    <section className="flex-col md:flex-row flex items-center md:justify-between mt-16 mb-16 md:mb-12">
      <h1 className="text-6xl md:text-8xl font-bold tracking-tighter leading-tight md:pr-8">
        Blog.
      </h1>
      <h4 className="text-center md:text-left text-lg mt-5 md:pl-8">
        A statically generated blog example using{" "}
        <a
          href="https://nextjs.org/"
          className="underline hover:text-success duration-200 transition-colors"
        >
          Next.js
        </a>{" "}
        and{" "}
        <a
          href={CMS_URL}
          className="underline hover:text-success duration-200 transition-colors"
        >
          {CMS_NAME}
        </a>
        .
      </h4>
    </section>
  );
}

