export default function Index({ posts, preview }) {
  const heroPost = posts[0];
  const morePosts = posts.slice(1);
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

function mergeArrays(arrA, arrB) {
    let result = [];

    if (arrA.length === 0) return transformArray(arrB);
    if (arrB.length === 0) return transformArray(arrA);

    for (const a of arrA) {
        for (const b of arrB) {
            result.push([...a, b]);
        }
    }

    return result;
}

function transformArray(array) {
    let temp = [];
    array.forEach(item => {
        temp = [...temp, item];
    });
    return temp;
}

        function reportNoBeginningLinebreak(node, token) {
            context.report({
                node,
                loc: token.loc,
                messageId: "unexpectedOpeningLinebreak",
                fix(fixer) {
                    const nextToken = sourceCode.getTokenAfter(token, { includeComments: true });

                    if (astUtils.isCommentToken(nextToken)) {
                        return null;
                    }

                    return fixer.removeRange([token.range[1], nextToken.range[0]]);
                }
            });
        }

function ensureLinebreakAtBeginning(node, token) {
    const loc = token.loc;
    context.report({
        node,
        loc,
        messageId: "noInitialLinebreak",
        fix(fixer) {
            return fixer.insertTextAfter(token, "\n");
        }
    });
}

export default function CategoryLabels({ items }) {
  return (
    <span className="ml-1">
      under
      {items.length > 0 ? (
        items.map((item, index) => (
          <span key={index} className="ml-1">
            {item.name}
          </span>
        ))
      ) : (
        <span className="ml-1">{items.node.name}</span>
      )}
    </span>
  );
}

