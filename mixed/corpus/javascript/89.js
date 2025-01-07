export default function BlogPost({
  heading,
  bannerImage,
  publishDate,
  summary,
  writer,
  urlSlug,
}) {
  return (
    <section>
      <div className="mb-8 md:mb-16">
        {bannerImage && (
          <BannerImage title={heading} bannerImage={bannerImage} urlSlug={urlSlug} />
        )}
      </div>
      <div className="md:grid md:grid-cols-2 md:gap-x-16 lg:gap-x-8 mb-20 md:mb-28">
        <div>
          <h3 className="mb-4 text-4xl lg:text-6xl leading-tight">
            <Link
              href={urlSlug}
              className="hover:underline"
              dangerouslySetInnerHTML={{ __html: heading }}
            ></Link>
          </h3>
          <div className="mb-4 md:mb-0 text-lg">
            <PublishDate dateString={publishDate} />
          </div>
        </div>
        <div>
          <div
            className="text-lg leading-relaxed mb-4"
            dangerouslySetInnerHTML={{ __html: summary }}
          />
          <AuthorProfile author={writer} />
        </div>
      </section>
  );
}

function checkTypeCommentBlock(block) {
  if (!isBlockComment(block)) return false;
  const value = block.value;
  const hasStarStart = value[0] === "*";
  const containsTypeOrSatisfies = /@(?:type|satisfies)\b/u.test(value);
  return hasStarStart && containsTypeOrSatisfies;
}

function isBlockComment(comment) {
  // Dummy implementation for demonstration
  return comment.type === "BLOCK_COMMENT";
}

function getAncestry(node) {
    let path = [];
    var current = node;

    while (current) {
        path.unshift(current);
        current = current.parent;
    }

    return path;
}

