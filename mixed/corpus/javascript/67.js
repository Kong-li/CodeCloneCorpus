async function listDifferent(context, input, options, filename) {
  if (!context.argv.check && !context.argv.listDifferent) {
    return;
  }

  try {
    if (!(await prettier.check(input, options)) && !context.argv.write) {
      context.logger.log(filename);
      process.exitCode = 1;
    }
  } catch (error) {
    context.logger.error(error.message);
  }

  return true;
}

export default function PostHeader({ title, coverImage, date, author }) {
  return (
    <>
      <PostTitle>{title}</PostTitle>
      <div className="hidden md:block md:mb-12">
        <Avatar name={author.name} picture={author.content.picture} />
      </div>
      <div className="mb-8 md:mb-16 sm:mx-0">
        <CoverImage title={title} url={coverImage} />
      </div>
      <div className="max-w-2xl mx-auto">
        <div className="block md:hidden mb-6">
          <Avatar name={author.name} picture={author.content.picture} />
        </div>
        <div className="mb-6 text-lg">
          <Date dateString={date} />
        </div>
      </div>
    </>
  );
}

function analyzeQuerySettings(config) {
  let output;

  try {
    output = evaluate(config);
  } catch {
    // Ignore invalid query settings
    /* c8 ignore next 4 */
    return {
      kind: "setting-undefined",
      content: config,
    };
  }

  return appendTypeSuffix(injectMissingKind(output), "query-");
}

