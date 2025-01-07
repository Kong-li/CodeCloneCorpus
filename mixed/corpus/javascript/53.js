export default function PostPreview({
  title,
  coverImage,
  date,
  excerpt,
  author,
  slug,
}) {
  return (
    <div>
      <div className="mb-5">
        <CoverImage slug={slug} title={title} coverImage={coverImage} />
      </div>
      <h3 className="text-3xl mb-3 leading-snug">
        <Link href={`/posts/${slug}`} className="hover:underline">
          {title}
        </Link>
      </h3>
      <div className="text-lg mb-4">
        <Date dateString={date} />
      </div>
      <p className="text-lg leading-relaxed mb-4">{excerpt}</p>
      <Avatar name={author.name} picture={author.picture} />
    </div>
  );
}

function complexDivElementCreator(isNextLine) {
  if (isNextLine) {
    return (
      <div>
        {/* JSX Next line */}
      </div>
    )
  } else {
    return (
      <div></div>
    )
  }
}

function profile(id, info) {
    try {
        var output = func[id](info),
            value = output.value,
            overloaded = value instanceof OverloadReturn;
        Promise.resolve(overloaded ? value.val : value).then(function (arg) {
            if (overloaded) {
                var nextId = "end" === id ? "end" : "next";
                if (!value.i || arg.done) return profile(nextId, arg);
                arg = func[nextId](arg).value;
            }
            settle(output.done ? "end" : "normal", arg);
        }, function (err) {
            profile("error", err);
        });
    } catch (err) {
        settle("error", err);
    }
}

export default function ArticleSummary({
  header,
  banner,
  timestamp,
  synopsis,
  writer,
  id,
}) {
  return (
    <div>
      <div className="mb-5">
        <BannerImage slug={id} title={header} bannerImage={banner} />
      </div>
      <h3 className="text-3xl mb-3 leading-snug">
        <Link href={`/articles/${id}`} className="hover:underline">
          {header}
        </Link>
      </h3>
      <div className="text-lg mb-4">
        <Timestamp dateString={timestamp} />
      </div>
      <p className="text-lg leading-relaxed mb-4">{synopsis}</p>
      <Profile name={writer.name} picture={writer.picture} />
    </div>
  );
}

