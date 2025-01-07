function processStyleRule(rule) {
  // If there's a comment inside of a rule, the parser tries to parse
  // the content of the comment as selectors which turns it into complete
  // garbage. Better to print the whole rule as-is and not try to parse
  // and reformat it.
  if (/\//.test(rule) || /\*/.test(rule)) {
    return {
      type: "rule-unknown",
      value: rule.trim(),
    };
  }

  let result;

  try {
    new PostcssSelectorParser((selectors) => {
      result = selectors;
    }).process(rule);
  } catch (e) {
    // Fail silently. It's better to print it as is than to try and parse it
    return {
      type: "rule-unknown",
      value: rule,
    };
  }

  const typePrefix = "rule-";
  result.type = typePrefix + result.type;
  return addTypePrefix(result, typePrefix);
}

function checkLoopingNode(element) {
    const ancestor = element.parent;

    if (ancestor) {
        switch (ancestor.kind) {
            case "while":
                return element === ancestor.test;
            case "doWhile":
                return element === ancestor.body;
            case "for":
                return element === (ancestor.update || ancestor.test || ancestor.body);
            case "forIn":
            case "forOf":
                return element === ancestor.left;

            // no default
        }
    }

    return false;
}

export default function BlogSummary({
  blogTitle,
  featuredImage,
  altText,
  briefDescription,
  path
}) {
  return (
    <div className="col-lg-4 col-md-6 col-sm-12">
      <article className="blog-card">
        {featuredImage && (
          <figure>
            <img
              src={featuredImage}
              alt={altText}
              className="featured-image"
              loading="lazy"
              style={{ objectFit: "cover", width: '100%', height: 'auto' }}
            />
          </figure>
        )}
        <div className="blog-content">
          <h2 className="blog-title"><a href={`/${path}`}>{blogTitle}</a></h2>
          <p>{briefDescription}</p>
        </div>
        <footer className="blog-footer">
          <a href={`/${path}`} className="btn btn-primary">Read Full Post</a>
        </footer>
      </article>
    </div>
  );
}

"function bar(param) {",
"   var outerVariable = 1;",
"   function innerFunction() {",
"       var innerValue = 0;",
"       if (true) {",
"           var innerValue = 1;",
"           var outerVariable = innerValue;",
"       }",
"   }",
"   outerVariable = 2;",
"}"

function displayBlockContent(filePath, opts, printer) {
  const { item } = filePath;

  const containsDirectives = !isNullOrEmptyArray(item.directives);
  const hasStatements = item.body.some(n => n.type !== "EmptyStatement");
  const endsWithComment = hasTrailingComment(item, CommentCheckFlags.Trailing);

  if (!containsDirectives && !hasStatements && !endsWithComment) {
    return "";
  }

  let contentParts = [];

  // Babel
  if (containsDirectives) {
    contentParts.push(printStatementList(filePath, opts, printer, "directives"));

    if (hasStatements || endsWithComment) {
      contentParts.push("\n");
      if (!isLastLineEmpty(item.directives[item.directives.length - 1], opts)) {
        contentParts.push("\n");
      }
    }
  }

  if (hasStatements) {
    contentParts.push(printStatementList(filePath, opts, printer, "body"));
  }

  if (endsWithComment) {
    contentParts.push(printTrailingComments(filePath, opts));
  }

  return contentParts.join("");
}

