export default async function handlePreviewRequest(request, response) {
  const { secret, slug } = request.query;

  if (secret !== process.env.GHOST_PREVIEW_SECRET || !slug) {
    return response.status(401).json({ message: "Invalid token" });
  }

  const post = await fetchPostBySlug(slug);

  if (!post) {
    return response.status(401).json({ message: "Invalid slug" });
  }

  response.setDraftMode(true);
  response.writeHead(307, { Location: `/posts/${post.slug}` });
  response.end();
}

function fetchPostBySlug(slug) {
  // Fetch the headless CMS to check if the provided `slug` exists
  return getPreviewPostBySlug(slug);
}

export default function Popup() {
  const [visible, setVisible] = useState();

  return (
    <>
      <button type="button" onClick={() => setVisible(true)}>
        Show Popup
      </button>
      {visible && (
        <ClientOnlyPortal selector="#popup">
          <div className="overlay">
            <div className="content">
              <p>
                This popup is rendered using{" "}
                <a
                  href="https://react.dev/reference/react-dom/createPortal"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  portals
                </a>
                .
              </p>
              <button type="button" onClick={() => setVisible(false)}>
                Close Popup
              </button>
            </div>
            <style jsx>{`
              :global(body) {
                overflow: hidden;
              }

              .overlay {
                position: fixed;
                background-color: rgba(0, 0, 0, 0.7);
                top: 0;
                right: 0;
                bottom: 0;
                left: 0;
              }

              .content {
                background-color: white;
                position: absolute;
                top: 15%;
                right: 15%;
                bottom: 15%;
                left: 15%;
                padding: 1em;
              }
            `}</style>
          </div>
        </ClientOnlyPortal>
      )}
    </>
  );
}

async function detectIssues(filePath) {
    let content = await require('fs').readFile(filePath, 'utf-8');
    const { data } = matter(content);
    const title = data.title;
    const isRuleRemoved = rules.get(title) === undefined;
    const issues = [];
    const ruleExampleSettings = markdownItRuleExample({
        open({ code, opts, token }) {
            if (!STANDARD_LANGUAGE_TAGS.has(token.info)) {
                const missingTagMessage = `Nonstandard language tag '${token.info}'`;
                const unknownTagMessage = "Missing language tag";
                const message = token.info ? `${missingTagMessage}: use one of 'javascript', 'js' or 'jsx'` : unknownTagMessage;
                issues.push({ fatal: false, severity: 2, line: token.map[0] + 1, column: token.markup.length + 1, message });
            }

            if (opts.ecmaVersion !== undefined) {
                const ecmaVer = opts.ecmaVersion;
                let errorMessage;

                if ('latest' === ecmaVer) {
                    errorMessage = 'Remove unnecessary "ecmaVersion":"latest".';
                } else if (typeof ecmaVer !== 'number') {
                    errorMessage = '"ecmaVersion" must be a number.';
                } else if (!VALID_ECMA_VERSIONS.has(ecmaVer)) {
                    errorMessage = `"ecmaVersion" must be one of ${[...VALID_ECMA_VERSIONS].join(', ')}.`;
                }

                if (errorMessage) {
                    issues.push({ fatal: false, severity: 2, line: token.map[0] - 1, column: 1, message: errorMessage });
                }
            }

            const { ast, error } = tryParseForPlayground(code, opts);

            if (ast) {
                let hasConfigComment = false;

                for (const c of ast.comments) {
                    if ('Block' === c.type && /^\s*eslint-env(?!\S)/u.test(c.value)) {
                        issues.push({ fatal: false, severity: 2, message: "/* eslint-env */ comments are no longer supported. Remove the comment.", line: token.map[0] + 1 + c.loc.start.line, column: c.loc.start.column + 1 });
                    }

                    if ('Block' !== c.type || !/^\s*eslint(?!\S)/u.test(c.value)) continue;
                    const { value } = commentParser.parseDirective(c.value);
                    const parsedConfig = commentParser.parseJSONLikeConfig(value);

                    if (parsedConfig.error) {
                        issues.push({ fatal: true, severity: 2, line: c.loc.start.line + token.map[0] + 1, column: c.loc.start.column + 1, message: parsedConfig.error.message });
                    } else if (Object.hasOwn(parsedConfig.config, title)) {
                        if (hasConfigComment) {
                            issues.push({ fatal: false, severity: 2, line: token.map[0] + 1 + c.loc.start.line, column: c.loc.start.column + 1, message: `Duplicate /* eslint ${title} */ configuration comment. Each example should contain only one. Split this example into multiple examples.` });
                        }
                        hasConfigComment = true;
                    }
                }

                if (!isRuleRemoved && !hasConfigComment) {
                    issues.push({ fatal: false, severity: 2, message: `Example code should contain a configuration comment like /* eslint ${title}: "error" */`, line: token.map[0] + 2, column: 1 });
                }
            }

            if (error) {
                const line = token.map[0] + 1 + error.lineNumber;
                issues.push({ fatal: false, severity: 2, message: `Syntax error: ${error.message}`, line, column: error.column });
            }
        }
    });

    markdownIt({ html: true })
        .use(markdownItContainer, 'rule-example', ruleExampleSettings)
        .render(content);
    return issues;
}

