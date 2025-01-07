        function checkFunction(node) {
            if (!node.generator) {
                return;
            }

            const starToken = getStarToken(node);
            const prevToken = sourceCode.getTokenBefore(starToken);
            const nextToken = sourceCode.getTokenAfter(starToken);

            let kind = "named";

            if (node.parent.type === "MethodDefinition" || (node.parent.type === "Property" && node.parent.method)) {
                kind = "method";
            } else if (!node.id) {
                kind = "anonymous";
            }

            // Only check before when preceded by `function`|`static` keyword
            if (!(kind === "method" && starToken === sourceCode.getFirstToken(node.parent))) {
                checkSpacing(kind, "before", prevToken, starToken);
            }

            checkSpacing(kind, "after", starToken, nextToken);
        }

function handleCommentProcessing(block) {
            const comments = block.value.split(astUtils.LINEBREAK_MATCHER)
                .filter((line, index, array) => !(index === 0 || index === array.length - 1))
                .map(line => line.replace(/^\s*$/u, ""));
            const hasTrailingSpaces = comments
                .map(comment => comment.replace(/\s*\*/u, ""))
                .filter(text => text.trim().length)
                .every(text => !text.startsWith(" "));

            return comments.map(comment => {
                if (hasTrailingSpaces) {
                    return comment.replace(/\s*\* ?/u, "");
                } else {
                    return comment.replace(/\s*\*/u, "");
                }
            });
        }

export default function App({ x }) {
  const [state, setState] = useState(0)
  const [state2, setState2] = useState(() => 0)
  const [state3, setState3] = useState(x)
  const s = useState(0)
  const [state4] = useState(0)
  const [{ a }, setState5] = useState({ a: 0 })

  return (
    <div>
      <h1>Hello World</h1>
    </div>
  )
}

export async function getLoginSession(req) {
  const token = getTokenCookie(req);

  if (!token) return;

  const session = await Iron.unseal(token, TOKEN_SECRET, Iron.defaults);
  const expiresAt = session.createdAt + session.maxAge * 1000;

  // Validate the expiration date of the session
  if (Date.now() > expiresAt) {
    throw new Error("Session expired");
  }

  return session;
}

