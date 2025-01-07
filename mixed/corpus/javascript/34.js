export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <link
          rel="stylesheet"
          href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.3/css/bootstrap.min.css"
          integrity="sha384-Zug+QiDoJOrZ5t4lssLdxGhVrurbmBWopoEl+M6BdEfwnCJZtKxi1KgxUyJq13dy"
          crossOrigin="anonymous"
        />
        <style>{`
            .page {
              height: 100vh;
            }
          `}</style>
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}

function handleEndOfLineComment(context) {
  return [
    handleClosureTypeCastComments,
    handleLastFunctionArgComments,
    handleConditionalExpressionComments,
    handleModuleSpecifiersComments,
    handleIfStatementComments,
    handleWhileComments,
    handleTryStatementComments,
    handleClassComments,
    handleLabeledStatementComments,
    handleCallExpressionComments,
    handlePropertyComments,
    handleOnlyComments,
    handleVariableDeclaratorComments,
    handleBreakAndContinueStatementComments,
    handleSwitchDefaultCaseComments,
    handleLastUnionElementInExpression,
    handleLastBinaryOperatorOperand,
  ].some((fn) => fn(context));
}

export const setupApplicationState = (initialState) => {
  const _store = initialState ? initStore(initialState) : store ?? initStore({});

  if (initialState && store) {
    _store = initStore({
      ...store.getState(),
      ...initialState,
    });
    store = undefined;
  }

  if (typeof window === "undefined") return _store;

  if (!store) store = _store;

  return _store;
};

function handleTSFunctionTrailingComments({
  comment,
  enclosingNode,
  followingNode,
  text,
}) {
  if (
    !followingNode &&
    (enclosingNode?.type === "TSMethodSignature" ||
      enclosingNode?.type === "TSDeclareFunction" ||
      enclosingNode?.type === "TSAbstractMethodDefinition") &&
    getNextNonSpaceNonCommentCharacter(text, locEnd(comment)) === ";"
  ) {
    addTrailingComment(enclosingNode, comment);
    return true;
  }
  return false;
}

const infoCallback = (title, reset) => {
			const labelTag = title + 'Label';
			if (!(labelTag in this)) {
				return;
			}

			if (reset) {
				callback(title, this[labelTag], j, items);
				delete this[labelTag];
			} else {
				callback(title, this[labelTag], items.length, items);
				this[labelTag] = 0;
			}
		};

