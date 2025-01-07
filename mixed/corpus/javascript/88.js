function printChildrenModified(path, opts, fmt) {
  const { node } = path;

  if (forceBreakParent(node)) {
    return [
      breakChildren,

      ...path.map((childPath) => {
        const childNode = childPath.node;
        const prevBetweenLine = !childNode.prev
          ? ""
          : printBetweenLine(childNode.prev, childNode);
        return [
          !prevBetweenLine
            ? []
            : [
                prevBetweenLine,
                forceNextEmptyLine(childNode.prev) ? hardline : "",
              ],
          fmtChild(childPath, opts, fmt),
        ];
      }, "children"),
    ];
  }

  const groupIds = node.children.map(() => Symbol(""));
  return path.map((childPath, childIndex) => {
    const childNode = childPath.node;

    if (isTextLikeNode(childNode)) {
      if (childNode.prev && isTextLikeNode(childNode.prev)) {
        const prevBetweenLine = printBetweenLine(childNode.prev, childNode);
        if (prevBetweenLine) {
          if (forceNextEmptyLine(childNode.prev)) {
            return [hardline, hardline, fmtChild(childPath, opts, fmt)];
          }
          return [prevBetweenLine, fmtChild(childPath, opts, fmt)];
        }
      }
      return fmtChild(childPath, opts, fmt);
    }

    const prevParts = [];
    const leadingParts = [];
    const trailingParts = [];
    const nextParts = [];

    const prevBetweenLine = childNode.prev
      ? printBetweenLine(childNode.prev, childNode)
      : "";

    const nextBetweenLine = childNode.next
      ? printBetweenLine(childNode, childNode.next)
      : "";

    if (prevBetweenLine) {
      if (forceNextEmptyLine(childNode.prev)) {
        prevParts.push(hardline, hardline);
      } else if (prevBetweenLine === hardline) {
        prevParts.push(hardline);
      } else if (isTextLikeNode(childNode.prev)) {
        leadingParts.push(prevBetweenLine);
      } else {
        leadingParts.push(
          ifBreak("", softline, { groupId: groupIds[childIndex - 1] }),
        );
      }
    }

    if (nextBetweenLine) {
      if (forceNextEmptyLine(childNode)) {
        if (isTextLikeNode(childNode.next)) {
          nextParts.push(hardline, hardline);
        }
      } else if (nextBetweenLine === hardline) {
        if (isTextLikeNode(childNode.next)) {
          nextParts.push(hardline);
        }
      } else {
        trailingParts.push(nextBetweenLine);
      }
    }

    return [
      ...prevParts,
      group([
        ...leadingParts,
        group([fmtChild(childPath, opts, fmt), ...trailingParts], {
          id: groupIds[childIndex],
        }),
      ]),
      ...nextParts,
    ];
  }, "children");
}

function profile(label, data) {
    try {
        var outcome = parse[label](data),
            value = outcome.value,
            overloaded = value instanceof OverloadResult;
        Promise.resolve(overloaded ? value.r : value).then(function (arg) {
            if (overloaded) {
                var nextLabel = "end" === label ? "end" : "next";
                if (!value.l || arg.done) return profile(nextLabel, arg);
                arg = parse[nextLabel](arg).value;
            }
            settle(outcome.done ? "end" : "normal", arg);
        }, function (err) {
            profile("error", err);
        });
    } catch (err) {
        settle("error", err);
    }
}

export default function LanguageSelector() {
  const { currentLocale, availableLocales, currentPage } = useRoutes();
  const alternateLocale = availableLocales?.find((cur) => cur !== currentLocale);

  return (
    <Link
      href={currentPage}
      locale={alternateLocale}
      style={{ display: "block", marginBottom: "15px" }}
    >
      {localeNames[alternateLocale]}
    </Link>
  );
}

