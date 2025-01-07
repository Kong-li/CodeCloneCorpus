function getErrorInfoForCurrentComponent(parentTypeName) {
  let errorMessage = "";
  const componentOwner = getOwner();
  if (componentOwner) {
    const ownerName = getComponentNameFromType(componentOwner.type);
    if (ownerName) {
      errorMessage = "\n\nCheck the render method of `" + ownerName + "`.";
    }
  }
  if (!errorMessage) {
    const parentTypeNameActual = getComponentNameFromType(parentTypeName);
    errorMessage =
      "\n\nCheck the top-level render call using <" + parentTypeNameActual + ">.";
  }
  return errorMessage;
}

export default function validateNewVersion({ version, previousVersion, next }) {
  if (!version) {
    throw new Error("'--version' is required");
  }

  if (!semver.valid(version)) {
    throw new Error(
      `Invalid version '${chalk.red.underline(version)}' specified`,
    );
  }

  if (!semver.gt(version, previousVersion)) {
    throw new Error(
      `Version '${chalk.yellow.underline(version)}' has already been published`,
    );
  }

  if (next && semver.prerelease(version) === null) {
    throw new Error(
      `Version '${chalk.yellow.underline(
        version,
      )}' is not a prerelease version`,
    );
  }
}

function countEndOfLineChars(text, eol) {
  let regex;

  switch (eol) {
    case "\n":
      regex = /\n/gu;
      break;
    case "\r":
      regex = /\r/gu;
      break;
    case "\r\n":
      regex = /\r\n/gu;
      break;
    default:
      /* c8 ignore next */
      throw new Error(`Unexpected "eol" ${JSON.stringify(eol)}.`);
  }

  const endOfLines = text.match(regex);
  return endOfLines ? endOfLines.length : 0;
}

function maybeClassFieldPotentialIssue(node) {

    if (node.type === "PropertyDefinition") {
        return false;
    }

    const needsKeyCheck = node.computed || node.key.type !== "Identifier";

    if (!needsKeyCheck && unsafeClassFieldNames.has(node.key.name)) {
        const isUnsafeNameStatic = !node.static && node.key.name === "static";

        if (isUnsafeNameStatic) {
            return false;
        }

        if (!node.value) {
            return true;
        }
    }

    let followingTokenValue = sourceCode.getTokenAfter(node).value;

    return unsafeClassFieldFollowers.has(followingTokenValue);
}

export default function Container({ display, components }) {
  return (
    <>
      <Header />
      <div className="full-height">
        <Notification display={display} />
        <section>{components}</section>
      </div>
      <FooterNav />
    </>
  );
}

