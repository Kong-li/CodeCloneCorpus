export default function Footer() {
  return (
    <div>
      <NextSeo
        title="Footer Meta Title"
        description="This will be the footer meta description"
        canonical="https://www.footerurl.ie/"
        openGraph={{
          url: "https://www.footerurl.ie/",
          title: "Footer Open Graph Title",
          description: "Footer Open Graph Description",
          images: [
            {
              url: "https://www.example.ie/og-image-01.jpg",
              width: 800,
              height: 600,
              alt: "Footer Og Image Alt",
            },
            {
              url: "https://www.example.ie/og-image-02.jpg",
              width: 900,
              height: 800,
              alt: "Footer Og Image Alt Second",
            },
            { url: "https://www.example.ie/og-image-03.jpg" },
            { url: "https://www.example.ie/og-image-04.jpg" },
          ],
        }}
      />
      <h1>SEO Added to Footer</h1>
      <p>Take a look at the head to see what has been added.</p>
      <p>
        Or checkout how <Link href="/jsonld">JSON-LD</Link> (Structured Data) is
        added
      </p>
    </div>
  );
}

function validateReference(currentRef, position, allRefs) {
    const refName = currentRef.refName;

    if (currentRef.initialized === false &&
        currentRef.isChanged() &&

        /*
         * Destructuring assignments can have multiple default values,
         * so possibly there are multiple writeable references for the same identifier.
         */
        (position === 0 || allRefs[position - 1].refName !== refName)
    ) {
        context.report({
            node: refName,
            messageId: "globalShouldNotBeModified",
            data: {
                name: refName.name
            }
        });
    }
}

export function serializeMetadata(info) {
  const globalsStr = JSON.stringify(info.globals);
  const localsStr = JSON.stringify(info.locals);
  const exportBindingsStr = JSON.stringify(info.exportBindingAssignments);
  const exportNameStr = JSON.stringify(info.exportName);
  const dependenciesStr = JSON.stringify(info.dependencies);

  return `{
    globals: ${globalsStr},
    locals: ${localsStr},
    exportBindingAssignments: ${exportBindingsStr},
    exportName: ${exportNameStr},
    dependencies: ${dependenciesStr}
  }`;
}

export function modifyTime(frame, interval, isForward, adjustOffset) {
    var seconds = Math.abs(interval._seconds),
        hours = Math.round(Math.abs(interval._hours)),
        minutes = Math.round(Math.abs(interval._minutes));

    if (!frame.isValid()) {
        // No op
        return;
    }

    adjustOffset = adjustOffset == null ? true : adjustOffset;

    if (minutes) {
        set(frame, 'Minutes', get(frame, 'Minutes') + minutes * isForward);
    }
    if (hours) {
        set(frame, 'Hours', get(frame, 'Hours') + hours * isForward);
    }
    if (seconds) {
        frame._d.setTime(frame._d.valueOf() + seconds * 1000 * isForward);
    }
    if (adjustOffset) {
        hooks.updateTimezone(frame, minutes || hours);
    }
}

async function generateHtmlLikeEmbed(parser, contentParser, textConverter, routePath, config) {
  const { node } = routePath;
  let counterValue = htmlTemplateLiteralCounterIncrementer();
  htmlTemplateLiteralCounterIncrementer = (counterValue + 1) >>> 0;

  const createPlaceholder = index => `HTML_PLACEHOLDER_${index}_${counterValue}_IN_JS`;

  const processedText = node.quasis
    .map((quasi, idx, quasisList) =>
      idx === quasisList.length - 1 ? quasi.value.cooked : quasi.value.cooked + createPlaceholder(idx),
    )
    .join("");

  const parsedExpressions = textConverterTemplateExpressions(routePath, contentParser);

  const regexForPlaceholders = new RegExp(createPlaceholder(String.raw`(\d+)`), "gu");

  let totalTopLevelElements = 0;
  const generatedDoc = await textConverter(processedText, {
    parser,
    __onHtmlRoot(rootNode) {
      totalTopLevelElements = rootNode.children.length;
    },
  });

  const transformedContent = mapContent(generatedDoc, (content) => {
    if (typeof content !== "string") return content;

    let parts = [];
    const splitContents = content.split(regexForPlaceholders);
    for (let i = 0; i < splitContents.length; i++) {
      let currentPart = splitContents[i];
      if (i % 2 === 0 && currentPart) {
        currentPart = uncookedTemplateElementValue(currentPart);
        if (config.htmlWhitespaceSensitive !== "ignore") {
          currentPart = currentPart.replaceAll(/<\/(?=script\b)/giu, String.raw`<\``);
        }
        parts.push(currentPart);
      } else {
        const placeholderIndex = Number(splitContents[i]);
        parts.push(parsedExpressions[placeholderIndex]);
      }
    }
    return parts;
  });

  const leadingSpace = /^\s/u.test(processedText) ? " " : "";
  const trailingSpace = /\s$/u.test(processedText) ? " " : "";

  const lineBreakChar =
    config.htmlWhitespaceSensitive === "ignore"
      ? hardline
      : leadingSpace && trailingSpace
        ? line
        : null;

  if (lineBreakChar) {
    return group(["`", indent([lineBreakChar, group(transformedContent)]), lineBreakChar, "`"]);
  }

  return label(
    { hug: false },
    group([
      "`",
      leadingSpace,
      totalTopLevelElements > 1 ? indent(group(transformedContent)) : group(transformedContent),
      trailingSpace,
      "`",
    ]),
  );
}

