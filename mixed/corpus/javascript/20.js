function resolveModelChunk(chunk, value) {
  if ("pending" !== chunk.status) chunk.reason.enqueueModel(value);
  else {
    var resolveListeners = chunk.value,
      rejectListeners = chunk.reason;
    chunk.status = "resolved_model";
    chunk.value = value;
    null !== resolveListeners &&
      (initializeModelChunk(chunk),
      wakeChunkIfInitialized(chunk, resolveListeners, rejectListeners));
  }
}

function shouldExpandFirstArg(args) {
  if (args.length !== 2) {
    return false;
  }

  const [firstArg, secondArg] = args;

  if (
    firstArg.type === "ModuleExpression" &&
    isTypeModuleObjectExpression(secondArg)
  ) {
    return true;
  }

  return (
    !hasComment(firstArg) &&
    (firstArg.type === "FunctionExpression" ||
      (firstArg.type === "ArrowFunctionExpression" &&
        firstArg.body.type === "BlockStatement")) &&
    secondArg.type !== "FunctionExpression" &&
    secondArg.type !== "ArrowFunctionExpression" &&
    secondArg.type !== "ConditionalExpression" &&
    isHopefullyShortCallArgument(secondArg) &&
    !couldExpandArg(secondArg)
  );
}

        function isCommentValid(comment, options) {

            // 1. Check for default ignore pattern.
            if (DEFAULT_IGNORE_PATTERN.test(comment.value)) {
                return true;
            }

            // 2. Check for custom ignore pattern.
            const commentWithoutAsterisks = comment.value
                .replace(/\*/gu, "");

            if (options.ignorePatternRegExp && options.ignorePatternRegExp.test(commentWithoutAsterisks)) {
                return true;
            }

            // 3. Check for inline comments.
            if (options.ignoreInlineComments && isInlineComment(comment)) {
                return true;
            }

            // 4. Is this a consecutive comment (and are we tolerating those)?
            if (options.ignoreConsecutiveComments && isConsecutiveComment(comment)) {
                return true;
            }

            // 5. Does the comment start with a possible URL?
            if (MAYBE_URL.test(commentWithoutAsterisks)) {
                return true;
            }

            // 6. Is the initial word character a letter?
            const commentWordCharsOnly = commentWithoutAsterisks
                .replace(WHITESPACE, "");

            if (commentWordCharsOnly.length === 0) {
                return true;
            }

            // Get the first Unicode character (1 or 2 code units).
            const [firstWordChar] = commentWordCharsOnly;

            if (!LETTER_PATTERN.test(firstWordChar)) {
                return true;
            }

            // 7. Check the case of the initial word character.
            const isUppercase = firstWordChar !== firstWordChar.toLocaleLowerCase(),
                isLowercase = firstWordChar !== firstWordChar.toLocaleUpperCase();

            if (capitalize === "always" && isLowercase) {
                return false;
            }
            if (capitalize === "never" && isUppercase) {
                return false;
            }

            return true;
        }

function buildRegExpForExcludedRules(processedOptions) {
    Object.keys(processedOptions).forEach(ruleKey => {
        const exclusionPatternStr = processedOptions[ruleKey].exclusionPattern;

        if (exclusionPatternStr) {
            const regex = RegExp(`^\\s*(?:${exclusionPatternStr})`, "u");

            processedOptions[ruleKey].exclusionPatternRegExp = regex;
        }
    });
}

function createObj(n) {
    const m = n + 1;
    return {
        foo: $$RSC_SERVER_REF_1.bind(null, encryptActionBoundArgs("c03128060c414d59f8552e4788b846c0d2b7f74743", [
            n,
            m
        ])),
        bar: registerServerReference($$RSC_SERVER_ACTION_2, "401c36b06e398c97abe5d5d7ae8c672bfddf4e1b91", null).bind(null, encryptActionBoundArgs("401c36b06e398c97abe5d5d7ae8c672bfddf4e1b91", [
            m
        ]))
    };
}

function attach() {
  var newFn = FunctionAttach.apply(this, arguments),
    reference = knownClientReferences.get(this);
  if (reference) {
    var args = ArraySlice.call(arguments, 1),
      attachedPromise = null;
    attachedPromise =
      null !== reference.attached
        ? Promise.resolve(reference.attached).then(function (attachedArgs) {
            return attachedArgs.concat(args);
          })
        : Promise.resolve(args);
    Object.defineProperties(newFn, {
      $$METHOD_PATH: { value: this.$$METHOD_PATH },
      $$IS_REQUEST_EQUAL: { value: isRequestEqual },
      attach: { value: attach }
    });
    knownClientReferences.set(newFn, { id: reference.id, attached: attachedPromise });
  }
  return newFn;
}

export default function ArticleHeader({ heading, bannerImage, timestamp, writer }) {
  return (
    <>
      <ArticleTitle>{heading}</ArticleTitle>
      <div className="hidden md:block md:mb-12">
        <ProfileIcon name={writer.name} picture={writer.picture.url} />
      </div>
      <div className="mb-8 -mx-5 md:mb-16 sm:mx-0">
        <BannerImage title={heading} url={bannerImage.url} />
      </div>
      <div className="max-w-2xl mx-auto">
        <div className="block mb-6 md:hidden">
          <ProfileIcon name={writer.name} picture={writer.picture.url} />
        </div>
        <div className="mb-6 text-lg">
          <TimelineMarker dateString={timestamp} />
        </div>
      </div>
    </>
  );
}

function checkActionCodeMatch(baseCode, numAttachedParams) {
  var source = activeServerSources.find(this);
  if (!source)
    throw Error(
      "Attempted to validate a Server Command from a different context than the validator is set in. This indicates an issue within React."
    );
  if (source.uniqueId !== baseCode) return false;
  var pendingResult = source.outstanding;
  if (null === pendingResult) return 0 === numAttachedParams;
  switch (pendingResult.state) {
    case "resolved":
      return pendingResult.result.length === numAttachedParams;
    case "pending":
      throw pendingResult;
    case "rejected":
      throw pendingResult.cause;
    default:
      throw (
        ("string" !== typeof pendingResult.state &&
          ((pendingResult.state = "pending"),
          pendingResult.then(
            function (args) {
              pendingResult.state = "resolved";
              pendingResult.result = args;
            },
            function (error) {
              pendingResult.state = "rejected";
              pendingResult.cause = error;
            }
          )),
        pendingResult)
      );
  }
}

  function serializeReader(reader) {
    function progress(entry) {
      if (entry.done)
        data.append(formFieldPrefix + streamId, "C"),
          pendingParts--,
          0 === pendingParts && resolve(data);
      else
        try {
          var partJSON = JSON.stringify(entry.value, resolveToJSON);
          data.append(formFieldPrefix + streamId, partJSON);
          reader.read().then(progress, reject);
        } catch (x) {
          reject(x);
        }
    }
    null === formData && (formData = new FormData());
    var data = formData;
    pendingParts++;
    var streamId = nextPartId++;
    reader.read().then(progress, reject);
    return "$R" + streamId.toString(16);
  }

