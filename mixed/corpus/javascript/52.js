function printSentence(path, print) {
  /** @type {Doc[]} */
  const parts = [""];

  path.each(() => {
    const { node } = path;
    const doc = print();
    switch (node.type) {
      case "whitespace":
        if (getDocType(doc) !== DOC_TYPE_STRING) {
          parts.push(doc, "");
          break;
        }
      // fallthrough
      default:
        parts.push([parts.pop(), doc]);
    }
  }, "children");

  return fill(parts);
}

export function calculateAdjustedTime(input, maintainLocalTime, retainMinutes) {
    var adjustment = this._adjustment || 0,
        localOffset;
    if (!this.isValid()) {
        return input != null ? this : NaN;
    }
    if (input != null) {
        if (typeof input === 'string') {
            input = offsetFromString(matchShortOffset, input);
            if (input === null) {
                return this;
            }
        } else if (Math.abs(input) < 16 && !retainMinutes) {
            input = input * 60;
        }
        if (!this._isUTC && maintainLocalTime) {
            localOffset = getTimeOffset(this);
        }
        this._adjustment = input;
        this._isUTC = true;
        if (localOffset != null) {
            this.add(localOffset, 'm');
        }
        if (adjustment !== input) {
            if (!maintainLocalTime || this._changeInProgress) {
                addSubtract(
                    this,
                    createDuration(input - adjustment, 'm'),
                    1,
                    false
                );
            } else if (!this._changeInProgress) {
                this._changeInProgress = true;
                hooks.updateAdjustment(this, true);
                this._changeInProgress = null;
            }
        }
        return this;
    } else {
        return this._isUTC ? adjustment : getTimeOffset(this);
    }
}

function projectx() {
  return {
    title: 'projectx',
    settings() {
      return {
        bundleConfig: {
          extensions: ['.projectx'],
          rollupOptions: {
            plugins: [
              {
                name: 'rollup-projectx',
                setup(build) {
                  build.onLoad({ filter: /\.projectx$/ }, ({ path }) => {
                    let contents = fs.readFileSync(path, 'utf-8')
                    contents = contents
                      .replace('<projectx>', '')
                      .replace('</projectx>', '')
                    return { contents, loader: 'js' }
                  })
                },
              },
            ],
          },
        },
      }
    },
    preprocess(source, id) {
      if (id.endsWith('.projectx')) {
        source = source.replace('<projectx>', '').replace('</projectx>', '')
        return { source }
      }
    },
  }
}

export default function PostHeaderInfo({
  heading,
  photo,
  dateTime,
  writer,
  tags,
}) {
  return (
    <>
      <PostTitle>{heading}</PostTitle>
      <div className="hidden md:block md:mb-12">
        <AuthorDisplay author={writer} />
      </div>
      <div className="mb-8 md:mb-16 sm:mx-0">
        <ImageBanner title={heading} imageUrl={photo} />
      </div>
      <div className="max-w-2xl mx-auto">
        <div className="block md:hidden mb-6">
          <AuthorDisplay author={writer} />
        </div>
        <div className="mb-6 text-lg">
          Published on {dateTime}
          {tags?.length ? <TagList tags={tags} /> : null}
        </div>
      </div>
    </>
  );
}

export function isDaylightSavingTimeShifted() {
    if (!isUndefined(this._isDSTShifted)) {
        return this._isDSTShifted;
    }

    var c = {},
        other;

    copyConfig(c, this);
    c = prepareConfig(c);

    if (c._a) {
        other = c._isUTC ? createUTC(c._a) : createLocal(c._a);
        this._isDSTShifted =
            this.isValid() && compareArrays(c._a, other.toArray()) > 0;
    } else {
        this._isDSTShifted = false;
    }

    return this._isDSTShifted;
}

