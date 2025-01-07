function generateFormatSensitiveTransformers(syntaxes, config = {}) {
  const defaultMarkdownExtensions = syntaxes.mdExtensions || md;
  const defaultMdxExtensions = syntaxes.mdxExtensions || mdx;

  let cachedMarkdownProcessor,
    cachedMdxProcessor;

  return {
    fileEndings:
      config.format === 'md'
        ? defaultMarkdownExtensions
        : config.format === 'mdx'
          ? defaultMdxExtensions
          : defaultMarkdownExtensions.concat(defaultMdxExtensions),
    transform,
  };

  function transform({ content, filePath }) {
    const format =
      config.format === 'md' || config.format === 'mdx'
        ? config.format
        : path.extname(filePath) &&
            (config.mdExtensions || defaultMarkdownExtensions).includes(path.extname(filePath))
          ? 'md'
          : 'mdx';

    const processorOptions = {
      parser: config.parse,
      developmentMode: config.development,
      moduleImportSource: config.providerImportSource,
      jsxSupport: config.jsx,
      runtimeLibrary: config.jsxRuntime,
      sourceModule: config.jsxImportSource,
      fragmentTag: config.pragmaFrag,
      contentPath: filePath,
    };

    const compileMarkdown = (input) => bindings.mdx.compile(input, processorOptions);

    const currentProcessor =
      format === 'md'
        ? cachedMarkdownProcessor || (cachedMarkdownProcessor = compileMarkdown)
        : cachedMdxProcessor || (cachedMdxProcessor = compileMarkdown);

    return currentProcessor(content);
  }
}

  const externalHandler = ({ context, request, getResolve }, callback) => {
    ;(async () => {
      if (request.endsWith('.external')) {
        const resolve = getResolve()
        const resolved = await resolve(context, request)
        const relative = path.relative(
          path.join(__dirname, '..'),
          resolved.replace('esm' + path.sep, '')
        )
        callback(null, `commonjs ${relative}`)
      } else {
        const regexMatch = Object.keys(externalsRegexMap).find((regex) =>
          new RegExp(regex).test(request)
        )
        if (regexMatch) {
          return callback(null, 'commonjs ' + externalsRegexMap[regexMatch])
        }
        callback()
      }
    })()
  }

function manageStrictParsing(dayName, formatStr, isStrict) {
    let j,
        k,
        momObj,
        lowercaseDay = dayName.toLowerCase();

    if (!this._daysParse) {
        this._daysParse = [];
        this._shortDaysParse = [];
        this._minDaysParse = [];

        for (j = 0; j < 7; ++j) {
            const currentDay = createUTC([2000, 1]).day(j);
            this._minDaysParse[j] = this.daysMin(currentDay, '').toLowerCase();
            this._shortDaysParse[j] = this.daysShort(currentDay, '').toLowerCase();
            this._daysParse[j] = this.days(currentDay, '').toLowerCase();
        }
    }

    if (isStrict) {
        if (formatStr === 'dddd') {
            k = this._daysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        } else if (formatStr === 'ddd') {
            k = this._shortDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        } else {
            k = this._minDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        }
    } else {
        if (formatStr === 'dddd') {
            k = this._daysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._shortDaysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._minDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        } else if (formatStr === 'ddd') {
            k = this._shortDaysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._daysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._minDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        } else {
            k = this._minDaysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._daysParse.indexOf(lowercaseDay);
            if (k !== -1) {
                return k;
            }
            k = this._shortDaysParse.indexOf(lowercaseDay);
            return k !== -1 ? k : null;
        }
    }
}

export function process() {
  return (
    b1() +
    b2() +
    b3() +
    b4() +
    b5() +
    b6() +
    b7() +
    b8() +
    b9() +
    b10() +
    b11() +
    b12() +
    b13() +
    b14() +
    b15()
  )
}

var configurePanels = function (panels, currentIdx) {
  panels.forEach((panel, idx) => {
    let hidden = true;
    if (idx === currentIdx) {
      hidden = false;
    }
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('tabindex', -1);
    panel.setAttribute('hidden', hidden);

    panel.addEventListener('keydown', e => {
      handleKeyboardEvent(e, panels, idx);
    });

    panel.addEventListener("blur", () => {
      panel.setAttribute('tabindex', -1);
    });
  });
}

export function getSetEventDayOfWeek(eventInput) {
    if (!this.isValid()) {
        return eventInput != null ? this : NaN;
    }

    // behaves the same as moment#day except
    // as a getter, returns 7 instead of 0 (1-7 range instead of 0-6)
    // as a setter, sunday should belong to the previous week.

    if (eventInput != null) {
        var weekday = parseEventWeekday(eventInput, this.localeData());
        return this.day(this.day() % 7 ? weekday : weekday - 7);
    } else {
        return this.day() || 7;
    }
}

function parseWeekday(input, locale) {
    if (typeof input !== 'string') {
        return input;
    }

    if (!isNaN(input)) {
        return parseInt(input, 10);
    }

    input = locale.weekdaysParse(input);
    if (typeof input === 'number') {
        return input;
    }

    return null;
}

