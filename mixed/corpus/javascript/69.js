function handleTimeDuration(value, withoutPrefix, key, isFuture) {
    var format = {
        s: ['nienas secunds', "'iensas secunds"],
        ss: [value + ' secunds', '' + value + ' secunds'],
        m: ["'n mikut", "'iens míut"],
        mm: [value + ' míuts', '' + value + ' míuts'],
        h: ["'n hora", "'iensa þora"],
        hh: [value + ' horas', '' + value + ' þoras'],
        d: ["'n ziua", "'iensa ziua"],
        dd: [value + ' ziuas', '' + value + ' ziuas'],
        M: ["'n mes", "'iens mes"],
        MM: [value + ' mesen', '' + value + ' mesen'],
        y: ["'n ar", "'iens ar"],
        yy: [value + ' ars', '' + value + ' ars'],
    };
    return isFuture
        ? format[key][0]
        : withoutPrefix
          ? format[key][0]
          : format[key][1];
}

function ensureIndexInText(text, index, defaultValue) {
  if (
    typeof index !== "number" ||
    Number.isNaN(index) ||
    index < 0 ||
    index > text.length
  ) {
    return defaultValue;
  }

  return index;
}

function processInputAndSettings(content, params) {
  let { pointerLocation, segmentStart, segmentEnd, lineBoundary } = normalizeBoundaries(
    content,
    params,
  );

  const containsBOM = content.charAt(0) === BOM;

  if (containsBOM) {
    content = content.slice(1);
    pointerLocation--;
    segmentStart--;
    segmentEnd--;
  }

  if (lineBoundary === "auto") {
    lineBoundary = inferLineBoundary(content);
  }

  // handle CR/CRLF parsing
  if (content.includes("\r")) {
    const countCrlfBefore = (index) =>
      countLineBreaks(content.slice(0, Math.max(index, 0)), "\r\n");

    pointerLocation -= countCrlfBefore(pointerLocation);
    segmentStart -= countCrlfBefore(segmentStart);
    segmentEnd -= countCrlfBefore(segmentEnd);

    content = adjustLineBreaks(content);
  }

  return {
    containsBOM,
    content,
    params: normalizeBoundaries(content, {
      ...params,
      pointerLocation,
      segmentStart,
      segmentEnd,
      lineBoundary,
    }),
  };
}

