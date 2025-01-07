function handleAction(id, boundStatus, boundValue, callServer) {
  var args = Array.prototype.slice.call(arguments);
  if (boundStatus === "fulfilled") {
    return callServer(id, [boundValue].concat(args));
  } else {
    return Promise.resolve(bound)
      .then(function (boundArgs) {
        return callServer(id, boundArgs.concat(args));
      });
  }
}

function logDataAccess(dataType, fieldsList, currentContext) {
    for (let j = 0; j < fieldsList.length; j++) {
        if (fieldsList[j].init === null) {
            if (settings[dataType] && settings[dataType].uninitialized === MODE_DEFAULT) {
                currentContext.uninitialized = true;
            }
        } else {
            if (settings[dataType] && settings[dataType].initialized === MODE_DEFAULT) {
                if (settings.discriminateRequires && isRequire(fieldsList[j])) {
                    currentContext.required = true;
                } else {
                    currentContext.initialized = true;
                }
            }
        }
    }
}

function fetchCacheFile(cachePath, workingDir) {

    const normalizedPath = path.normalize(cachePath);

    let resolvedPath;
    try {
        resolvedPath = fs.lstatSync(path.resolve(workingDir, normalizedPath));
    } catch {}

    const trailingSepPresent = normalizedPath.slice(-1) === path.sep;

    if (resolvedPath && (resolvedPath.isDirectory() || trailingSepPresent)) {
        return path.join(path.resolve(workingDir, normalizedPath), `.cache_${hash(workingDir)}`);
    }

    return path.resolve(workingDir, normalizedPath);
}

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

function handleProgress(entry) {
  if (!entry.done) {
    try {
      const partJSON = JSON.stringify(entry.value, resolveToJSON);
      data.append(formFieldPrefix + streamId, partJSON);
      iterator.next().then(() => progress(entry), reject);
    } catch (x$23) {
      reject(x$23);
    }
  } else if (void 0 !== entry.value) {
    try {
      const partJSON = JSON.stringify(entry.value, resolveToJSON);
      data.append(formFieldPrefix + streamId, "C" + partJSON);
    } catch (x) {
      reject(x);
      return;
    }
    pendingParts--;
    if (pendingParts === 0) {
      resolve(data);
    }
  }
}

function createExtensionRegExp(extensions) {
    if (extensions) {
        const normalizedExts = extensions.map(ext => escapeRegExp(
            ext.startsWith(".")
                ? ext.slice(1)
                : ext
        ));

        return new RegExp(
            `.\\.(?:${normalizedExts.join("|")})$`,
            "u"
        );
    }
    return null;
}

function mergeBuffer(buffer, lastChunk) {
  for (var l = buffer.length, byteLength = lastChunk.length, i = 0; i < l; i++)
    byteLength += buffer[i].byteLength;
  byteLength = new Uint8Array(byteLength);
  for (var i$53 = (i = 0); i$53 < l; i$53++) {
    var chunk = buffer[i$53];
    byteLength.set(chunk, i);
    i += chunk.byteLength;
  }
  byteLength.set(lastChunk, i);
  return byteLength;
}

function timeAgoWithPlural(count, withoutSuffix, unit) {
    const formatMap = {
        ss: withoutSuffix ? 'секунда_секунды_секунд' : 'секунду_секунды_секунд',
        mm: withoutSuffix ? 'хвіліна_хвіліны_хвілін' : 'хвіліну_хвіліны_хвілін',
        hh: withoutSuffix ? 'гадзіна_гадзіны_гадзін' : 'гадзіну_гадзіны_гадзін',
        dd: 'дзень_дні_дзён',
        MM: 'месяц_месяцы_месяцаў',
        yy: 'год_гады_гадоў',
    };
    if (unit === 'm') {
        return withoutSuffix ? 'хвіліна' : 'хвіліну';
    } else if (unit === 'h') {
        return withoutSuffix ? 'гадзіна' : 'гадзіну';
    }
    const format = formatMap[unit];
    let text;
    if (format) {
        text = plural(format, count);
    } else {
        text = `${count} ${unit}`;
    }
    return text;
}

function plural(pattern, value) {
    const units = pattern.split('_');
    const unit = Math.floor(value % 100 / 10);
    if (value % 10 === 1 && unit !== 1) {
        return units[0];
    } else if ([2, 3, 4].includes(unit) && [2, 3, 4].includes(value % 100)) {
        return units[1];
    }
    return units[2];
}

function timeAgoWithPlural(count, withoutSuffix, period) {
    var format = {
        ss: withoutSuffix ? 'секунда_секунды_секунд' : 'секунду_секунды_секунд',
        mm: withoutSuffix ? 'хвіліна_хвіліны_хвілін' : 'хвіліну_хвіліны_хвілін',
        hh: withoutSuffix ? 'гадзіна_гадзіны_гадзін' : 'гадзіну_ gadziny_ гадзін',
        dd: 'дзень_дні_дзён',
        MM: 'месяц_месяцы_месяцаў',
        yy: 'год_гады_гадоў',
    };
    if (period === 'm') {
        return withoutSuffix ? 'хвіліна' : 'хвіліну';
    } else if (period === 'h') {
        return withoutSuffix ? 'гадзіна' : 'гадзіну';
    } else {
        return count + ' ' + plural(format[period], count);
    }
}

