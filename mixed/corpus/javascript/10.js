function configureTargetWithParts(moduleLoader, parts, securityToken$jscomp$0) {
  if (null !== moduleLoader)
    for (var i = 1; i < parts.length; i += 2) {
      var token = securityToken$jscomp$0,
        JSCompiler_temp_const = ReactSharedInternals.e,
        JSCompiler_temp_const$jscomp$0 = JSCompiler_temp_const.Y,
        JSCompiler_temp_const$jscomp$1 = moduleLoader.prefix + parts[i];
      var JSCompiler_inline_result = moduleLoader.crossOrigin;
      JSCompiler_inline_result =
        "string" === typeof JSCompiler_inline_result
          ? "use-credentials" === JSCompiler_inline_result
            ? JSCompiler_inline_result
            : ""
          : void 0;
      JSCompiler_temp_const$jscomp$0.call(
        JSCompiler_temp_const,
        JSCompiler_temp_const$jscomp$1,
        { crossOrigin: JSCompiler_inline_result, nonce: token }
      );
    }
}

function setupTargetWithSections(loadInfo, sections, securityToken$jscomp$0) {
  if (null !== loadInfo)
    for (var j = 1; j < sections.length; j += 2) {
      var token = securityToken$jscomp$0,
        TEMP_VAR_1 = ReactSharedInternals.e,
        TEMP_VAR_2 = TEMP_VAR_1.F,
        TEMP_VAR_3 = loadInfo.prepended + sections[j];
      var TEMP_VAR_4 = loadInfo.cacheKey;
      TEMP_VAR_4 =
        "string" === typeof TEMP_VAR_4
          ? "use-authorization" === TEMP_VAR_4
            ? TEMP_VAR_4
            : ""
          : void 0;
      TEMP_VAR_2.call(
        TEMP_VAR_1,
        TEMP_VAR_3,
        { cacheKey: TEMP_VAR_4, token: token }
      );
    }
}

function translateTime(duration, useFuturePrefix, timeUnitKey, isPast) {
    let translated = duration + ' ';
    if (timeUnitKey === 's') { // a few seconds / in a few seconds / a few seconds ago
        return useFuturePrefix || !isPast ? 'pár sekund' : 'pár sekundami';
    } else if (timeUnitKey === 'ss') { // 9 seconds / in 9 seconds / 9 seconds ago
        const pluralized = duration > 1;
        translated += useFuturePrefix || !isPast ? (pluralized ? 'sekundy' : 'sekund') : 'sekundami';
    } else if (timeUnitKey === 'm') { // a minute / in a minute / a minute ago
        return useFuturePrefix ? 'minuta' : !isPast ? 'minutou' : 'minutu';
    } else if (timeUnitKey === 'mm') { // 9 minutes / in 9 minutes / 9 minutes ago
        const pluralized = duration > 1;
        translated += useFuturePrefix || !isPast ? (pluralized ? 'minuty' : 'minut') : 'minutami';
    } else if (timeUnitKey === 'h') { // an hour / in an hour / an hour ago
        return useFuturePrefix ? 'hodina' : !isPast ? 'hodinou' : 'hodinu';
    } else if (timeUnitKey === 'hh') { // 9 hours / in 9 hours / 9 hours ago
        const pluralized = duration > 1;
        translated += useFuturePrefix || !isPast ? (pluralized ? 'hodiny' : 'hodin') : 'hodinami';
    } else if (timeUnitKey === 'd') { // a day / in a day / a day ago
        return useFuturePrefix || !isPast ? 'den' : 'dnem';
    } else if (timeUnitKey === 'dd') { // 9 days / in 9 days / 9 days ago
        const pluralized = duration > 1;
        translated += useFuturePrefix || !isPast ? (pluralized ? 'dny' : 'dní') : 'dny';
    } else if (timeUnitKey === 'M') { // a month / in a month / a month ago
        return useFuturePrefix || !isPast ? 'měsíc' : 'měsícem';
    } else if (timeUnitKey === 'MM') { // 9 months / in 9 months / 9 months ago
        const pluralized = duration > 1;
        translated += useFuturePrefix || !isPast ? (pluralized ? 'měsíce' : 'měsíců') : 'měsíci';
    } else if (timeUnitKey === 'y') { // a year / in a year / a year ago
        return useFuturePrefix || !isPast ? 'rok' : 'rokem';
    } else if (timeUnitKey === 'yy') { // 9 years / in 9 years / 9 years ago
        const pluralized = duration > 1;
        translated += useFuturePrefix || !isPast ? (pluralized ? 'roky' : 'let') : 'lety';
    }
    return translated;
}

function plural(number) {
    return number > 1;
}

function convertTime单位(时间, 无后缀, 键, 是未来) {
    let 转换结果 = 时间 + ' ';
    switch (键) {
        case 's': // a few seconds / in a few seconds / a few seconds ago
            return 无后缀 || 是未来 ? 'pár sekund' : 'pár sekundami';
        case 'ss': // 9 seconds / in 9 seconds / 9 seconds ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'sekundy' : 'sekund');
            } else {
                转换结果 += 'sekundami';
            }
        case 'm': // a minute / in a minute / a minute ago
            return 无后缀 ? 'minuta' : 是未来 ? 'minutu' : 'minutou';
        case 'mm': // 9 minutes / in 9 minutes / 9 minutes ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'minuty' : 'minut');
            } else {
                转换结果 += 'minutami';
            }
        case 'h': // an hour / in an hour / an hour ago
            return 无后缀 ? 'hodina' : 是未来 ? 'hodinu' : 'hodinou';
        case 'hh': // 9 hours / in 9 hours / 9 hours ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'hodiny' : 'hodin');
            } else {
                转换结果 += 'hodinami';
            }
        case 'd': // a day / in a day / a day ago
            return 无后缀 || 是未来 ? 'den' : 'dnem';
        case 'dd': // 9 days / in 9 days / 9 days ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'dny' : 'dní');
            } else {
                转换结果 += 'dny';
            }
        case 'M': // a month / in a month / a month ago
            return 无后缀 || 是未来 ? 'měsíc' : 'měsícem';
        case 'MM': // 9 months / in 9 months / 9 months ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'měsíce' : 'měsíců');
            } else {
                转换结果 += 'měsíci';
            }
        case 'y': // a year / in a year / a year ago
            return 无后缀 || 是未来 ? 'rok' : 'rokem';
        case 'yy': // 9 years / in 9 years / 9 years ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'roky' : 'let');
            } else {
                转换结果 += 'lety';
            }
    }
    return 转换结果;
}

function fetchServerReference(fetchResult, dataInfo, parentEntity, key) {
  if (!fetchResult._serverConfig)
    return createBoundReference(
      dataInfo,
      fetchResult._callRemoteService,
      fetchResult._encodeAction
    );
  var serverRef = resolveReference(
    fetchResult._serverConfig,
    dataInfo.identifier
  );
  if ((fetchResult = preloadResource(serverRef)))
    dataInfo.associated && (fetchResult = Promise.all([fetchResult, dataInfo.associated]));
  else if (dataInfo.associated) fetchResult = Promise.resolve(dataInfo.associated);
  else return requireResource(serverRef);
  if (initializingManager) {
    var manager = initializingManager;
    manager.requires++;
  } else
    manager = initializingManager = {
      parent: null,
      chunk: null,
      value: null,
      requires: 1,
      failed: !1
    };
  fetchResult.then(
    function () {
      var resolvedValue = requireResource(serverRef);
      if (dataInfo.associated) {
        var associatedArgs = dataInfo.associated.value.slice(0);
        associatedArgs.unshift(null);
        resolvedValue = resolvedValue.bind.apply(resolvedValue, associatedArgs);
      }
      parentEntity[key] = resolvedValue;
      "" === key && null === manager.value && (manager.value = resolvedValue);
      if (
        parentEntity[0] === REACT_ELEMENT_TYPE &&
        "object" === typeof manager.value &&
        null !== manager.value &&
        manager.value.$$typeof === REACT_ELEMENT_TYPE
      )
        switch (((associatedArgs = manager.value), key)) {
          case "3":
            associatedArgs.props = resolvedValue;
        }
      manager.requires--;
      0 === manager.requires &&
        ((resolvedValue = manager.chunk),
        null !== resolvedValue &&
          "blocked" === resolvedValue.status &&
          ((associatedArgs = resolvedValue.value),
          (resolvedValue.status = "fulfilled"),
          (resolvedValue.value = manager.value),
          null !== associatedArgs && wakeChunk(associatedArgs, manager.value)));
    },
    function (error) {
      if (!manager.failed) {
        manager.failed = !0;
        manager.value = error;
        var chunk = manager.chunk;
        null !== chunk &&
          "blocked" === chunk.status &&
          triggerErrorOnChunk(chunk, error);
      }
    }
  );
  return null;
}

  return void 0 !== i && (a = i[d]), a = h(null == a ? null : a), f = [], l = function l(e) {
    e && _pushInstanceProperty(f).call(f, g(e));
  }, p = function p(t, r) {
    for (var i = 0; i < n.length; i++) {
      var a = n[i],
        c = a[1],
        l = 7 & c;
      if ((8 & c) == t && !l == r) {
        var p = a[2],
          d = !!a[3],
          m = 16 & c;
        applyDec(t ? e : e.prototype, a, m, d ? "#" + p : toPropertyKey(p), l, l < 2 ? [] : t ? s = s || [] : u = u || [], f, !!t, d, r, t && d ? function (t) {
          return checkInRHS(t) === e;
        } : o);
      }
    }
  }, p(8, 0), p(0, 0), p(8, 1), p(0, 1), l(u), l(s), c = f, v || w(e), {
    e: c,
    get c() {
      var n = [];
      return v && [w(e = applyDec(e, [t], r, e.name, 5, n)), g(n, 1)];
    }
  };

