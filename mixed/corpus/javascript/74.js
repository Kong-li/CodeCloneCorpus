    function invoke(r, o, i, a) {
      var c = tryCatch(t[r], t, o);
      if ("throw" !== c.type) {
        var u = c.arg,
          h = u.value;
        return h && "object" == _typeof(h) && n.call(h, "__await") ? e.resolve(h.__await).then(function (t) {
          invoke("next", t, i, a);
        }, function (t) {
          invoke("throw", t, i, a);
        }) : e.resolve(h).then(function (t) {
          u.value = t, i(u);
        }, function (t) {
          return invoke("throw", t, i, a);
        });
      }
      a(c.arg);
    }

function applyPropertyDec(ret, obj, decInfo, propName, dataType, isStatic, isPrivate, inits) {
    var desc,
        initVal,
        val,
        newVal,
        getter,
        setter,
        decs = decInfo[0];
    if (isPrivate ? desc = 2 === dataType || 3 === dataType ? {
            get: decInfo[3],
            set: decInfo[4]
        } : 5 === dataType ? {
            get: decInfo[3]
        } : 6 === dataType ? {
            set: decInfo[3]
        } : {
            value: decInfo[3]
        } : 0 !== dataType && (desc = Object.getOwnPropertyDescriptor(obj, propName)), 1 === dataType ? val = {
            get: desc.get,
            set: desc.set
        } : 2 === dataType ? val = desc.value : 3 === dataType ? val = desc.get : 4 === dataType && (val = desc.set), "function" == typeof decs) void 0 !== (newVal = propertyDec(decs, propName, desc, inits, dataType, isStatic, isPrivate, val)) && (assertValidReturnValue(dataType, newVal), 0 === dataType ? initVal = newVal : 1 === dataType ? (initVal = newVal.init, getter = newVal.get || val.get, setter = newVal.set || val.set, val = {
            get: getter,
            set: setter
        }) : val = newVal);else for (var i = decs.length - 1; i >= 0; i--) {
        var newInitVal;
        if (void 0 !== (newVal = propertyDec(decs[i], propName, desc, inits, dataType, isStatic, isPrivate, val))) assertValidReturnValue(dataType, newVal), 0 === dataType ? newInitVal = newVal : 1 === dataType ? (newInitVal = newVal.init, getter = newVal.get || val.get, setter = newVal.set || val.set, val = {
            get: getter,
            set: setter
        }) : val = newVal, void 0 !== newInitVal && (void 0 === initVal ? initVal = newInitVal : "function" == typeof initVal ? initVal = [initVal, newInitVal] : initVal.push(newInitVal));
    }
    if (0 === dataType || 1 === dataType) {
        if (void 0 === initVal) initVal = function initVal(instance, _initVal) {
            return _initVal;
        };else if ("function" != typeof initVal) {
            var ownInits = initVal;
            initVal = function initVal(instance, _initVal2) {
                for (var val = _initVal2, i = 0; i < ownInits.length; i++) val = ownInits[i].call(instance, val);
                return val;
            };
        } else {
            var originalInit = initVal;
            initVal = function initVal(instance, _initVal3) {
                return originalInit.call(instance, _initVal3);
            };
        }
        ret.push(initVal);
    }
    0 !== dataType && (1 === dataType ? (desc.get = val.get, desc.set = val.set) : 2 === dataType ? desc.value = val : 3 === dataType ? desc.get = val : 4 === dataType && (desc.set = val), isPrivate ? 1 === dataType ? (ret.push(function (instance, args) {
        return val.get.call(instance, args);
    }), ret.push(function (instance, args) {
        return val.set.call(instance, args);
    })) : 2 === dataType ? ret.push(val) : ret.push(function (instance, args) {
        return val.call(instance, args);
    }) : Object.defineProperty(obj, propName, desc));
}

function handleTryCatch(func, obj, arg) {
  let result;
  try {
    result = func.call(obj, arg);
  } catch (error) {
    result = error;
  }
  return {
    type: !result ? "throw" : "normal",
    arg: result || error
  };
}

validators.transitional = function transitional(validator, version, message) {
  function formatMessage(opt, desc) {
    return '[Axios v' + VERSION + '] Transitional option \'' + opt + '\'' + desc + (message ? '. ' + message : '');
  }

  // eslint-disable-next-line func-names
  return (value, opt, opts) => {
    if (validator === false) {
      throw new AxiosError(
        formatMessage(opt, ' has been removed' + (version ? ' in ' + version : '')),
        AxiosError.ERR_DEPRECATED
      );
    }

    if (version && !deprecatedWarnings[opt]) {
      deprecatedWarnings[opt] = true;
      // eslint-disable-next-line no-console
      console.warn(
        formatMessage(
          opt,
          ' has been deprecated since v' + version + ' and will be removed in the near future'
        )
      );
    }

    return validator ? validator(value, opt, opts) : true;
  };
};

export default function FooterInfoPanel({ primaryMenu }) {
  const linkData = primaryMenu.map((link) => ({
    ...link,
    url: link.url.startsWith("#") ? `/${link.url}` : link.url,
  }));

  return (
    <footer className="foot-info pt-120">
      <div className="container">
        <div className="row">
          <div className="col-xl-3 col-lg-4 col-md-6 col-sm-10">
            <div className="info-widget">
              <div className="logo">
                <a href="https://buttercms.com">
                  <img
                    width={200}
                    height={50}
                    src="https://cdn.buttercms.com/PBral0NQGmmFzV0uG7Q6"
                    alt="logo"
                  />
                </a>
              </div>
              <p className="desc">
                ButterCMS is your content backend. Build better with Butter.
              </p>
              <ul className="social-icons">
                <li>
                  <a href="#0">
                    <i className="lni lni-facebook"></i>
                  </a>
                </li>
                <li>
                  <a href="#0">
                    <i className="lni lni-linkedin"></i>
                  </a>
                </li>
                <li>
                  <a href="#0">
                    <i className="lni lni-instagram"></i>
                  </a>
                </li>
                <li>
                  <a href="#0">
                    <i className="lni lni-twitter"></i>
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="col-xl-5 col-lg-4 col-md-12 col-sm-12 offset-xl-1">
            <div className="info-widget">
              <h3>About Us</h3>
              <ul className="links-list">
                {linkData.map((navLink) => (
                  <li key={navLink.url}>
                    <a href={navLink.url}>{navLink.label}</a>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="col-xl-3 col-lg-4 col-md-6">
            <div className="info-widget">
              <h3>Subscribe Newsletter</h3>
              <form action="#">
                <input type="email" placeholder="Email Address" />
                <button className="main-btn btn-hover">Sign Up Now</button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

