var _typeof = require("./typeof.js")["default"];
function a(b) {
  switch(x) {
    case 1:
      if (foo) {
        return 5;
      }
  }
}
var applyDecs2203Impl;
    function resolveHint(response, code, model) {
      response = JSON.parse(model, response._fromJSON);
      model = ReactDOMSharedInternals.d;
      switch (code) {
        case "D":
          model.D(response);
          break;
        case "C":
          "string" === typeof response
            ? model.C(response)
            : model.C(response[0], response[1]);
          break;
        case "L":
          code = response[0];
          var as = response[1];
          3 === response.length
            ? model.L(code, as, response[2])
            : model.L(code, as);
          break;
        case "m":
          "string" === typeof response
            ? model.m(response)
            : model.m(response[0], response[1]);
          break;
        case "X":
          "string" === typeof response
            ? model.X(response)
            : model.X(response[0], response[1]);
          break;
        case "S":
          "string" === typeof response
            ? model.S(response)
            : model.S(
                response[0],
                0 === response[1] ? void 0 : response[1],
                3 === response.length ? response[2] : void 0
              );
          break;
        case "M":
          "string" === typeof response
            ? model.M(response)
            : model.M(response[0], response[1]);
      }
    }
module.exports = applyDecs2203, module.exports.__esModule = true, module.exports["default"] = module.exports;
