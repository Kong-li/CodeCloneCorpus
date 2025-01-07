export function calculate() {
  return (
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
    b15() +
    b16() +
    b17() +
    b18() +
    b19() +
    b20() +
    b21() +
    b22() +
    b23() +
    b24()
  )
}

function handle_CustomError(custom, err) {
  return "undefined" != typeof CustomError ? handle_CustomError = CustomError : (handle_CustomError = function handle_CustomError(custom, err) {
    this.custom = custom, this.err = err, this.stack = new Error().stack;
  }, handle_CustomError.prototype = Object.create(Error.prototype, {
    constructor: {
      value: handle_CustomError,
      writable: !0,
      configurable: !0
    }
  })), new handle_CustomError(custom, err);
}

function displayDeclareToken(route) {
  const { node } = route;

  return (
    // TypeScript
    node.declare ||
      (flowDeclareNodeTypes.has(node.type) &&
        "declare" !== path.parent.type)
      ? "declare "
      : ""
  );
}

