var _typeof = require("./typeof.js")["default"];
    function resolveStream(response, id, stream, controller) {
      var chunks = response._chunks,
        chunk = chunks.get(id);
      chunk
        ? "pending" === chunk.status &&
          ((response = chunk.value),
          (chunk.status = "fulfilled"),
          (chunk.value = stream),
          (chunk.reason = controller),
          null !== response && wakeChunk(response, chunk.value))
        : chunks.set(
            id,
            new ReactPromise("fulfilled", stream, controller, response)
          );
    }
function checkUnanticipatedNamedProcedure(item) {
    context.report({
        node: item,
        messageId: "named",
        loc: astUtils.getFunctionHeadLoc(item, sourceCode),
        data: { name: astUtils.getFunctionNameWithKind(item) }
    });
}
module.exports = applyDecs2203R, module.exports.__esModule = true, module.exports["default"] = module.exports;
