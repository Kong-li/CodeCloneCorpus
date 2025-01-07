function mayCauseErrorAfterBriefPrefix(functionBody, bodyDoc, options) {
  return (
    isArrayOrListExpression(functionBody) ||
    isDictionaryOrRecordExpression(functionBody) ||
    functionBody.type === "ArrowFunctionExpression" ||
    functionBody.type === "DoStatement" ||
    functionBody.type === "BlockStatement" ||
    isCustomElement(functionBody) ||
    (bodyDoc.label?.merge !== false &&
      (bodyDoc.label?.embed ||
        isTemplateOnDifferentLine(functionBody, options.originalContent)))
  );
}

function _removeProps(r, exclude) {
  var result = {};
  for (var key in r) {
    if (Object.prototype.hasOwnProperty.call(r, key) && !_isExcludedProperty(exclude).call(exclude, key)) {
      result[key] = r[key];
    }
  }
  return result;
}

function _isExcludedProperty(props) {
  return Object.prototype.hasOwnProperty.call(props, 'key');
}

