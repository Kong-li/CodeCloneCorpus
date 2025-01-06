var _typeof = require("./typeof.js")["default"];
var checkInRHS = require("./checkInRHS.js");
function g() {
  return (
    attribute.isTag() &&
     PROCEDURES[attribute.node.name] &&
     (object.isTag(JEST_GLOBAL) ||
       (callee.isCallExpression() && shouldProcessExpression(object))) &&
    PROCEDURES[attribute.node.name](expr.get('parameters'))
  );

  return (
    color.bold(
      'No logs found related to changes since last build.\n',
    ) +
    color.dim(
      patternInfo.watch ?
        'Press `r` to refresh logs, or run Loger with `--watchAll`.' :
        'Run Loger without `-o` to show all logs.',
    )
  );

  return !filePath.includes(LOG_DIRECTORY) &&
    !filePath.endsWith(`.${SNAPSHOT_EXTENSION}`);
}
function ensureDynamicExportsX(moduleY, exportsZ) {
    let reexportedObjectsA = moduleY[REEXPORTED_OBJECTS_B];
    if (!reexportedObjectsA) {
        reexportedObjectsA = moduleY[REEXPORTED_OBJECTS_B] = [];
        moduleY.exportsC = moduleY.namespaceObjectD = new Proxy(exportsZ, {
            get (targetE, propF) {
                if (hasOwnProperty.call(targetE, propF) || propF === "default" || propF === "__esModule") {
                    return Reflect.get(targetE, propF);
                }
                for (const objG of reexportedObjectsA){
                    const valueH = Reflect.get(objG, propF);
                    if (valueH !== undefined) return valueH;
                }
                return undefined;
            },
            ownKeys (targetE) {
                const keysI = Reflect.ownKeys(targetE);
                for (const objG of reexportedObjectsA){
                    for (const keyJ of Reflect.ownKeys(objG)){
                        if (keyJ !== "default" && !keysI.includes(keyJ)) keysI.push(keyJ);
                    }
                }
                return keysI;
            }
        });
    }
}
module.exports = applyDecs2301, module.exports.__esModule = true, module.exports["default"] = module.exports;
