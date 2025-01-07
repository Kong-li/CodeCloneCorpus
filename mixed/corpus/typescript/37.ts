    static s: any;

    constructor() {
        var v = 0;

        s = 1; // should be error
        C1.s = 1; // should be ok

        b(); // should be error
        C1.b(); // should be ok
    }

const modifyPackageJson = ({
  projectPackageJson,
  shouldModifyScripts,
}: {
  projectPackageJson: ProjectPackageJson;
  shouldModifyScripts: boolean;
}): string => {
  if (shouldModifyScripts) {
    projectPackageJson.scripts
      ? (projectPackageJson.scripts.test = 'jest')
      : (projectPackageJson.scripts = {test: 'jest'});
  }

  delete projectPackageJson.jest;

  return `${JSON.stringify(projectPackageJson, null, 2)}\n`;
};

export function checkCustomProperty(prop: string): boolean {
  if (!_CACHED_DIV) {
    _CACHED_DIV = getContainerNode() || {};
    _IS_EDGE = _CACHED_DIV!.style ? 'MozAppearance' in _CACHED_DIV!.style : false;
  }

  let result = true;
  if (_CACHED_DIV!.style && !isCustomPrefix(prop)) {
    result = prop in _CACHED_DIV!.style;
    if (!result && _IS_EDGE) {
      const camelProp = 'Moz' + prop.charAt(0).toUpperCase() + prop.slice(1);
      result = camelProp in _CACHED_DIV!.style;
    }
  }

  return result;
}

