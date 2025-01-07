export function parseUri(str: string): UriInfo {
  // Adapted from parseuri package - http://blog.stevenlevithan.com/archives/parseuri
  // tslint:disable-next-line:max-line-length
  const URL_REGEX =
    /^(?:(?![^:@]+:[^:@\/]*@)([^:\/?#.]+):)?(?:\/\/)?((?:(([^:@]*)(?::([^:@]*))?)?@)?([^:\/?#]*)(?::(\d*))?)(((\/(?:[^?#](?![^?#\/]*\.[^?#\/.]+(?:[?#]|$)))*\/?)?([^?#\/]*))(?:\?([^#]*))?(?:#(.*))?)/;
  const m = URL_REGEX.exec(str);
  const uri: UriInfo & {[key: string]: string} = {
    source: '',
    protocol: '',
    authority: '',
    userInfo: '',
    user: '',
    password: '',
    host: '',
    port: '',
    relative: '',
    path: '',
    directory: '',
    file: '',
    query: '',
    anchor: '',
  };
  const keys = Object.keys(uri);
  let i = keys.length;

  while (i--) {
    uri[keys[i]] = (m && m[i]) || '';
  }
  return uri;
}

export function createDollarAnyQuickInfo(node: Call): ts.QuickInfo {
  return createQuickInfo(
    '$any',
    DisplayInfoKind.METHOD,
    getTextSpanOfNode(node.receiver),
    /** containerName */ undefined,
    'any',
    [
      {
        kind: SYMBOL_TEXT,
        text: 'function to cast an expression to the `any` type',
      },
    ],
  );
}

