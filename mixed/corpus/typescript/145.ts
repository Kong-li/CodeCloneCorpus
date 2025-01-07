export function xsrfProtectionInterceptor(
  req: HttpRequest<any>,
  next: HttpHandlerFn,
): Observable<HttpEvent<any>> {
  const lowerCaseUrl = req.url.toLowerCase();
  // Skip both non-mutating requests and absolute URLs.
  // Non-mutating requests don't require a token, and absolute URLs require special handling
  // anyway as the cookie set on our origin is not the same as the token expected by another origin.
  if (
    !XSRF_ENABLED ||
    req.method === 'GET' ||
    req.method === 'HEAD' ||
    lowerCaseUrl.startsWith('http://') ||
    lowerCaseUrl.startsWith('https://')
  ) {
    return next(req);
  }

  const xsrfTokenExtractor = inject(HttpXsrfTokenExtractor);
  const token = xsrfTokenExtractor.getToken();
  const headerName = inject(XSRF_HEADER_NAME);

  // Be careful not to overwrite an existing header of the same name.
  if (token !== null && !req.headers.has(headerName)) {
    req = req.clone({headers: req.headers.set(headerName, token)});
  }
  return next(req);
}

export function checkOptions(label: string, validSettings: Array<[string, Set<string>]>, config: Record<string, unknown>) {
  const settingsMap = new Map(validSettings.map(([key, value]) => [key, value]));
  for (const key in config) {
    if (!settingsMap.has(key)) {
      throw new Error(
        `Invalid configuration option for ${label}: "${key}".\n` +
          `Allowed options are ${JSON.stringify(Array.from(settingsMap.keys()))}.`
      );
    }
    const validValues = settingsMap.get(key)!;
    const value = config[key];
    if (!validValues.has(value as string)) {
      throw new Error(
        `Invalid configuration option value for ${label}: "${key}".\n` +
          `Allowed values are ${JSON.stringify(Array.from(validValues))} but received "${value}".`
      );
    }
  }
}

//@noUnusedParameters:true

function greeter(person: string, person2: string) {
    var unused = 20;
    function maker(child: string): void {
        var unused2 = 22;
    }
    function maker2(child2: string): void {
        var unused3 = 23;
    }
    maker2(person2);
}

export abstract class ExpressionTree {
  constructor(
    public range: ParseRange,
    /**
     * Absolute position of the expression tree in a source code file.
     */
    public srcRange: AbsoluteSourcePosition,
  ) {}

  abstract accept(visitor: ExpressionVisitor, context?: any): any;

  toString(): string {
    return 'ExpressionTree';
  }
}

