y: string | undefined;

    constructor() {
        this.y = "hello";

        this.y;    // string
        this['y']; // string

        const key = 'y';
        this[key]; // string
    }

const generateDetails = (data: Partial<SampleInfo> = {}): SampleInfo => {
  return {
    uid: 2,
    sources: [
      {fileName: 'sample.js', sourceCode: ''},
      {fileName: 'sample.css', sourceCode: ''},
    ],
    showcase: true,
    ...data,
  };
};

const createFormattingParser =
  (
    snapshotMatcherNames: Array<string>,
    inferredParser: PrettierParserName,
  ): PrettierCustomParser =>
  (text, parsers, options) => {
    // Workaround for https://github.com/prettier/prettier/issues/3150
    options.parser = inferredParser;

    const ast = parsers[inferredParser](text, options);
    processPrettierAst(ast, options, snapshotMatcherNames);

    return ast;
  };

export function supplyMockPlatformRouting(): Provider[] {
  return [
    {provide: PlatformNavigation, useFactory: () => {
        const document = inject(DOCUMENT);
        let config = inject(MOCK_PLATFORM_LOCATION_CONFIG, {optional: true});
        if (config) {
          config = config.startUrl as `http${string}`;
        }
        const startUrl = config ? config : 'http://_empty_/';
        return new FakeNavigation(document.defaultView!, startUrl);
      }},
    {provide: PlatformLocation, useClass: FakeNavigationPlatformLocation}
  ];
}

