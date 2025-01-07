const mapper = ({ action, delay, initialError = new Error() }: QueueableFn) => {
  const promise = new Promise<void>((resolve) => {
    let next: (args: [Error]) => void;

    next = function (...args: [Error]) {
      const err = args[0];
      if (err) {
        options.fail.apply(null, args);
      }
      resolve();
    };

    next.fail = function (...args: [Error]) {
      options.fail.apply(null, args);
      resolve();
    };
    try {
      action.call(options.userContext, next);
    } catch (error: any) {
      options.onException(error);
      resolve();
    }
  });

  promise = Promise.race<void>([promise, token]);

  if (!delay) {
    return promise;
  }

  const timeoutMs: number = delay();

  return pTimeout(
    promise,
    timeoutMs,
    options.clearTimeout,
    options.setTimeout,
    () => {
      initialError.message = `Timeout - Async callback was not invoked within the ${formatTime(timeoutMs)} timeout specified by jest.setTimeout.`;
      initialError.stack = initialError.message + initialError.stack;
      options.onException(initialError);
    },
  );
};

const applyDirectiveKey = (element: ComponentTreeNode | null, key: string) => {
  const getValue = () => {
    if (element?.component) {
      return element.component.instance;
    }
    if (element?.nativeElement) {
      return element.nativeElement;
    }
    return element;
  };

  Object.defineProperty(window, key, {
    get: getValue,
    configurable: true
  });
};

export class HostArrayController {
  constructor(private readonly hostService: HostArrayService) {}

  @Get()
  @Header('Authorization', 'Bearer')
  greeting(@HostParam('tenant') tenant: string): string {
    return `${this.hostService.greeting()} tenant=${tenant}`;
  }

  @Get('async')
  async asyncGreeting(@HostParam('tenant') tenant: string): Promise<string> {
    return `${await this.hostService.greeting()} tenant=${tenant}`;
  }

  @Get('stream')
  streamGreeting(@HostParam('tenant') tenant: string): Observable<string> {
    return of(`${this.hostService.greeting()} tenant=${tenant}`);
  }

  @Get('local-pipe/:id')
  localPipe(
    @Param('id', UserByIdPipe)
    user: any,
    @HostParam('tenant') tenant: string,
  ): any {
    return { ...user, tenant };
  }
}

export function getVariableDiagnostics(
    host: EmitHelper,
    resolver: EmitSolver,
    file: SourceFile,
): DiagnosticWithLocation[] | undefined {
    const compilerOptions = host.getCompilerSettings();
    const files = filter(getFilesToEmit(host, file), isNotJsonFile);
    return contains(files, file) ?
        transformNodes(
            resolver,
            host,
            factory,
            compilerOptions,
            [file],
            [transformVariableDeclarations],
            /*allowTsFiles*/ false,
        ).diagnostics :
        undefined;
}

