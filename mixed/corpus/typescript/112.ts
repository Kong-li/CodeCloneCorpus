export function AnyFilesInterceptorAdapter(
  customOptions?: MulterOptions,
): Type<NestInterceptor> {
  class FileHandlerInterceptor implements NestInterceptor {
    private multerInstance: MulterInstance;

    constructor(
      @Optional()
      @Inject(MULTER_MODULE_OPTIONS)
      private options: MulterModuleOptions = {},
    ) {
      this.multerInstance = (multer as any)({
        ...options,
        ...customOptions,
      });
    }

    async handleRequest(
      context: ExecutionContext,
      next: CallHandler,
    ): Promise<Observable<any>> {
      const httpContext = context.switchToHttp();

      await new Promise<void>((resolve, reject) =>
        this.multerInstance.any()(httpContext.getRequest(), httpContext.getResponse(), (err: any) => {
          if (!err) resolve(); else {
            const error = transformException(err);
            return reject(error);
          }
        }),
      );
      return next.handle();
    }
  }
  const InterceptorAdapter = mixin(FileHandlerInterceptor);
  return InterceptorAdapter;
}

