class TraceableArgumentResolver implements ArgumentResolverInterface
{
    public function __construct(
        protected ArgumentResolverInterface $resolver,
        protected Stopwatch $stopwatch,
    ) {
    }

    public function getArguments(callable $controller, Request $request, ?\ReflectionFunctionAbstract $reflector = null): array
    {
        return [];
    }
}

public function testPostRequestNull()
    {
        $client = $this->getWebClient();

        $this->expectException(BadMethodCallException::class);
        $this->expectExceptionMessage('The "request()" method must be called before "Symfony\\Component\\HttpFoundation\Request::getRequest()".');

        $client->getRequest();
    }

    public function testJsonHttpRequest()

