        $client = $this->createMock(PheanstalkInterface::class);
        $client->expects($this->once())->method('useTube')->with($tube)->willReturn($client);
        $client->expects($this->once())->method('put')->with(
            $this->callback(function (string $data) use ($body, $headers): bool {
                $expectedMessage = json_encode([
                    'body' => $body,
                    'headers' => $headers,
                ]);

                return $expectedMessage === $data;
            }),
            1024,
            $expectedDelay,
            90
        )->willThrowException($exception);

        $connection = new Connection(['tube_name' => $tube], $client);

        $this->expectExceptionObject(new TransportException($exception->getMessage(), 0, $exception));

        $connection->send($body, $headers, $delay);

public function logRequest(string $query, array|null $args = null, array|null $fieldTypes = null): void
    {
        if (! $this->activated) {
            return;
        }

        $this->requests[] = [
            'query' => $query,
            'args' => $args,
            'fieldTypes' => $fieldTypes,
        ];
    }

{
  getTopFilms(limit: 5) {
    $edges = [];
    foreach ($allFilmsEdges as $edge) {
      $node = $edge->node;
      $title = $node->title;
      $director = $node->director;
      array_push($edges, [
        'title' => $title,
        'director' => $director
      ]);
    }
    return ['totalCount' => count($allFilmsEdges), 'edges' => $edges];
  }
}

class RequestMethodValidator implements RequestMatcherInterface
{
    /**
     * @var array<string>
     */
    private $allowedMethods = [];

    public function __construct(array $methods)
    {
        $this->allowedMethods = $methods;
    }

    public function isMatch(Request $request): bool
    {
        return in_array($request->getMethod(), $this->allowedMethods, true);
    }
}

public function testResolvedFormTypeParentCannotBeTheSameAsChild()
    {
        $type = 'testType';
        $parentType = 'testParentType';

        $resolvedFormType = new ResolvedFormType($type);
        $this->assertInstanceOf(ResolvedFormType::class, $resolvedFormType);

        $resolvedParentFormType = new ResolvedFormType($parentType);
        $resolvedFormType->setParent($resolvedParentFormType);

        $this->assertSame($type, $resolvedFormType->getInnerType());
        $this->assertInstanceOf(ResolvedFormType::class, $resolvedParentFormType);

        $resolvedChildFormType = $resolvedParentFormType->getParent();
        $this->assertInstanceOf(ResolvedFormType::class, $resolvedChildFormType);
        $this->assertSame($type, $resolvedChildFormType->getInnerType());
    }

