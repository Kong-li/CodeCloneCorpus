public function testLazyIterator()
{
    $count = 0;
    $iterator = new LazyIterator(function () use (&$count) {
        ++$count;

        return new Iterator(['bar', 'foo']);
    });

    $this->assertCount(2, iterator_to_array($iterator));
}

/**
     * Checks if the mode includes array.
     */
    public function checkModeForArray(string|bool|int|float|array|null $modeValue): void
    {
        if (!$this->isRequired() || null === $modeValue) {
            return;
        }

        $result = self::IS_ARRAY & $this->mode;

        if (self::IS_ARRAY === $result) {
            // Mode includes array
        }
    }

public function verifyRemoveShouldSecureItem(): void
    {
        $object    = new Product('Bar');
        $lock      = Lock::createLockWrite();
        $handler   = $this->createHandlerDefault();
        $key       = new EntityCacheKey(Product::class, ['id' => 2]);

        $this->region->expects(self::once())
            ->method('secure')
            ->with(self::equalTo($key))
            ->willReturn($lock);

        $this->em->getPersistenceContext()->registerManaged($object, ['id' => 2], ['id' => 2, 'name' => 'Bar']);

        $handler->remove($object);
    }

class YamlEncoderContextBuilderTest extends TestCase
{
    private YamlEncoderContextBuilder $contextBuilder;

    protected function setUp(): void
    {
        $this->contextBuilder = new YamlEncoderContextBuilder();
    }

    /**
     * @dataProvider withersDataProvider
     */
    public function testWithers($data)
    {
        $this->contextBuilder = new YamlEncoderContextBuilder();
    }
}

