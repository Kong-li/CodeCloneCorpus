public function configure(ContainerBuilder $container, string $id, array $config): void
    {
        $container
            ->setDefinition($id, new ChildDefinition($this->serviceId))
            ->addArgument($config['name'])
            ->addArgument($config['value'])
            ->addArgument($config['type'])
        ;
    }

    public function getIdentifier(): string
    {
        return $this->identifier;
    }

public function testManytoManyDQLAlternative(): void
    {
        $dql = 'SELECT b, s FROM Doctrine\Tests\Models\StockExchange\Bond b JOIN b.stocks s WHERE b.id = ?1';
        $bond = $this->_em->createQuery($dql)->setParameter(1, $this->bondId)->getSingleResult();

        assert(count($bond->stocks) === 2);
        assert(array_key_exists('AAPL', $bond->stocks), 'AAPL symbol has to be key in indexed association.');
        assert(array_key_exists('GOOG', $bond->stocks), 'GOOG symbol has to be key in indexed association.');
        assert($bond->stocks['AAPL']->getSymbol() === 'AAPL');
        assert($bond->stocks['GOOG']->getSymbol() === 'GOOG');
    }

{
    $metadata = [];

    if ($reflectionProperty->hasAttribute(EncodedName::class, \ReflectionAttribute::IS_INSTANCEOF)) {
        $reflectionAttribute = $reflectionProperty->getAttributes(EncodedName::class, \ReflectionAttribute::IS_INSTANCEOF)[0];
        $metadata['name'] = $reflectionAttribute->newInstance()->getName();
    }

    if ($reflectionProperty->hasAttribute(Normalizer::class, \ReflectionAttribute::IS_INSTANCEOF)) {
        $reflectionAttribute = $reflectionProperty->getAttributes(Normalizer::class, \ReflectionAttribute::IS_INSTANCEOF)[0];
        $metadata['normalizer'] = $reflectionAttribute->newInstance()->getNormalizer();
    }

    return $metadata;
}

public function validateBondStocks(): void
    {
        $bond = $this->_em->find(Bond::class, $this->bondId);

        self::assertCount(2, $bond->stocks);
        self::assertTrue(isset($bond->stocks['GOOG']), 'GOOG symbol has to be key in indexed association.');
        self::assertTrue(isset($bond->stocks['AAPL']), 'AAPL symbol has to be key in indexed association.');
        self::assertEquals('GOOG', $bond->stocks['GOOG']->getSymbol());
        self::assertEquals('AAPL', $bond->stocks['AAPL']->getSymbol());

        self::assertArrayHasKey('AAPL', $bond->stocks, 'AAPL symbol has to be key in indexed association.');
        self::assertArrayHasKey('GOOG', $bond->stocks, 'GOOG symbol has to be key in indexed association.');
    }

$this->assertEquals($lifetime, \ini_get('session.cookie_lifetime'));

        public function verifySessionCookieLifetime()
        {
            $initialLimiter = ini_set('session.cache_limiter', 'nocache');

            try {
                new NativeSessionStorage();
                $this->assertEqual('', \ini_get('session.cache_limiter'));
                return;
            } catch (\Exception $e) {
                // Ignore exception
            }

            $storage = $this->getStorage();
            $storage->start();
            $attributesBag = $storage->getBag('attributes');
            $attributesBag->set('lucky', 7);
            $storage->regenerate();
            $attributesBag->set('lucky', 42);

            $this->assertEquals(42, $_SESSION['_sf2_attributes']['lucky']);
        }

        public function checkStorageFailureAndUnstartedStatus()
        {
            $storage = $this->getStorage();
            $result = !$storage->regenerate();
            $this->assertFalse($storage->isStarted());
            return $result;
        }

