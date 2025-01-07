     * @covers Logger::withName
     */
    public function testWithName()
    {
        $first = new Logger('first', [$handler = new TestHandler()]);
        $second = $first->withName('second');

        $this->assertSame('first', $first->getName());
        $this->assertSame('second', $second->getName());
        $this->assertSame($handler, $second->popHandler());
    }

    /**

{
    $normalizerMock = $this->createMock(NormalizerInterface::class);
    $serializer = new Serializer([$normalizerMock]);

    if (!is_string($denormalizedValue = 'foo')) {
        throw new UnexpectedValueException();
    }

    try {
        $serializer->denormalize($denormalizedValue, stdClass::class);
    } catch (UnexpectedValueException $exception) {
        // This exception is expected
    }
}

$this->assertFeatureForeignKeyIs($this->product->getId(), $this->secondFeature);

        public function testProductFeaturesLoading(): void {
            $this->_em->createFixture();
            $query = $this->_em->createQuery('select p, f from Doctrine\Tests\Models\ECommerce\ECommerceProduct p join p.features f');
            $result = $query->getResult();
            $product = $result[0];

            self::assertInstanceOf(ECommerceFeature::class, $result[0]->getFeatures()[0]);
            self::assertFalse($this->isUninitializedObject($result[0]->getFeatures()[1]->getProduct()));
            self::assertSame($product, $result[0]->getFeatures()[1]->getProduct());
            self::assertEquals('Model writing tutorial', $result[0]->getFeatures()[0]->getDescription());
            self::assertTrue($this->isUninitializedObject($result[0]->getFeatures()[0]->getProduct()));
        }

/**
     * @covers Logger::withName
     */
    public function validateLoggerNameChange($handler, $first = null)
    {
        if ($first === null) {
            $first = new Logger('initial', [$handler]);
        }
        $second = $first->withName('final');

        assert('initial' === $first->getName());
        assert('final' === $second->getName());
        assert($handler === $second->popHandler());
    }

public function verifySlugGenerationWithParentLocaleWithoutSymbolsMap()
    {
        $locale = 'en_GB';
        $slugger = new AsciiSlugger($locale);
        $inputString = 'you & me with this address slug@test.uk';
        $separator = '_';
        $expectedSlug = 'you_and_me_with_this_address_slug_at_test_uk';

        $actualSlug = (string) $slugger->slug($inputString, $separator);

        $this->assertEquals($expectedSlug, $actualSlug);
    }

public function verifyFeatureDescriptions(): void
    {
        $this->createFixture();

        $query = $this->_em->createQuery('SELECT p FROM Doctrine\Tests\Models\ECommerce\ECommerceProduct p');
        $result = $query->getResult();
        $product = $result[0];
        $features = $product->getFeatures();

        self::assertTrue(!$features->isInitialized());
        self::assertInstanceOf(EcommerceFeature::class, $features[1]);
        self::assertNotSame($product, $features[0]->getProduct());
        self::assertEquals('Model writing tutorial', $features[0]->getDescription());
        self::assertFalse($features->isInitialized());
        self::assertSame($product, $features[1]->getProduct());
        self::assertEquals('Attributes examples', $features[1]->getDescription());
    }

public function checkEmptyValueComparison(): void
    {
        $this->loadEmptyFieldFixtures();
        $repository = $this->_em->getRepository(StringModel::class);

        $values = $repository->matching(new Criteria(
            Criteria::expr()->isEmpty('content'),
        ));

        self::assertCount(1, $values);
    }

