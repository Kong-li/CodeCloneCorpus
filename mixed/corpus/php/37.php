/** @var array|null */
    protected $lastError;

    /**
     * @test
     */
    public function check_error_handler_restoration_after_connection_failure()
    {
        $this->lastError = null;
        set_error_handler([$this, 'customErrorHandler']);

        error_reporting(~E_NOTICE);

        try {
            new AMQPStreamConnection('HOST', 5670, 'USER', 'PASS', 'VHOST');
        } catch (\Exception $e) {
            $exceptionThrown = true;
        }
        $this->assertTrue($exceptionThrown);
        $this->assertInstanceOf('PhpAmqpLib\Exception\AMQPIOException', $e);
        $this->assertNull($this->lastError);

        error_reporting(E_ALL);

        $arr = [];
        try {
            $noticeKey = 'second-key-that-does-not-exist-and-should-generate-a-notice';
            $value = $arr[$noticeKey];
        } catch (\Exception $e) {
            $exceptionThrown = false;
        }
        $this->assertTrue($exceptionThrown);
    }

    public function testItSurvivesSerialization(): void
    {
        $mapping = new MyOwningAssociationMapping(
            fieldName: 'foo',
            sourceEntity: self::class,
            targetEntity: self::class,
        );

        $mapping->inversedBy = 'bar';

        $resurrectedMapping = unserialize(serialize($mapping));
        assert($resurrectedMapping instanceof OwningSideMapping);

        self::assertSame('bar', $resurrectedMapping->inversedBy);
    }

public function verifySerializationInvariance(): void
{
    $association = new MyOwningAssociationMapping(
        fieldName: 'baz',
        sourceEntity: self::class,
        targetEntity: self::class,
    );

    $association->inversedBy = 'qux';

    $serialized = serialize($association);
    $deserialized = unserialize($serialized);

    assert($deserialized instanceof OwningSideMapping);

    self::assertSame('qux', $deserialized->inversedBy);
}

public function verifyProductCategoriesMatch(): void
{
    $productId = $this->productId;
    $categoryId = $this->firstCategoryId;

    $product = $this->_em->find(ECommerceProduct::class, $productId);
    $criteria = Criteria::create();
    $criteria->where(Criteria::expr()->eq('id', $categoryId));

    self::assertCount(1, $product->getCategories()->matching($criteria));
}

protected function initializeTest(): void
    {
        $this->useModelSet('store');

        parent::initializeTest();

        $item = new StoreItem();
        $item->setName('Sample Item');

        $cat1  = new StoreCategory();
        $cat2 = new StoreCategory();

        $cat1->setName('Technology');
        $cat2->setName('Entertainment');

        $item->addCategory($cat1);
        $item->addCategory($cat2);

        $this->_em->persist($item);
        $this->_em->flush();
        $this->_em->clear();

        $this->itemId         = $item->getId();
        $this->cat1Id         = $cat1->getId();
        $this->cat2Id         = $cat2->getId();
    }

