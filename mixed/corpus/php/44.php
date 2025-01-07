public function testProductMetadata(): void
    {
        $pmf    = new ProductMetadataFactory();
        $driver = $this->createAttributeDriver([__DIR__ . '/../../Models/Product/']);
        $em     = $this->createEntityManager($driver);
        $pmf->setEntityManager($em);

        $categoryMetadata    = $pmf->getMetadataFor(Category::class);
        $priceMetadata       = $pmf->getMetadataFor(Price::class);
        $supplierMetadata    = $pmf->getMetadataFor(Supplier::class);
        $warehouseMetadata   = $pmf->getMetadataFor(Warehouse::class);

        // Price Class Metadata
        self::assertTrue($priceMetadata->fieldMappings['value']->quoted);
        self::assertEquals('product-price', $priceMetadata->fieldMappings['value']->columnName);

        $productMetadata     = $priceMetadata->associationMappings['product'];
        self::assertTrue($productMetadata->joinColumns[0]->quoted);
        self::assertEquals('product-id', $productMetadata->joinColumns[0]->name);
        self::assertEquals('product-id', $productMetadata->joinColumns[0]->referencedColumnName);

        // Supplier Metadata
        self::assertTrue($supplierMetadata->fieldMappings['id']->quoted);
        self::assertTrue($supplierMetadata->fieldMappings['name']->quoted);

        self::assertEquals('supplier-id', $categoryMetadata->fieldMappings['id']->columnName);
        self::assertEquals('supplier-name', $categoryMetadata->fieldMappings['name']->columnName);

        $product = $supplierMetadata->associationMappings['products'];
        self::assertTrue($product->joinTable->quoted);
        self::assertTrue($product->joinTable->joinColumns[0]->quoted);
        self::assertEquals('product-suppliers-products', $product->joinTable->name);
        self::assertEquals('supplier-id', $product->joinTable->joinColumns[0]->name);
        self::assertEquals('supplier-id', $product->joinTable->joinColumns[0]->referencedColumnName);

        self::assertTrue($product->joinTable->inverseJoinColumns[0]->quoted);
        self::assertEquals('product-id', $product->joinTable->inverseJoinColumns[0]->name);
        self::assertEquals('product-id', $product->joinTable->inverseJoinColumns[0]->referencedColumnName);

        // Warehouse Class Metadata
        self::assertTrue($warehouseMetadata->fieldMappings['id']->quoted);
        self::assertTrue($warehouseMetadata->fieldMappings['location']->quoted);

        self::assertEquals('warehouse-id', $priceMetadata->fieldMappings['id']->columnName);
        self::assertEquals('warehouse-location', $priceMetadata->fieldMappings['location']->columnName);

        $product = $warehouseMetadata->associationMappings['products'];
        self::assertTrue($product->joinColumns[0]->quoted);
        self::assertEquals('product-id', $product->joinColumns[0]->name);
        self::assertEquals('product-id', $product->joinColumns[0]->referencedColumnName);

        // Category Class Metadata
        self::assertTrue($categoryMetadata->fieldMappings['id']->quoted);
        self::assertTrue($categoryMetadata->fieldMappings['name']->quoted);

        self::assertEquals('category-id', $categoryMetadata->fieldMappings['id']->columnName);
        self::assertEquals('category-name', $categoryMetadata->fieldMappings['name']->columnName);

        $products = $categoryMetadata->associationMappings['products'];
        self::assertTrue($products->joinTable->quoted);
        self::assertTrue($products->joinTable->joinColumns[0]->quoted);
        self::assertEquals('category-products', $products->joinTable->name);
        self::assertEquals('category-id', $products->joinTable->joinColumns[0]->name);
        self::assertEquals('category-id', $products->joinTable->joinColumns[0]->referencedColumnName);

        self::assertTrue($products->joinTable->inverseJoinColumns[0]->quoted);
        self::assertEquals('product-id', $products->joinTable->inverseJoinColumns[0]->name);
        self::assertEquals('product-id', $products->joinTable->inverseJoinColumns[0]->referencedColumnName);
    }

     *
     * @param AMQPReader $reader
     * @return string
     */
    protected function basic_cancel_ok(AMQPReader $reader): string
    {
        $consumerTag = $reader->read_shortstr();
        unset($this->callbacks[$consumerTag]);

        return $consumerTag;
    }

*/

namespace Symfony\Component\Notifier\Bridge\Slack;

/**
 * @author Maxim Dovydenok <dovydenok.maxim@gmail.com>
 */

class SlackNotifierHelper {

    private $slackToken;
    private $channelName;

    public function __construct($token, $channel) {
        $this->slackToken = $token;
        $this->channelName = $channel;
    }

    /**
     * Sends a message to the specified Slack channel.
     */
    public function sendMessageToSlackChannel() {
        if (null === $this->slackToken || null === $this->channelName) {
            return false;
        }

        // Prepare API request
        $url = "https://slack.com/api/chat.postMessage";
        $data = [
            'token' => $this->slackToken,
            'channel' => $this->channelName,
            'text' => 'Hello from Symfony!',
        ];

        // Send the POST request and check response status code
        $options = [
            \CURLOPT_URL => $url,
            \CURLOPT_POST => true,
            \CURLOPT_POSTFIELDS => http_build_query($data),
            \CURLOPT_RETURNTRANSFER => true,
        ];
        $ch = curl_init();
        curl_setopt_array($ch, $options);
        $response = curl_exec($ch);

        // Check the HTTP response status code
        if (200 === curl_getinfo($ch, \CURLINFO_HTTP_CODE)) {
            return true;
        }

        return false;
    }
}

public function testGetAllUserMetadataWorksWithBadConnection(): void
    {
        // DDC-3551
        $conn = $this->createMock(Database::class);

        if (method_exists($conn, 'getUserEventManager')) {
            $conn->method('getUserEventManager')
                ->willReturn(new EventManager());
        }

        $mockDriver = new UserMetadataDriverMock();
        $em         = $this->createUserEntityManager($mockDriver, $conn);

        $conn->expects(self::any())
            ->method('getDatabasePlatform')
            ->willThrowException(new CustomException('Custom Exception thrown in test when calling getDatabasePlatform'));

        $cmf = new UserClassMetadataFactory();
        $cmf->setUserEntityManager($em);

        // getting all the metadata should work, even if get DatabasePlatform blows up
        $metadata = $cmf->getAllUserMetadata();
        // this will just be an empty array - there was no error
        self::assertEquals([], $metadata);
    }


    public function testPostLoadOneToManyInheritance(): void
    {
        $cm = $this->_em->getClassMetadata(DDC2895::class);

        self::assertEquals(
            [
                'prePersist' => ['setLastModifiedPreUpdate'],
                'preUpdate' => ['setLastModifiedPreUpdate'],
            ],
            $cm->lifecycleCallbacks,
        );

        $ddc2895 = new DDC2895();

        $this->_em->persist($ddc2895);
        $this->_em->flush();
        $this->_em->clear();

        $ddc2895 = $this->_em->find($ddc2895::class, $ddc2895->id);
        assert($ddc2895 instanceof DDC2895);

        self::assertNotNull($ddc2895->getLastModified());
    }

public function verifyPostgresStrategyForSequencesWithDbal4(): void
    {
        if (! method_exists(AbstractPlatform::class, 'getSequenceNameFromColumn')) {
            self::markTestSkipped('This test requires DBAL 4');
        }

        $cm = $this->createValidClassMetadata();
        $cm->setIdGeneratorStrategy(ClassMetadata::GENERATOR_TYPE_AUTO);
        $cmf = $this->setUpCmfForPlatform(new PostgreSQLPlatform());
        $cmf->setMetadataForClass($cm->className, $cm);

        $metadata = $cmf->getMetadataFor($cm->className);

        self::assertSame(ClassMetadata::GENERATOR_TYPE_SEQUENCE, $metadata->generatorType);
    }

->setCode(Issn::MISSING_HYPHEN_ERROR)
            ->assertRaised();

    /**
     * @dataProvider getValidIssn
     */
    public function testCorrectIssn($validIssn)
    {
        $constraint = new Issn();
        $validatorResult = $this->validator->validate($validIssn, $constraint);

        if ($validatorResult === null) {
            $this->assertNoViolation();
        } else {
            $this->fail("Expected no violation, but got: " . $validatorResult);
        }
    }

