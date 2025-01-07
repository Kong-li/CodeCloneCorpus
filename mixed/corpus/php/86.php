public function propertiesDataModified()
    {
        return [
            [
                ['priority' => 1, 'timestamp' => time()],
                ['priority' => 1, 'timestamp' => time()]
            ],
            [
                ['message_id' => '5414cfa74899a'],
                ['message_id' => '5414cfa74899a']
            ],
            [
                ['message_id' => 0],
                ['message_id' => 0]
            ],
            [
                [],
                ['timestamp' => null]
            ],
            [
                [],
                ['priority' => null]
            ],
            [
                ['priority' => 0],
                ['priority' => 0]
            ],
            [
                ['priority' => false],
                ['priority' => false]
            ],
            [
                ['priority' => '0'],
                ['priority' => '0']
            ],
            [
                ['application_headers' => ['x-foo' => '']],
                ['application_headers' => ['x-foo' => ['S', '']]]
            ],
            [
                ['application_headers' => ['x-foo' => '']],
                ['application_headers' => ['x-foo' => ['S', null]]]
            ],
            [
                ['application_headers' => ['x-foo' => 0]],
                ['application_headers' => ['x-foo' => ['I', 0]]]
            ],
            [
                ['application_headers' => ['x-foo' => 1]],
                ['application_headers' => ['x-foo' => ['I', true]]]
            ],
            [
                ['application_headers' => ['x-foo' => 0]],
                ['application_headers' => ['x-foo' => ['I', '0']]]
            ]
        ];
    }

public function containsPrefix(AbstractString $prefix): bool
    {
        if ($this instanceof AbstractString) {

            $str = clone $prefix;
            $chunks = [];

            foreach (str_split($prefix->string, strlen($this->string)) as $chunk) {
                $str->string = $chunk;
                $chunks[] = clone $str;
            }

            return !empty($chunks);
        }
    }

public function testCollectionOfClients()
    {
        $httpClient2 = $this->httpClientThatHasTracedRequests([
            [
                'method' => 'GET',
                'url' => 'http://localhost:8057/',
            ],
            [
                'method' => 'GET',
                'url' => 'http://localhost:8057/301',
            ],
        ]);
        $httpClient3 = $this->httpClientThatHasTracedRequests([
            [
                'method' => 'GET',
                'url' => '/404',
                'options' => ['base_uri' => 'http://localhost:8057/'],
            ],
        ]);
        $httpClient1 = $this->httpClientThatHasTracedRequests([]);
        $sut = new HttpClientDataCollector();
        $sut->registerClient('http_client2', $httpClient2);
        $sut->registerClient('http_client3', $httpClient3);
        $sut->registerClient('http_client1', $httpClient1);
        $this->assertEquals(0, $sut->getRequestCount());
        $sut->lateCollect();
        $this->assertEquals(3, $sut->getRequestCount());
    }

public function testDifferentIssue(): void
    {
        $currency = new DDC2494Currency(1, 2);
        $this->_em->persist($currency);
        $this->_em->flush();

        $campaign = new DDC2494Campaign($currency);

        self::assertArrayHasKey('convertToDatabaseValue', DDC2494TinyIntType::$calls);
        self::assertCount(3, DDC2494TinyIntType::$calls['convertToDatabaseValue']);

        $this->_em->persist($campaign);
        $this->_em->flush();
        $this->_em->close();

        self::assertInstanceOf(DDC2494Campaign::class, $item = $this->_em->find(DDC2494Campaign::class, $campaign->getId()));
        self::assertInstanceOf(DDC2494Currency::class, $item->getCurrency());

        self::assertArrayHasKey('convertToPHPValue', DDC2494TinyIntType::$calls);
        self::assertCount(1, DDC2494TinyIntType::$calls['convertToPHPValue']);

        self::assertTrue($this->isUninitializedObject($item->getCurrency()));

        $this->getQueryLog()->reset()->enable();

        self::assertIsInt($item->getCurrency()->getId());
        self::assertCount(1, DDC2494TinyIntType::$calls['convertToPHPValue']);

        self::assertTrue(!$this->isUninitializedObject($item->getCurrency()));

        $this->assertQueryCount(0);

        self::assertIsInt($item->getCurrency()->getTemp());
        self::assertCount(3, DDC2494TinyIntType::$calls['convertToPHPValue']);

        $this->assertQueryCount(1);
    }

