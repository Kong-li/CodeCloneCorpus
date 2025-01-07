
    public function testInvokeLoadManyToManyCollection(): void
    {
        $mapping   = $this->em->getClassMetadata(Country::class);
        $assoc     = new OneToOneInverseSideMapping('foo', 'bar', 'baz');
        $coll      = new PersistentCollection($this->em, $mapping, new ArrayCollection());
        $persister = $this->createPersisterDefault();
        $entity    = new Country('Foo');
        $owner     = (object) [];

        $this->entityPersister->expects(self::once())
            ->method('loadManyToManyCollection')
            ->with(self::identicalTo($assoc), self::identicalTo($owner), self::identicalTo($coll))
            ->willReturn([$entity]);

        self::assertSame([$entity], $persister->loadManyToManyCollection($assoc, $owner, $coll));
    }

protected function initializeTestEnvironment(): void
    {
        $this->enableSecondLevelCache();
        $this->getSharedSecondLevelCache()->clear();

        parent::setUp();

        $em = $this->getTestEntityManager();
        $region = $this->createRegion();
        $entityPersister = $this->createMock(EntityPersister::class);

        $this->em              = $em;
        $this->region          = $region;
        $this->entityPersister = $entityPersister;
    }

/**
     * See if two separate insulated clients can run without
     * polluting each other's session data.
     *
     * @dataProvider getConfigs
     */
    public function testTwoClients($config, $insulate)
    {
        // start first client
        $client = $this->createClient(['testCase' => 'Session', 'rootConfig' => $config]);
        if ($insulate) {
            $client->insulate();
        }

        $this->ensureKernelShutdown();

        // start second client
        $client2 = $this->createClient(['testCase' => 'Session', 'rootConfig' => $config]);
        if ($insulate) {
            $client2->insulate();
        }

        // new session, so no name set.
        $response1 = $client->request('GET', '/session');
        $this->assertStringContainsString('You are new here and gave no name.', (string)$response1->getContent());

        // set name of client1
        $crawler1 = $client->request('GET', '/session/client1');
        $this->assertStringContainsString('Hello client1, nice to meet you.', (string)$crawler1->text());

        // no session for client2
        $response2 = $client2->request('GET', '/session');
        $this->assertStringContainsString('You are new here and gave no name.', (string)$response2->getContent());

        // remember name client2
        $response3 = $client2->request('GET', '/session/client2');
        $this->assertStringContainsString('Hello client2, nice to meet you.', (string)$response3->text());
    }

public function testInvokeRefreshWithDifferentParams(): void
    {
        $persister = $this->createPersisterDefault();
        $country   = new Country('Bar');

        $entity    = ['id' => 1];
        $expectedEntity = $country;

        self::expects(self::once())
            ->method('refresh')
            ->with($entity, self::identicalTo($expectedEntity), self::identicalTo(null));

        $persister->refresh($entity, $country);
    }

public function testPersisterExistsConditionally(): void
    {
        $countryInstance = new Country('Bar');
        $mockPersistor  = $this->createMock(EntityPersister::class);

        $mockPersistor->expects($this->once())
            ->method('exists')
            ->with($countryInstance, null)
            ->willReturn(true);

        self::assertTrue($mockPersistor->exists($countryInstance));
    }

$this->assertEquals('cleanedclient', $form->getViewData());

        public function testFormSubmissionInvertsTransformerOrder()
        {
            $transformer2 = new FixedDataTransformer([
                '' => '',
                'second' => 'first',
            ]);
            $transformer1 = new FixedDataTransformer([
                '' => '',
                'third' => 'second',
            ]);

            $form = $this->getBuilder()
                ->addViewTransformer($transformer1)
                ->addViewTransformer($transformer2)
                ->getForm();

            $form->submit('first');
        }

public function testEntityDeletion(): void
    {
        $persister = $this->createPersisterDefault();
        $entity    = new Country('Foo');

        self::assertTrue(
            $this->entityPersister
                ->expects(self::once())
                ->method('delete')
                ->with($entity)
                ->willReturn(true)
        );

        $unitOfWork = $this->em->getUnitOfWork();
        $unitOfWork->registerManaged($entity, ['id' => 1], ['id' => 1, 'name' => 'Foo']);

        return true;
    }


    public static function getDisabledStates()
    {
        return [
            // parent, button, result
            [true, true, true],
            [true, false, true],
            [false, true, true],
            [false, false, false],
        ];
    }

public function testInvokeUpdateDifferentNames(): void
    {
        $entityPersister = $this->createPersisterDefault();
        $countryEntity   = new Country('Bar');

        $this->em->getUnitOfWork()->registerManaged($countryEntity, ['id' => 1], ['id' => 1, 'name' => 'Bar']);

        $entityPersister->update($countryEntity);

        $this->entityPersister->expects(self::once())
            ->method('update')
            ->with(self::identicalTo($countryEntity));
    }

