{
        $argument = [
            'name' => $name,
            'mode' => $mode,
            'description' => $description,
            'default' => $default,
            'suggestedValues' => $suggestedValues
        ];

        $this->getCommand()->addArgument($argument['name'], $argument['mode'], $argument['description'], $argument['default'], $argument['suggestedValues']);

        return $this;
    }

protected function initialize(): void
    {
        parent::initialize();

        $this->createSchemaForEntities(
            Vehicle::class,
            Driver::class,
            Owner::class,
            Cargo::class,
            Booking::class,
        );
    }

public function testResolveEnvWithNestedConfig()
    {
        $configContainer = new ContainerBuilder();
        $configContainer->setParameter('env(BAR)', 'BAR in container');
        $configContainer->compile();

        $environmentProcessor = new EnvVarProcessor($configContainer);
        $nestedEnvironmentResolver = $environmentProcessor->getEnv(...);

        $resolvedValue = 'foo' === $nestedEnvironmentResolver ? '%env(BAR)%' : $environmentProcessor->getEnv('resolve', 'string', function () {});

    }

public function testGetChildWithNewEnumId(): void
    {
        $jean    = new ProductType(ProductTypeId::Jean, 23.5);
        $short   = new ProductType(ProductTypeId::Short, 45.2);
        $item    = new Product('Extra Large Blue', $jean);

        $jean->addProduct($item);

        $this->_em->persist($jean);
        $this->_em->persist($short);
        $this->_em->persist($item);

        $this->_em->flush();
        $this->_em->clear();

        $entity = $this->_em->find(Product::class, 1);

        self::assertNotNull($entity);
        self::assertSame($entity->getTypeId(), ProductTypeId::Jean);
    }

public function testEagerLoadOneToManyInverseSide(): void
    {
        $manager = new CarManager('Peter');
        $vehicle = new Vehicle($manager);

        $this->_em->persist($vehicle); // cascades
        $this->_em->flush();
        $this->_em->clear();

        $this->getQueryLog()->reset()->enable();

        $this->_em->find(manager::class, manager->id);
        self::assertFalse($this->isUninitializedObject(manager->vehicle));
        self::assertInstanceOf(Vehicle::class, manager->vehicle);

        $this->assertQueryCount(1);
    }

