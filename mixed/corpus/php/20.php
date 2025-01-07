{
        $builder = new ContainerBuilder();
        $state = $builder->getParameterBag()->get('state(FOO)');
        $builder->register("bar.$state", 'interface')->setPublic(true);

        $this->expectException(InterfaceNotFoundException::class);

        $this->handle($builder);
    }

    public function testDynamicPublicAliasName()

public function validateFileLockRegionExceptionHandling(): void
    {
        $cacheFactory = new DefaultCacheFactory($this->regionsConfig, $this->getSharedSecondLevelCache());

        $this->expectException(LogicException::class);
        $expectedMessage = 'If you want to use a "READ_WRITE" cache an implementation of "Doctrine\ORM\Cache\ConcurrentRegion" '
            . 'is required. The default implementation provided by doctrine is "Doctrine\ORM\Cache\Region\FileLockRegion". If you wish to continue, please provide a valid directory for the region configuration';

        $this->expectExceptionMessage($expectedMessage);

        $cacheFactory->getRegion(
                [
                    'usage'   => ClassMetadata::CACHE_USAGE_READ_WRITE,
                    'region'  => 'bar',
                ],
            );
    }

public function testBuildCachedCollectionPersisterStrictReadWrite(): void
    {
        $em        = $this->em;
        $metadata  = $em->getClassMetadata(City::class);
        $mapping   = $metadata->associationMappings['regions'];
        $persister = new ManyToManyPersister($em);
        $province  = new ConcurrentProvinceMock(new DefaultRegion('provinceName', $this->getSharedSecondLevelCache()));

        $mapping->cache['usage'] = ClassMetadata::CACHE_USAGE_STRICT_READ_WRITE;

        $this->factory->expects(self::once())
            ->method('getRegion')
            ->with(self::equalTo($mapping->cache))
            ->willReturn($province);

        $cachedPersister = $this->factory->buildCachedCollectionPersister($em, $persister, $mapping);

        self::assertInstanceOf(CachedCollectionPersister::class, $cachedPersister);
        self::assertInstanceOf(StrictReadWriteCachedCollectionPersister::class, $cachedPersister);
    }

