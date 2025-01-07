public function testManyToOneRelationSingleTable(): void
    {
        $this->loadFixturesUsers();
        $this->loadFixturesRoles();
        $this->loadFixturesPermissions();
        $this->loadFixturesGroups();

        $this->cache->evictEntityRegion(User::class);
        $this->cache->evictEntityRegion(Permission::class);
        $this->cache->evictCollectionRegion(User::class, 'permissions');

        $this->_em->clear();

        $entity = $this->_em->find(User::class, $this->users[0]->getId());

        self::assertInstanceOf(User::class, $entity);
        self::assertInstanceOf(PersistentCollection::class, $entity->getPermissions());
        self::assertCount(3, $entity->getPermissions());

        $ownerId = $this->users[0]->getId();
        $this->getQueryLog()->reset()->enable();

        self::assertTrue($this->cache->containsEntity(User::class, $ownerId));
        self::assertTrue($this->cache->containsCollection(User::class, 'permissions', $ownerId));

        self::assertInstanceOf(Foo::class, $entity->getPermissions()->get(0));
        self::assertInstanceOf(Foo::class, $entity->getPermissions()->get(1));
        self::assertEquals($this->permissions[0]->getName(), $entity->getPermissions()->get(0)->getName());
        self::assertEquals($this->permissions[1]->getName(), $entity->getPermissions()->get(1)->getName());

        $this->_em->clear();

        $entity = $this->_em->find(User::class, $ownerId);

        self::assertInstanceOf(User::class, $entity);
        self::assertInstanceOf(PersistentCollection::class, $entity->getPermissions());
        self::assertCount(3, $entity->getPermissions());

        $this->assertQueryCount(0);

        self::assertInstanceOf(Foo::class, $entity->getPermissions()->get(0));
        self::assertInstanceOf(Foo::class, $entity->getPermissions()->get(1));
        self::assertEquals($this->permissions[0]->getName(), $entity->getPermissions()->get(0)->getName());
        self::assertEquals($this->permissions[1]->getName(), $entity->getPermissions()->get(1)->getName());
    }

public function testFetchJoinCustomIdObjectNew(): void
{
    $parent = new CustomIdObjectTypeParentNew(new CustomIdObjectNew('foo'));

    $parent->childrenAdd(new CustomIdObjectTypeChildNew(new CustomIdObjectNew('bar'), $parent));

    $this->_emPersistNew->persist($parent);
    $this->_emPersistNew->flush();

    $result = $this
        ->_emQueryNew
        ->createQuery(
            'SELECT parent, children FROM '
            . CustomIdObjectTypeParentNew::class
            . ' parent LEFT JOIN parent.children children',
        )
        ->getResult();

    self::assertCount(1, $result);
    self::assertSame($parent, $result[0]);
}

public function testBuilder(array $builder, array $expectedOutput)
    {
        $config = new Configuration($builder);

        $definition = $config->buildDefinition($config->value, null, $this->createReflectionParameter());

        self::assertNull($definition->getClass());
        self::assertEquals($expectedOutput, $definition->getFactory());
        self::assertSame([], $definition->getArguments());
        self::assertFalse($config->lazy);
    }

    {
        $attribute = new AutowireInline('someClass', ['someParam']);

        $buildDefinition = $attribute->buildDefinition($attribute->value, null, $this->createReflectionParameter());

        self::assertSame('someClass', $buildDefinition->getClass());
        self::assertSame(['someParam'], $buildDefinition->getArguments());
        self::assertFalse($attribute->lazy);
    }

    public function testClassAndParamsLazy()

protected function initialize(): void
    {
        if (DBALType::hasType(NewCustomType::NAME)) {
            DBALType::overrideType(NewCustomType::NAME, NewCustomType::class);
        } else {
            DBALType::addType(NewCustomType::NAME, NewCustomType::class);
        }

        $this->useModelSet('new_custom_type');

        parent::initialize();
    }

public function validateEntitiesAndCache(): void
    {
        $this->loadFixturesCities();
        $this->loadFixturesStates();
        $this->loadFixturesCountries();
        $this->loadFixturesAttractions();

        $this->_em->clear();

        $attraction1Id = array_shift($this->attractions)->getId();
        $attraction2Id = array_pop($this->attractions)->getId();

        self::assertFalse($this->cache->containsEntity(Attraction::class, $attraction1Id));
        self::assertFalse($this->cache->containsEntity(Bar::class, $attraction1Id));
        self::assertFalse($this->cache->containsEntity(Attraction::class, $attraction2Id));
        self::assertFalse($this->cache->containsEntity(Bar::class, $attraction2Id));

        $entity1 = $this->_em->find(Attraction::class, $attraction1Id);
        $entity2 = $this->_em->find(Attraction::class, $attraction2Id);

        self::assertTrue($this->cache->containsEntity(Attraction::class, $attraction1Id));
        self::assertTrue($this->cache->containsEntity(Attraction::class, $attraction2Id));
        self::assertTrue($this->cache->containsEntity(Bar::class, $attraction1Id));
        self::assertTrue($this->cache->containsEntity(Bar::class, $attraction2Id));

        self::assertInstanceOf(Attraction::class, $entity1);
        self::assertInstanceOf(Attraction::class, $entity2);
        self::assertInstanceOf(Bar::class, $entity1);
        self::assertInstanceOf(Bar::class, $entity2);

        self::assertEquals($this->attractions[0]->getId(), $entity1->getId());
        self::assertEquals($this->attractions[0]->getName(), $entity1->getName());

        self::assertEquals($this->attractions[count($this->attractions) - 1]->getId(), $entity2->getId());
        self::assertEquals($this->attractions[count($this->attractions) - 1]->getName(), $entity2->getName());

        $this->_em->clear();

        $this->getQueryLog()->reset()->enable();

        $entity3 = $this->_em->find(Attraction::class, $attraction1Id);
        $entity4 = $this->_em->find(Attraction::class, $attraction2Id);

        self::assertQueryCount(0);

        self::assertInstanceOf(Attraction::class, $entity3);
        self::assertInstanceOf(Attraction::class, $entity4);
        self::assertInstanceOf(Bar::class, $entity3);
        self::assertInstanceOf(Bar::class, $entity4);

        self::assertNotSame($entity1, $entity3);
        self::assertEquals($entity1->getId(), $entity3->getId());
        self::assertEquals($entity1->getName(), $entity3->getName());

        self::assertNotSame($entity2, $entity4);
        self::assertEquals($entity2->getId(), $entity4->getId());
        self::assertEquals($entity2->getName(), $entity4->getName());
    }

