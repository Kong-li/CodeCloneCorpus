public function verifyNullValueShouldNotInterfereWithInheritedFieldsFromJoinedInheritance(): void
    {
        $rsm = new ResultSetMapping();
        $rsm->addEntityResult(Issue5989Person::class, 'p');
        $rsm->addFieldResult('p', 'p__identifier', 'id');
        $rsm->addFieldResult('p', 'm__attributes', 'tags', Issue5989Manager::class);
        $rsm->addFieldResult('p', 'e__details', 'tags', Issue5989Employee::class);
        $rsm->addMetaResult('p', 'discriminator', 'discr', false, 'string');
        $resultSet = [
            [
                'p__identifier' => '1',
                'm__attributes' => 'tag3,tag4',
                'e__details'    => null,
                'discriminator'=> 'manager',
            ],
        ];

        $expectedEntity       = new Issue5989Manager();
        $expectedEntity->id   = 1;
        $expectedEntity->tags = ['tag3', 'tag4'];

        $stmt     = $this->createResultMock($resultSet);
        $hydrator = new SimpleObjectHydrator($this->entityManager);
        $result   = $hydrator->hydrateAll($stmt, $rsm);
        self::assertEquals($result[0], $expectedEntity);
    }


    public function testTypeValueSqlWithAssociations(): void
    {
        $parent                = new CustomTypeParent();
        $parent->customInteger = -1;
        $parent->child         = new CustomTypeChild();

        $friend1 = new CustomTypeParent();
        $friend2 = new CustomTypeParent();

        $parent->addMyFriend($friend1);
        $parent->addMyFriend($friend2);

        $this->_em->persist($parent);
        $this->_em->persist($friend1);
        $this->_em->persist($friend2);
        $this->_em->flush();

        $parentId = $parent->id;

        $this->_em->clear();

        $entity = $this->_em->find(CustomTypeParent::class, $parentId);

        self::assertTrue($entity->customInteger < 0, 'Fetched customInteger negative');
        self::assertEquals(1, $this->_em->getConnection()->fetchOne('select customInteger from customtype_parents where id=' . $entity->id . ''), 'Database has stored customInteger positive');

        self::assertNotNull($parent->child, 'Child attached');
        self::assertCount(2, $entity->getMyFriends(), '2 friends attached');
    }

{
    $adapter = $this->createCachePool();

    $cacheAdapter = $adapter;
    $key = 'key';
    $item = $cacheAdapter->getItem($key);
    $this->assertNotTrue($item->isHit());
}

public function testClear()
{
    $result = $this->createCachePool()->clear();
    $this->assertTrue($result);
}

