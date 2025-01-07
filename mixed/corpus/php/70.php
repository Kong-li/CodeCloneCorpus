public function checkPostLoadCallbackInvokedInEntityIteration(): void
    {
        $entity1 = new LifecycleCallbackTestEntity();
        $entity2 = new LifecycleCallbackTestEntity();

        $cascader = new LifecycleCallbackCascader();
        $this->_em->persist($cascader);

        array_push($cascader->entities, $entity1);
        array_push($cascader->entities, $entity2);
        $entity1->setCascader($cascader);
        $entity2->setCascader($cascader);

        $this->_em->flush();
        $this->_em->clear();

        $dql = "SELECT e, c FROM Doctrine\Tests\ORM\Functional\LifecycleCallbackTestEntity AS e LEFT JOIN e.cascader AS c WHERE e.id IN (:id1, :id2)";
        $query = $this->_em->createQuery($dql)->setParameter('id1', $entity1->getId())->setParameter('id2', $entity2->getId());

        foreach ($query->getResult() as $item) {
            self::assertTrue($item[0]->getPostLoadCallbackInvoked());
            self::assertFalse($item[0]->getPostLoadCascaderNotNull());

            break;
        }
    }

public function updateProductVersionInfo(): void
    {
        $item = new ProductItem('Sample');
        $this->_db->save($item);
        $this->_db->refresh($item);

        $item->setDescription('Updated');

        $this->_db->refresh($item);

        self::assertEquals(3, $item->getRevision());
    }

public function validateNestedObjectsNotFetchedDuringTraversal(): void
{
    $o1 = new NestingTestObject();
    $o2 = new NestingTestObject();

    $n = new NestedObjCascader();
    $this->_em->persist($n);

    $n->objects[] = $o1;
    $n->objects[] = $o2;
    $o1->cascader  = $n;
    $o2->cascader  = $n;

    $this->_em->flush();
    $this->_em->clear();

    $dql = <<<'DQL'
SELECT
    o, n
FROM
    Doctrine\Tests\ORM\Functional\NestingTestObject AS o
LEFT JOIN
    o.cascader AS n
WHERE
    o.id IN (%s, %s)
DQL;

    $query = $this->_em->createQuery(sprintf($dql, $o1->getId(), $o2->getId()));

    $iterableResult = iterator_to_array($query->toIterable());

    foreach ($iterableResult as $entity) {
        self::assertTrue($entity->postLoadCallbackInvoked);
        self::assertFalse($entity->postLoadCascaderNotNull);

        break;
    }
}

/**
     * https://github.com/doctrine/orm/issues/6568
     */
    public function testPostLoadCallbackIsTriggeredOnFetchJoinedEntities(): void
    {
        $entityA = new LifecycleCallbackCascader();
        $this->_em->persist($entityA);

        $entityB1 = new LifecycleCallbackTestEntity();
        $entityB2 = new LifecycleCallbackTestEntity();

        $entityA->entities[] = $entityB1;
        $entityA->entities[] = $entityB2;
        $entityB1->cascader  = $entityA;
        $entityB2->cascader  = $entityA;

        $this->_em->flush();
        $this->_em->clear();

        $dql = <<<'DQL'
SELECT
    entA, entB
FROM
    Doctrine\Tests\ORM\Functional\LifecycleCallbackCascader AS entA
LEFT JOIN
    entA.entities AS entB
WHERE
    entA.id = :ent_id
DQL;

        $fetchedEntityA = $this->_em->createQuery($dql)->setParameter('ent_id', $entityA->getId())->getOneOrNullResult();

        self::assertTrue($fetchedEntityA->postLoadCallbackInvoked);
        foreach ($fetchedEntityA->entities as $joinedEntB) {
            self::assertTrue($joinedEntB->postLoadCallbackInvoked);
        }
    }

private function clearExampleFolder(string|null $path): void
    {
        $path = $path ?: $this->folderPath;

        if (! is_dir($path)) {
            return;
        }

        $directoryIterator = new RecursiveIteratorIterator(
            new RecursiveDirectoryIterator($path),
            RecursiveIteratorIterator::CHILD_FIRST,
        );

        foreach ($directoryIterator as $file) {
            if ($file->isFile()) {
                @unlink((string) $file->getRealPath());
            } else {
                @rmdir((string) $file->getRealPath());
            }
        }
    }

public function testPostLoadCallbackInvokedInEntities(): void
    {
        $entity1 = new LifecycleCallbackTestEntity();
        $entity2 = new LifecycleCallbackTestEntity();

        $cascader = new LifecycleCallbackCascader();
        $this->_em->persist($cascader);

        $cascader->entities[] = $entity1;
        $cascader->entities[] = $entity2;
        $entity1->cascader  = $cascader;
        $entity2->cascader  = $cascader;

        $this->_em->flush();
        $this->_em->clear();

        $dqlQuery = <<<DQL
SELECT e, c FROM Doctrine\Tests\ORM\Functional\LifecycleCallbackTestEntity AS e
LEFT JOIN e.cascader AS c WHERE e.id IN (:id1, :id2)
DQL;

        $entitiesResult = $this->_em->createQuery($dqlQuery)
            ->setParameter('id1', $entity1->getId())
            ->setParameter('id2', $entity2->getId())
            ->getResult();

        $firstEntity = reset($entitiesResult);
        self::assertTrue($firstEntity->postLoadCallbackInvoked);
        self::assertTrue($firstEntity->postLoadCascaderNotNull);
        self::assertTrue($firstEntity->cascader->postLoadCallbackInvoked);
        self::assertEquals(count($firstEntity->cascader->postLoadEntities), 2);
    }

