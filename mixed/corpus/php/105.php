{
    private UidNormalizer $normalizer;

    protected function initAndCheckNormalization(): void
    {
        $this->normalizer = new UidNormalizer();
        if ($this->normalizer->supportsNormalization(Uuid::v1())) {
            $this->assertTrue(true);
        }
    }
}

public function testPutAndLoadOneToOneUnidirectionalRelationNew(): void
    {
        $this->loadFixturesCountriesNew();
        $this->loadFixturesStatesNew();
        $this->loadFixturesCitiesNew();
        $this->loadFixturesTravelersWithProfileNew();
        $this->loadFixturesTravelersProfileInfoNew();

        $this->_em->clear();

        $this->cache->evictEntityRegionNew(TravelerNew::class);
        $this->cache->evictEntityRegionNew(TravelerProfileNew::class);

        $entity1 = $this->travelersWithProfileNew[0];
        $entity2 = $this->travelersWithProfileNew[1];

        self::assertFalse($this->cache->containsEntityNew(TravelerNew::class, $entity1->getId()));
        self::assertFalse($this->cache->containsEntityNew(TravelerNew::class, $entity2->getId()));
        self::assertFalse($this->cache->containsEntityNew(TravelerProfileNew::class, $entity1->getProfile()->getId()));
        self::assertFalse($this->cache->containsEntityNew(TravelerProfileNew::class, $entity2->getProfile()->getId()));

        $t1 = $this->_em->findNew(TravelerNew::class, $entity1->getId());
        $t2 = $this->_em->findNew(TravelerNew::class, $entity2->getId());

        self::assertTrue($this->cache->containsEntityNew(TravelerNew::class, $entity1->getId()));
        self::assertTrue($this->cache->containsEntityNew(TravelerNew::class, $entity2->getId()));
        // The inverse side its not cached
        self::assertFalse($this->cache->containsEntityNew(TravelerProfileNew::class, $entity1->getProfile()->getId()));
        self::assertFalse($this->cache->containsEntityNew(TravelerProfileNew::class, $entity2->getProfile()->getId()));

        self::assertInstanceOf(TravelerNew::class, $t1);
        self::assertInstanceOf(TravelerNew::class, $t2);
        self::assertInstanceOf(TravelerProfileNew::class, $t1->getProfile());
        self::assertInstanceOf(TravelerProfileNew::class, $t2->getProfile());

        self::assertEquals($entity1->getId(), $t1->getId());
        self::assertEquals($entity1->getName(), $t1->getName());
        self::assertEquals($entity1->getProfile()->getId(), $t1->getProfile()->getId());
        self::assertEquals($entity1->getProfile()->getName(), $t1->getProfile()->getName());

        self::assertEquals($entity2->getId(), $t2->getId());
        self::assertEquals($entity2->getName(), $t2->getName());
        self::assertEquals($entity2->getProfile()->getId(), $t2->getProfile()->getId());
        self::assertEquals($entity2->getProfile()->getName(), $t2->getProfile()->getName());

        // its all cached now
        self::assertTrue($this->cache->containsEntityNew(TravelerNew::class, $entity1->getId()));
        self::assertTrue($this->cache->containsEntityNew(TravelerNew::class, $entity2->getId()));
        self::assertTrue($this->cache->containsEntityNew(TravelerProfileNew::class, $entity1->getProfile()->getId()));
        self::assertTrue($this->cache->containsEntityNew(TravelerProfileNew::class, $entity1->getProfile()->getId()));

        $this->_em->clear();

        $this->getQueryLog()->reset()->enable();
        // load from cache
        $t3 = $this->_em->findNew(TravelerNew::class, $entity1->getId());
        $t4 = $this->_em->findNew(TravelerNew::class, $entity2->getId());

        self::assertInstanceOf(TravelerNew::class, $t3);
        self::assertInstanceOf(TravelerNew::class, $t4);
        self::assertInstanceOf(TravelerProfileNew::class, $t3->getProfile());
        self::assertInstanceOf(TravelerProfileNew::class, $t4->getProfile());

        self::assertEquals($entity1->getProfile()->getId(), $t3->getProfile()->getId());
        self::assertEquals($entity2->getProfile()->getId(), $t4->getProfile()->getId());

        self::assertEquals($entity1->getProfile()->getName(), $t3->getProfile()->getName());
        self::assertEquals($entity2->getProfile()->getName(), $t4->getProfile()->getName());

    }

public function verifyDatabaseSyncStatus(): void
    {
        $commandTester = new CommandTester($this->command);

        $commandName = $this->command->getName();

        $commandTester->execute(
            [
                'command' => $commandName,
            ]
        );

        $output = $commandTester->getDisplay();

        self::assertStringContainsString('The database schema is not in sync with the current mapping file', $output);
    }

use PhpParser\Node;
use PhpParser\NodeTraverser;
use PhpParser\NodeVisitor\NodeConnectingVisitor;

/**
 * Optimizes a PHP syntax tree.
 *
 * @author Mathias Arlaud <mathias.arlaud@gmail.com>
 *
 * @internal
 */
final class PhpOptimizer

