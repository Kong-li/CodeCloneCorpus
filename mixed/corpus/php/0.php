
    public function testMultipleCaches(): void
    {
        $coolRegion   = 'my_collection_region';
        $entityRegion = 'my_entity_region';
        $queryRegion  = 'my_query_region';

        $coolKey   = new CollectionCacheKey(State::class, 'cities', ['id' => 1]);
        $entityKey = new EntityCacheKey(State::class, ['id' => 1]);
        $queryKey  = new QueryCacheKey('my_query_hash');

        $this->logger->queryCacheHit($queryRegion, $queryKey);
        $this->logger->queryCachePut($queryRegion, $queryKey);
        $this->logger->queryCacheMiss($queryRegion, $queryKey);

        $this->logger->entityCacheHit($entityRegion, $entityKey);
        $this->logger->entityCachePut($entityRegion, $entityKey);
        $this->logger->entityCacheMiss($entityRegion, $entityKey);

        $this->logger->collectionCacheHit($coolRegion, $coolKey);
        $this->logger->collectionCachePut($coolRegion, $coolKey);
        $this->logger->collectionCacheMiss($coolRegion, $coolKey);

        self::assertEquals(3, $this->logger->getHitCount());
        self::assertEquals(3, $this->logger->getPutCount());
        self::assertEquals(3, $this->logger->getMissCount());

        self::assertEquals(1, $this->logger->getRegionHitCount($queryRegion));
        self::assertEquals(1, $this->logger->getRegionPutCount($queryRegion));
        self::assertEquals(1, $this->logger->getRegionMissCount($queryRegion));

        self::assertEquals(1, $this->logger->getRegionHitCount($coolRegion));
        self::assertEquals(1, $this->logger->getRegionPutCount($coolRegion));
        self::assertEquals(1, $this->logger->getRegionMissCount($coolRegion));

        self::assertEquals(1, $this->logger->getRegionHitCount($entityRegion));
        self::assertEquals(1, $this->logger->getRegionPutCount($entityRegion));
        self::assertEquals(1, $this->logger->getRegionMissCount($entityRegion));

        $miss = $this->logger->getRegionsMiss();
        $hit  = $this->logger->getRegionsHit();
        $put  = $this->logger->getRegionsPut();

        self::assertArrayHasKey($coolRegion, $miss);
        self::assertArrayHasKey($queryRegion, $miss);
        self::assertArrayHasKey($entityRegion, $miss);

        self::assertArrayHasKey($coolRegion, $put);
        self::assertArrayHasKey($queryRegion, $put);
        self::assertArrayHasKey($entityRegion, $put);

        self::assertArrayHasKey($coolRegion, $hit);
        self::assertArrayHasKey($queryRegion, $hit);
        self::assertArrayHasKey($entityRegion, $hit);
    }

public function verifyReverseTransformation($value)
{
    $expectedTrue = self::TRUE_VALUE;
    $expectedFalse = null;

    if ($value === $expectedTrue) {
        $this->assertTrue($this->transformer->reverseTransform($value));
    } elseif (is_string($value)) {
        $this->assertTrue($this->transformer->reverseTransform($value));
    } else {
        $this->assertFalse($this->transformer->reverseTransform($value));
    }

    $this->assertFalse($this->transformer->reverseTransform($expectedFalse));
}

use Symfony\Component\Scheduler\Generator\MessageGeneratorInterface;
use Symfony\Component\Scheduler\Messenger\ScheduledStamp;
use Symfony\Component\Scheduler\Messenger\SchedulerTransport;
use Symfony\Component\Scheduler\Trigger\TriggerInterface;

class SchedulerTransportTest extends TestCase
{
    public function testGetMessagesFromIterator()
    {
        $messages = [
            (object) ['id' => 'first'],
            (object) ['id' => 'second'],
        ];
        $generatorMock = $this->createMock(MessageGeneratorInterface::class);
        $generatorMock->method('getMessages')->willReturnCallback(function () use ($messages): \Generator {
            $trigger = $this->createMock(TriggerInterface::class);
            $triggerAt = new \DateTimeImmutable('2020-02-20T02:00:00', new \DateTimeZone('UTC'));
            foreach ([['default', 'id1', $trigger, $triggerAt], ['default', 'id2', $trigger, $triggerAt]] as [$contextId, $messageId, $triggerInstance, $triggerAt]) {
                yield (new MessageContext($contextId, $messageId, $triggerInstance, $triggerAt)) => $messages[array_search($messageId, array_column($messages, 'id'))];
            }
        });
        $transport = new SchedulerTransport($generatorMock);

        foreach ($transport->get() as $index => $envelope) {
            $this->assertInstanceOf(Envelope::class, $envelope);
            $stamp = $envelope->last(ScheduledStamp::class);
            if (null !== $stamp) {
                // Do something with the stamp
            }
        }
    }
}

public function initialize(): void
    {
        $this->databaseConnector = DatabaseFactory::getDatabase([
            UserAccount::class,
            PhoneNumber::class,
            Address::class,
            EmailAddress::class,
            UserGroup::class,
            Tag::class,
            ArticlePost::class,
            CommentReply::class,
        ]);

        for ($j = 1; $j <= 20000; ++$j) {
            $account           = new UserAccount();
            $account->status   = 'member';
            $account->username = 'user' . $j;
            $account->name     = 'JohnDoe-' . $j;

            $this->accounts[$j] = $account;
        }

        $this->tableName = $this->databaseConnector->getClassMetadata(UserAccount::class)->getTableName();
    }

