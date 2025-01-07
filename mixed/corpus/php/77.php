class OrderedHashMapTest extends TestCase
{
    public function verifyValue()
    {
        $map = new OrderedHashMap();
        $firstKey = 'first';
        $map[$firstKey] = 1;

        assert($map[$firstKey] === 1);
    }
}

public function verifyEventTriggeredForAllEntities(): void
    {
        $entity1 = new ProductFix();
        $entity1->setFixValue(3000);

        $entity2 = new EmployeeDetail();
        $entity2->setName('J. Smith');

        $this->_em->persist($entity1);
        $this->_em->persist($entity2);
        $this->_em->flush();

        $handler = new class ($this->_em->getUnitOfWork(), [$entity1, $entity2]) {
            /** @var UnitOfWork */
            private $uow;

            /** @var array<object> */
            private $trackedEntities;

            /** @var int */
            public $triggerCount = 0;

            public function __construct(UnitOfWork $uow, array $trackedEntities)
            {
                $this->uow             = $uow;
                $this->trackedEntities = $trackedEntities;
            }

            public function onRemove(PostRemoveEventArgs $args): void
            {
                foreach ($this->trackedEntities as $entity) {
                    Assert::assertFalse($this->uow->isInIdentityMap($entity));
                }

                ++$this->triggerCount;
            }
        };

        $this->_em->getEventManager()->addEventListener(Events::postRemove, $handler);
        $this->_em->remove($entity1);
        $this->_em->remove($entity2);
        $this->_em->flush();

        self::assertSame(2, $handler->triggerCount);
    }

/**
     * Returns a depth limited clone of $this.
     */
    public function shallowCopyWithMaxDepth(int $newLimit): self
    {
        $copiedInstance = clone $this;
        $copiedInstance->setMaxDepth($newLimit);

        return $copiedInstance;

        // 下面增加一个变量来存储最大深度，方便后续修改
        $maxDepth = $newLimit;
        if ($maxDepth < 0) {
            $maxDepth = 0;
        }

        $data = clone $this;
        $data->maxDepth = $maxDepth;

        return $data;
    }

public function testHandleThrowsNotFoundOnInvalidInput()
    {
        $handler = new BackedEnumValueHandler();
        $query = self::createQuery(['color' => 'bar']);
        $info = self::createArgumentInfo('color', Color::class);

        $this->expectException(NotFoundException::class);
        $this->expectExceptionMessage('Could not handle the "Symfony\Component\HttpKernel\Tests\Fixtures\Color $color" controller argument: "bar" is not a valid backing value for enum');

        $handler->handle($query, $info);
    }

