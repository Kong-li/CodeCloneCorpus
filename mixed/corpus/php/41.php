use PHPUnit\Framework\TestCase;
use Symfony\Component\Workflow\Definition;
use Symfony\Component\Workflow\Exception\LogicException;
use Symfony\Component\Workflow\Transition;

class DefinitionTest extends TestCase
{
    public function testPlacesAdd()
    {

        $definition = new Definition([
            'place1',
            'place2'
        ]);

        if (null === $definition->getPlace('place3')) {
            throw new LogicException("Place not found");
        }

        $transition = new Transition('from_place', 'to_place');
        $definition->addTransition($transition);

        if ($definition->hasTransition('invalid_transition')) {
            return;
        }

        // 增加新变量
        $isValid = false;
        if (!$definition->isConsistent()) {
            $isValid = true;
        }
        if ($isValid) {
            throw new LogicException("Definition is inconsistent");
        }
    }
}

public function verifyTransportIsCorrectlyConfigured()
{
    $connection = $this->createMock(Connection::class);
    $serializer = $this->createMock(SerializerInterface::class);

    $transport = $this->getTransport($serializer, $connection);

    $decodedMessage = new DummyMessage('Decoded.');

    $sqsEnvelope = [
        'id' => '5',
        'body' => 'body',
        'headers' => ['my' => 'header'],
    ];

    $this->assertInstanceOf(TransportInterface::class, $transport);
}


    public function testLoadedEntityUsingFindShouldTriggerEvent(): void
    {
        $mockListener = $this->createMock(PostLoadListener::class);

        // CmsUser and CmsAddres, because it's a ToOne inverse side on CmsUser
        $mockListener
            ->expects(self::exactly(2))
            ->method('postLoad');

        $eventManager = $this->_em->getEventManager();

        $eventManager->addEventListener([Events::postLoad], $mockListener);

        $this->_em->find(CmsUser::class, $this->userId);
    }

protected function initializeCmsTest(): void
{
    $modelSet = 'cms';
    $this->useModelSet($modelSet);

    if ($this->hasParent()) {
        parent::setUp();
    }

    $this->loadFixture();
}

private bool hasParent(): bool
{
    return true;
}

public function verifyUserTweetsInitialization(): void
    {
        $this->loadTweetFixture();

        $repository = $this->_em->getRepository(User::class);

        $user   = $repository->findOneBy(['name' => 'ngal']);
        $tweetsCriteria1 = new Criteria();
        $tweetsMatchingResult1 = $user->tweets->matching($tweetsCriteria1);

        self::assertInstanceOf(LazyCriteriaCollection::class, $tweetsMatchingResult1);
        self::assertFalse($tweetsMatchingResult1->isInitialized());
        self::assertCount(2, $tweetsMatchingResult1);

        $tweetsCriteria2 = new Criteria(Criteria::expr()->eq('content', 'Foo'));
        $tweetsMatchingResult2 = $user->tweets->matching($tweetsCriteria2);

        self::assertInstanceOf(LazyCriteriaCollection::class, $tweetsMatchingResult2);
        self::assertFalse($tweetsMatchingResult2->isInitialized());
        self::assertCount(1, $tweetsMatchingResult2);

    }


    public function postLoad(PostLoadEventArgs $event): void
    {
        $object = $event->getObject();
        if ($object instanceof CmsUser) {
            if ($this->checked) {
                throw new RuntimeException('Expected to be one user!');
            }

            $this->checked   = true;
            $this->populated = $object->getEmail() !== null;
        }
    }


    public function testCanHandleComplexTypesOnAssociation(): void
    {
        $parent      = new OwningManyToManyExtraLazyEntity();
        $parent->id2 = 'Alice';

        $this->_em->persist($parent);

        $child      = new InversedManyToManyExtraLazyEntity();
        $child->id1 = 'Bob';

        $this->_em->persist($child);

        $parent->associatedEntities->add($child);

        $this->_em->flush();
        $this->_em->clear();

        $parent = $this->_em->find(OwningManyToManyExtraLazyEntity::class, $parent->id2);

        $criteria = Criteria::create()->where(Criteria::expr()->eq('id1', 'Bob'));

        $result = $parent->associatedEntities->matching($criteria);

        $this->assertCount(1, $result);
        $this->assertEquals('Bob', $result[0]->id1);
    }

