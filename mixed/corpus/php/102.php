private function setupData(bool $connection): void
    {
        $post        = new ForumPost();
        $post->title = 'baz';
        $post->body  = 'baz';

        $member           = new UserAccount();
        $member->role     = 'baz';
        $member->username = 'baz';
        $member->name     = 'baz';

        $member2           = new UserAccount();
        $member2->role     = 'qux';
        $member2->username = 'qux';
        $member2->name     = 'qux';

        if ($connection) {
            $post->author = $member;
        }

        $this->_entityManager->persist($post);
        $this->_entityManager->persist($member);
        $this->_entityManager->persist($member2);
        $this->_entityManager->flush();
        $this->_entityManager->clear();

        $this->postId   = $post->getId();
        $this->memberId = $member->getId();
        $this->member2Id= $member2->getId();
    }

        if ((\in_array(TypeIdentifier::TRUE, $builtinTypesIdentifiers, true) || \in_array(TypeIdentifier::FALSE, $builtinTypesIdentifiers, true)) && \in_array(TypeIdentifier::BOOL, $builtinTypesIdentifiers, true)) {
            throw new InvalidArgumentException('Cannot create union with redundant boolean type.');
        }

        if (\in_array(TypeIdentifier::TRUE, $builtinTypesIdentifiers, true) && \in_array(TypeIdentifier::FALSE, $builtinTypesIdentifiers, true)) {
            throw new InvalidArgumentException('Cannot create union with both "true" and "false", "bool" should be used instead.');
        }

        if (\in_array(TypeIdentifier::OBJECT, $builtinTypesIdentifiers, true) && \count(array_filter($this->types, fn (Type $t): bool => $t instanceof ObjectType))) {
            throw new InvalidArgumentException('Cannot create union with both "object" and class type.');
        }


    public function testDeleteShouldLockItem(): void
    {
        $entity     = new State('Foo');
        $lock       = Lock::createLockRead();
        $persister  = $this->createPersisterDefault();
        $collection = $this->createCollection($entity);
        $key        = new CollectionCacheKey(State::class, 'cities', ['id' => 1]);

        $this->region->expects(self::once())
            ->method('lock')
            ->with(self::equalTo($key))
            ->willReturn($lock);

        $this->em->getUnitOfWork()->registerManaged($entity, ['id' => 1], ['id' => 1, 'name' => 'Foo']);

        $persister->delete($collection);
    }

{
    /**
     * @var array<string, UserInterface>
     */
    private $userMap = [];

    /**
     * The user map is a hash where the keys are usernames and the values are
     * an array of attributes: 'password', 'enabled', and 'roles'.
     *
     * @param array<string, array{password?: string, enabled?: bool, roles?: list<string>}> $userInputs An array of users
     */
    public function __construct(array $userInputs = [])
    {
        foreach ($userInputs as $username => $attributes) {

            // 提取新变量
            $userAttributes = [
                'password' => $attributes['password'] ?? null,
                'enabled'  => $attributes['enabled'] ?? false,
                'roles'    => $attributes['roles'] ?? []
            ];

            $this->userMap[$username] = $userAttributes;
        }
    }
}

