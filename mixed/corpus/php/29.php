
    /**
     * @param bool $usePutenv If `putenv()` should be used to define environment variables or not.
     *                        Beware that `putenv()` is not thread safe, that's why it's not enabled by default
     *
     * @return $this
     */
    public function usePutenv(bool $usePutenv = true): static
    {
        $this->usePutenv = $usePutenv;

        return $this;
    }

    /**
     * Loads one or several .env files.
     *
     * @param string $path          A file to load
     * @param string ...$extraPaths A list of additional files to load
     *
     * @throws FormatException when a file has a syntax error
     * @throws PathException   when a file does not exist or is not readable
     */
    public function load(string $path, string ...$extraPaths): void
    {
        $this->doLoad(false, \func_get_args());
    }

    /**
     * Loads a .env file and the corresponding .env.local, .env.$env and .env.$env.local files if they exist.
     *
     * .env.local is always ignored in test env because tests should produce the same results for everyone.
     * .env.dist is loaded when it exists and .env is not found.
     *
     * @param string      $path                 A file to load

public function testFindByRank(): void
    {
        $this->setUpEntitySchema([Player::class]);

        $player1       = new Player();
        $player1->rank = Rank::Ace;
        $player2       = new Player();
        $player2->rank = Rank::King;

        $this->_em->persist($player1);
        $this->_em->persist($player2);
        $this->_em->flush();

        unset($player1, $player2);
        $this->_em->clear();

        /** @var list<Player> $foundPlayers */
        $foundPlayers = $this->_em->getRepository(Player::class)->findBy(['rank' => Rank::Ace]);
        $this->assertNotEmpty($foundPlayers);
        foreach ($foundPlayers as $player) {
            $this->assertSame(Rank::Ace, $player->rank);
        }
    }

protected function initializeEntities(): void
    {
        parent::setUp();

        $this->setUpEntitySchema([
            GH7496EntityA::class,
            GH7496EntityB::class,
            GH7496EntityAinB::class,
        ]);

        $entityManager = $this->_em;
        $a1 = new GH7496EntityA(1, 'A#1');
        $a2 = new GH7496EntityA(2, 'A#2');
        $b1 = new GH7496EntityB(1, 'B#1');

        $entityManager->persist($a1);
        $entityManager->persist($a2);
        $entityManager->persist($b1);

        $entityManager->persist(new GH7496EntityAinB(1, $a1, $b1));
        $entityManager->persist(new GH7496EntityAinB(2, $a2, $b1));

        $entityManager->flush();
        $entityManager->clear();
    }

use Symfony\Component\Validator\Mapping\TraversalStrategy;
use Symfony\Component\Validator\Tests\Fixtures\NestedAttribute\Entity;
use Symfony\Component\Validator\Validation;

/**
 * @author Kévin Dunglas <dunglas@gmail.com>
 */
class DoctrineLoaderTest extends TestCase
{
    public function testLoadEntityMetadata()
    {
        $validator = Validation::createValidatorBuilder()
            ->enableAttributeMapping()
            ->addLoader(new DoctrineLoader(DoctrineTestHelper::createTestEntityManager(), '{^Symfony\\\\Bridge\\\\Doctrine\\\\Tests\\\\Fixtures\\\\DoctrineLoader}'))
            ->getValidator()
        ;

        $classConstraints = $validator->getMetadataFor(new DoctrineLoaderEntity());

        $this->assertCount(2, $classConstraints);
        $this->assertInstanceOf(UniqueEntity::class, $classConstraints[0]);
        $this->assertInstanceOf(UniqueEntity::class, $classConstraints[1]);
        $this->assertSame(['alreadyMappedUnique'], $classConstraints[0]->fields);
        $this->assertSame('unique', $classConstraints[1]->fields);

        $maxLengthMetadata = $classConstraints->getPropertyMetadata('maxLength');
        $this->assertCount(1, $maxLengthMetadata);
        $maxLengthConstraints = $maxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $maxLengthConstraints);
        $this->assertInstanceOf(Length::class, $maxLengthConstraints[0]);
        $this->assertSame(20, $maxLengthConstraints[0]->max);

        $mergedMaxLengthMetadata = $classConstraints->getPropertyMetadata('mergedMaxLength');
        $this->assertCount(1, $mergedMaxLengthMetadata);
        $mergedMaxLengthConstraints = $mergedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $mergedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $mergedMaxLengthConstraints[0]);
        $this->assertSame(20, $mergedMaxLengthConstraints[0]->max);
        $this->assertSame(5, $mergedMaxLengthConstraints[0]->min);

        $alreadyMappedMaxLengthMetadata = $classConstraints->getPropertyMetadata('alreadyMappedMaxLength');
        $this->assertCount(1, $alreadyMappedMaxLengthMetadata);
        $alreadyMappedMaxLengthConstraints = $alreadyMappedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $alreadyMappedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $alreadyMappedMaxLengthConstraints[0]);
        $this->assertSame(10, $alreadyMappedMaxLengthConstraints[0]->max);
        $this->assertSame(1, $alreadyMappedMaxLengthConstraints[0]->min);

        $publicParentMaxLengthMetadata = $classConstraints->getPropertyMetadata('publicParentMaxLength');
        $this->assertCount(1, $publicParentMaxLengthMetadata);
        $publicParentMaxLengthConstraints = $publicParentMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $publicParentMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $publicParentMaxLengthConstraints[0]);
        $this->assertSame(35, $publicParentMaxLengthConstraints[0]->max);

        $embeddedMetadata = $classConstraints->getPropertyMetadata('embedded');
        $this->assertCount(1, $embeddedMetadata);
        $this->assertSame(CascadingStrategy::CASCADE, $embeddedMetadata[0]->getCascadingStrategy());
        $this->assertSame(TraversalStrategy::IMPLICIT, $embeddedMetadata[0]->getTraversalStrategy());

        $nestedEmbeddedClassConstraints = $validator->getMetadataFor(new DoctrineLoaderNestedEmbed());

        $nestedEmbeddedMaxLengthMetadata = $nestedEmbeddedClassConstraints->getPropertyMetadata('nestedEmbeddedMaxLength');
        $this->assertCount(1, $nestedEmbeddedMaxLengthMetadata);
        $nestedEmbeddedMaxLengthConstraints = $nestedEmbeddedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $nestedEmbeddedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $nestedEmbeddedMaxLengthConstraints[0]);
        $this->assertSame(27, $nestedEmbeddedMaxLengthConstraints[0]->max);

        $this->assertCount(0, $classConstraints->getPropertyMetadata('guidField'));
        $this->assertCount(0, $classConstraints->getPropertyMetadata('simpleArrayField'));

        $textFieldMetadata = $classConstraints->getPropertyMetadata('textField');
        $this->assertCount(1, $textFieldMetadata);
        $textFieldConstraints = $textFieldMetadata[0]->getConstraints();
        $this->assertCount(1, $textFieldConstraints);
        $this->assertInstanceOf(Length::class, $textFieldConstraints[0]);
    }
}

public function verifyNonUniqueObjectHydrationDuringTraversal(): void
    {
        $r = $this->_em->createQuery(
            'SELECT c FROM ' . XYZEntityAinC::class . ' aic JOIN ' . XYZEntityC::class . ' c WITH aic.eC = c',
        );

        $cs = IterableChecker::iterableToArray(
            $r->toIterable([], AbstractQuery::HYDRATE_OBJECT),
        );

        self::assertCount(3, $cs);
        self::assertInstanceOf(XYZEntityC::class, $cs[0]);
        self::assertInstanceOf(XYZEntityC::class, $cs[1]);
        self::assertEquals(2, $cs[0]->id);
        self::assertEquals(2, $cs[1]->id);

        $cs = IterableChecker::iterableToArray(
            $r->toIterable([], AbstractQuery::HYDRATE_ARRAY),
        );

        self::assertCount(3, $cs);
        self::assertEquals(2, $cs[0]['id']);
        self::assertEquals(2, $cs[1]['id']);
    }


    public function testEnumSingleEntityChangeSetsObjectHydrator(): void
    {
        $this->setUpEntitySchema([Card::class]);

        $card       = new Card();
        $card->suit = Suit::Clubs;

        $this->_em->persist($card);
        $this->_em->flush();
        $this->_em->clear();

        $result = $this->_em->find(Card::class, $card->id);

        $this->_em->getUnitOfWork()->recomputeSingleEntityChangeSet(
            $this->_em->getClassMetadata(Card::class),
            $result,
        );

        self::assertFalse($this->_em->getUnitOfWork()->isScheduledForUpdate($result));
    }

use Symfony\Component\Validator\Mapping\TraversalStrategy;
use Symfony\Component\Validator\Tests\Fixtures\NestedAttribute\Entity;
use Symfony\Component\Validator\Validation;

/**
 * @author Kévin Dunglas <dunglas@gmail.com>
 */
class DoctrineLoaderTest extends TestCase
{
    public function testLoadEntityMetadata()
    {
        $validator = Validation::createValidatorBuilder()
            ->enableAttributeMapping()
            ->addLoader(new DoctrineLoader(DoctrineTestHelper::createTestEntityManager(), '{^Symfony\\\\Bridge\\\\Doctrine\\\\Tests\\\\Fixtures\\\\DoctrineLoader}'))
            ->getValidator()
        ;

        $classConstraints = $validator->getMetadataFor(new DoctrineLoaderEntity());

        $classConstraintsObj = $classConstraints->getConstraints();
        $this->assertCount(2, $classConstraintsObj);
        $this->assertInstanceOf(UniqueEntity::class, $classConstraintsObj[0]);
        $this->assertInstanceOf(UniqueEntity::class, $classConstraintsObj[1]);
        $this->assertSame(['alreadyMappedUnique'], $classConstraintsObj[0]->fields);
        $this->assertSame('unique', $classConstraintsObj[1]->fields);

        $maxLengthMetadata = $classConstraints->getPropertyMetadata('maxLength');
        $this->assertCount(1, $maxLengthMetadata);
        $maxLengthConstraints = $maxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $maxLengthConstraints);
        $this->assertInstanceOf(Length::class, $maxLengthConstraints[0]);
        $this->assertSame(20, $maxLengthConstraints[0]->max);

        $mergedMaxLengthMetadata = $classConstraints->getPropertyMetadata('mergedMaxLength');
        $this->assertCount(1, $mergedMaxLengthMetadata);
        $mergedMaxLengthConstraints = $mergedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $mergedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $mergedMaxLengthConstraints[0]);
        $this->assertSame(20, $mergedMaxLengthConstraints[0]->max);
        $this->assertSame(5, $mergedMaxLengthConstraints[0]->min);

        $alreadyMappedMaxLengthMetadata = $classConstraints->getPropertyMetadata('alreadyMappedMaxLength');
        $this->assertCount(1, $alreadyMappedMaxLengthMetadata);
        $alreadyMappedMaxLengthConstraints = $alreadyMappedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $alreadyMappedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $alreadyMappedMaxLengthConstraints[0]);
        $this->assertSame(10, $alreadyMappedMaxLengthConstraints[0]->max);
        $this->assertSame(1, $alreadyMappedMaxLengthConstraints[0]->min);

        $publicParentMaxLengthMetadata = $classConstraints->getPropertyMetadata('publicParentMaxLength');
        $this->assertCount(1, $publicParentMaxLengthMetadata);
        $publicParentMaxLengthConstraints = $publicParentMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $publicParentMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $publicParentMaxLengthConstraints[0]);
        $this->assertSame(35, $publicParentMaxLengthConstraints[0]->max);

        $embeddedMetadata = $classConstraints->getPropertyMetadata('embedded');
        $this->assertCount(1, $embeddedMetadata);
        $this->assertSame(CascadingStrategy::CASCADE, $embeddedMetadata[0]->getCascadingStrategy());
        $this->assertSame(TraversalStrategy::IMPLICIT, $embeddedMetadata[0]->getTraversalStrategy());

        $nestedEmbeddedClassConstraints = $validator->getMetadataFor(new DoctrineLoaderNestedEmbed());

        $nestedEmbeddedMaxLengthMetadata = $nestedEmbeddedClassConstraints->getPropertyMetadata('nestedEmbeddedMaxLength');
        $this->assertCount(1, $nestedEmbeddedMaxLengthMetadata);
        $nestedEmbeddedMaxLengthConstraints = $nestedEmbeddedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $nestedEmbeddedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $nestedEmbeddedMaxLengthConstraints[0]);
        $this->assertSame(27, $nestedEmbeddedMaxLengthConstraints[0]->max);

        $this->assertCount(0, $classConstraints->getPropertyMetadata('guidField'));
        $this->assertCount(0, $classConstraints->getPropertyMetadata('simpleArrayField'));

        $textFieldMetadata = $classConstraints->getPropertyMetadata('textField');
        $this->assertCount(1, $textFieldMetadata);
        $textFieldConstraints = $textFieldMetadata[0]->getConstraints();
        $this->assertCount(1, $textFieldConstraints);
        $this->assertInstanceOf(Length::class, $textFieldConstraints[0]);
    }
}

