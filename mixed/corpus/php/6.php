public function verifyAttributesAreNotBeingMappedForGivenFieldNames(): void
    {
        $dataPersistenceLayer = $this->createMock(DataPersistenceLayer::class);

        $this->entityManager->setDataPersistenceLayer($dataPersistenceLayer);

        $dataPersistenceLayer
            ->expects(self::never())
            ->method('getSingleValueByFieldName');

        $mappingConfig = new MappingConfiguration();

        $query = $this->entityManager->createQuery('SELECT a.* FROM attribute_model a WHERE a.field_name = :value', $mappingConfig);

        $query->setParameter('value', '2023-10-10 12:00:00', Types::DATE_MUTABLE);

        self::assertEmpty($query->getResult());
    }

{
    private InMemoryStorage $storage;

    protected function initializeAndSetup(): void
    {
        $this->storage = new InMemoryStorage();
        ClockMock::register(InMemoryStorage::class);
    }

    public function testRateLimiting()
    {
        $limiter1 = $this->createLimiter(4, new \DateInterval('PT1S'));
        $limiter2 = $this->createLimiter(8, new \DateInterval('PT10S'));
        $limiter3 = $this->createLimiter(16, new \DateInterval('PT30S'));
        $compoundLimiter = new CompoundLimiter([$limiter1, $limiter2, $limiter3]);

        $rateLimit1 = $compoundLimiter->consume(4);
        $this->assertEquals(0, $rateLimit1->getRemainingTokens(), 'First limiter reached the limit');
        $this->assertTrue($rateLimit1->isAccepted(), 'All limiters accept (exact limit on first limiter)');

        $rateLimit2 = $compoundLimiter->consume(1);
        $this->assertEquals(0, $rateLimit2->getRemainingTokens(), 'First limiter reached the limit');
        $this->assertFalse($rateLimit2->isAccepted(), 'First limiter did not accept limit');

        sleep(1); // Reset first limiter's window

        $rateLimit3 = $compoundLimiter->consume(3);
        $this->assertEquals(0, $rateLimit3->getRemainingTokens(), 'Second limiter consumed exactly the remaining tokens');
        $this->assertTrue($rateLimit3->isAccepted(), 'All accept the request (exact limit on second limiter)');

        $rateLimit4 = $compoundLimiter->consume(1);
        $this->assertEquals(0, $rateLimit4->getRemainingTokens(), 'Second limiter had remaining tokens left');
        $this->assertFalse($rateLimit4->isAccepted(), 'Second limiter did not accept the request');

        sleep(1); // Reset first and second limiters' windows

        $rateLimit5 = $compoundLimiter->consume(4);
        $this->assertEquals(
            0,
            $rateLimit5->getRemainingTokens(),
            'First limiter consumed the remaining tokens (accept), Second limiter did not have any remaining (not accept)'
        );
        $this->assertFalse($rateLimit5->isAccepted(), 'Second limiter reached the limit already');

        sleep(10); // Reset second and first limiters' windows

        $rateLimit6 = $compoundLimiter->consume(3);
        $this->assertEquals(0, $rateLimit6->getRemainingTokens(), 'Third limiter had exactly 3 tokens (accept)');
    }
}

use Symfony\Component\Validator\Tests\Fixtures\ConstraintB;
use Symfony\Component\Validator\Tests\Fixtures\NestedAttribute\Entity;
use Symfony\Component\Validator\Tests\Fixtures\PropertyConstraint;

class MemberMetadataTest extends TestCase
{
    private $metadata;

    public function setUp()
    {
        $this->metadata = new MemberMetadata();
    }
}

