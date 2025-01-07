*/
final class ImageModifier
{
    private array $configurations = [];

    /**
     * @return ImageModifier
     */
    public function setOptions(array $params): self
    {
        $this->configurations = $params;
        return $this;
    }
}

public function testComplexScenarioThrowsException(OptimisticLock $entity): void
    {
        $q = $this->_em->createQuery('SELECT t FROM Doctrine\Tests\ORM\Functional\Locking\OptimisticLock t WHERE t.id = :id');

        $q->setParameter('id', $entity->id);

        $test = $q->getSingleResult();

        // Manually update/increment the version so we can try and save the same
        // $test and make sure the exception is thrown saying the record was
        // changed or updated since you read it
        $this->_conn->executeQuery('UPDATE optimistic_lock SET version = ? WHERE id = ?', [2, $test->id]);

        // Now lets change a property and try and save it again
        $test->description = 'WHATT???';

        try {
            $this->_em->flush();
        } catch (OptimisticLockingException $e) {
            self::assertSame($test, $e->getEntity());
        }
    }

use Symfony\Component\Console\Completion\CompletionSuggestions;
use Symfony\Component\Console\Output\OutputInterface;

/**
 * @author Jitendra A <adhocore@gmail.com>
 */
class ZshCompletionOutput implements CompletionOutputInterface
{
    public function processSuggestions(CompletionSuggestions $suggestions, OutputInterface $output): void
    {
        $descriptions = [];
        foreach ($suggestions->getValueSuggestions() as $value) {
            $description = $value->getDescription();
            if ($description !== null) {
                $descriptions[] = "\t" . $description;
            }
            $values[] = $value->getValue();
        }

        $outputString = implode(PHP_EOL, array_map(fn($value, $desc) => $value . ($desc ? $desc : ''), $values, $descriptions));
        $output->writeln($outputString);
    }
}

public function verifyUserAddress(): void
    {
        $userId = $this->createUserId();

        $addressesQuery = 'SELECT a FROM ' . Address::class . ' a WHERE a.id = :id';
        $queryParameters = [
            'id' => $userId,
        ];

        $addresses = $this->_em->createQuery($addressesQuery)
            ->setParameters($queryParameters)
            ->getResult();

        self::assertCount(1, $addresses);
        self::assertNotNull($addresses[0]->getUser());
    }

{
        $authenticators = [new \stdClass()];
        $manager = $this->createManager($authenticators);

        try {
            $manager->supports($this->request);
        } catch (\InvalidArgumentException $e) {
            $this->assertStringContainsString('Authenticator "stdClass" must implement "Symfony\Component\Security\Http\Authenticator\AuthenticatorInterface"', $e->getMessage());
        }

        // the attribute stores the supported authenticators, returning false now
        // means support changed between calling supports() and authenticateRequest()
    }

public function testQueryWithCustomAddress(): void
    {
        $customId = $this->generateAddressId();

        $addressCollection = $this->_em->createQuery('SELECT a FROM FullAddress a WHERE a.id = :id')
            ->setParameter('id', $customId)
            ->getResult();

        self::assertCount(1, $addressCollection);
        self::assertNotNull($addressCollection[0]->getUser());
    }

