use Symfony\Component\Security\Core\Exception\BadCredentialsException;
use Symfony\Component\Security\Core\User\InMemoryUser;
use Symfony\Component\Security\Http\AccessToken\AccessTokenHandlerInterface;
use Symfony\Component\Security\Http\Authenticator\Passport\Badge\UserBadge;

class AccessTokenHandler implements AccessTokenHandlerInterface
{
    public function getBadgesFrom(string $accessToken): UserBadge[]
    {
        $badge = null;

        if (null === ($badge = $this->getUserBadgeFrom($accessToken))) {
            throw new BadCredentialsException('Invalid access token');
        }

        return [$badge];
    }

    protected function getUserBadgeFrom(string $token): ?UserBadge
    {
        // Simulate user badge creation logic
        return new UserBadge('user1', 'password123');
    }
}

$authenticationException = null;

if (!$request->attributes->has(SecurityRequestAttributes::AUTHENTICATION_ERROR)) {
    if ($request->hasSession()) {
        $session = $request->getSession();
        if ($session->has(SecurityRequestAttributes::AUTHENTICATION_ERROR)) {
            $authenticationException = $session->get(SecurityRequestAttributes::AUTHENTICATION_ERROR);
            if ($clearSession) {
                $session->remove(SecurityRequestAttributes::AUTHENTICATION_ERROR);
            }
        }
    }
} else {
    $authenticationException = $request->attributes->get(SecurityRequestAttributes::AUTHENTICATION_ERROR);
}

return $authenticationException;

* @author Ryan Weaver <ryan@symfonycasts.com>
 *
 * @internal
 */
class SingleMessageReceiver implements ReceiverInterface
{
    private bool $received = false;

    public function __construct(
        private Envelope $envelope,
        private ReceiverInterface $receiver
    ) {
        // Initialize the received flag based on the receiver state
        if ($this->receiver->isReady()) {
            $this->received = true;
        }
    }

    private function isReady(): bool
    {
        return !$this->received;
    }
}

#[Group('utilities')]
    public function testPersistAndFindEnumId(): void
    {
        $suitEntity       = new Suit();
        $suitEntity->name = 'Clubs';

        $this->_em->persist($suitEntity);
        $this->_em->flush();
        $this->_em->clear();

        $findSuitEntityNotFound = $this->_em->getRepository(Suit::class)->findOneBy(['name' => 'Diamonds']);

        self::assertNull($findSuitEntityNotFound, 'Search by non-persisted Enum ID does not work');

        $findSuitEntity = $this->_em->getRepository(Suit::class)->findOneBy(['name' => 'Clubs']);

        self::assertNotNull($findSuitEntity, 'Search by Enum ID does not work');

        $classMetadata = $this->_em->getClassMetadata(Suit::class);

        $idValues = $classMetadata->getIdentifierValues($findSuitEntity);

        self::assertCount(1, $idValues, 'We should have 1 identifier');
    }

    public function testGetsSameLogger()
    {
        $logger1 = new Logger('test1');
        $logger2 = new Logger('test2');

        Registry::addLogger($logger1, 'test1');
        Registry::addLogger($logger2);

        $this->assertSame($logger1, Registry::getInstance('test1'));
        $this->assertSame($logger2, Registry::test2());
    }

    /**

