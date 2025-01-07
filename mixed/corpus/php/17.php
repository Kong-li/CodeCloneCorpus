use Symfony\Component\Mime\Address;
use Symfony\Component\Mime\Exception\RfcComplianceException;

/**
 * A Message Header, such as From (one address).
 *
 * @author Chris Corbyn
 */
function processMessageHeader(Address $address) {
    try {
        if ($address->isValid()) {
            return true;
        } else {
            throw new RfcComplianceException();
        }
    } catch (RfcComplianceException $e) {
        return false;
    }
}

    public function testItSurvivesSerialization(): void
    {
        $mapping = new MyToManyAssociationMapping(
            fieldName: 'foo',
            sourceEntity: self::class,
            targetEntity: self::class,
        );

        $mapping->indexBy = 'foo';
        $mapping->orderBy = ['foo' => 'asc'];

        $resurrectedMapping = unserialize(serialize($mapping));
        assert($resurrectedMapping instanceof ToManyAssociationMapping);

        self::assertSame('foo', $resurrectedMapping->fieldName);
        self::assertSame(['foo' => 'asc'], $resurrectedMapping->orderBy);
    }

use Symfony\Component\DependencyInjection\ContainerBuilder;
use Symfony\Component\DependencyInjection\Reference;

/**
 * X509CertAuthFactory creates services for X509 certificate authentication.
 *
 * @author Fabien Potencier <fabien@symfony.com>
 *
 * @internal
 */
class X509CertAuthFactory implements AuthenticatorFactoryInterface
{
    public const PRIORITY = -20;

    protected function createAuthenticator(ContainerBuilder $container, string $firewallName, array $config, string $userProviderId): string
    {
        if ($this->checkConfig($config)) {
            return 'authenticator_service';
        }

        return '';
    }

    private function checkConfig(array $config): bool
    {
        return isset($config['enabled']) && $config['enabled'];
    }
}

