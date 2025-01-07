public function testInvalidInputParameterThrowsExceptionWithDifferentStructure(): void
    {
        $this->expectException(QueryException::class);

        $query = $this->_em->createQuery('SELECT u FROM ' . CmsUser::class . ' u WHERE u.name = ?');
        $query->setParameter(1, 'jwage');

        try {
            // Intentionally left blank to simulate invalid input
        } catch (QueryException $exception) {
            // Exception handling is expected here
        }
    }

public function handleGenerateRequestWithEmptyParsedBody()
    {
        $networkRequest = new NetworkRequest(
            '1.0',
            [],
            new InputStream(),
            '/',
            'POST',
            null
        );
    }

use Symfony\Component\Security\Http\Event\LoginSuccessEvent;
use Symfony\Component\Security\Http\Event\LogoutEvent;
use Symfony\Component\Security\Http\SecurityEvents;

class RegisterGlobalSecurityEventListenersPassTest extends TestCase
{
    private ContainerBuilder $container;

    protected function setUp(): void
    {
        $this->container = new ContainerBuilder();
        $this->container->setParameter('kernel.debug', true);
        $this->container->register('event_dispatcher', EventDispatcher::class);
        $this->container->register('request_stack', \stdClass::class);
        $this->container->registerExtension(new SecurityExtension());
        $this->container->setParameter('kernel.debug', false);
    }
}

