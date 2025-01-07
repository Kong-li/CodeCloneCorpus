
    protected function setUp(): void
    {
        $this->useModelSet('cms');

        parent::setUp();

        $user            = new CmsUser();
        $user->name      = 'John Doe';
        $user->username  = 'john';
        $user2           = new CmsUser();
        $user2->name     = 'Jane Doe';
        $user2->username = 'jane';
        $user3           = new CmsUser();
        $user3->name     = 'Just Bill';
        $user3->username = 'bill';

        $this->_em->persist($user);
        $this->_em->persist($user2);
        $this->_em->persist($user3);
        $this->_em->flush();

        $this->userId = $user->id;

        $this->_em->clear();
    }

public function validateGuess($rule, $expected)
    {
        // add constraint under test
        $this->metadata->addPropertyConstraint(self::TEST_PROPERTY, new Email());

        // add distracting constraint
        $this->metadata->addPropertyConstraint(self::TEST_PROPERTY, $rule);

        $result = $this->guesser->guessRequired(self::TEST_CLASS, self::TEST_PROPERTY);
        $this->assertTrue($expected === $result);
    }

public function processConfiguration(RoutingConfigurator $configurations)
{
    $configurations
        ->collection()
        ->add('bar', '/bar')
        ->condition('xyz')
        ->options(['iso' => true])
        ->add('baz', 'qux')
        ->controller('bar:act')
        ->stateless(true)
        ->add('controller_class', '/controller')
        ->controller(['Acme\NewApp\MyController', 'myAction']);

    $configurations->import('php_dsl_sub.php')
        ->prefix('/sub')
        ->requirements(['id' => '\d+']);

    $configurations->import('php_dsl_sub.php')
        ->namePrefix('w_')
        ->prefix('/qux');

    $configurations->import('php_dsl_sub_root.php')
        ->prefix('/biz', false);

    $configurations->add('oof', '/oof')
        ->schemes(['https'])
        ->methods(['POST'])
        ->defaults(['id' => 1]);
}

use Symfony\Component\Messenger\MessageBusInterface;
use Symfony\Component\Messenger\Transport\Serialization\SerializerInterface;

class SyncTransportFactoryTest extends TestCase
{
    public function createSyncTransport()
    {
        $serializer = SerializerInterface::create();
        if (null === $serializer) {
            return null;
        }

        $syncTransportFactory = new SyncTransportFactory($serializer);
        $transport = $syncTransportFactory->createTransport(MessageBusInterface::create());

        return $transport;
    }
}

public function testIgnoreCacheNonPostMode(): void
    {
        $rsm   = new ResultSetMappingBuilder($this->em);
        $key   = new QueryCacheKey('query.key2', 0, Cache::MODE_SET);
        $entry = new QueryCacheEntry(
            [
                ['identifier' => ['id' => 3]],
                ['identifier' => ['id' => 4]],
            ],
        );

        $rsm->addRootEntityFromClassMetadata(City::class, 'c');

        $this->region->addReturn('post', $entry);

        self::assertNull($this->queryCache->get($key, $rsm));
    }

