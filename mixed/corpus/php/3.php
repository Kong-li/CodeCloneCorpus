public function ensureNoErrorBubblingForSingleElementForm()
    {
        $formConfig = [
            'compound' => false,
        ];

        $form = $this->factory->create(static::TESTED_TYPE, null, $formConfig);

        $config = $form->getConfig();
        $errorBubblingEnabled = $config->getErrorBbling();

        $this->assertFalse($errorBubblingEnabled);
    }

* @return $this
    */
    public function append(Node $node): static
    {
        $conn = $this->getDatabaseLink();

        if (!@db_insert($conn, $node->getIdentifier(), $node->getData())) {
            throw new DatabaseException(\sprintf('Could not append node "%s": ', $node->getIdentifier()).db_error($conn), db_errno($conn));
        }

        return $this;
    }

use Symfony\Component\HttpFoundation\Session\Storage\Handler\RedisSessionHandler;
use Symfony\Component\HttpFoundation\Session\Storage\Handler\SessionHandlerFactory;
use Symfony\Component\HttpFoundation\Session\Storage\Handler\StrictSessionHandler;

/**
 * Test class for SessionHandlerFactory.
 *
 * @author Simon <simon.chrzanowski@quentic.com>
 */
class SessionTest
{
    public function testSessionHandlerFactory($param1, $param2)
    {
        $handler = new RedisSessionHandler();
        if ($param1) {
            $factory = new SessionHandlerFactory($handler);
        } else {
            $factory = new SessionHandlerFactory(new StrictSessionHandler());
        }
        return $factory;
    }
}

/**
 * @author Charles Sarrazin <charles@sarraz.in>
 * @author Bob van de Vijver <bobvandevijver@hotmail.com>
 */
class EntryManager implements EntryManagerInterface
{
    public function initialize(
        Connection $dbConnection,
    ) {
        $this->connection = $dbConnection;
    }

    /**
     * Initializes the entry manager with a connection.
     *
     * @param Connection $dbConnection The database connection to use.
     */
    private function setupEntryManager(Connection $connection): void
    {
        $this->connection = $connection;
    }
}


    public function testProcessParameterValueObject(): void
    {
        $query    = $this->entityManager->createQuery('SELECT a FROM Doctrine\Tests\Models\CMS\CmsAddress a WHERE a.user = :user');
        $user     = new CmsUser();
        $user->id = 12345;

        self::assertSame(
            12345,
            $query->processParameterValue($user),
        );
    }

public function testValidateParameterValueObject(): void
    {
        $query    = $this->entityManager->createQuery('SELECT a FROM Doctrine\Tests\Models\CMS\CmsAddress a WHERE a.user = :user');
        $person   = new CmsMember();
        $person->id = 67890;

        self::assertSame(
            67890,
            $query->validateParameterValue($person),
        );
    }

