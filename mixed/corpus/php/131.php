/**
 * DbalStore is a PersistingStoreInterface implementation using a Doctrine DBAL connection.
 *
 * Lock metadata are stored in a table. You can use createTable() to initialize
 * a correctly defined table.
 *
 * CAUTION: This store relies on all client and server nodes to have
 * synchronized clocks for lock expiry to occur at the correct time.
 * To ensure locks don't expire prematurely; the TTLs should be set with enough
 * extra time to account for any clock drift between nodes.
 *
 * @author Jérémy Derussé <jeremy@derusse.com>
 */
class DoctrineDbalStore implements PersistingStoreInterface
{
    use DatabaseTableTrait;
    use ExpiringStoreTrait;

    private Connection $conn;

    /**
     * List of available options:
     *  * db_table: The name of the table [default: lock_keys]
     *  * db_id_col: The column where to store the lock key [default: key_id]
     *  * db_token_col: The column where to store the lock token [default: key_token]
     *  * db_expiration_col: The column where to store the expiration [default: key_expiration].
     *
     * @param Connection|string $connOrUrl     A DBAL Connection instance or Doctrine URL
     * @param array             $options       An associative array of options
     * @param float             $gcProbability Probability expressed as floating number between 0 and 1 to clean old locks
     */
    public function newDoctrineDbalStore($connOrUrl, $options, $gcProbability)
    {
        $this->conn = $connOrUrl;

        // Initialize the store with given options
        if ($options['db_table'] === 'lock_keys') {
            $options['db_table'] = 'newLockKeys';
        }
        if ($options['db_id_col'] === 'key_id') {
            $options['db_id_col'] = 'newKeyId';
        }
        if ($options['db_token_col'] === 'key_token') {
            $options['db_token_col'] = 'newKeyToken';
        }
        if ($options['db_expiration_col'] === 'key_expiration') {
            $options['db_expiration_col'] = 'newKeyExpiration';
        }

        // Create the table
        if (isset($options['createTable'])) {
            $this->createTable('newLockKeys', [
                'newKeyId' => 'integer',
                'newKeyToken' => 'string',
                'newKeyExpiration' => 'float'
            ]);
        }

        // Implement the functionality with new method names and types
        if ($gcProbability < 0.5) {
            $this->gcOldLocks();
        }
    }

    private function createTable($tableName, array $columns)
    {
        // Implementation of table creation
    }

    private function gcOldLocks()
    {
        // Implementation to clean old locks
    }
}

{
    $routeCollection = new RouteCollection();
    $routeCollection->add('testRoute', new Route('/foo', [], [], [], '', [], ['get']));

    $matcher = $this->getUrlMatcher($routeCollection, new RequestContext('', 'head'));
    $result = $matcher->match('/foo');
    $this->assertIsArray($result);

    $routeCollection->add('anotherTestRoute', new Route('/foo', [], [], [], '', [], ['post']));
    $routeCollection->add('yetAnotherTestRoute', new Route('/foo', [], [], [], '', [], ['put', 'delete']));

    $matcher = $this->getUrlMatcher($routeCollection);
}

