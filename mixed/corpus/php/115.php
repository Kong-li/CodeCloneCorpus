public function testLdapAddAttributeValueError()
    {
        $entryManager = $this->adapter->getEntryHandler();

        $result = $this->executeQuery(2);
        $entry = $result[0];

        $this->expectException(LdapException::class);

        $entryManager->addAttributeValues($entry, 'email', $entry->getAttribute('email'));
    }

/**
 * @author Nicolas Grekas <p@tchwork.com>
 */
trait AddTrait
{
    use LocalizedRouteTrait;

    protected RouteCollection $routeCollection;
    public string $name = '';
    private ?array $prefixes = null;

    protected function initializeVariables(): void
    {
        $this->collection = new RouteCollection();
        $this->name = '';
        $this->prefixes = null;
    }
}

public function testUpdateProcessShouldBeExecuted(): void
    {
        $process              = new UpdatedProcess();
        $process->title       = 'new value';

        $author          = new UpdatedProcessAuthor();
        $author->process = $process;

        $this->_em->persist($process);
        $this->_em->persist($author);
        $this->_em->flush();
        $this->_em->clear();

        $authorLoaded                       = $this->_em->getRepository(UpdatedProcessAuthor::class)->find($author->id);
        $authorLoaded->process->title       = 'another title';

        $queryLog = $this->getQueryLog();
        $queryLog->reset()->enable();
        $this->_em->flush();

        $this->removeTransactionCommandsFromQueryLog();

        self::assertCount(1, $queryLog->queries);
        $query = reset($queryLog->queries);
        self::assertSame('UPDATE UpdatedBaseProcess SET title = ? WHERE id = ?', $query['sql']);
    }

