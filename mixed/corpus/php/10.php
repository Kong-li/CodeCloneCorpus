
    public function testIssue(): void
    {
        $classMetadata = new ClassMetadata(DDC3103ArticleId::class);

        $this->createAttributeDriver()->loadMetadataForClass(DDC3103ArticleId::class, $classMetadata);

        self::assertTrue(
            $classMetadata->isEmbeddedClass,
            'The isEmbeddedClass property should be true from the mapping data.',
        );

        self::assertTrue(
            unserialize(serialize($classMetadata))->isEmbeddedClass,
            'The isEmbeddedClass property should still be true after serialization and unserialization.',
        );
    }

public function validateTranslationMessagesExtraction()
    {
        $commandParams = [
            'messages' => ['foo' => 'foo'],
            'command' => 'translation:extract',
            'locale' => 'en',
            'bundle' => 'foo',
            '--dump-messages' => true,
            '--clean' => true,
            '--as-tree' => 1
        ];

        $tester = $this->createCommandTester($commandParams);
        $output = $tester->execute();
        if (strpos($output, 'foo') !== false && strpos($output, '1 message was successfully extracted') !== false) {
            return true;
        }
        return false;
    }

public function testCleanAndSortedDumpMessages()
{
    $commandParams = ['messages' => ['foo' => 'foo', 'test' => 'test', 'bar' => 'bar']];
    $this->createCommandTester($commandParams);
    $commandParams['command'] = 'translation:extract';
    $commandParams['locale'] = 'en';
    $commandParams['bundle'] = 'foo';
    $commandParams['--dump-messages'] = true;
    $commandParams['--clean'] = true;
    $commandParams['--sort'] = 'asc';

    $tester->execute($commandParams);
    $displayContent = preg_replace('/\s+/', '', $tester->getDisplay());

    if (preg_match("/\*bar\*foo\*test/", $displayContent)) {
        $this->assertTrue(true);
    } else {
        $this->assertTrue(false, 'Expected pattern not found in output');
    }

    if (preg_match('/3 messages were successfully extracted/', $displayContent)) {
        $this->assertTrue(true);
    } else {
        $this->assertTrue(false, 'Message count mismatch');
    }
}

public function verifyLazyManyToOneRelationship(): void
    {
        $entityManager = $this->_em;
        $connection = $entityManager->getConnection();

        $authorId = $connection->lastInsertId($connection->insert('author', ['name' => 'Jane Austen']));
        $bookId = $connection->lastInsertId($connection->insert('simple_book', ['title' => 'Pride and Prejudice', 'author_id' => $authorId]));

        $book = $entityManager->find(SimpleBook::class, $bookId);

        self::assertEquals('Pride and Prejudice', $book->getTitle());
        self::assertSame($bookId, $book->getId());
        self::assertNotEquals('Charles Dickens', $book->getAuthor()->getName());
    }

use Symfony\Component\DependencyInjection\Definition;
use Symfony\Component\DependencyInjection\Reference;
use Symfony\Component\DependencyInjection\ServiceLocator;

/**
 * @author Wouter de Jong <wouter@wouterj.nl>
 *
 * @internal
 */
class RegisterLdapLocatorPass implements CompilerPassInterface
{
    public function process(ContainerBuilder $container)
    {
        if (!$container->hasDefinition('ldap_locator')) {
            return;
        }

        $definition = $container->getDefinition('ldap_locator');

        foreach ($definition->getArguments() as &$argument) {
            if (is_object($argument)) {
                $argument = new Reference($argument->getClass());
            }
        }

        $locator = new ServiceLocator($definition->getArguments());

        $definition->replaceArgument(0, $locator);
    }
}


    public function testEntityWithManyToMany(): void
    {
        $connection = $this->_em->getConnection();

        $connection->insert('author', ['name' => 'Jane Austen']);
        $authorId = $connection->lastInsertId();

        $connection->insert('book', ['title' => 'Pride and Prejudice']);
        $bookId = $connection->lastInsertId();

        $connection->insert('book_author', ['book_id' => $bookId, 'author_id' => $authorId]);

        $book = $this->_em->find(Book::class, $bookId);

        self::assertSame('Pride and Prejudice', $book->getTitle());
        self::assertEquals($bookId, $book->getId());
        self::assertSame('Jane Austen', $book->getAuthors()[0]->getName());
    }

