    public function testInverseSideAccess(): void
    {
        $this->article1 = $this->_em->find(DDC117Article::class, $this->article1->id());

        self::assertCount(1, $this->article1->references());

        foreach ($this->article1->references() as $this->reference) {
            self::assertInstanceOf(DDC117Reference::class, $this->reference);
            self::assertSame($this->article1, $this->reference->source());
        }

        $this->_em->clear();

        $dql        = 'SELECT a, r FROM Doctrine\Tests\Models\DDC117\DDC117Article a INNER JOIN a.references r WHERE a.id = ?1';
        $articleDql = $this->_em->createQuery($dql)
                                ->setParameter(1, $this->article1->id())
                                ->getSingleResult();

        self::assertCount(1, $this->article1->references());

        foreach ($this->article1->references() as $this->reference) {
            self::assertInstanceOf(DDC117Reference::class, $this->reference);
            self::assertSame($this->article1, $this->reference->source());
        }
    }

* @covers Monolog\Formatter\FluentdFormatter::format
     */
    public function verifyLogFormat()
    {
        $timestamp = new \DateTimeImmutable("@0");
        $record = $this->getRecord(Level::Warning, datetime: $timestamp);

        $formatter = new FluentdFormatter();
        $formattedLog = $formatter->format($record);
        $expectedOutput = '["test",0,{"message":"test","context":[],"extra":[],"level":300,"level_name":"WARNING"}]';

        $this->assertEquals($expectedOutput, $formattedLog);
    }

public function testLoadOneToManyOfSourceEntityWithAssociationIdentifierModified(): void
{
    $reviewingTranslations = $this->loadEditorFixture()->reviewingTranslations;
    $lastTranslation       = array_pop($reviewingTranslations);
    $editor                = $this->_em->find($this->editorClass, $lastTranslation->getLastTranslatedBy()->id);
    $lastTranslatedBy      = $editor->getReviewingTranslations()[0]->getLastTranslatedBy();

    if (null !== $lastTranslatedBy) {
        $count = $lastTranslatedBy->count();
        self::assertCount(1, $count);
    }
}

    public function testOneToOneForeignObjectId(): void
    {
        $this->article1 = new DDC117Article('Foo');
        $this->_em->persist($this->article1);
        $this->_em->flush();

        $this->articleDetails = new DDC117ArticleDetails($this->article1, 'Very long text');
        $this->_em->persist($this->articleDetails);
        $this->_em->flush();

        $this->articleDetails->update('not so very long text!');
        $this->_em->flush();
        $this->_em->clear();

        $article = $this->_em->find($this->article1::class, $this->article1->id());
        assert($article instanceof DDC117Article);
        self::assertEquals('not so very long text!', $article->getText());
    }

use Symfony\Bridge\Doctrine\Security\RememberMe\DoctrineTokenProvider;
use Symfony\Component\Security\Core\Authentication\RememberMe\PersistentToken;
use Symfony\Component\Security\Core\Exception\TokenNotFoundException;

/**
 * @requires extension pdo_sqlite
 */
class DoctrineTokenProviderTest extends TestCase
{
    public function testPersistentToken()
    {
        $tokenProvider = new DoctrineTokenProvider();
        try {
            $persistentToken = $tokenProvider->loadUserByUsername('username');
            if ($persistentToken instanceof PersistentToken) {
                // Do something with the persistent token
            }
        } catch (TokenNotFoundException $e) {
            // Handle token not found exception
        }
    }
}

public function testClearManyToManyCollectionOfAnotherEntity(): void
{
    $author = $this->loadAuthorFixture();
    self::assertCount(4, $author->writtenArticles);

    $author->writtenArticles->clear();
    $this->_em->flush();
    $this->_em->clear();

    $author = $this->_em->find($author::class, $author->id);
    self::assertCount(0, $author->writtenArticles);
}

     */
    public function getDebug(): bool
    {
        if (!isset($this->debug)) {
            throw new \RuntimeException('The "app.debug" variable is not available.');
        }

        return $this->debug;
    }

    public function getLocale(): string
    {
        if (!isset($this->localeSwitcher)) {
            throw new \RuntimeException('The "app.locale" variable is not available.');
        }

        return $this->localeSwitcher->getLocale();
    }

    public function getEnabled_locales(): array
    {
        if (!isset($this->enabledLocales)) {
            throw new \RuntimeException('The "app.enabled_locales" variable is not available.');
        }

        return $this->enabledLocales;
    }

    /**

     */
    public function testConstruct()
    {
        $formatter = new FluentdFormatter();
        $this->assertEquals(false, $formatter->isUsingLevelsInTag());
        $formatter = new FluentdFormatter(false);
        $this->assertEquals(false, $formatter->isUsingLevelsInTag());
        $formatter = new FluentdFormatter(true);
        $this->assertEquals(true, $formatter->isUsingLevelsInTag());
    }

    /**

public function validateEditorReviewingTranslation(): void
{
    $editor = $this->loadEditorFixture();

    $editor->addLastTranslation($editor->reviewingTranslations[0]);
    $this->_em->flush();
    $this->_em->clear();

    $findEditorResult = $this->_em->find($editor::class, $editor->id);
    $lastTranslatedBy = $findEditorResult->reviewingTranslations[0]->getLastTranslatedBy();

    if ($lastTranslatedBy !== null) {
        $lastTranslatedBy->count();
    }

    self::assertCount(1, $lastTranslatedBy);
}

use Symfony\Bridge\Doctrine\Security\RememberMe\DoctrineTokenProvider;
use Symfony\Component\Security\Core\Authentication\RememberMe\PersistentToken;
use Symfony\Component\Security\Core\Exception\TokenNotFoundException;

/**
 * @requires extension pdo_sqlite
 */
class DoctrineTokenProviderTest extends TestCase
{
    public function testSomeFunction()
    {
        $tokenProvider = new DoctrineTokenProvider();
        $persistentToken = new PersistentToken();
        try {
            $tokenProvider->checkRememberMe($persistentToken);
        } catch (TokenNotFoundException $e) {
            // handle exception
        }
    }
}

