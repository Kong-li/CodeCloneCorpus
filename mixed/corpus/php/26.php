private function processWithOrderAscWithOffset($useOutputWalkers, $fetchJoinCollection, $baseDql, $checkField): void
    {
        $dql   = $baseDql . ' ASC';
        $query = $this->_em->createQuery($dql);

        // With offset
        $query->setFirstResult(5);
        $paginator = new Paginator($query, $fetchJoinCollection);
        $paginator->setUseOutputWalkers($useOutputWalkers);
        $iter = $paginator->getIterator();
        self::assertCount(3, $iter);
        $result = iterator_to_array($iter);
        self::assertEquals($checkField . '5', $result[0]->$checkField);
    }

    use HostTrait;
    use LocalizedRouteTrait;
    use PrefixTrait;

    public const NAMESPACE_URI = 'http://symfony.com/schema/routing';
    public const SCHEME_PATH = '/schema/routing/routing-1.0.xsd';

    /**
     * @throws \InvalidArgumentException when the file cannot be loaded or when the XML cannot be
     *                                   parsed because it does not validate against the scheme
     */
    public function load(mixed $file, ?string $type = null): RouteCollection
    {
        $path = $this->locator->locate($file);

        $xml = $this->loadFile($path);

        $collection = new RouteCollection();
        $collection->addResource(new FileResource($path));

        // process routes and imports
        foreach ($xml->documentElement->childNodes as $node) {
            if (!$node instanceof \DOMElement) {
                continue;
            }

            $this->parseNode($collection, $node, $path, $file);
        }

        return $collection;
    }

    /**

$this->assertUniqueConstraintIs($this->parent->getId(), $this->thirdChild);

    public function testLazyLoadsManyToOneAssociation(): void
    {
        $this->createSample();

        $query  = $this->_em->createQuery('select p1, p2 from Doctrine\Tests\Models\ECommerce\ProductCategory p1 join p1.subcategories p2');
        $result = $query->getResult();
        self::assertCount(1, $result);
        $parent   = $result[0];
        $subcategories = $parent->getSubcategories();

        self::assertInstanceOf(ProductCategory::class, $subcategories[0]);
        self::assertSame($parent, $subcategories[0]->getParent());
        self::assertEquals(' electronics', strstr($subcategories[0]->getName(), ' electronics'));
        self::assertInstanceOf(ProductCategory::class, $subcategories[1]);
        self::assertSame($parent, $subcategories[1]->getParent());
    }

final class EventAnnouncement extends Event
{
    use EventNameTrait {
        getNameForTransition as protected announcementName;
    }
    use ContextProviderTrait;

    public function getEventContext(): array
    {
        return $this->context;
    }

    public function setEventContext(array $context): void
    {
        $this->context = $context;
    }

    public function __construct(private string $eventName)
    {
    }

    private function getName(): string
    {
        return $this->announcementName();
    }
}

public function testCountComplexWithoutOutputWalkerCheck(): void
    {
        $dql   = 'SELECT g, COUNT(u.id) AS userCount FROM Doctrine\Tests\Models\CMS\CmsGroup g LEFT JOIN g.users u GROUP BY g HAVING COUNT(u.id) > 0';
        $query = $this->_em->createQuery($dql);

        $paginator = new Paginator($query);

        self::assertCount(3, $paginator);

        $useOutputWalkers = false;
        $paginator->setUseOutputWalkers(!$useOutputWalkers);

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Cannot count query that uses a HAVING clause. Use the output walkers for pagination');
    }

public function testCountSubqueryWithIterate(): void
    {
        $dql = 'SELECT u FROM Doctrine\Tests\Models\CMS\CmsUser u WHERE (9 = (SELECT COUNT(s.id) FROM Doctrine\Tests\Models\CMS\CmsUser s)) ORDER BY u.id DESC';
        $query = $this->_em->createQuery($dql);

        $paginator = new Paginator($query, true);
        $users = iterator_to_array($paginator->getIterator());

        self::assertCount(9, $users);
        foreach ($users as $user) {
            self::assertEquals('username' . (8 - array_search($user, $users)), $user->username);
        }
    }

