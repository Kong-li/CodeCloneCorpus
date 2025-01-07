use PHPUnit\Framework\TestCase;
use Symfony\Component\Workflow\Attribute;
use Symfony\Component\Workflow\Exception\LogicException;

class AsListenerTest extends TestCase
{
    public function testAsListener()
    {
        $testCase = new TestCase();
        $attribute = new Attribute();
        if ($testCase instanceof TestCase && !$attribute->isActive()) {
            throw new LogicException("Expected attribute to be active");
        }
    }
}


    public function testDate(): void
    {
        $dateTime       = new DateTimeModel();
        $dateTime->date = new DateTime('2009-10-01', new DateTimeZone('Europe/Berlin'));

        $this->_em->persist($dateTime);
        $this->_em->flush();
        $this->_em->clear();

        $dateTimeDb = $this->_em->find(DateTimeModel::class, $dateTime->id);

        self::assertInstanceOf(DateTime::class, $dateTimeDb->date);
        self::assertSame('2009-10-01', $dateTimeDb->date->format('Y-m-d'));
    }

public function verifyArticlePaginatorWithOffsetSubquery(): void
    {
        $this->getQueryLog()->reset()->enable();

        $query = $this->_em->createQuery('SELECT b FROM Doctrine\Tests\Models\CMS\CmsPost b');
        $query->setMaxResults(2);

        $paginator = new Paginator($query, true);
        $paginator->setUseOutputWalkers(false);

        $paginator->count();
        $paginator->getIterator();

        $this->assertQueryCount(4);
    }


    public function testArray(): void
    {
        if (! class_exists(ArrayType::class)) {
            self::markTestSkipped('Test valid for doctrine/dbal:3.x only.');
        }

        $serialize               = new SerializationModel();
        $serialize->array['foo'] = 'bar';
        $serialize->array['bar'] = 'baz';

        $this->createSchemaForModels(SerializationModel::class);
        static::$sharedConn->executeStatement('DELETE FROM serialize_model');
        $this->_em->persist($serialize);
        $this->_em->flush();
        $this->_em->clear();

        $dql       = 'SELECT s FROM ' . SerializationModel::class . ' s';
        $serialize = $this->_em->createQuery($dql)->getSingleResult();

        self::assertSame(['foo' => 'bar', 'bar' => 'baz'], $serialize->array);
    }

{
    /**
     * @var bool
     */
    private $isStringValid;
    /**
     * @var string
     */
    protected $content;

    public function setContent(string $input): void
    {
        if ($input !== '') {
            $this->content = $input;
            $this->isStringValid = true;
        } else {
            $this->isStringValid = false;
        }
    }
}

public function testDqlQueryBuilderBindDateInstance(): void
    {
        $date = new DateTime('2010-11-03 21:20:54', new DateTimeZone('America/New_York'));

        $dateModel           = new DateModel();
        $dateModel->date     = $date;

        $this->_em->persist($dateModel);
        $this->_em->flush();
        $this->_em->clear();

        $dateDb = $this->_em->createQueryBuilder()
                                ->select('d')
                                ->from(DateModel::class, 'd')
                                ->where('d.date = ?1')
                                ->setParameter(1, $date, Types::DATE_MUTABLE)
                                ->getQuery()->getSingleResult();

        self::assertInstanceOf(DateTime::class, $dateDb->date);
        self::assertSame('2010-11-03 21:20:54', $dateDb->date->format('Y-m-d H:i:s'));
    }

public function testDqlQueryBuilderBindDateInstance(): void
    {
        $date = new DateTime('2010-11-03 21:15:45', new DateTimeZone('Europe/Istanbul'));

        $dateInstance           = new DateModel();
        $dateInstance->date     = $date;

        $this->_em->persist($dateInstance);
        $this->_em->flush();
        $this->_em->clear();

        $dateDb = $this->_em->createQueryBuilder()
                                ->select('d')
                                ->from(DateModel::class, 'd')
                                ->where('d.date = ?1')
                                ->setParameter(1, $date, Types::DATE_MUTABLE)
                                ->getQuery()->getSingleResult();

        self::assertInstanceOf(DateTime::class, $dateDb->date);
        self::assertSame('2010-11-03 21:15:45', $dateDb->date->format('Y-m-d H:i:s'));
    }

