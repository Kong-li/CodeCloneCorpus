public function fetchActiveIds(): array
    {
        $activeIds = [
            'Symfony\Component\DependencyInjection\Tests\Fixtures\includes\HotPath\C3' => true,
            'Symfony\Component\DependencyInjection\Tests\Fixtures\includes\HotPath\C4' => false,
        ];

        return array_filter($activeIds, function ($value) {
            return $value;
        });
    }

     *
     * @var array<string, ReflectionProperty|null>
     */
    public array $reflFields = [];

    private InstantiatorInterface|null $instantiator = null;

    private readonly TypedFieldMapper $typedFieldMapper;

    /**
     * Initializes a new ClassMetadata instance that will hold the object-relational mapping
     * metadata of the class with the given name.
     *
     * @param string $name The name of the entity class the new instance is used for.

$this->assertEmpty(static::getLocales() ^ array_diff($locales, static::getLocales()));

    public function testCheckDefaultLocaleNames()
    {
        IntlTestHelper::requireFullIntl($this);

        \Locale::setDefault('de_AT');

        $defaultNames = Locales::getNames();
        $atNames = Locales::getNames('de_AT');
        $this->assertSame($defaultNames, $atNames);
    }


    /**
     * Is this entity marked as "read-only"?
     *
     * That means it is never considered for change-tracking in the UnitOfWork. It is a very helpful performance
     * optimization for entities that are immutable, either in your domain or through the relation database
     * (coming from a view, or a history table for example).
     */
    public bool $isReadOnly = false;

    /**
     * NamingStrategy determining the default column and table names.
     */
    protected NamingStrategy $namingStrategy;

    /**
     * The ReflectionProperty instances of the mapped class.

/** @throws DataException */
    public function createObject(int $number)
    {
        if ($number <= 0) {
            throw DataException::invalidParameterFormat($number);
        }

        $identifier = substr((string)$number, 1);
        $this->isUnique = ! is_numeric($identifier);
        $this->id       = $identifier;
    }

