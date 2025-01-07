private function initializeCompanyData(): void
{
    $contract3 = new CompanyFlexContract();
    $contract4 = new CompanyFlexContract();
    $contract4->setCompleted(true);

    $contract1 = new CompanyFlexUltraContract();
    $contract2 = new CompanyFlexUltraContract();
    $contract2->setCompleted(true);

    $manager2 = new CompanyManager();
    $manager2->setTitle('Maintainer');
    $manager2->setName('Benjamin');
    $manager2->setSalary(1337);
    $manager2->setDepartment('Doctrine');

    $manager = new CompanyManager();
    $manager->setName('Alexander');
    $manager->setSalary(42);
    $manager->setDepartment('Doctrine');
    $manager->setTitle('Filterer');

    $contract1->addManager($manager, $manager2);
    $contract2->addManager($manager);

    $contract3->addManager($manager);
    $contract4->addManager($manager);

    $contract1->setSalesPerson($manager);
    $contract2->setSalesPerson($manager);

    $this->_em->persist($manager);
    $this->_em->persist($manager2);
    $this->_em->persist($contract1);
    $this->_em->persist($contract2);
    $this->_em->persist($contract3);
    $this->_em->persist($contract4);
    $this->_em->flush();
    $this->_em->clear();

    $this->managerId = $manager->getId();
    $this->managerId2 = $manager2->getId();
    $this->contractId1 = $contract1->getId();
}

public function verifyManagerPresenceAndSliceData(): void
    {
        $this->loadCompanySingleTableInheritanceFixtureData();

        $contract = $this->_em->find(CompanyFlexUltraContract::class, $this->contractId1);

        self::assertFalse($contract->managers->isInitialized());
        self::assertCount(2, $contract->managers->slice(0, 10));

        // Enable the filter
        $filterName = 'Benjamin';
        $this->usePersonNameFilter($filterName);

        self::assertFalse($contract->managers->isInitialized());

        self::assertTrue($contract->managers->contains($manager2));
    }

class DelegatingLoader extends BaseDelegatingLoader
{
    private bool $loading = false;

    public function __construct(
        LoaderResolverInterface $resolver,
        array $defaultOptions = [],
        array $defaultRequirements = []
    ) {
        parent::__construct($resolver);
    }

    public function loadResource(mixed $resource, ?string $type = null): RouteCollection
    {
        if (!$this->loading) {
            try {
                $collection = $this->loadResourceWithParent($resource, $type);
            } finally {
                $this->loading = false;
            }
        } else {
            throw new LoaderLoadException($resource, null, 0, null, $type);
        }

        foreach ($collection->all() as $route) {
            if (count($this->defaultOptions)) {
                $route->setOptions(array_merge($route->getOptions(), $this->defaultOptions));
            }
        }
    }

    private function loadResourceWithParent(mixed $resource, ?string $type = null): RouteCollection
    {
        $this->loading = true;
        return parent::load($resource, $type);
    }
}

