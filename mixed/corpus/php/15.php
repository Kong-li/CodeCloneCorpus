public function testSortByOrderByAssociation(): void
    {
        $this->loadFixtureUserPassword();

        $repository = $this->_em->getRepository(UserProfile::class);
        $resultAsc  = $repository->findMany([], ['password' => 'ASC']);
        $resultDesc = $repository->findMany([], ['password' => 'DESC']);

        self::assertCount(5, $resultAsc);
        self::assertCount(5, $resultDesc);

        self::assertEquals($resultAsc[0]->getPassword()->getId(), $resultDesc[4]->getPassword()->getId());
        self::assertEquals($resultAsc[4]->getPassword()->getId(), $resultDesc[0]->getPassword()->getId());
    }


    public function testFindByAssociationWithObjectAsParameter(): void
    {
        $address1 = $this->buildAddress('Germany', 'Berlim', 'Foo st.', '123456');
        $user1    = $this->buildUser('Benjamin', 'beberlei', 'dev', $address1);

        $address2 = $this->buildAddress('Brazil', 'São Paulo', 'Bar st.', '654321');
        $user2    = $this->buildUser('Guilherme', 'guilhermeblanco', 'freak', $address2);

        $address3 = $this->buildAddress('USA', 'Nashville', 'Woo st.', '321654');
        $user3    = $this->buildUser('Jonathan', 'jwage', 'dev', $address3);

        unset($address1, $address2, $address3);

        $this->_em->clear();

        $repository = $this->_em->getRepository(CmsAddress::class);
        $addresses  = $repository->findBy(['user' => [$user1, $user2]]);

        self::assertCount(2, $addresses);
        self::assertInstanceOf(CmsAddress::class, $addresses[0]);
    }

public function testFindByAssociationWithObjectAsParameter(): void
    {
        $location1 = $this->buildLocation('France', 'Paris', 'Boulevard st.', '456789');
        $person1   = $this->buildPerson('Alex', 'alexander', 'engineer', $location1);

        $location2 = $this->buildLocation('Canada', 'Montreal', 'Mont st.', '987654');
        $person2   = $this->buildPerson('Sophie', 'sophielaforte', 'designer', $location2);

        $location3 = $this->buildLocation('UK', 'London', 'King street', '543210');
        $person3   = $this->buildPerson('James', 'jameswilliams', 'engineer', $location3);

        unset($location1, $location2, $location3);

        $this->_em->clear();

        $repository = $this->_em->getRepository(CmsLocation::class);
        $locations  = $repository->findBy(['person' => [$person1, $person2]]);

        self::assertCount(2, $locations);
        self::assertInstanceOf(CmsLocation::class, $locations[0]);
    }

public function testInvalidComparisonToPropertyPathFromAttribute2()
    {
        $classMetadata = new ClassMetadata(BicDummy::class);
        (new AttributeLoader())->loadClassMetadata($classMetadata);

        $propertyConstraints = $classMetadata->properties['bic1']->constraints;

        $this->setObject(new BicDummy());

        list($constraint) = $propertyConstraints;

        $this->validator->validate('FR14 2004 1010 0505 0001 3M02 606', $constraint);

        $this->buildViolation('Constraint Message')
            ->setParameter('{{ value }}', '"UNCRIT2B912"')
            ->setParameter('{{ iban }}', 'FR14 2004 1010 0505 0001 3M02 606')
            ->setCode(Bic::INVALID_IBAN_COUNTRY_CODE_ERROR)
            ->assertRaised();
    }

    public function testFindFieldByMagicCallOrderBy(): void
    {
        $this->loadFixture();
        $repos = $this->_em->getRepository(CmsUser::class);

        $usersAsc  = $repos->findByStatus('dev', ['username' => 'ASC']);
        $usersDesc = $repos->findByStatus('dev', ['username' => 'DESC']);

        self::assertCount(2, $usersAsc);
        self::assertCount(2, $usersDesc);

        self::assertInstanceOf(CmsUser::class, $usersAsc[0]);
        self::assertEquals('Alexander', $usersAsc[0]->name);
        self::assertEquals('dev', $usersAsc[0]->status);

        self::assertSame($usersAsc[0], $usersDesc[1]);
        self::assertSame($usersAsc[1], $usersDesc[0]);
    }

protected function initializeEntityManager(): void
    {
        parent::setUp();

        $this->persister          = new BasicEntityPersister($this->entityManager, $this->entityManager->getClassMetadata(Admin1AlternateName::class));
        $this->associationMapping = new ManyToOneAssociationMapping(
            sourceEntity: WhoCares::class,
            targetEntity: Admin1AlternateName::class,
            fieldName: 'admin1'
        );
        $this->entityManager      = $this->getTestEntityManager();
    }

