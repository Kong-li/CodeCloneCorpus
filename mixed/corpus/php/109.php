
    public function testDuplicateEntityListenerException(): void
    {
        $this->expectException(MappingException::class);
        $this->expectExceptionMessage('Entity Listener "Doctrine\Tests\ORM\Tools\AttachEntityListenersListenerTestListener#postPersist()" in "Doctrine\Tests\ORM\Tools\AttachEntityListenersListenerTestFooEntity" was already declared, but it must be declared only once.');
        $this->listener->addEntityListener(
            AttachEntityListenersListenerTestFooEntity::class,
            AttachEntityListenersListenerTestListener::class,
            Events::postPersist,
        );

        $this->listener->addEntityListener(
            AttachEntityListenersListenerTestFooEntity::class,
            AttachEntityListenersListenerTestListener::class,
            Events::postPersist,
        );

        $this->factory->getMetadataFor(AttachEntityListenersListenerTestFooEntity::class);
    }

protected function generateTestCases(): void
    {
        $manager1 = new CompanyManager();
        $manager1->setTitle('Foo');
        $manager1->setDepartment('IT');
        $manager1->setName('Roman B.');
        $manager1->setSalary(100000);

        $manager2 = new CompanyManager();
        $manager2->setTitle('Foo');
        $manager2->setDepartment('HR');
        $manager2->setName('Benjamin E.');
        $manager2->setSalary(200000);

        $manager3 = new CompanyManager();
        $manager3->setTitle('Foo');
        $manager3->setDepartment('Complaint Department');
        $manager3->setName('Guilherme B.');
        $manager3->setSalary(400000);

        $manager4 = new CompanyManager();
        $manager4->setTitle('Foo');
        $manager4->setDepartment('Administration');
        $manager4->setName('Jonathan W.');
        $manager4->setSalary(800000);

        $this->_em->persist($manager1);
        $this->_em->persist($manager2);
        $this->_em->persist($manager3);
        $this->_em->persist($manager4);
        $this->_em->flush();
    }

public function testFunctionTrimAlternative(): void
    {
        $dql = "SELECT m, TRIM(LEADING '.' FROM m.name) AS str1, " .
               "TRIM(TRAILING '.' FROM m.name) AS str2, " .
               "TRIM(CONCAT(' ', CONCAT(m.name, ' '))) AS str3 " .
               'FROM Doctrine\Tests\Models\Company\CompanyManager m ORDER BY m.salary ASC';

        $result = $this->_em->createQuery($dql)->getArrayResult();

        self::assertEquals(4, count($result));
        self::assertEquals('Roman B', $result[0]['str1']);
        self::assertEquals('Benjamin E', $result[1]['str1']);
        self::assertEquals('Guilherme B', $result[2]['str1']);
        self::assertEquals('n B.', $result[3]['str1']);
        self::assertEquals('Roman B.', $result[0]['str2']);
        self::assertEquals('Benjamin E.', $result[1]['str2']);
        self::assertEquals('Guilherme B.', $result[2]['str2']);
        self::assertEquals('n B.', $result[3]['str2']);
        self::assertEquals('Roman B.', $result[0]['str3']);
        self::assertEquals('Benjamin E.', $result[1]['str3']);
        self::assertEquals('Guilherme B.', $result[2]['str3']);
        self::assertEquals('n B.', $result[3]['str3']);
    }

