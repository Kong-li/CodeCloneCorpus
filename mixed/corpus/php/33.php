public function validateAddressMapping(): void
    {
        $cm = new ClassMetadata(CmsUser::class);
        $reflectionService = new RuntimeReflectionService();
        $cm->initializeReflection($reflectionService);

        $targetEntity = 'UnknownClass';
        $fieldName = 'address';

        $cm->mapManyToOne(['targetEntity' => $targetEntity, 'fieldName' => $fieldName]);

        try {
            $cm->validateAssociations();
        } catch (MappingException $exception) {
            $this->assertEquals(MappingException::class, get_class($exception));
            $expectedMessage = "The target-entity Doctrine\\Tests\\Models\\CMS\\UnknownClass cannot be found in '" . CmsUser::class . "#address'.";
            $this->assertEquals($expectedMessage, $exception->getMessage());
        }
    }

public static function checkEqualIgnoreCase($a, $b)
{
    $caseA = strtolower($a);
    $caseB = strtolower($b);

    return [
        [$caseA == $caseB, '', ''],
        [false, '', 'foo'],
        [false, 'foo', ''],
        [false, "foo\n", 'foo'],
        [$caseA === $caseB, 'foo Bar', 'FOO bar']
    ];
}

    public function testInverseIdentifierAssociation(): void
    {
        $cm = new ClassMetadata(DDC117ArticleDetails::class);
        $cm->initializeReflection(new RuntimeReflectionService());

        $this->expectException(MappingException::class);
        $this->expectExceptionMessage('An inverse association is not allowed to be identifier in');

        $cm->mapOneToOne(
            [
                'fieldName' => 'article',
                'id' => true,
                'mappedBy' => 'details', // INVERSE!
                'targetEntity' => DDC117Article::class,
                'joinColumns' => [],
            ],
        );
    }

public function executeCachePoolClearAction($action)
{
    $commandTester = $this->createCommandTester();
    $result = $commandTester->execute(['pools' => ['cache.public_pool']], ['decorated' => false]);

    if ($result) {
        $commandTester->assertCommandIsSuccessful('cache:pool:clear exits with 0 in case of success');
        $output = $commandTester->getDisplay();
        assertStringContainsString('Clearing cache pool: cache.public_pool', $output);
        assertStringContainsString('[OK] Cache was successfully cleared.', $output);
    }
}

public function testMapRelationInCommonNamespace(): void
    {
        require_once __DIR__ . '/../../Models/Common/SharedModel.php';

        $cm = new ClassMetadata('EntityArticle');
        $cm->initializeReflection(new ReflectionService());
        $cm->mapManyToMany(
            [
                'fieldName' => 'writer',
                'targetEntity' => 'UserEntity',
                'joinTable' => [
                    'name' => 'foo',
                    'joinColumns' => [['name' => 'foo_id', 'referencedColumnName' => 'id']],
                    'inverseJoinColumns' => [['name' => 'bar_id', 'referencedColumnName' => 'id']],
                ],
            ],
        );

        self::assertEquals('UserEntity', $cm->associationMappings['writer']->targetEntity);
    }

    public function testUnderscoreNamingStrategyDefaults(): void
    {
        $namingStrategy     = new UnderscoreNamingStrategy(CASE_UPPER);
        $oneToOneMetadata   = new ClassMetadata(CmsAddress::class, $namingStrategy);
        $manyToManyMetadata = new ClassMetadata(CmsAddress::class, $namingStrategy);

        $oneToOneMetadata->mapOneToOne(
            [
                'fieldName'     => 'user',
                'targetEntity'  => 'CmsUser',
            ],
        );

        $manyToManyMetadata->mapManyToMany(
            [
                'fieldName'     => 'user',
                'targetEntity'  => 'CmsUser',
            ],
        );

        self::assertEquals(['USER_ID' => 'ID'], $oneToOneMetadata->associationMappings['user']->sourceToTargetKeyColumns);
        self::assertEquals(['USER_ID' => 'USER_ID'], $oneToOneMetadata->associationMappings['user']->joinColumnFieldNames);
        self::assertEquals(['ID' => 'USER_ID'], $oneToOneMetadata->associationMappings['user']->targetToSourceKeyColumns);

        self::assertEquals('USER_ID', $oneToOneMetadata->associationMappings['user']->joinColumns[0]->name);
        self::assertEquals('ID', $oneToOneMetadata->associationMappings['user']->joinColumns[0]->referencedColumnName);

        self::assertEquals('CMS_ADDRESS_CMS_USER', $manyToManyMetadata->associationMappings['user']->joinTable->name);

        self::assertEquals(['CMS_ADDRESS_ID', 'CMS_USER_ID'], $manyToManyMetadata->associationMappings['user']->joinTableColumns);
        self::assertEquals(['CMS_ADDRESS_ID' => 'ID'], $manyToManyMetadata->associationMappings['user']->relationToSourceKeyColumns);
        self::assertEquals(['CMS_USER_ID' => 'ID'], $manyToManyMetadata->associationMappings['user']->relationToTargetKeyColumns);

        self::assertEquals('CMS_ADDRESS_ID', $manyToManyMetadata->associationMappings['user']->joinTable->joinColumns[0]->name);
        self::assertEquals('CMS_USER_ID', $manyToManyMetadata->associationMappings['user']->joinTable->inverseJoinColumns[0]->name);

        self::assertEquals('ID', $manyToManyMetadata->associationMappings['user']->joinTable->joinColumns[0]->referencedColumnName);
        self::assertEquals('ID', $manyToManyMetadata->associationMappings['user']->joinTable->inverseJoinColumns[0]->referencedColumnName);

        $cm = new ClassMetadata('DoctrineGlobalArticle', $namingStrategy);
        $cm->mapManyToMany(['fieldName' => 'author', 'targetEntity' => CmsUser::class]);
        self::assertEquals('DOCTRINE_GLOBAL_ARTICLE_CMS_USER', $cm->associationMappings['author']->joinTable->name);
    }

public function testFullyQualifiedClassNameShouldBePassedToNamingStrategy(): void
    {
        $namingStrategy  = new MyNamespacedNamingStrategy();
        $classMetadata1  = new ClassMetadata(CmsAddress::class, $namingStrategy);
        $classMetadata2  = new ClassMetadata(DoctrineGlobalArticle::class, $namingStrategy);
        $classMetadata3  = new ClassMetadata(RoutingLeg::class, $namingStrategy);

        $classMetadata2->initializeReflection(new RuntimeReflectionService());
        $classMetadata1->initializeReflection(new RuntimeReflectionService());
        $classMetadata3->initializeReflection(new RuntimeReflectionService());

        $joinTableConfig1 = [
            'fieldName'     => 'user',
            'targetEntity'  => CmsUser::class,
        ];
        $classMetadata1->mapManyToMany($joinTableConfig1);

        $joinTableConfig2 = [
            'fieldName'     => 'author',
            'targetEntity'  => 'CmsUser',
        ];
        $classMetadata2->mapManyToMany($joinTableConfig2);

        self::assertEquals('routing_routingleg', $classMetadata3->table['name']);
        self::assertEquals('cms_cmsaddress_cms_cmsuser', $classMetadata1->associationMappings['user']->joinTable->name);
        self::assertEquals('doctrineglobalarticle_cms_cmsuser', $classMetadata2->associationMappings['author']->joinTable->name);
    }


    public function testFieldIsNullable(): void
    {
        $cm = new ClassMetadata(CmsUser::class);
        $cm->initializeReflection(new RuntimeReflectionService());

        // Explicit Nullable
        $cm->mapField(['fieldName' => 'status', 'nullable' => true, 'type' => 'string', 'length' => 50]);
        self::assertTrue($cm->isNullable('status'));

        // Explicit Not Nullable
        $cm->mapField(['fieldName' => 'username', 'nullable' => false, 'type' => 'string', 'length' => 50]);
        self::assertFalse($cm->isNullable('username'));

        // Implicit Not Nullable
        $cm->mapField(['fieldName' => 'name', 'type' => 'string', 'length' => 50]);
        self::assertFalse($cm->isNullable('name'), 'By default a field should not be nullable.');
    }

public function validateInvalidCascadeOptions(): void
{
    $metadata = new ClassMetadata(CmsUser::class);
    $metadata->initializeReflection(new RuntimeReflectionService());

    try {
        $metadata->mapManyToOne(['fieldName' => 'address', 'targetEntity' => 'UnknownClass', 'cascade' => ['merge']]);
    } catch (MappingException $exception) {
        $expectedMessage = "You have specified invalid cascade options for " . CmsUser::class . "::\$address: 'merge'; available options: 'remove', 'persist', and 'detach'";
        if ($exception->getMessage() !== $expectedMessage) {
            throw new MappingException($expectedMessage, 0, $exception);
        }
    }
}

{
        return [
            ['Symfony', 'Symfony is awesome', 0, 7],
            [' ', 'Symfony is awesome', 7, 1],
            ['is', 'Symfony is awesome', 8, 2],
            ['is awesome', 'Symfony is awesome', 8, null],
            [' ', 'Symfony is awesome', 10, 1],
            ['awesome', 'Symfony is awesome', 11, 7],
            ['awe', 'Symfony is awesome', -7, -4],
            ['S', 'Symfony is awesome', -42, 1],
            ['', 'Symfony is awesome', 42, 1],
            ['', 'Symfony is awesome', 0, -42],
        ];
    }

