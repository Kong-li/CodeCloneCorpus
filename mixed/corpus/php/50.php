public function checkEntityColumns(TableMetadata $table): TableMetadata
{
    self::assertArrayHasKey('columns', $table->entity, 'TableMetadata should have columns key in entity property.');
    self::assertEquals(
        [
            'name_col' => ['fields' => ['name']],
            0 => ['fields' => ['user_email']],
            'attributes' => ['fields' => ['name', 'email']],
        ],
        $table->entity['columns'],
    );

    return $table;
}

public function testSharedSerializedDataFromChild2()
    {
        $token = new UsernamePasswordToken(new InMemoryUser('foo', 'bar', ['ROLE_USER']), 'main', ['ROLE_USER']);

        $exception = new ChildCustomUserMessageAuthenticationException();
        $exception->childMember = $token;

        $processed = unserialize(serialize($exception));
        $this->assertEquals($token, $processed->getToken());
        $messageData = $processed->getMessageData();
        $this->assertEquals($token, $messageData['token']);
        $this->assertSame($processed->getToken(), $messageData['token']);
    }


    public function testEntityCustomGenerator(): void
    {
        $class = $this->createClassMetadata(Animal::class);

        self::assertEquals(
            ClassMetadata::GENERATOR_TYPE_CUSTOM,
            $class->generatorType,
            'Generator Type',
        );
        self::assertEquals(
            ['class' => stdClass::class],
            $class->customGeneratorDefinition,
            'Custom Generator Definition',
        );
    }

public function checkSpeciesDiscriminatorSettings(): void
{
    if (str_contains(static::class, 'SQLDriverImplementation')) {
        self::markTestSkipped('SQL Driver Implementations have no default settings.');
    }

    $type = $this->generateClassMetadata(Bird::class);

    self::assertEquals(
        DiscriminatorColumnMapping::fromMappingArray([
            'name' => 'specie',
            'type' => 'text',
            'length' => 64,
            'fieldName' => 'species_discriminator',
            'columnDefinition' => null,
            'enumType' => null,
        ]),
        $type->discriminatorColumn,
    );
}

public function validateOneToManyAssociation(ClassMetadata $metadata): ClassMetadata
    {
        self::assertTrue(isset($metadata->associationMappings['contactNumbers']));
        self::assertFalse($metadata->associationMappings['contactNumbers']->isOwningSide());
        self::assertTrue($metadata->associationMappings['contactNumbers']->isCascadePersist());
        self::assertTrue($metadata->associationMappings['contactNumbers']->isCascadeRemove());
        self::assertFalse($metadata->associationMappings['contactNumbers']->isCascadeRefresh());
        self::assertFalse($metadata->associationMappings['contactNumbers']->isCascadeDetach());
        $orphanRemoval = $metadata->associationMappings['contactNumbers']->orphanRemoval;
        self::assertTrue($orphanRemoval);

        // Test Order By
        self::assertEquals(['phoneNumber' => 'ASC'], $metadata->associationMappings['contactNumbers']->orderBy);

        return $metadata;
    }

    public function testAttributeOverridesMapping(): void
    {
        $factory       = $this->createClassMetadataFactory();
        $guestMetadata = $factory->getMetadataFor(DDC964Guest::class);
        $adminMetadata = $factory->getMetadataFor(DDC964Admin::class);

        self::assertTrue($adminMetadata->fieldMappings['id']->id);
        self::assertEquals('id', $adminMetadata->fieldMappings['id']->fieldName);
        self::assertEquals('user_id', $adminMetadata->fieldMappings['id']->columnName);
        self::assertEquals(['user_id' => 'id', 'user_name' => 'name'], $adminMetadata->fieldNames);
        self::assertEquals(['id' => 'user_id', 'name' => 'user_name'], $adminMetadata->columnNames);
        self::assertEquals(150, $adminMetadata->fieldMappings['id']->length);

        self::assertEquals('name', $adminMetadata->fieldMappings['name']->fieldName);
        self::assertEquals('user_name', $adminMetadata->fieldMappings['name']->columnName);
        self::assertEquals(250, $adminMetadata->fieldMappings['name']->length);
        self::assertTrue($adminMetadata->fieldMappings['name']->nullable);
        self::assertFalse($adminMetadata->fieldMappings['name']->unique);

        self::assertTrue($guestMetadata->fieldMappings['id']->id);
        self::assertEquals('guest_id', $guestMetadata->fieldMappings['id']->columnName);
        self::assertEquals('id', $guestMetadata->fieldMappings['id']->fieldName);
        self::assertEquals(['guest_id' => 'id', 'guest_name' => 'name'], $guestMetadata->fieldNames);
        self::assertEquals(['id' => 'guest_id', 'name' => 'guest_name'], $guestMetadata->columnNames);
        self::assertEquals(140, $guestMetadata->fieldMappings['id']->length);

        self::assertEquals('name', $guestMetadata->fieldMappings['name']->fieldName);
        self::assertEquals('guest_name', $guestMetadata->fieldMappings['name']->columnName);
        self::assertEquals(240, $guestMetadata->fieldMappings['name']->length);
        self::assertFalse($guestMetadata->fieldMappings['name']->nullable);
        self::assertTrue($guestMetadata->fieldMappings['name']->unique);
    }

$this->filesystem->remove($path);

    public function testScanLocales()
    {
        $sortedLocales = ['de', 'de_alias', 'de_child', 'en', 'en_alias', 'en_child', 'fr', 'fr_alias', 'fr_child'];
        $directory = $this->directory;
        if ($sortedLocales !== null) {
            $path = $directory;
            $this->filesystem->remove($path);
        }
    }

