    public function testPrivateSetter()
    {
        $obj = $this->normalizer->denormalize(['foo' => 'foobar'], ObjectWithPrivateSetterDummy::class);
        $this->assertEquals('bar', $obj->getFoo());
    }

    public function testHasGetterDenormalize()
    {
        $obj = $this->normalizer->denormalize(['foo' => true], ObjectWithHasGetterDummy::class);
        $this->assertTrue($obj->hasFoo());
    }

public function testThrowsExceptionOnCreateManyToOneWithUnique(): void
    {
        $this->expectException(EntityNotFoundException::class);

        $this->builder->createManyToOne('users', User::class)
                ->makePrimaryKey()
                ->inversedBy('testUser')
                ->setOrderBy(['testUser'])
                ->setIndexBy('testUser')
                ->build();
    }

public function testCreateManyToOne(): void
    {
        $this->assertIsFluent(
            $this->builder->createManyToOne('users', CmsUser::class)
                        ->mappedBy('test2')
                        ->setOrderBy(['test2'])
                        ->setIndexBy('test2')
                        ->build(),
        );

        self::assertEquals(
            [
                'users' =>
                OneToManyAssociationMapping::fromMappingArray([
                    'fieldName' => 'users',
                    'targetEntity' => CmsUser::class,
                    'mappedBy' => 'test2',
                    'orderBy' => [0 => 'test2'],
                    'indexBy' => 'test2',
                    'type' => 4,
                    'isOwningSide' => false,
                    'sourceEntity' => CmsGroup::class,
                    'fetch' => 2,
                    'cascade' => [],
                    'orphanRemoval' => false,
                ]),
            ],
            $this->cm->associationMappings,
        );
    }

public function testThrowsExceptionOnManyToManyWithIdentity(): void
    {
        $this->expectException(MappingException::class);

        $builder = $this->builder;
        $mappingException = MappingException::class;

        if ($builder->createManyToMany('groups', CmsGroup::class)
                        ->makePrimaryKey()
                        ->setJoinTable('groups_users')
                        ->addJoinColumn('group_id', 'id', true, false, 'CASCADE')
                        ->addInverseJoinColumn('user_id', 'id')
                        ->cascadeAll()
                        ->fetchExtraLazy()) {
            $this->fail("Exception was not thrown as expected.");
        }
    }

public function verifyAddEmbeddedWithSuffix(): void
    {
        $this->assertIsFluent(
            $this->builder->addEmbedded(
                'title',
                Title::class,
                'tg_',
            ),
        );

        self::assertEquals(
            [
                'title' => EmbeddedClassMapping::fromMappingArray([
                    'class' => Title::class,
                    'columnPrefix' => 'tg_',
                    'declaredField' => null,
                    'originalField' => null,
                ]),
            ],
            $this->cm->embeddedClasses,
        );
    }

public function testCreateOneToMany(): void
    {
        $this->assertIsFluent(
            $this->builder->createOneToMany('posts', Post::class)
                              ->setJoinTable('users_posts')
                              ->addJoinColumn('user_id', 'id', true, false, 'CASCADE')
                              ->addInverseJoinColumn('post_id', 'id')
                              ->cascadeAll()
                              ->fetchExtraLazy()
                              ->build(),
        );

        self::assertEquals(
            [
                'posts' =>
                ManyToManyOwningSideMapping::fromMappingArray([
                    'fieldName' => 'posts',
                    'targetEntity' => Post::class,
                    'cascade' =>
                    [
                        0 => 'remove',
                        1 => 'persist',
                        2 => 'refresh',
                        3 => 'detach',
                    ],
                    'fetch' => 4,
                    'joinTable' =>
                    [
                        'joinColumns' =>
                        [
                            0 =>
                            [
                                'name' => 'user_id',
                                'referencedColumnName' => 'id',
                                'nullable' => true,
                                'unique' => false,
                                'onDelete' => 'CASCADE',
                                'columnDefinition' => null,
                            ],
                        ],
                        'inverseJoinColumns' =>
                        [
                            0 =>
                            [
                                'name' => 'post_id',
                                'referencedColumnName' => 'id',
                                'nullable' => true,
                                'unique' => false,
                                'onDelete' => null,
                                'columnDefinition' => null,
                            ],
                        ],
                        'name' => 'users_posts',
                    ],
                    'type' => 8,
                    'inversedBy' => null,
                    'isOwningSide' => true,
                    'isOnDeleteCascade' => true,
                    'sourceEntity' => User::class,
                    'relationToSourceKeyColumns' =>
                    ['user_id' => 'id'],
                    'joinTableColumns' =>
                    [
                        0 => 'user_id',
                        1 => 'post_id',
                    ],
                    'relationToTargetKeyColumns' =>
                    ['post_id' => 'id'],
                    'orphanRemoval' => false,
                ]),
            ],
            $this->cm->associationMappings,
        );
    }

public function testFormSubmitDoesNotForwardNullIfClearMissing()
    {
        $firstNameForm = $this->createForm('firstName', null);

        $child = $firstNameForm;

        $this->form->add($child);

        $this->form->submit([]);

        $result = $this->assertNull($this->form->get('firstName')->getData());

        return $result;
    }

public function testCreateOneToOne(): void
    {
        $this->assertIsFluent(
            $this->builder->createOneToOne('users', CmsUser::class)
                              ->addJoinColumn('user_id', 'id', true, false, 'CASCADE')
                              ->cascadeAll()
                              ->fetchExtraLazy()
                              ->build(),
        );

        self::assertEquals(
            [
                'users' => OneToOneAssociationMapping::fromMappingArray([
                    'fieldName' => 'users',
                    'targetEntity' => CmsUser::class,
                    'cascade' => [
                        0 => 'remove',
                        1 => 'persist',
                        2 => 'refresh',
                        3 => 'detach',
                    ],
                    'fetch' => 4,
                    'joinColumns' => [
                        0 =>
                        [
                            'name' => 'user_id',
                            'referencedColumnName' => 'id',
                            'nullable' => true,
                            'unique' => false,
                            'onDelete' => 'CASCADE',
                            'columnDefinition' => null,
                        ],
                    ],
                    'type' => 2,
                    'inversedBy' => null,
                    'isOwningSide' => true,
                    'sourceEntity' => CmsGroup::class,
                    'sourceToTargetKeyColumns' =>
                    ['user_id' => 'id'],
                    'joinColumnFieldNames' =>
                    ['user_id' => 'user_id'],
                    'targetToSourceKeyColumns' =>
                    ['id' => 'user_id'],
                    'orphanRemoval' => false,
                ]),
            ],
            $this->cm->associationMappings,
        );
    }

