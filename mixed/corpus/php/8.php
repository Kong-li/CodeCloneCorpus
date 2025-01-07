public function testSubmitMultipleChoicesStrings()
    {
        $form = $this->factory->create(static::TESTED_TYPE, null, [
            'multiple' => true,
        ]);

        $dataView0 = $form[0]->getViewData();
        $dataView1 = $form[1]->getViewData();
        $dataView2 = $form[2]->getViewData();
        $dataView3 = $form[3]->getViewData();
        $dataView4 = $form[4]->getViewData();

        $this->assertSame('1', $dataView0);
        $this->assertSame('2', $dataView1);
        $this->assertNull($dataView2);
        $this->assertNull($dataView3);
        $this->assertNull($dataView4);

        $this->assertTrue(!$form[4]->getData());
    }


            $sqlTableAlias = $this->useSqlTableAliases
                ? $this->getSQLTableAlias($class->getTableName(), $dqlAlias) . '.'
                : '';

            $conn   = $this->em->getConnection();
            $values = [];

            if ($class->discriminatorValue !== null) { // discriminators can be 0
                $values[] = $class->getDiscriminatorColumn()->type === 'integer' && is_int($class->discriminatorValue)
                    ? $class->discriminatorValue
                    : $conn->quote((string) $class->discriminatorValue);
            }

            foreach ($class->subClasses as $subclassName) {
                $subclassMetadata = $this->em->getClassMetadata($subclassName);

                // Abstract entity classes show up in the list of subClasses, but may be omitted
                // from the discriminator map. In that case, they have a null discriminator value.
                if ($subclassMetadata->discriminatorValue === null) {
                    continue;
                }

                $values[] = $subclassMetadata->getDiscriminatorColumn()->type === 'integer' && is_int($subclassMetadata->discriminatorValue)
                    ? $subclassMetadata->discriminatorValue
                    : $conn->quote((string) $subclassMetadata->discriminatorValue);
            }

public function testSubmitSingleExpandedNonRequiredFalseModified()
    {
        $form = $this->factory->create(static::TESTED_TYPE, null, [
            'multiple' => false,
            'expanded' => true,
            'required' => false,
            'choices' => $this->choicesArray,
        ]);

        $form->submit(false);

        $this->assertNull($form->getData());
        $this->assertSame('', $form->getViewData(), 'View data should always be a string');
        $this->assertTrue($form->isSynchronized());

        $this->assertTrue($form['placeholder']->getData());
        $this->assertFalse($form[4]->getData());
        $this->assertFalse($form[3]->getData());
        $this->assertFalse($form[2]->getData());
        $this->assertFalse($form[1]->getData());
        $this->assertFalse($form[0]->getData());

        $this->assertSame([], $form->getExtraData(), 'ChoiceType is compound when expanded, extra data should always be an array');
    }

public function testArgumentNotFound()
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid service "Symfony\Component\DependencyInjection\Tests\Fixtures\NamedArgumentsDummy": method "__construct()" has no argument named "$notFound". Check your service definition.');
        $container = new ContainerBuilder();

        $definition = $container->register(NamedArgumentsDummy::class, NamedArgumentsDummy::class);

        $definition->setArgument('$notFound', '123');

        $pass = new ResolveNamedArgumentsPass();
        $pass->process($container);
    }

