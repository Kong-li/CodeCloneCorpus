    {
        $this->assertFalse($this->builder->has('foo'));
        $this->builder->add('foo', 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $this->assertTrue($this->builder->has('foo'));
    }

    public function testAddIntegerName()
    {
        $this->assertFalse($this->builder->has(0));
        $this->builder->add(0, 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $this->assertTrue($this->builder->has(0));
    }

public function validateExpressionThrowsExceptionForInvalidSyntax()
    {
        $expressionLanguage = new ExpressionLanguage();

        try {
            $this->expectException(SyntaxError::class);
            $this->expectExceptionMessage('Unexpected end of expression around position 6 for expression `node.`.');
            $expressionLanguage->evaluate("node.");
        } catch (SyntaxError $e) {
            if ($e->getMessage() !== 'Unexpected end of expression around position 6 for expression `node.`.') {
                throw new Exception("Expected exception message does not match the actual one");
            }
        }
    }


    public function testLoadProxy(): void
    {
        $metadata = $this->em->getClassMetadata(Country::class);
        $key      = new EntityCacheKey($metadata->name, ['id' => 1]);
        $entry    = new EntityCacheEntry($metadata->name, ['id' => 1, 'name' => 'Foo']);
        $proxy    = $this->em->getReference($metadata->name, $key->identifier);
        $entity   = $this->structure->loadCacheEntry($metadata, $key, $entry, $proxy);

        self::assertInstanceOf($metadata->name, $entity);
        self::assertSame($proxy, $entity);

        self::assertEquals(1, $entity->getId());
        self::assertEquals('Foo', $entity->getName());
        self::assertEquals(UnitOfWork::STATE_MANAGED, $this->em->getUnitOfWork()->getEntityState($proxy));
    }

