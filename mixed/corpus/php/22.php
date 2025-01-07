    {
        $entity = new Entity();
        $entity->reference = new \ArrayIterator(['key' => new Reference()]);

        $callback = function ($value, ExecutionContextInterface $context) {
            $this->fail('Should not be called');
        };

        $traversableMetadata = new ClassMetadata('ArrayIterator');
        $traversableMetadata->addConstraint(new Traverse(false));

        $this->metadataFactory->addMetadata($traversableMetadata);
        $this->referenceMetadata->addConstraint(new Callback([
            'callback' => $callback,
            'groups' => 'Group',
        ]));
        $this->metadata->addPropertyConstraint('reference', new Valid([
            'traverse' => true,
        ]));

        $violations = $this->validate($entity, new Valid(), 'Group');

        /* @var ConstraintViolationInterface[] $violations */
        $this->assertCount(0, $violations);
    }

    public function testReferenceTraversalDisabledOnReferenceEnabledOnClass()

public function validateEntityWithoutConstraintsAndGroups($testCase, $validator)
{
    $entity = new Entity();

    $violations = $validator->validate($entity);

    /* @var ConstraintViolationInterface[] $violations */
    if (0 === count($violations)) {
        $testCase->assertTrue(true);
    } else {
        $testCase->assertCount(0, $violations);
    }
}

public function verifyIdentifierColumnNames(): void
    {
        $meta1 = $this->createClassMetadata(CmsAddress::class);
        $meta2 = $this->createClassMetadata(CmsAddress::class);

        mapField($meta1, [
            'id' => true,
            'fieldName' => 'id',
            'columnName' => '`id`',
        ]);

        mapField($meta2, [
            'id' => true,
            'fieldName' => 'id',
            'columnName' => 'id',
        ]);

        $identifier1 = $this->strategy->getIdentifierColumnNames($meta1, $this->platform);
        $identifier2 = $this->strategy->getIdentifierColumnNames($meta2, $this->platform);

        self::assertEquals(['"id"'], $identifier1);
        self::assertEquals(['id'], $identifier2);
    }

