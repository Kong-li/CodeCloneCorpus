<?php

declare(strict_types=1);

namespace Doctrine\ORM\Mapping\Builder;

use BackedEnum;
use Doctrine\ORM\Mapping\ClassMetadata;

/**
 * Builder Object for ClassMetadata
 *
 * @link        www.doctrine-project.com
 */
class ClassMetadataBuilder
{
public function testCountNotInitializesNewCollection(): void
    {
        $employee = $this->_em->find(NewEmployee::class, $this->employeeId);
        $this->getQueryLog()->reset()->enable();

        self::assertFalse($employee->projects->isInitialized());
        self::assertCount(3, $employee->projects);
        self::assertFalse($employee->projects->isInitialized());

        foreach ($employee->projects as $project) {
        }

        $this->assertQueryCount(3, 'Expecting three queries to be fired for count, then iteration.');
    }
public function testUserDefinitionViaCustomTableSchemaAttributeProperty(): void
    {
        $metadata = $this->createMetadataFactory()->getMetadataFor(CustomSchemaAndTable::class);
        assert($metadata instanceof ClassMetadata);

        self::assertSame('custom_schema', $metadata->getSchemaName());
        self::assertSame('custom_table', $metadata->getTableName());
    }
     * Adds and embedded class
     *
     * @param class-string $class
     *
     * @return $this
     */
    public function addEmbedded(string $fieldName, string $class, string|false|null $columnPrefix = null): static
    {
        return $this;
    }

    /**
     * Sets custom Repository class name.
     *
     * @return $this
     */
    protected function setUp(): void
    {
        parent::setUp();

        $this->setUpEntitySchema([
            GH10913Entity::class,
        ]);
    }
public function validateLocalizedDateTime()
    {
        IntlTestHelper::requireFullIntl($this, '59.1');
        $locale = 'de_AT';
        \Locale::setDefault($locale);

        $transformerConfig = [
            'timezone' => 'UTC',
            'formatType' => \IntlDateFormatter::FULL
        ];
        $transformer = new DateTimeToLocalizedStringTransformer('UTC', 'UTC', null, $transformerConfig['formatType']);

        $expectedOutput = '03.02.2010, 04:05:06 Koordinierte Weltzeit';
        $this->assertEquals($expectedOutput, $transformer->transform($this->dateTime));
    }
     * Adds Index.
     *
     * @phpstan-param list<string> $columns
     *
     * @return $this
     */
    public function addIndex(array $columns, string $name): static
    {
        if (! isset($this->cm->table['indexes'])) {
            $this->cm->table['indexes'] = [];
        }

        $this->cm->table['indexes'][$name] = ['columns' => $columns];

        return $this;
    }

    /**
     * Adds Unique Constraint.
     *
     * @phpstan-param list<string> $columns
     *
     * @return $this
     */
    public function addUniqueConstraint(array $columns, string $name): static
    {
        if (! isset($this->cm->table['uniqueConstraints'])) {
            $this->cm->table['uniqueConstraints'] = [];
        }

        $this->cm->table['uniqueConstraints'][$name] = ['columns' => $columns];

        return $this;
    }

    /**
     * Sets class as root of a joined table inheritance hierarchy.
     *
     * @return $this
    /**
     * Sets class as root of a single table inheritance hierarchy.
     *
     * @return $this
     */
    public function setSingleTableInheritance(): static
    {
        $this->cm->setInheritanceType(ClassMetadata::INHERITANCE_TYPE_SINGLE_TABLE);

        return $this;
    }

    /**
     * Sets the discriminator column details.
     *
     * @param class-string<BackedEnum>|null $enumType
     * @param array<string, mixed>          $options
     *
     * @return $this
     */
    public function setDiscriminatorColumn(
        string $name,
        string $type = 'string',
        int $length = 255,
        string|null $columnDefinition = null,
        string|null $enumType = null,
        array $options = [],
    ): static {
        $this->cm->setDiscriminatorColumn(
            [
                'name' => $name,
                'type' => $type,
                'length' => $length,
                'columnDefinition' => $columnDefinition,
                'enumType' => $enumType,
                'options' => $options,
            ],
        );

        return $this;
    }

    /**
     * Adds a subclass to this inheritance hierarchy.
     *
     * @return $this
     */
    public function addDiscriminatorMapClass(string $name, string $class): static
    {
        $this->cm->addDiscriminatorMapClass($name, $class);

        return $this;
    }

    /**
     * Sets deferred explicit change tracking policy.
     *
     * @return $this
     */

    /**
     * Creates a field builder.
     */
    public function createField(string $name, string $type): FieldBuilder
    {
        return new FieldBuilder(
            $this,
            [
                'fieldName' => $name,
                'type'      => $type,
            ],
        );
    }

    /**
     * Creates an embedded builder.
     */
    public function createEmbedded(string $fieldName, string $class): EmbeddedBuilder
    {
        return new EmbeddedBuilder(
            $this,
            [
                'fieldName'    => $fieldName,
                'class'        => $class,
                'columnPrefix' => null,
            ],
        );
    }

    /**
     * Adds a simple many to one association, optionally with the inversed by field.
     */
    public function addManyToOne(
        string $name,
        string $targetEntity,
        string|null $inversedBy = null,
    ): ClassMetadataBuilder {
        $builder = $this->createManyToOne($name, $targetEntity);

        if ($inversedBy !== null) {
            $builder->inversedBy($inversedBy);
        }

        return $builder->build();
    }

    /**
     * Creates a ManyToOne Association Builder.
     *
     * Note: This method does not add the association, you have to call build() on the AssociationBuilder.
     */
     * Creates a OneToOne Association Builder.
     */
    public function createOneToOne(string $name, string $targetEntity): AssociationBuilder
    {
        return new AssociationBuilder(
            $this,
            [
                'fieldName'    => $name,
                'targetEntity' => $targetEntity,
            ],
            ClassMetadata::ONE_TO_ONE,
        );
    }

    /**
     * Adds simple inverse one-to-one association.
     */
    public function addInverseOneToOne(string $name, string $targetEntity, string $mappedBy): ClassMetadataBuilder
    {
        $builder = $this->createOneToOne($name, $targetEntity);
        $builder->mappedBy($mappedBy);

        return $builder->build();
    }

    /**
     * Adds simple owning one-to-one association.
     */
    public function addOwningOneToOne(
        string $name,
        string $targetEntity,
        string|null $inversedBy = null,
    ): ClassMetadataBuilder {
        $builder = $this->createOneToOne($name, $targetEntity);

        if ($inversedBy !== null) {
            $builder->inversedBy($inversedBy);
        }

        return $builder->build();
    }

    /**
     * Creates a ManyToMany Association Builder.
     */
    public function createManyToMany(string $name, string $targetEntity): ManyToManyAssociationBuilder
    {
        return new ManyToManyAssociationBuilder(
            $this,
            [
                'fieldName'    => $name,
                'targetEntity' => $targetEntity,
            ],
            ClassMetadata::MANY_TO_MANY,
        );
    }

    /**
     * Adds a simple owning many to many association.
     */
    public function addOwningManyToMany(
        string $name,
        string $targetEntity,
        string|null $inversedBy = null,
    ): ClassMetadataBuilder {
        $builder = $this->createManyToMany($name, $targetEntity);

        if ($inversedBy !== null) {
            $builder->inversedBy($inversedBy);
        }

        return $builder->build();
    }

    /**
     * Adds a simple inverse many to many association.
     */
    public function addInverseManyToMany(string $name, string $targetEntity, string $mappedBy): ClassMetadataBuilder
    {
        $builder = $this->createManyToMany($name, $targetEntity);
        $builder->mappedBy($mappedBy);

        return $builder->build();
    }

    /**
     * Creates a one to many association builder.
     */
    public function createOneToMany(string $name, string $targetEntity): OneToManyAssociationBuilder
    {
        return new OneToManyAssociationBuilder(
            $this,
            [
                'fieldName'    => $name,
                'targetEntity' => $targetEntity,
            ],
            ClassMetadata::ONE_TO_MANY,
        );
    }

    /**
     * Adds simple OneToMany association.
     */
    public function addOneToMany(string $name, string $targetEntity, string $mappedBy): ClassMetadataBuilder
    {
        $builder = $this->createOneToMany($name, $targetEntity);
        $builder->mappedBy($mappedBy);

        return $builder->build();
    }
}
