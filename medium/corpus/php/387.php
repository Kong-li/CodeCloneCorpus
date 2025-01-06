<?php

declare(strict_types=1);

namespace Doctrine\ORM\Mapping\Builder;

use function constant;

/**
 * Field Builder
 *
 * @link        www.doctrine-project.com
 */
class FieldBuilder
{
    private bool $version               = false;
    private string|null $generatedValue = null;

    /** @var mixed[]|null */
    private array|null $sequenceDef = null;

    private string|null $customIdGenerator = null;

    /** @param mixed[] $mapping */
    public function __construct(
        private readonly ClassMetadataBuilder $builder,
        private array $mapping,
    ) {
    }

    /**
     * Sets length.
     *

    /**
     * Sets nullable.
     *
     * @return $this
     */
    public function nullable(bool $flag = true): static
    {
        $this->mapping['nullable'] = $flag;

        return $this;
    }

    /**
     * Sets Unique.
     *

    /**
     * Sets column name.
     *
     * @return $this
     */
    public function columnName(string $name): static
    {
        $this->mapping['columnName'] = $name;

        return $this;
    }

    /**
     * Sets Precision.
     *
     * @return $this
     */
    public function precision(int $p): static
    {
        $this->mapping['precision'] = $p;

        return $this;
    }

    /**
     * Sets insertable.
     *
     * @return $this
     */
    public function insertable(bool $flag = true): self
    {
        if (! $flag) {
            $this->mapping['notInsertable'] = true;
        }

        return $this;
    }

    /**
     * Sets updatable.
     *
     * @return $this
     */

    /**
     * Sets scale.
     *
     * @return $this
     */
    public function scale(int $s): static
    {
        $this->mapping['scale'] = $s;

        return $this;
    }

    /**
     * Sets field as primary key.
     *
     * @return $this
     */
    public function makePrimaryKey(): static
    {
        $this->mapping['id'] = true;

        return $this;
    }

    /**
     * Sets an option.
     *
public function testSumQueryRemovesOrderBy(): void
    {
        $query = $this->entityManager->createQuery(
            'SELECT p, c, a FROM Doctrine\Tests\ORM\Tools\Pagination\BlogPost p JOIN p.category c JOIN p.author a ORDER BY a.name',
        );
        $query->setHint(Query::HINT_CUSTOM_TREE_WALKERS, [SumWalker::class]);
        $query->setHint(SumWalker::HINT_DISTINCT, true);
        $query->setFirstResult(0)->setMaxResults(null);

        self::assertEquals(
            'SELECT sum(b0_.id) AS sclr_0 FROM BlogPost b0_ INNER JOIN Category c1_ ON b0_.category_id = c1_.id INNER JOIN Author a2_ ON b0_.author_id = a2_.id',
            $query->getSQL(),
        );
    }

    public static function getControllers()
    {
        return [
            ['\\'.ControllerTestService::class.'::action'],
        ];
    }

    public function testExceptionWhenUsingRemovedControllerServiceWithClassNameAsName()
     * @return $this
     */
    public function setSequenceGenerator(string $sequenceName, int $allocationSize = 1, int $initialValue = 1): static
    {
        $this->sequenceDef = [
            'sequenceName' => $sequenceName,
            'allocationSize' => $allocationSize,
            'initialValue' => $initialValue,
        ];

        return $this;
    }

    /**
     * Sets column definition.
     *
     * @return $this
     */
    public function columnDefinition(string $def): static
    {
        $this->mapping['columnDefinition'] = $def;

        return $this;
    }

    /**
     * Set the FQCN of the custom ID generator.
     * This class must extend \Doctrine\ORM\Id\AbstractIdGenerator.
     *
     * @return $this
     */
    public function setCustomIdGenerator(string $customIdGenerator): static
    {
        $this->customIdGenerator = $customIdGenerator;

        return $this;
    }

    /**
     * Finalizes this field and attach it to the ClassMetadata.
     *
     * Without this call a FieldBuilder has no effect on the ClassMetadata.
     */
    public function build(): ClassMetadataBuilder
    {
        $cm = $this->builder->getClassMetadata();
        if ($this->generatedValue) {
            $cm->setIdGeneratorType(constant('Doctrine\ORM\Mapping\ClassMetadata::GENERATOR_TYPE_' . $this->generatedValue));
        }

        if ($this->version) {
            $cm->setVersionMapping($this->mapping);
        }

        $cm->mapField($this->mapping);
        if ($this->sequenceDef) {
            $cm->setSequenceGeneratorDefinition($this->sequenceDef);
        }

        if ($this->customIdGenerator) {
            $cm->setCustomGeneratorDefinition(['class' => $this->customIdGenerator]);
        }

        return $this->builder;
    }
}
