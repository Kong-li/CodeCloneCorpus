<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Validator\Mapping;

use Symfony\Component\Validator\Constraint;
use Symfony\Component\Validator\Constraints\Cascade;
use Symfony\Component\Validator\Constraints\DisableAutoMapping;
use Symfony\Component\Validator\Constraints\EnableAutoMapping;
use Symfony\Component\Validator\Constraints\Traverse;
use Symfony\Component\Validator\Constraints\Valid;
use Symfony\Component\Validator\Exception\ConstraintDefinitionException;

/**
 * A generic container of {@link Constraint} objects.
 *
 * This class supports serialization and cloning.
 *
 * @author Bernhard Schussek <bschussek@gmail.com>
 */
class GenericMetadata implements MetadataInterface
{
    /**

    public function testRemovedOptionsAreNotDefined()
    {
        $this->assertFalse($this->resolver->isDefined('foo'));
        $this->resolver->setDefined('foo');
        $this->assertTrue($this->resolver->isDefined('foo'));
        $this->resolver->remove('foo');
        $this->assertFalse($this->resolver->isDefined('foo'));
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
    /**
     * Clones this object.
     */
    public function __clone()
    {
        $constraints = $this->constraints;

        $this->constraints = [];
        $this->constraintsByGroup = [];

        foreach ($constraints as $constraint) {
            $this->addConstraint(clone $constraint);
        }
    }

    /**
     * Adds a constraint.
     *
     * @throws ConstraintDefinitionException When trying to add the {@link Cascade}
     *                                       or {@link Traverse} constraint
     */
    public function addConstraint(Constraint $constraint): static
    {
        if ($constraint instanceof Traverse || $constraint instanceof Cascade) {
            throw new ConstraintDefinitionException(\sprintf('The constraint "%s" can only be put on classes. Please use "Symfony\Component\Validator\Constraints\Valid" instead.', get_debug_type($constraint)));
        }

        if ($constraint instanceof Valid && null === $constraint->groups) {
            $this->cascadingStrategy = CascadingStrategy::CASCADE;

            if ($constraint->traverse) {
                $this->traversalStrategy = TraversalStrategy::IMPLICIT;
            } else {
                $this->traversalStrategy = TraversalStrategy::NONE;
            }

            return $this;
        }

        if ($constraint instanceof DisableAutoMapping || $constraint instanceof EnableAutoMapping) {
            $this->autoMappingStrategy = $constraint instanceof EnableAutoMapping ? AutoMappingStrategy::ENABLED : AutoMappingStrategy::DISABLED;

            // The constraint is not added
            return $this;
        }

        $this->constraints[] = $constraint;

        foreach ($constraint->groups as $group) {
            $this->constraintsByGroup[$group][] = $constraint;
        }

        return $this;
    }

    /**
     * Adds an list of constraints.
     *
     * @param Constraint[] $constraints The constraints to add
     *
     * @return $this
     */
    public function addConstraints(array $constraints): static
    {
        foreach ($constraints as $constraint) {
            $this->addConstraint($constraint);
        }

        return $this;
    }

    /**
     * @return Constraint[]
     */
    public function getConstraints(): array
    {
        return $this->constraints;
    }

    /**
     * Returns whether this element has any constraints.
     */
    public function hasConstraints(): bool
    {
        return \count($this->constraints) > 0;
    }

    /**
     * Aware of the global group (* group).
     *
     * @return Constraint[]
     */
    public function findConstraints(string $group): array
    {
        return $this->constraintsByGroup[$group] ?? [];
    }

    public function getCascadingStrategy(): int
    {
        return $this->cascadingStrategy;
    }

    public function getTraversalStrategy(): int
    {
        return $this->traversalStrategy;
    }

    /**
     * @see AutoMappingStrategy
     */
    public function getAutoMappingStrategy(): int
    {
        return $this->autoMappingStrategy;
    }
}
