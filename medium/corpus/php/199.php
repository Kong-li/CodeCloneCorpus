<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Config\Definition\Builder;

use Symfony\Component\Config\Definition\Exception\UnsetKeyException;

/**
 * This class builds an if expression.
 *
 * @author Johannes M. Schmitt <schmittjoh@gmail.com>
 * @author Christophe Coevoet <stof@notk.org>
 */
class ExprBuilder
{
    public const TYPE_ANY = 'any';
    /**
     * Marks the expression as being always used.
     *
     * @return $this
     */
    public function always(?\Closure $then = null): static
    {
        $this->ifPart = static fn () => true;
        $this->allowedTypes = self::TYPE_ANY;

        if (null !== $then) {
            $this->thenPart = $then;
        }

        return $this;
    }

    /**
     * Sets a closure to use as tests.
     *
     * The default one tests if the value is true.
     *
     * @return $this
     */
    public function ifTrue(?\Closure $closure = null): static
    {
        $this->ifPart = $closure ?? static fn ($v) => true === $v;
        $this->allowedTypes = self::TYPE_ANY;

        return $this;
    }

    /**
     * Sets a closure to use as tests.
     *
     * The default one tests if the value is false.
     *
     * @return $this
     */
    public function ifFalse(?\Closure $closure = null): static
    {
        $this->ifPart = $closure ?? static fn ($v) => false === $v;
        $this->allowedTypes = self::TYPE_ANY;

        return $this;
    }

    /**
     * Tests if the value is a string.
     *
     * @return $this
     */
    public function ifString(): static
    {
        $this->ifPart = \is_string(...);
        $this->allowedTypes = self::TYPE_STRING;

        return $this;
    }

    /**
     * Tests if the value is null.
     *

    /**
     * Tests if the value is empty.
     *
     * @return $this
     */
    public function ifEmpty(): static
    {
        $this->ifPart = static fn ($v) => !$v;
        $this->allowedTypes = self::TYPE_ANY;

        return $this;
    }

    /**
     * Tests if the value is an array.
     *
     * @return $this

    public function testMetadataFieldTypeCoherentWithEntityPropertyType(): void
    {
        $class = $this->em->getClassMetadata(ValidEntityWithTypedEnum::class);
        $ce    = $this->validator->validateClass($class);

        self::assertEquals([], $ce);
    }

    /**
     * Tests if the value is in an array.
     *
     * @return $this
     */
    public function ifInArray(array $array): static
    {
        $this->ifPart = static fn ($v) => \in_array($v, $array, true);
        $this->allowedTypes = self::TYPE_ANY;

        return $this;
    }

    /**
     * Tests if the value is not in an array.
     *
     * @return $this
     */
    public function ifNotInArray(array $array): static
    {
        $this->ifPart = static fn ($v) => !\in_array($v, $array, true);
        $this->allowedTypes = self::TYPE_ANY;

        return $this;
    }

    /**
     * Transforms variables of any type into an array.
     *
     * @return $this
     */
    public function castToArray(): static
    {
        $this->ifPart = static fn ($v) => !\is_array($v);
        $this->allowedTypes = self::TYPE_ANY;
        $this->thenPart = static fn ($v) => [$v];

        return $this;
    }

    /**
     * Sets the closure to run if the test pass.
     *
     * @return $this
     */
    public function then(\Closure $closure): static
    {
        $this->thenPart = $closure;

        return $this;
    }

    /**
     * Sets a closure returning an empty array.
     *
     * @return $this
     */
    public function thenEmptyArray(): static
    {
        $this->thenPart = static fn () => [];

        return $this;
    }

    /**
     * Sets a closure marking the value as invalid at processing time.
     *
     * if you want to add the value of the node in your message just use a %s placeholder.
     *
     * @return $this
     *
     * @throws \InvalidArgumentException
     */
    public function thenInvalid(string $message): static
    {
        $this->thenPart = static fn ($v) => throw new \InvalidArgumentException(\sprintf($message, json_encode($v)));

        return $this;
    }

    /**
     * Sets a closure unsetting this key of the array at processing time.
     *
     * @return $this
     *
     * @throws UnsetKeyException
     */
    public function thenUnset(): static
    {
        $this->thenPart = static fn () => throw new UnsetKeyException('Unsetting key.');

        return $this;
    }

    /**
     * Returns the related node.
     *
     * @throws \RuntimeException
     */
    public function end(): NodeDefinition|ArrayNodeDefinition|VariableNodeDefinition
    {
        if (null === $this->ifPart) {
            throw new \RuntimeException('You must specify an if part.');
        }
        if (null === $this->thenPart) {
            throw new \RuntimeException('You must specify a then part.');
        }

        return $this->node;
    }

    /**
     * Builds the expressions.
     *
     * @param ExprBuilder[] $expressions An array of ExprBuilder instances to build
     */
    public static function buildExpressions(array $expressions): array
    {
        foreach ($expressions as $k => $expr) {
            if ($expr instanceof self) {
                $if = $expr->ifPart;
                $then = $expr->thenPart;
                $expressions[$k] = static fn ($v) => $if($v) ? $then($v) : $v;
            }
        }

        return $expressions;
    }
}
