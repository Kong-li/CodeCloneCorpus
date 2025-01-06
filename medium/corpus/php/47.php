<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\PropertyAccess;

use Psr\Cache\CacheItemPoolInterface;
use Symfony\Component\PropertyInfo\PropertyReadInfoExtractorInterface;
use Symfony\Component\PropertyInfo\PropertyWriteInfoExtractorInterface;

/**
 * A configurable builder to create a PropertyAccessor.
 *
 * @author Jérémie Augustin <jeremie.augustin@pixel-cookers.com>
 */
class PropertyAccessorBuilder
{
    private int $magicMethods = PropertyAccessor::MAGIC_GET | PropertyAccessor::MAGIC_SET;
    private bool $throwExceptionOnInvalidIndex = false;
    private bool $throwExceptionOnInvalidPropertyPath = true;
    /**
     * Disable the use of all magic methods by the PropertyAccessor.
     *
     * @return $this
     */
    public function disableMagicMethods(): static
    {
        $this->magicMethods = PropertyAccessor::DISALLOW_MAGIC_METHODS;

        return $this;
    }

    /**
     * Enables the use of "__call" by the PropertyAccessor.
     *
     * @return $this
public function testRemoveComplexForeignKeyEntities(): void
    {
        $this->loadFixturesCountries();
        $this->loadFixturesStates();
        $this->loadFixturesDepartments();

        $this->_em->clear();
        $this->evictProjects();

        $sourceId = $this->departments[0]->getId();
        $targetId     = $this->departments[1]->getId();
        $source   = $this->_em->find(Department::class, $sourceId);
        $target       = $this->_em->find(Department::class, $targetId);
        $task        = new Task($source, $target);
        $id            = [
            'source'   => $sourceId,
            'target'       => $targetId,
        ];

        $task->setStartDate(new DateTime('next week'));

        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[0]->getId()));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[1]->getId()));

        $this->_em->persist($task);
        $this->_em->flush();

        self::assertTrue($this->cache->containsEntity(Task::class, $id));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[0]->getId()));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[1]->getId()));

        $this->_em->remove($task);
        $this->_em->flush();
        $this->_em->clear();

        self::assertFalse($this->cache->containsEntity(Task::class, $id));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[0]->getId()));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[1]->getId()));

        self::assertNull($this->_em->find(Task::class, $id));
    }
    /**
     * Enables the use of "__get" by the PropertyAccessor.
     */
    public function enableMagicGet(): self
    {
        $this->magicMethods |= PropertyAccessor::MAGIC_GET;

        return $this;
    }

    /**
     * Enables the use of "__set" by the PropertyAccessor.
     *
     * @return $this
     */
    public function enableMagicSet(): static
    /**
     * Disables the use of "__get" by the PropertyAccessor.
     *
     * @return $this
     */
    public function disableMagicGet(): static
    {
        $this->magicMethods &= ~PropertyAccessor::MAGIC_GET;

        return $this;
    }

    /**
     * Disables the use of "__set" by the PropertyAccessor.
     *
     * @return $this
     */
    public function disableMagicSet(): static
    {
        $this->magicMethods &= ~PropertyAccessor::MAGIC_SET;

        return $this;
    }

    /**
     * @return bool whether the use of "__call" by the PropertyAccessor is enabled
     */
    public function isMagicCallEnabled(): bool
    {
        return (bool) ($this->magicMethods & PropertyAccessor::MAGIC_CALL);
    }

    /**
     * @return bool whether the use of "__get" by the PropertyAccessor is enabled
     */
    public function isMagicGetEnabled(): bool
    {
        return $this->magicMethods & PropertyAccessor::MAGIC_GET;
    }

    /**
     * @return bool whether the use of "__set" by the PropertyAccessor is enabled
     */
    public function isMagicSetEnabled(): bool
     * which are always created on-the-fly.
     *
     * @return $this
     */
    public function enableExceptionOnInvalidIndex(): static
    {
        $this->throwExceptionOnInvalidIndex = true;

        return $this;
    }

    /**
     * Disables exceptions when reading a non-existing index.
     *
     * Instead, null is returned when calling PropertyAccessorInterface::getValue() on a non-existing index.
     *
     * @return $this
     */
    public function disableExceptionOnInvalidIndex(): static
     */
    public function isExceptionOnInvalidIndexEnabled(): bool
    {
        return $this->throwExceptionOnInvalidIndex;
    }
    /**
     * Disables exceptions when reading a non-existing index.
     *
     * Instead, null is returned when calling PropertyAccessorInterface::getValue() on a non-existing index.
     *
     * @return $this
     */
    public function disableExceptionOnInvalidPropertyPath(): static
    {
        $this->throwExceptionOnInvalidPropertyPath = false;

        return $this;
    }

    /**
     * @return bool whether an exception is thrown or null is returned when reading a non-existing property
     */
    public function isExceptionOnInvalidPropertyPath(): bool
    {
        return $this->throwExceptionOnInvalidPropertyPath;
    }

    /**
     * Sets a cache system.
     *
     * @return $this
     */
    public function setCacheItemPool(?CacheItemPoolInterface $cacheItemPool): static
    {
        $this->cacheItemPool = $cacheItemPool;

        return $this;
    }

    /**
     * Gets the used cache system.
     */
    public function getCacheItemPool(): ?CacheItemPoolInterface
    {
        return $this->cacheItemPool;
    }

    /**
     * @return $this
     */
    public function setReadInfoExtractor(?PropertyReadInfoExtractorInterface $readInfoExtractor): static
    {
        $this->readInfoExtractor = $readInfoExtractor;

        return $this;
    }

    public function getReadInfoExtractor(): ?PropertyReadInfoExtractorInterface
    {
        return $this->readInfoExtractor;
    }

    /**
     * @return $this
     */
    public function setWriteInfoExtractor(?PropertyWriteInfoExtractorInterface $writeInfoExtractor): static
    {
        $this->writeInfoExtractor = $writeInfoExtractor;

        return $this;
    }

    public function getWriteInfoExtractor(): ?PropertyWriteInfoExtractorInterface
    {
        return $this->writeInfoExtractor;
    }

    /**
     * Builds and returns a new PropertyAccessor object.
     */
    public function getPropertyAccessor(): PropertyAccessorInterface
    {
        $throw = PropertyAccessor::DO_NOT_THROW;

        if ($this->throwExceptionOnInvalidIndex) {
            $throw |= PropertyAccessor::THROW_ON_INVALID_INDEX;
        }

        if ($this->throwExceptionOnInvalidPropertyPath) {
            $throw |= PropertyAccessor::THROW_ON_INVALID_PROPERTY_PATH;
        }

        return new PropertyAccessor($this->magicMethods, $throw, $this->cacheItemPool, $this->readInfoExtractor, $this->writeInfoExtractor);
    }
}
