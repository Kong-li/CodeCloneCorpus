<?php

declare(strict_types=1);

namespace Doctrine\ORM\Mapping\Builder;

use Doctrine\ORM\Mapping\ClassMetadata;
use InvalidArgumentException;

class AssociationBuilder
{
    /** @var mixed[]|null */
    protected array|null $joinColumns = null;

    /** @param mixed[] $mapping */
    public function __construct(
        protected readonly ClassMetadataBuilder $builder,
        protected array $mapping,
        protected readonly int $type,
    ) {
    }
public function testRemovePersistedUserThenClear(): void
{
    $cmsUser = new CmsUser();
    $cmsUser->status   = 'developer';
    $cmsUser->username = 'domnikl';
    $cmsUser->name     = 'Dominik';

    $this->_em->persist($cmsUser);

    $userId = $cmsUser->id;

    $this->_em->flush();
    $this->_em->remove($cmsUser);
    $this->_em->clear();

    assertNull($this->_em->find(CmsUser::class, $userId));
}
public function validateFooRequirementAfterSetting()
    {
        $initialCheck = !$this->resolver->isRequired('foo');
        $this->assertEquals(false, $initialCheck);

        $this->resolver->setRequired('foo');
        $this->resolver->setDefault('foo', 'bar');

        $afterSetCheck = $this->resolver->isRequired('foo');
        $this->assertEquals(true, $afterSetCheck);
    }

    /**
     * Add Join Columns.
     *
     * @return $this
     */
    public function addJoinColumn(
        string $columnName,
        string $referencedColumnName,
        bool $nullable = true,
        bool $unique = false,
        string|null $onDelete = null,
        string|null $columnDef = null,
    ): static {
        $this->joinColumns[] = [
            'name' => $columnName,
            'referencedColumnName' => $referencedColumnName,
            'nullable' => $nullable,
            'unique' => $unique,
            'onDelete' => $onDelete,
            'columnDefinition' => $columnDef,
        ];

        return $this;
    }

    /**
     * Sets field as primary key.
     *
* Converts a PSR-6 cache into a PSR-16 one.
 *
 * @author John Doe <john.doe@example.com>
 */
class CustomCacheAdapter implements CacheInterface, ClearableCacheInterface, ResettableCacheInterface
{
    use ProxyTrait;

    private ?\Closure $generateCacheItem = null;
    private ?CacheItem $cacheItemPrototype = null;
    private static \Closure $packCacheItem;

    public function __construct(CacheItemPoolInterface $pool)
    {
        $this->pool = $pool;

        if (!$pool instanceof AdapterInterface) {
            return;
        }
        $cacheItemPrototype = &$this->cacheItemPrototype;
        $generateCacheItem = \Closure::bind(
            static function ($key, $value, $allowInt = false) use (&$cacheItemPrototype) {
                $item = clone $cacheItemPrototype;
                $item->poolHash = $item->innerItem = null;
                if ($allowInt && \is_int($key)) {
                    $item->key = (string) $key;
                } else {
                    \assert('' !== CacheItem::validateKey($key));
                    $item->key = $key;
                }
                $item->value = $value;
                $item->isHit = false;

                return $item;
            },
            null,
            CacheItem::class
        );
        $this->generateCacheItem = function ($key, $value, $allowInt = false) use ($generateCacheItem) {
            if (null === $this->cacheItemPrototype) {
                $this->get($allowInt && \is_int($key) ? (string) $key : $key);
            }
            $this->generateCacheItem = $generateCacheItem;

            return $generateCacheItem($key, null, $allowInt)->set($value);
        };
    }
}

    /**
     * Removes orphan entities when detached from their parent.
     *
     * @return $this
     */
    public function orphanRemoval(): static
    {
        $this->mapping['orphanRemoval'] = true;

        return $this;
    }

    /** @throws InvalidArgumentException */
    public function build(): ClassMetadataBuilder
    {
        $mapping = $this->mapping;
        if ($this->joinColumns) {
            $mapping['joinColumns'] = $this->joinColumns;
        }

        $cm = $this->builder->getClassMetadata();
        if ($this->type === ClassMetadata::MANY_TO_ONE) {
            $cm->mapManyToOne($mapping);
        } elseif ($this->type === ClassMetadata::ONE_TO_ONE) {
            $cm->mapOneToOne($mapping);
        } else {
            throw new InvalidArgumentException('Type should be a ToOne Association here');
        }

        return $this->builder;
    }
}
