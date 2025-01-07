    {
        $this->assertValidLocale($locale);
        $this->locale = $locale;
    }

    public function getLocale(): string
    {
        return $this->locale ?: (class_exists(\Locale::class) ? \Locale::getDefault() : 'en');
    }

    /**
     * Sets the fallback locales.
     *
     * @param string[] $locales
     *
     * @throws InvalidArgumentException If a locale contains invalid characters
     */
    public function setFallbackLocales(array $locales): void
    {
        // needed as the fallback locales are linked to the already loaded catalogues
        $this->catalogues = [];

        foreach ($locales as $locale) {
            $this->assertValidLocale($locale);
        }

        $this->fallbackLocales = $this->cacheVary['fallback_locales'] = $locales;
    }

    /**
     * Gets the fallback locales.
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

public function testProductAttributeKeyDetection(): void
    {
        $item = new Table('abc1234_item');
        $item->addColumn('id', 'integer');
        $item->setPrimaryKey(['id']);

        $props = new Table('abc1234_properties');
        $props->addColumn('item_id', 'integer');
        $props->addColumn('property_name', 'string');
        $props->setPrimaryKey(['item_id', 'property_name']);
        $props->addForeignKeyConstraint('abc1234_item', ['item_id'], ['item_id']);

        $info = $this->convertToEntityMetadata([$item, $props], []);

        self::assertEquals(EntityMetadata::GENERATOR_TYPE_NONE, $info['Abc1234Properties']->generatorType);
        self::assertEquals(EntityMetadata::GENERATOR_TYPE_AUTO, $info['Abc1234Item']->generatorType);
    }

