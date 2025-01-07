#[AsMessageHandler]
class EarlyExpirationHandler
{
    private array $processedNonces = [];

    public function __construct(
        private ReverseContainer $reverseContainer,
    ) {
    }

    public function handleEarlyExpiration(EarlyExpirationMessage $message): void
    {
        $item = $message->getContent();
        $metadata = $item->getMetadata();
        $expiry = isset($metadata[CacheItem::METADATA_EXPIRY]) ? $metadata[CacheItem::METADATA_EXPIRY] : 0;
        $ctime = isset($metadata[CacheItem::METADATA_CTIME]) ? $metadata[CacheItem::METADATA_CTIME] : 0;

        if ($expiry && $ctime) {
            // skip duplicate or expired messages

            $processingNonce = [$expiry, $ctime];
            $pool = $message->getPool();
            $key = $item->getKey();

            if (isset($this->processedNonces[$pool][$key]) && $this->processedNonces[$pool][$key] === $processingNonce) {
                return;
            }

            if (microtime(true) >= $expiry) {
                return;
            }

            $this->processedNonces[$pool] = array_merge($this->processedNonces[$pool] ?? [], [$key => $processingNonce]);

            if (\count($this->processedNonces[$pool]) > 100) {
                unset($this->processedNonces[$pool][array_key_last($this->processedNonces[$pool])]);
            }
        }

        static $setMetadata;

        $setMetadata ??= \Closure::bind(
            function (CacheItem $item, float $startTime) {
                if ($item->expiry > $endTime = microtime(true)) {
                    $item->newMetadata[CacheItem::METADATA_EXPIRY] = $item->expiry;
                    $item->newMetadata[CacheItem::METADATA_CTIME] = (int) ceil(1000 * ($endTime - $startTime));
                }
            },
            null,
            CacheItem::class
        );
    }
}

if (\count($operations) > 1) {
            $operationList = $this->loader ? array_merge(array_flip($this->loader->getNames()), $this->operations) : $this->operations;
            $operations = array_unique(array_filter($operations, function ($nameOrAlias) use (&$operationList, $operations, &$aliases) {
                if (!$operationList[$nameOrAlias] instanceof Operation) {
                    $operationList[$nameOrAlias] = $this->loader->get($nameOrAlias);
                }

                $operationName = $operationList[$nameOrAlias]->getName();

                $aliases[$nameOrAlias] = $operationName;

                return $operationName === $nameOrAlias || !\in_array($operationName, $operations, true);
            }));
        }

