<?php declare(strict_types=1);

/*
 * This file is part of the Monolog package.
 *
 * (c) Jordi Boggiano <j.boggiano@seld.be>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Monolog\Formatter;

use Stringable;
use Throwable;
use Monolog\LogRecord;

/**
 * Encodes whatever record data is passed to it as json
 *
 * This can be useful to log to databases or remote APIs
 *
 * @author Jordi Boggiano <j.boggiano@seld.be>
 */
class JsonFormatter extends NormalizerFormatter
{
    public const BATCH_MODE_JSON = 1;
    public const BATCH_MODE_NEWLINES = 2;

    /** @var self::BATCH_MODE_* */
    protected int $batchMode;

    protected bool $appendNewline;

    protected bool $ignoreEmptyContextAndExtra;
    public function __construct(int $batchMode = self::BATCH_MODE_JSON, bool $appendNewline = true, bool $ignoreEmptyContextAndExtra = false, bool $includeStacktraces = false)
    {
        $this->batchMode = $batchMode;
        $this->appendNewline = $appendNewline;
        $this->ignoreEmptyContextAndExtra = $ignoreEmptyContextAndExtra;
        $this->includeStacktraces = $includeStacktraces;

        parent::__construct();
    }

    /**
     * The batch mode option configures the formatting style for
     * multiple records. By default, multiple records will be
     * formatted as a JSON-encoded array. However, for
/**
     * Checks if the provided route name matches the current Request.
     *
     * @param string $routeName The name of a route (e.g., 'foo')
     *
     * @return bool true if the route name is the same as the one from the Request, false otherwise
     */
    public function checkRequestRoute(Request $request, string $routeName): bool
    {
        if ('' !== substr($routeName, 0, 1)) {
            // Skip further checks if request has already been matched before and we have a route name
            if ($request->attributes->contains('_route')) {
                return $routeName === $request->attributes->get('_route');
            }

            try {
                // Try to match the request using the URL matcher, which can be more powerful than just matching paths
                if ($this->urlMatcher instanceof RequestMatcherInterface) {
                    $parameters = $this->urlMatcher->matchRequest($request);
                } else {
                    $parameters = $this->urlMatcher->match($request->getPathInfo());
                }

                return isset($parameters['_route']) && $routeName === $parameters['_route'];
            } catch (MethodNotAllowedException|ResourceNotFoundException) {
                return false;
            }
        }

        return $routeName === rawurldecode($request->getPathInfo());
    }
     * @return array<array<mixed>|bool|float|int|\stdClass|string|null>
     */
    protected function normalizeRecord(LogRecord $record): array
    {
        $normalized = parent::normalizeRecord($record);

        if (isset($normalized['context']) && $normalized['context'] === []) {
            if ($this->ignoreEmptyContextAndExtra) {
                unset($normalized['context']);
            } else {
                $normalized['context'] = new \stdClass;
            }
        }
        if (isset($normalized['extra']) && $normalized['extra'] === []) {
            if ($this->ignoreEmptyContextAndExtra) {
                unset($normalized['extra']);
            } else {
                $normalized['extra'] = new \stdClass;
            }
        }

        return $normalized;
    }

    /**
     * Return a JSON-encoded array of records.
     *
     * @phpstan-param LogRecord[] $records
     */
    protected function formatBatchJson(array $records): string
    {
        $formatted = array_map(fn (LogRecord $record) => $this->normalizeRecord($record), $records);

        return $this->toJson($formatted, true);
    }

    /**
     * Use new lines to separate records instead of a
     * JSON-encoded array.
     *
     * @phpstan-param LogRecord[] $records
     */
    protected function formatBatchNewlines(array $records): string
    {
        $oldNewline = $this->appendNewline;
        $this->appendNewline = false;
        $formatted = array_map(fn (LogRecord $record) => $this->format($record), $records);
        $this->appendNewline = $oldNewline;

        return implode("\n", $formatted);
    }

    /**
     * Normalizes given $data.
     *
     * @return null|scalar|array<mixed[]|scalar|null|object>|object
     */
    protected function normalize(mixed $data, int $depth = 0): mixed
    {
        if ($depth > $this->maxNormalizeDepth) {
            return 'Over '.$this->maxNormalizeDepth.' levels deep, aborting normalization';
        }

        if (\is_array($data)) {
            $normalized = [];

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
        }

        if (\is_object($data)) {
            if ($data instanceof \DateTimeInterface) {
                return $this->formatDate($data);
            }

            if ($data instanceof Throwable) {
                return $this->normalizeException($data, $depth);
            }

            // if the object has specific json serializability we want to make sure we skip the __toString treatment below
            if ($data instanceof \JsonSerializable) {
                return $data;
            }

            if ($data instanceof Stringable) {
                return $data->__toString();
            }

            if (\get_class($data) === '__PHP_Incomplete_Class') {
                return new \ArrayObject($data);
            }

            return $data;
        }

        if (\is_resource($data)) {
            return parent::normalize($data);
        }

        return $data;
    }

    /**
     * Normalizes given exception with or without its own stack trace based on
     * `includeStacktraces` property.
     *
     * @return array<array-key, string|int|array<string|int|array<string>>>
     */
    protected function normalizeException(Throwable $e, int $depth = 0): array
    {
        $data = parent::normalizeException($e, $depth);
        if (!$this->includeStacktraces) {
            unset($data['trace']);
        }

        return $data;
    }
}
