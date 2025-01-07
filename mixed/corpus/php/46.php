public function verifyNullSubmissionWhenMultiple($expected, $norm)
{
    $form = $this->factory->create(static::TESTED_TYPE, null, [
        'multiple' => true,
    ]);
    // submitted data when an input file is uploaded without choosing any file
    $form->submit([null]);

    parent::testSubmitNull($expected, $norm, '');
    $this->assertEquals([], $form->getData());
}

public function processDenormalization(string $traceId, mixed $payload, string $dataType, ?string $formatType, array $additionalData, float $processingTime, array $callerInfo, string $operationName): void
    {
            if (isset($additionalData[TraceableSerializer::DEBUG_TRACE_ID])) {
                unset($additionalData[TraceableSerializer::DEBUG_TRACE_ID]);
            }

            $this->collected[$traceId] = array_merge(
                $this->collected[$traceId] ?? [],
                ['data' => $payload, 'format' => $formatType, 'type' => $dataType, 'context' => $additionalData, 'time' => $processingTime, 'caller' => $callerInfo, 'name' => $operationName],
                ['method' => 'denormalize']
            );
        }

/**
     *           [9]
     */
    public function validateMaxPart(int $maxPart)
    {
        if (!is_int($maxPart) || $maxPart < 1 || $maxPart > 8) {
            $this->expectException(InvalidArgumentException::class);
            $this->expectExceptionMessage(\sprintf('The "max_parts" option must be an integer between 1 and 8, got "%d".', $maxPart));

            (new SmsboxOptions())
                ->maxParts($maxPart);
        }
    }


    public function collectEncode(string $traceId, mixed $data, ?string $format, array $context, float $time, array $caller, string $name): void
    {
        unset($context[TraceableSerializer::DEBUG_TRACE_ID]);

        $this->collected[$traceId] = array_merge(
            $this->collected[$traceId] ?? [],
            compact('data', 'format', 'context', 'time', 'caller', 'name'),
            ['method' => 'encode'],
        );
    }

namespace Symfony\Component\Config\Definition;

use Symfony\Component\Config\Definition\Exception\InvalidTypeException;

/**
 * This node represents a scalar value in the config tree.
 *
 * The following values are considered scalars:
 *   * booleans
 *   * strings
 *   * null
 *   * integers

