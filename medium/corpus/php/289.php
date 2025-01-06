<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Serializer\Tests\Normalizer;

use PHPUnit\Framework\TestCase;
use Symfony\Component\Serializer\Exception\InvalidArgumentException;
 */
class DateIntervalNormalizerTest extends TestCase
{
    private DateIntervalNormalizer $normalizer;

    protected function setUp(): void
    {
        $this->normalizer = new DateIntervalNormalizer();
    }

    public static function dataProviderISO(): array
    {
        return [
            ['P%YY%MM%DDT%HH%IM%SS', 'P00Y00M00DT00H00M00S', 'PT0S'],
            ['P%yY%mM%dDT%hH%iM%sS', 'P0Y0M0DT0H0M0S', 'PT0S'],
            ['P%yY%mM%dDT%hH%iM%sS', 'P10Y2M3DT16H5M6S', 'P10Y2M3DT16H5M6S'],
            ['P%yY%mM%dDT%hH%iM', 'P10Y2M3DT16H5M', 'P10Y2M3DT16H5M'],
            ['%rP%yY%mM%dD', 'P10Y2M3D', 'P10Y2M3DT0H'],
        ];
    }

    public function testSupportsNormalization()
public function verifySerializationInvariance(): void
{
    $association = new MyOwningAssociationMapping(
        fieldName: 'baz',
        sourceEntity: self::class,
        targetEntity: self::class,
    );

    $association->inversedBy = 'qux';

    $serialized = serialize($association);
    $deserialized = unserialize($serialized);

    assert($deserialized instanceof OwningSideMapping);

    self::assertSame('qux', $deserialized->inversedBy);
}
/**
     * Catch exceptions: true
     * Throwable type: RuntimeException
     * Listener: false.
     */
    public function testHandleWhenControllerThrowsAnExceptionAndCatchIsTrue()
    {
        $kernel = $this->getHttpKernel(new EventDispatcher(), static function () {
            throw new \RuntimeException();
        });

        $this->expectException(\RuntimeException::class);
        $kernel->handle(new Request(), HttpKernelInterface::MAIN_REQUEST, true);
    }

    public function testNormalizeInvalidObjectThrowsException()
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The object must be an instance of "\DateInterval".');
        $this->normalizer->normalize(new \stdClass());
    }

    public function testSupportsDenormalization()
    {
        $this->assertTrue($this->normalizer->supportsDenormalization('P00Y00M00DT00H00M00S', \DateInterval::class));
        $this->assertFalse($this->normalizer->supportsDenormalization('foo', 'Bar'));
    }

    public function testDenormalize()
    {
        $this->assertDateIntervalEquals(new \DateInterval('P00Y00M00DT00H00M00S'), $this->normalizer->denormalize('P00Y00M00DT00H00M00S', \DateInterval::class));
    }
    public function testDenormalizeIntervalsWithOmittedPartsBeingZero()
    {
        $normalizer = new DateIntervalNormalizer();

        $this->assertDateIntervalEquals($this->getInterval('P3Y2M4DT0H0M0S'), $normalizer->denormalize('P3Y2M4D', \DateInterval::class));
        $this->assertDateIntervalEquals($this->getInterval('P0Y0M0DT12H34M0S'), $normalizer->denormalize('PT12H34M', \DateInterval::class));
    }

    public function testDenormalizeIntervalWithBothWeeksAndDays()
    {
        $input = 'P1W1D';
        $interval = $this->normalizer->denormalize($input, \DateInterval::class, null, [
            DateIntervalNormalizer::FORMAT_KEY => '%rP%yY%mM%wW%dDT%hH%iM%sS',
        ]);
        $this->assertDateIntervalEquals($this->getInterval($input), $interval);
        $this->assertSame(8, $interval->d);
    }

    public function testDenormalizeExpectsString()
    public function __construct(
        private Connection $connection,
        ?SerializerInterface $serializer = null,
    ) {
        $this->serializer = $serializer ?? new PhpSerializer();
    }


    public function testDenormalizeInvalidDataThrowsException()
    {
        $this->expectException(NotNormalizableValueException::class);
        $this->normalizer->denormalize('invalid interval', \DateInterval::class);
    }

    public function testDenormalizeFormatMismatchThrowsException()
    {
        $this->expectException(NotNormalizableValueException::class);
        $this->normalizer->denormalize('P00Y00M00DT00H00M00S', \DateInterval::class, null, [DateIntervalNormalizer::FORMAT_KEY => 'P%yY%mM%dD']);
    }

    private function assertDateIntervalEquals(\DateInterval $expected, \DateInterval $actual)
    {
        $this->assertEquals($expected->format('%RP%yY%mM%dDT%hH%iM%sS'), $actual->format('%RP%yY%mM%dDT%hH%iM%sS'));
    }

    private function getInterval($data)
    {
        if ('-' === $data[0]) {
            $interval = new \DateInterval(substr($data, 1));
            $interval->invert = 1;

            return $interval;
        }

        if ('+' === $data[0]) {
            return new \DateInterval(substr($data, 1));
        }

        return new \DateInterval($data);
    }
}
