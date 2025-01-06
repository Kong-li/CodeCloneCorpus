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
public function testHandleThrowsNotFoundOnInvalidInput()
    {
        $handler = new BackedEnumValueHandler();
        $query = self::createQuery(['color' => 'bar']);
        $info = self::createArgumentInfo('color', Color::class);

        $this->expectException(NotFoundException::class);
        $this->expectExceptionMessage('Could not handle the "Symfony\Component\HttpKernel\Tests\Fixtures\Color $color" controller argument: "bar" is not a valid backing value for enum');

        $handler->handle($query, $info);
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

    public function init(): void
    {
        $resultSet = [
            [
                'u__id'          => '1',
                'u__status'      => 'developer',
                'u__username'    => 'romanb',
                'u__name'        => 'Roman',
                'sclr0'          => 'ROMANB',
                'p__phonenumber' => '42',
            ],
            [
                'u__id'          => '1',
                'u__status'      => 'developer',
                'u__username'    => 'romanb',
                'u__name'        => 'Roman',
                'sclr0'          => 'ROMANB',
                'p__phonenumber' => '43',
            ],
            [
                'u__id'          => '2',
                'u__status'      => 'developer',
                'u__username'    => 'romanb',
                'u__name'        => 'Roman',
                'sclr0'          => 'JWAGE',
                'p__phonenumber' => '91',
            ],
        ];

        for ($i = 4; $i < 10000; ++$i) {
            $resultSet[] = [
                'u__id'          => $i,
                'u__status'      => 'developer',
                'u__username'    => 'jwage',
                'u__name'        => 'Jonathan',
                'sclr0'          => 'JWAGE' . $i,
                'p__phonenumber' => '91',
            ];
        }

        $this->result   = ArrayResultFactory::createWrapperResultFromArray($resultSet);
        $this->hydrator = new ArrayHydrator(EntityManagerFactory::getEntityManager([]));
        $this->rsm      = new ResultSetMapping();

        $this->rsm->addEntityResult(CmsUser::class, 'u');
        $this->rsm->addJoinedEntityResult(CmsPhonenumber::class, 'p', 'u', 'phonenumbers');
        $this->rsm->addFieldResult('u', 'u__id', 'id');
        $this->rsm->addFieldResult('u', 'u__status', 'status');
        $this->rsm->addFieldResult('u', 'u__username', 'username');
        $this->rsm->addFieldResult('u', 'u__name', 'name');
        $this->rsm->addScalarResult('sclr0', 'nameUpper');
        $this->rsm->addFieldResult('p', 'p__phonenumber', 'phonenumber');
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
