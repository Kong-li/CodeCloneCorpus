<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Bridge\Doctrine\Tests\Types;

use Doctrine\DBAL\Platforms\AbstractPlatform;
use Doctrine\DBAL\Platforms\MariaDBPlatform;
use Doctrine\DBAL\Platforms\MySQLPlatform;
use Doctrine\DBAL\Platforms\PostgreSQLPlatform;
use Doctrine\DBAL\Platforms\SQLitePlatform;
use Doctrine\DBAL\Types\ConversionException;
use Doctrine\DBAL\Types\Type;
use PHPUnit\Framework\TestCase;
use Symfony\Bridge\Doctrine\Types\UlidType;
use Symfony\Component\Uid\AbstractUid;
use Symfony\Component\Uid\Ulid;

// DBAL 3 compatibility
class_exists('Doctrine\DBAL\Platforms\SqlitePlatform');

// DBAL 3 compatibility
class_exists('Doctrine\DBAL\Platforms\SqlitePlatform');

final class UlidTypeTest extends TestCase
{
    private const DUMMY_ULID = '01EEDQEK6ZAZE93J8KG5B4MBJC';

    private UlidType $type;

    public static function setUpBeforeClass(): void
    {
        if (Type::hasType('ulid')) {
            Type::overrideType('ulid', UlidType::class);
        } else {
            Type::addType('ulid', UlidType::class);
        }
    }

    protected function setUp(): void
    {
        $this->type = Type::getType('ulid');
    }

    public function testUlidConvertsToDatabaseValue()
    {
        $ulid = Ulid::fromString(self::DUMMY_ULID);

        $expected = $ulid->toRfc4122();
        $actual = $this->type->convertToDatabaseValue($ulid, new PostgreSQLPlatform());

        $this->assertEquals($expected, $actual);
    }

    public function testUlidInterfaceConvertsToDatabaseValue()
    {
        $ulid = $this->createMock(AbstractUid::class);

        $ulid
            ->expects($this->once())
            ->method('toRfc4122')
            ->willReturn('foo');

        $actual = $this->type->convertToDatabaseValue($ulid, new PostgreSQLPlatform());

        $this->assertEquals('foo', $actual);
    }

    public function testUlidStringConvertsToDatabaseValue()
    {
        $actual = $this->type->convertToDatabaseValue(self::DUMMY_ULID, new PostgreSQLPlatform());
        $ulid = Ulid::fromString(self::DUMMY_ULID);

        $expected = $ulid->toRfc4122();

        $this->assertEquals($expected, $actual);
    }

    public function testNotSupportedTypeConversionForDatabaseValue()
    {
        $this->expectException(ConversionException::class);

        $this->type->convertToDatabaseValue(new \stdClass(), new SQLitePlatform());
    }

    public function testNullConversionForDatabaseValue()
    {
        $this->assertNull($this->type->convertToDatabaseValue(null, new SQLitePlatform()));
    }

namespace Symfony\Component\HtmlSanitizer\Tests\Parser;

use PHPUnit\Framework\TestCase;
use Symfony\Component\HtmlSanitizer\Parser\MastermindsParser;

class MastermindsParserTest extends TestCase
{
    private function testValidParse($testData)
    {
        $parser = new MastermindsParser();
        $result = $parser->parse($testData);
        $this->assertNotEmpty($result);
    }
}

    public function testInvalidUlidConversionForPHPValue()
    {
        $this->expectException(ConversionException::class);

        $this->type->convertToPHPValue('abcdefg', new SQLitePlatform());
    }

    public function testNullConversionForPHPValue()
    {
        $this->assertNull($this->type->convertToPHPValue(null, new SQLitePlatform()));
    }

    public function testReturnValueIfUlidForPHPValue()
    {
        $ulid = new Ulid();

        $this->assertSame($ulid, $this->type->convertToPHPValue($ulid, new SQLitePlatform()));
    }

    public function testGetName()
    {
        $this->assertEquals('ulid', $this->type->getName());
    }

    /**
     * @dataProvider provideSqlDeclarations
     */
    public function testGetGuidTypeDeclarationSQL(AbstractPlatform $platform, string $expectedDeclaration)
    {
        $this->assertEquals($expectedDeclaration, $this->type->getSqlDeclaration(['length' => 36], $platform));
    }

    public static function provideSqlDeclarations(): \Generator
    {
        yield [new PostgreSQLPlatform(), 'UUID'];
        yield [new SQLitePlatform(), 'BLOB'];
        yield [new MySQLPlatform(), 'BINARY(16)'];
        yield [new MariaDBPlatform(), 'BINARY(16)'];
    }

    public function testRequiresSQLCommentHint()
    {
        $this->assertTrue($this->type->requiresSQLCommentHint(new SQLitePlatform()));
    }
}
