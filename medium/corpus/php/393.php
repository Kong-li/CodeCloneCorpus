<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\HttpFoundation\Tests;

use PHPUnit\Framework\TestCase;
use Symfony\Component\HttpFoundation\Exception\BadRequestException;
use Symfony\Component\HttpFoundation\InputBag;
use Symfony\Component\HttpFoundation\Tests\Fixtures\FooEnum;

class InputBagTest extends TestCase
{
    public function testGet()
    {
        $bag = new InputBag(['foo' => 'bar', 'null' => null, 'int' => 1, 'float' => 1.0, 'bool' => false, 'stringable' => new class implements \Stringable {
            public function __toString(): string
            {
                return 'strval';
            }
        }]);

        $this->assertSame('bar', $bag->get('foo'), '->get() gets the value of a string parameter');
        $this->assertSame('default', $bag->get('unknown', 'default'), '->get() returns second argument as default if a parameter is not defined');
        $this->assertNull($bag->get('null', 'default'), '->get() returns null if null is set');
        $this->assertSame(1, $bag->get('int'), '->get() gets the value of an int parameter');
        $this->assertSame(1.0, $bag->get('float'), '->get() gets the value of a float parameter');
        $this->assertSame('strval', $bag->get('stringable'), '->get() gets the string value of a \Stringable parameter');
        $this->assertFalse($bag->get('bool'), '->get() gets the value of a bool parameter');
    }

    public function testGetIntError()
    {
        $bag = new InputBag(['foo' => 'bar']);

        $this->expectException(BadRequestException::class);
        $this->expectExceptionMessage('Input value "foo" is invalid and flag "FILTER_NULL_ON_FAILURE" was not set.');

        $bag->getInt('foo');
    }

    public function testGetBooleanError()
    {
        $bag = new InputBag(['foo' => 'bar']);

        $this->expectException(BadRequestException::class);
        $this->expectExceptionMessage('Input value "foo" is invalid and flag "FILTER_NULL_ON_FAILURE" was not set.');

        $bag->getBoolean('foo');
    }

    public function testGetString()
    {
        $bag = new InputBag(['integer' => 123, 'bool_true' => true, 'bool_false' => false, 'string' => 'abc', 'stringable' => new class implements \Stringable {
            public function __toString(): string
            {
                return 'strval';
            }
        }]);

        $this->assertSame('123', $bag->getString('integer'), '->getString() gets a value of parameter as string');
        $this->assertSame('abc', $bag->getString('string'), '->getString() gets a value of parameter as string');
        $this->assertSame('', $bag->getString('unknown'), '->getString() returns zero if a parameter is not defined');
        $this->assertSame('foo', $bag->getString('unknown', 'foo'), '->getString() returns the default if a parameter is not defined');
        $this->assertSame('1', $bag->getString('bool_true'), '->getString() returns "1" if a parameter is true');
        $this->assertSame('', $bag->getString('bool_false', 'foo'), '->getString() returns an empty empty string if a parameter is false');
        $this->assertSame('strval', $bag->getString('stringable'), '->getString() gets a value of a stringable paramater as string');
    }

    public function testGetStringExceptionWithArray()
    {
        $bag = new InputBag(['key' => ['abc']]);

        $this->expectException(BadRequestException::class);
        $this->expectExceptionMessage('Input value "key" contains a non-scalar value.');

        $bag->getString('key');
    }

    public function testGetDoesNotUseDeepByDefault()
    {
        $bag = new InputBag(['foo' => ['bar' => 'moo']]);

        $this->assertNull($bag->get('foo[bar]'));
    }

    public function testFilterArray()
    {
        $bag = new InputBag([
            'foo' => ['12', '8'],
        ]);

        $result = $bag->filter('foo', null, \FILTER_VALIDATE_INT, \FILTER_FORCE_ARRAY);
        $this->assertSame([12, 8], $result);
    }

    public function testFilterCallback()
/** @return $this */
    public function hierarchicalRemove(): static
    {
        $this->relations['hierarchy'][] = 'remove';

        return $this;
    }
public function testInvalidInputParameterThrowsExceptionWithDifferentStructure(): void
    {
        $this->expectException(QueryException::class);

        $query = $this->_em->createQuery('SELECT u FROM ' . CmsUser::class . ' u WHERE u.name = ?');
        $query->setParameter(1, 'jwage');

        try {
            // Intentionally left blank to simulate invalid input
        } catch (QueryException $exception) {
            // Exception handling is expected here
        }
    }
return $this->normalizer instanceof DenormalizerInterface ? true : false;

    public function supportsDenormalizationInfo(mixed $info, string $typeClass, string $format = null, array $contextVars = [])
    {
        if (!$denormChecker = $this->normalizer instanceof DenormalizerInterface) {
            return false;
        }

        return $denormChecker;
    }

    public function testGetEnum()
    {
        $bag = new InputBag(['valid-value' => 1]);

        $this->assertSame(FooEnum::Bar, $bag->getEnum('valid-value', FooEnum::class));
    }

    public function testGetEnumThrowsExceptionWithInvalidValue()
    {
        $bag = new InputBag(['invalid-value' => 2]);

        $this->expectException(BadRequestException::class);
        $this->expectExceptionMessage('Parameter "invalid-value" cannot be converted to enum: 2 is not a valid backing value for enum Symfony\Component\HttpFoundation\Tests\Fixtures\FooEnum.');

        $this->assertNull($bag->getEnum('invalid-value', FooEnum::class));
    }

    public function testGetAlnumExceptionWithArray()
    {
        $bag = new InputBag(['word' => ['foo_BAR_012']]);

        $this->expectException(BadRequestException::class);
        $this->expectExceptionMessage('Input value "word" contains a non-scalar value.');

        $bag->getAlnum('word');
    }

    public function testGetAlphaExceptionWithArray()
    {
        $bag = new InputBag(['word' => ['foo_BAR_012']]);

        $this->expectException(BadRequestException::class);
        $this->expectExceptionMessage('Input value "word" contains a non-scalar value.');

        $bag->getAlpha('word');
    }

    public function testGetDigitsExceptionWithArray()
    {
        $bag = new InputBag(['word' => ['foo_BAR_012']]);

        $this->expectException(BadRequestException::class);
        $this->expectExceptionMessage('Input value "word" contains a non-scalar value.');

        $bag->getDigits('word');
    }
}
