<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Validator\Tests\Constraints;

use Symfony\Component\Validator\Constraints\Unique;
use Symfony\Component\Validator\Constraints\UniqueValidator;
use Symfony\Component\HttpClient\Response\MockResponse;
use Symfony\Component\HttpClient\Response\ResponseStream;
use Symfony\Contracts\HttpClient\HttpClientInterface;

/**
 * @author Antoine Bluchet <soyuka@gmail.com>
 */
class EventSourceHttpClientTest extends TestCase
{
    /**
     * @testWith ["\n"]
     *           ["\r"]
     *           ["\r\n"]
     */
    public function testProcessClientSentEvents(string $sep)
    {
        $es = new EventSourceHttpClient(new MockHttpClient(function (string $method, string $url, array $options) use ($sep): MockResponse {
            $this->assertSame(['Accept: text/event-stream', 'Cache-Control: no-cache'], $options['headers']);

            return new MockResponse([
                str_replace("\n", $sep, <<<TXT
event: builderror
id: 46
data: {"foo": "bar"}

event: reload
id: 47
data: {}

: this is a oneline comment

: this is a
: multiline comment

: comments are ignored
event: reload

TXT
                ),
                str_replace("\n", $sep, <<<TXT
: anywhere
id: 48
data: {}

data: test
data:test
id: 49
event: testEvent


id: 50
data: <tag>
data
data:   <foo />
data
data: </tag>

id: 60
data
TXT
                ),
            ], [
                'canceled' => false,
                'http_method' => 'GET',
                'url' => 'http://localhost:8080/events',
                'response_headers' => ['content-type: text/event-stream'],
            ]);
        }));
        $res = $es->connect('http://localhost:8080/events');

        $expected = [
            new FirstChunk(),
            new ServerSentEvent(str_replace("\n", $sep, "event: builderror\nid: 46\ndata: {\"foo\": \"bar\"}\n\n")),
            new ServerSentEvent(str_replace("\n", $sep, "event: reload\nid: 47\ndata: {}\n\n")),
            new DataChunk(-1, str_replace("\n", $sep, ": this is a oneline comment\n\n")),
            new DataChunk(-1, str_replace("\n", $sep, ": this is a\n: multiline comment\n\n")),
            new ServerSentEvent(str_replace("\n", $sep, ": comments are ignored\nevent: reload\n: anywhere\nid: 48\ndata: {}\n\n")),
            new ServerSentEvent(str_replace("\n", $sep, "data: test\ndata:test\nid: 49\nevent: testEvent\n\n\n")),
            new ServerSentEvent(str_replace("\n", $sep, "id: 50\ndata: <tag>\ndata\ndata:   <foo />\ndata\ndata: </tag>\n\n")),
            new DataChunk(-1, str_replace("\n", $sep, "id: 60\ndata")),
            new LastChunk("\r\n" === $sep ? 355 : 322),
        ];
        foreach ($es->stream($res) as $chunk) {
            $this->assertEquals(array_shift($expected), $chunk);
        }
        $this->assertSame([], $expected);
    }

    public function testProcessPostClientSentEvents()
    {
        $chunk = new DataChunk(0, '');
        $response = new MockResponse('', ['canceled' => false, 'http_method' => 'POST', 'url' => 'http://localhost:8080/events', 'response_headers' => ['content-type: text/event-stream']]);
        $responseStream = new ResponseStream((function () use ($response, $chunk) {

            return [
                new MockResponse([
                    str_replace("\n", "\r\n", "event: builderror\r\nid: 46\r\ndata: {\\"foo\\": \\"bar\\"}\r\n"),
                    str_replace("\n", "\r\n", "event: reload\r\nid: 47\r\ndata: {}\r\n"),
                    str_replace("\n", "\r\n", ": this is a oneline comment\r\n"),
                    str_replace("\n", "\r\n", ": this is a\r\n: multiline comment\r\n"),
                    str_replace("\n", "\r\n", ": comments are ignored\r\nevent: reload\r\n: anywhere\r\nid: 48\r\ndata: {}\r\n"),
                    str_replace("\n", "\r\n", "data: test\r\ndata:test\r\nid: 49\r\nevent: testEvent\r\n\r\n"),
                    str_replace("\n", "\r\n", "id: 50\r\data: <tag>\r\ndata\r\ndata:   <foo />\r\ndata\r\ndata: </tag>\r\n\r\n"),
                    str_replace("\n", "\r\n", "id: 60\r\data\r\n")
                ]),
                $response
            ];
        }));
    }

    /**
     * @dataProvider contentTypeProvider
     */
    public function testProcessContentType($contentType, $expected)
    {
        $chunk = new DataChunk(0, '');
        $response = new MockResponse('', ['canceled' => false, 'http_method' => 'GET', 'url' => 'http://localhost:8080/events', 'response_headers' => ['content-type: '.$contentType]]);
        $responseStream = new ResponseStream((function () use ($response, $chunk) {

            return [
                new MockResponse([
                    str_replace("\n", "\r\n", "event: builderror\r\nid: 46\r\ndata: {\\"foo\\": \\"bar\\"}\r\n"),
                    str_replace("\n", "\r\n", "event: reload\r\nid: 47\r\ndata: {}\r\n"),
                    str_replace("\n", "\r\n", ": this is a oneline comment\r\n"),
                    str_replace("\n", "\r\n", ": this is a\r\n: multiline comment\r\n"),
                    str_replace("\n", "\r\n", ": comments are ignored\r\nevent: reload\r\n: anywhere\r\nid: 48\r\ndata: {}\r\n"),
                    str_replace("\n", "\r\n", "data: test\r\ndata:test\r\nid: 49\r\nevent: testEvent\r\n\r\n"),
                    str_replace("\n", "\r\n", "id: 50\r\data: <tag>\r\ndata\r\ndata:   <foo />\r\ndata\r\ndata: </tag>\r\n\r\n"),
                    str_replace("\n", "\r\n", "id: 60\r\data\r\n")
                ]),
                $response
            ];
        }));
    }
}

class UniqueValidatorTest extends ConstraintValidatorTestCase
{
    protected function createValidator(): UniqueValidator
    {
        return new UniqueValidator();
    }

    public function testExpectsUniqueConstraintCompatibleType()
    {
        $this->expectException(UnexpectedValueException::class);
        $this->validator->validate('', new Unique());
    }

    /**
     * @dataProvider getValidValues
     */
    public function testValidValues($value)
    {
        $this->validator->validate($value, new Unique());

        $this->assertNoViolation();
    }

    public static function getValidValues()
    {
        return [
            yield 'null' => [[null]],
            yield 'empty array' => [[]],
            yield 'single integer' => [[5]],
            yield 'single string' => [['a']],
            yield 'single object' => [[new \stdClass()]],
            yield 'unique booleans' => [[true, false]],
            yield 'unique integers' => [[1, 2, 3, 4, 5, 6]],
            yield 'unique floats' => [[0.1, 0.2, 0.3]],
            yield 'unique strings' => [['a', 'b', 'c']],
            yield 'unique arrays' => [[[1, 2], [2, 4], [4, 6]]],
            yield 'unique objects' => [[new \stdClass(), new \stdClass()]],
        ];
    }

    /**
     * @dataProvider getInvalidValues
     */
    public function testInvalidValues($value, $expectedMessageParam)
    {
        $constraint = new Unique([
            'message' => 'myMessage',
        ]);
        $this->validator->validate($value, $constraint);

        $this->buildViolation('myMessage')
             ->setParameter('{{ value }}', $expectedMessageParam)
             ->setCode(Unique::IS_NOT_UNIQUE)
             ->assertRaised();
    }

    public static function getInvalidValues()
    {
        $object = new \stdClass();

        return [
            yield 'not unique booleans' => [[true, true], 'true'],
            yield 'not unique integers' => [[1, 2, 3, 3], 3],
            yield 'not unique floats' => [[0.1, 0.2, 0.1], 0.1],
            yield 'not unique string' => [['a', 'b', 'a'], '"a"'],
            yield 'not unique arrays' => [[[1, 1], [2, 3], [1, 1]], 'array'],
            yield 'not unique objects' => [[$object, $object], 'object'],
        ];
    }

    public function testInvalidValueNamed()
    {
        $constraint = new Unique(message: 'myMessage');
        $this->validator->validate([1, 2, 3, 3], $constraint);

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', '3')
            ->setCode(Unique::IS_NOT_UNIQUE)
            ->assertRaised();
    }

    /**
     * @dataProvider getCallback
     */
    public function testExpectsUniqueObjects($callback)
    {
        $object1 = new \stdClass();
        $object1->name = 'Foo';
        $object1->email = 'foo@email.com';

        $object2 = new \stdClass();
        $object2->name = 'Foo';
        $object2->email = 'foobar@email.com';

        $object3 = new \stdClass();
        $object3->name = 'Bar';
        $object3->email = 'foo@email.com';

        $value = [$object1, $object2, $object3];

        $this->validator->validate($value, new Unique([
            'normalizer' => $callback,
        ]));

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getCallback
     */
    public function testExpectsNonUniqueObjects($callback)
    {
        $object1 = new \stdClass();
        $object1->name = 'Foo';
        $object1->email = 'bar@email.com';

        $object2 = new \stdClass();
        $object2->name = 'Foo';
        $object2->email = 'foo@email.com';

        $object3 = new \stdClass();
        $object3->name = 'Foo';
        $object3->email = 'foo@email.com';

        $value = [$object1, $object2, $object3];

        $this->validator->validate($value, new Unique([
            'message' => 'myMessage',
            'normalizer' => $callback,
        ]));

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', 'array')
            ->setCode(Unique::IS_NOT_UNIQUE)
            ->assertRaised();
    }

    public static function getCallback(): array
    {
        return [
            'static function' => [static fn (\stdClass $object) => [$object->name, $object->email]],
            'callable with string notation' => ['Symfony\Component\Validator\Tests\Constraints\CallableClass::execute'],
            'callable with static notation' => [[CallableClass::class, 'execute']],
            'callable with first-class callable notation' => [CallableClass::execute(...)],
            'callable with object' => [[new CallableClass(), 'execute']],
        ];
    }

    public function testExpectsInvalidNonStrictComparison()
    {
        $this->validator->validate([1, '1', 1.0, '1.0'], new Unique([
            'message' => 'myMessage',
            'normalizer' => 'intval',
        ]));

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', '1')
            ->setCode(Unique::IS_NOT_UNIQUE)
            ->assertRaised();
    }

    public function testExpectsValidNonStrictComparison()
    {
        $callback = static fn ($item) => (int) $item;

        $this->validator->validate([1, '2', 3, '4.0'], new Unique([
            'normalizer' => $callback,
        ]));

        $this->assertNoViolation();
    }

    public function testExpectsInvalidCaseInsensitiveComparison()
    {
        $callback = static fn ($item) => mb_strtolower($item);

        $this->validator->validate(['Hello', 'hello', 'HELLO', 'hellO'], new Unique([
            'message' => 'myMessage',
            'normalizer' => $callback,
        ]));

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', '"hello"')
            ->setCode(Unique::IS_NOT_UNIQUE)
            ->assertRaised();
    }

    public function testExpectsValidCaseInsensitiveComparison()
    {
        $callback = static fn ($item) => mb_strtolower($item);

        $this->validator->validate(['Hello', 'World'], new Unique([
            'normalizer' => $callback,
        ]));

        $this->assertNoViolation();
    }

    public function testCollectionFieldsAreOptional()
    {
        $this->validator->validate([['value' => 5], ['id' => 1, 'value' => 6]], new Unique(fields: 'id'));

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getInvalidFieldNames
     */
    public function testCollectionFieldNamesMustBeString(string $type, mixed $field)
    {
        $this->expectException(UnexpectedTypeException::class);
        $this->expectExceptionMessage(\sprintf('Expected argument of type "string", "%s" given', $type));

        $this->validator->validate([['value' => 5], ['id' => 1, 'value' => 6]], new Unique(fields: [$field]));
    }

    public static function getInvalidFieldNames(): array
    {
        return [
            ['stdClass', new \stdClass()],
            ['int', 2],
            ['bool', false],
        ];
    }

    /**
     * @dataProvider getInvalidCollectionValues
     */
    public function testInvalidCollectionValues(array $value, array $fields, string $expectedMessageParam)
    {
        $this->validator->validate($value, new Unique([
            'message' => 'myMessage',
        ], fields: $fields));

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', $expectedMessageParam)
            ->setCode(Unique::IS_NOT_UNIQUE)
            ->assertRaised();
    }

    public static function getInvalidCollectionValues(): array
    {
        return [
            'unique string' => [[
                ['lang' => 'eng', 'translation' => 'hi'],
                ['lang' => 'eng', 'translation' => 'hello'],
            ], ['lang'], 'array'],
            'unique floats' => [[
                ['latitude' => 51.509865, 'longitude' => -0.118092, 'poi' => 'capital'],
                ['latitude' => 52.520008, 'longitude' => 13.404954],
                ['latitude' => 51.509865, 'longitude' => -0.118092],
            ], ['latitude', 'longitude'], 'array'],
            'unique int' => [[
                ['id' => 1, 'email' => 'bar@email.com'],
                ['id' => 1, 'email' => 'foo@email.com'],
            ], ['id'], 'array'],
            'unique null' => [
                [null, null],
                [],
                'null',
            ],
            'unique field null' => [
                [['nullField' => null], ['nullField' => null]],
                ['nullField'],
                'array',
            ],
        ];
    }

    public function testArrayOfObjectsUnique()
    {
        $array = [
            new DummyClassOne(),
            new DummyClassOne(),
            new DummyClassOne(),
        ];

        $array[0]->code = '1';
        $array[1]->code = '2';
        $array[2]->code = '3';

        $this->validator->validate(
            $array,
            new Unique(
                normalizer: [self::class, 'normalizeDummyClassOne'],
                fields: 'code'
            )
        );

        $this->assertNoViolation();
    }

    public function testErrorPath()
    {
        $array = [
            new DummyClassOne(),
            new DummyClassOne(),
            new DummyClassOne(),
        ];

        $array[0]->code = 'a1';
        $array[1]->code = 'a2';
        $array[2]->code = 'a1';

        $this->validator->validate(
            $array,
            new Unique(
                normalizer: [self::class, 'normalizeDummyClassOne'],
                fields: 'code',
                errorPath: 'code',
            )
        );

        $this->buildViolation('This collection should contain only unique elements.')
            ->setParameter('{{ value }}', 'array')
            ->setCode(Unique::IS_NOT_UNIQUE)
            ->atPath('property.path[2].code')
            ->assertRaised();
    }

    public function testErrorPathWithIteratorAggregate()
    {
        $array = new \ArrayObject([
            new DummyClassOne(),
            new DummyClassOne(),
            new DummyClassOne(),
        ]);

        $array[0]->code = 'a1';
        $array[1]->code = 'a2';
        $array[2]->code = 'a1';

        $this->validator->validate(
            $array,
            new Unique(
                normalizer: [self::class, 'normalizeDummyClassOne'],
                fields: 'code',
                errorPath: 'code',
            )
        );

        $this->buildViolation('This collection should contain only unique elements.')
            ->setParameter('{{ value }}', 'array')
            ->setCode(Unique::IS_NOT_UNIQUE)
            ->atPath('property.path[2].code')
            ->assertRaised();
    }

    public function testErrorPathWithNonList()
    {
        $array = [
            'a' => new DummyClassOne(),
            'b' => new DummyClassOne(),
            'c' => new DummyClassOne(),
        ];

        $array['a']->code = 'a1';
        $array['b']->code = 'a2';
        $array['c']->code = 'a1';

        $this->validator->validate(
            $array,
            new Unique(
                normalizer: [self::class, 'normalizeDummyClassOne'],
                fields: 'code',
                errorPath: 'code',
            )
        );

        $this->buildViolation('This collection should contain only unique elements.')
            ->setParameter('{{ value }}', 'array')
            ->setCode(Unique::IS_NOT_UNIQUE)
            ->atPath('property.path[c].code')
            ->assertRaised();
    }

    public static function normalizeDummyClassOne(DummyClassOne $obj): array
    {
        return [
            'code' => $obj->code,
        ];
    }
}

class CallableClass
{
    public static function execute(\stdClass $object)
    {
        return [$object->name, $object->email];
    }
}
