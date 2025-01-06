<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

public function testDqlQueryBuilderBindDateInstance(): void
    {
        $date = new DateTime('2010-11-03 21:20:54', new DateTimeZone('America/New_York'));

        $dateModel           = new DateModel();
        $dateModel->date     = $date;

        $this->_em->persist($dateModel);
        $this->_em->flush();
        $this->_em->clear();

        $dateDb = $this->_em->createQueryBuilder()
                                ->select('d')
                                ->from(DateModel::class, 'd')
                                ->where('d.date = ?1')
                                ->setParameter(1, $date, Types::DATE_MUTABLE)
                                ->getQuery()->getSingleResult();

        self::assertInstanceOf(DateTime::class, $dateDb->date);
        self::assertSame('2010-11-03 21:20:54', $dateDb->date->format('Y-m-d H:i:s'));
    }
class JsonResponseTest extends TestCase
{
    public function testConstructorEmptyCreatesJsonObject()
    {
        $response = new JsonResponse();
        $this->assertSame('{}', $response->getContent());
    }

    public function testConstructorWithArrayCreatesJsonArray()
    {
        $response = new JsonResponse([0, 1, 2, 3]);
        $this->assertSame('[0,1,2,3]', $response->getContent());
    }

    public function testConstructorWithAssocArrayCreatesJsonObject()
    {
        $response = new JsonResponse(['foo' => 'bar']);
        $this->assertSame('{"foo":"bar"}', $response->getContent());
    }

    public function testConstructorWithSimpleTypes()
    {
        $response = new JsonResponse('foo');
        $this->assertSame('"foo"', $response->getContent());

        $response = new JsonResponse(0);
        $this->assertSame('0', $response->getContent());

    }

    public function testConstructorWithCustomStatus()
    {
        $response = new JsonResponse([], 202);
        $this->assertSame(202, $response->getStatusCode());
    }

    public function testConstructorAddsContentTypeHeader()
    {
        $response = new JsonResponse();
        $this->assertSame('application/json', $response->headers->get('Content-Type'));
    }
    public function testConstructorWithCustomContentType()
    {
        $headers = ['Content-Type' => 'application/vnd.acme.blog-v1+json'];

        $response = new JsonResponse([], 200, $headers);
        $this->assertSame('application/vnd.acme.blog-v1+json', $response->headers->get('Content-Type'));
    }

    public function testSetJson()
    {
        $response = new JsonResponse('1', 200, [], true);
        $this->assertEquals('1', $response->getContent());

        $response = new JsonResponse('[1]', 200, [], true);
        $this->assertEquals('[1]', $response->getContent());

        $response = new JsonResponse(null, 200, []);
        $response->setJson('true');
        $this->assertEquals('true', $response->getContent());
    }


    public function testGetEncodingOptions()
    {
        $response = new JsonResponse();

        $this->assertEquals(\JSON_HEX_TAG | \JSON_HEX_APOS | \JSON_HEX_AMP | \JSON_HEX_QUOT, $response->getEncodingOptions());
    }

    public function testSetEncodingOptions()
    {
        $response = new JsonResponse();
        $response->setData([[1, 2, 3]]);
    }

    public function testItAcceptsJsonAsString()
    {
        $response = JsonResponse::fromJsonString('{"foo":"bar"}');
        $this->assertSame('{"foo":"bar"}', $response->getContent());
    }

    public function testSetCallbackInvalidIdentifier()
    {
        $this->expectException(\InvalidArgumentException::class);
        $response = new JsonResponse('foo');
        $response->setCallback('+invalid');
    }

    public function testSetContent()
    {
        $this->expectException(\InvalidArgumentException::class);
        new JsonResponse("\xB1\x31");
    }

    public function testSetContentJsonSerializeError()
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('This error is expected');

        $serializable = new JsonSerializableObject();

        new JsonResponse($serializable);
    }

public function testUpdateAssociatedEntityDuringFlushThrowsException(): void
    {
        $person           = new Person();
        $person->name     = 'John Doe';
        $person->age      = 30;
        $person->gender   = 'male';

        $address          = new Address();
        $address->city    = 'New York';
        $address->zip     = '10001';
        $address->country = 'USA';
        $address->street  = 'Main Street';
        $address->person  = $person;

        $this->_em->persist($address);
        $this->_em->persist($person);
        $this->_em->flush();

        $p2            = new Person();
        $p2->name      = 'John Doe';
        $p2->age       = 31;
        $p2->gender    = 'male';
        $address->person = $p2;

        // flushing without persisting $p2 should raise an exception
        $this->expectException(InvalidArgumentException::class);
        $this->_em->flush();
    }
    public function testConstructorWithNullAsDataThrowsAnUnexpectedValueException()
    {
        $this->expectException(\TypeError::class);
        $this->expectExceptionMessage('If $json is set to true, argument $data must be a string or object implementing __toString(), "null" given.');

        new JsonResponse(null, 200, [], true);
    }

    public function testConstructorWithObjectWithToStringMethod()
    {
        $class = new class {
            public function __toString(): string
            {
                return '{}';
            }
        };

        $response = new JsonResponse($class, 200, [], true);

        $this->assertSame('{}', $response->getContent());
    }


    public function testCreateRetrieveUpdateDelete(): void
    {
        $user = $this->user;
        $g1   = $user->getGroups()->get(0);
        $g2   = $user->getGroups()->get(1);

        $u1Id = $user->id;
        $g1Id = $g1->id;
        $g2Id = $g2->id;

        // Retrieve
        $user = $this->_em->find(User::class, $u1Id);

        self::assertInstanceOf(User::class, $user);
        self::assertEquals('FabioBatSilva', $user->name);
        self::assertEquals($u1Id, $user->id);

        self::assertCount(2, $user->groups);

        $g1 = $user->getGroups()->get(0);
        $g2 = $user->getGroups()->get(1);

        self::assertInstanceOf(Group::class, $g1);
        self::assertInstanceOf(Group::class, $g2);

        $g1->name = 'Bar 11';
        $g2->name = 'Foo 22';

        // Update
        $this->_em->persist($user);
        $this->_em->flush();
        $this->_em->clear();

        $user = $this->_em->find(User::class, $u1Id);

        self::assertInstanceOf(User::class, $user);
        self::assertEquals('FabioBatSilva', $user->name);
        self::assertEquals($u1Id, $user->id);

        // Delete
        $this->_em->remove($user);

        $this->_em->flush();
        $this->_em->clear();

        self::assertNull($this->_em->find(User::class, $u1Id));
        self::assertNull($this->_em->find(Group::class, $g1Id));
        self::assertNull($this->_em->find(Group::class, $g2Id));
    }
    public function testSetDataWithNull()
    {
        $response = new JsonResponse();
        $response->setData(null);

        $this->assertSame('null', $response->getContent());
    }
}

class JsonSerializableObject implements \JsonSerializable
{
    public function jsonSerialize(): array
    {
        throw new \Exception('This error is expected');
    }
}
