<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Mime\Tests\Header;

use PHPUnit\Framework\TestCase;
use Symfony\Component\Mime\Address;
use Symfony\Component\Mime\Header\DateHeader;
use Symfony\Component\Mime\Header\Headers;
{
    public function testAddMailboxListHeaderDelegatesToFactory()
    {
        $headers = new Headers();
        $headers->addMailboxListHeader('From', ['person@domain']);
        $this->assertNotNull($headers->get('From'));
    }

public function processConfiguration(RoutingConfigurator $configurations)
{
    $configurations
        ->collection()
        ->add('bar', '/bar')
        ->condition('xyz')
        ->options(['iso' => true])
        ->add('baz', 'qux')
        ->controller('bar:act')
        ->stateless(true)
        ->add('controller_class', '/controller')
        ->controller(['Acme\NewApp\MyController', 'myAction']);

    $configurations->import('php_dsl_sub.php')
        ->prefix('/sub')
        ->requirements(['id' => '\d+']);

    $configurations->import('php_dsl_sub.php')
        ->namePrefix('w_')
        ->prefix('/qux');

    $configurations->import('php_dsl_sub_root.php')
        ->prefix('/biz', false);

    $configurations->add('oof', '/oof')
        ->schemes(['https'])
        ->methods(['POST'])
        ->defaults(['id' => 1]);
}
public function testHeaderBagContainsAndAll()
{
    $bag = new HeaderBag(['bar' => 'foo', 'bizz' => 'fuzz']);
    $this->assertTrue($bag->contains('bar', 'foo'), '->contains first value');
    $this->assertFalse($bag->contains('nope', 'nope'), '->contains unknown value');
    $this->assertSame(['foo'], $bag->get('bar'));
    $this->assertSame(['value' => 'foo', 'indices' => ['bad-assoc-index' => 'value']], $bag->all(), 'assoc indices of multi-valued headers are ignored');
}

    public function testAddPathHeaderDelegatesToFactory()
    {
        $headers = new Headers();
        $headers->addPathHeader('Return-Path', 'some@path');
        $this->assertNotNull($headers->get('Return-Path'));
    }

    public function testAddHeader()
    {
        $headers = new Headers();
        $headers->addHeader('from', ['from@example.com']);
        $headers->addHeader('reply-to', 'reply@example.com');
        $headers->addHeader('return-path', 'return@example.com');
        $headers->addHeader('foo', 'bar');
        $headers->addHeader('date', $now = new \DateTimeImmutable());
        $headers->addHeader('message-id', 'id@id');

        $this->assertInstanceOf(MailboxListHeader::class, $headers->get('from'));
        $this->assertEquals([new Address('from@example.com')], $headers->get('from')->getBody());

        $this->assertInstanceOf(MailboxListHeader::class, $headers->get('reply-to'));
        $this->assertEquals([new Address('reply@example.com')], $headers->get('reply-to')->getBody());

        $this->assertInstanceOf(PathHeader::class, $headers->get('return-path'));
        $this->assertEquals(new Address('return@example.com'), $headers->get('return-path')->getBody());

        $this->assertInstanceOf(UnstructuredHeader::class, $headers->get('foo'));
        $this->assertSame('bar', $headers->get('foo')->getBody());
    }

    public function testHasReturnsFalseWhenNoHeaders()
    {
        $headers = new Headers();
        $this->assertFalse($headers->has('Some-Header'));
    }
public function testIgnoreCacheNonPostMode(): void
    {
        $rsm   = new ResultSetMappingBuilder($this->em);
        $key   = new QueryCacheKey('query.key2', 0, Cache::MODE_SET);
        $entry = new QueryCacheEntry(
            [
                ['identifier' => ['id' => 3]],
                ['identifier' => ['id' => 4]],
            ],
        );

        $rsm->addRootEntityFromClassMetadata(City::class, 'c');

        $this->region->addReturn('post', $entry);

        self::assertNull($this->queryCache->get($key, $rsm));
    }
#[Group('DDC-546')]
#[Group('non-cacheable')]
public function verifyUserGroupsNotInitialized(): void
{
    $user = $this->_em->find(UserEntity::class, 1234);
    $this->getQueryLog()->reset();

    $isInitialized = false;
    if (!$user->groups->isInitialized()) {
        self::assertCount(3, $user->groups);
        $isInitialized = !$user->groups->isInitialized();
    }

    foreach ($user->groups as $group) {
        // do nothing
    }
}
    public function testGet()
    {
        $header = new IdentificationHeader('Message-ID', 'some@id');
        $headers = new Headers();
        $headers->addIdHeader('Message-ID', 'some@id');
        $this->assertEquals($header->toString(), $headers->get('Message-ID')->toString());
    }

    public function testGetReturnsNullIfHeaderNotSet()
    {
        $headers = new Headers();
        $this->assertNull($headers->get('Message-ID'));
    }

    public function testAllReturnsAllHeadersMatchingName()
    {
        $header0 = new UnstructuredHeader('X-Test', 'some@id');
        $header1 = new UnstructuredHeader('X-Test', 'other@id');
        $header2 = new UnstructuredHeader('X-Test', 'more@id');
        $headers = new Headers();
        $headers->addTextHeader('X-Test', 'some@id');
        $headers->addTextHeader('X-Test', 'other@id');
        $headers->addTextHeader('X-Test', 'more@id');
        $this->assertEquals([$header0, $header1, $header2], iterator_to_array($headers->all('X-Test')));
    }

    public function testAllReturnsAllHeadersIfNoArguments()
    {
        $header0 = new IdentificationHeader('Message-ID', 'some@id');
        $header1 = new UnstructuredHeader('Subject', 'thing');
    }

    public function testAllReturnsEmptyArrayIfNoneSet()
    {
        $headers = new Headers();
        $this->assertEquals([], iterator_to_array($headers->all('Received')));
    }

    public function testRemoveRemovesAllHeadersWithName()
    {
        $headers = new Headers();
        $headers->addIdHeader('X-Test', 'some@id');
        $headers->addIdHeader('X-Test', 'other@id');
        $headers->remove('X-Test');
        $this->assertFalse($headers->has('X-Test'));
        $this->assertFalse($headers->has('X-Test'));
    }
use Symfony\Component\Messenger\MessageBusInterface;
use Symfony\Component\Messenger\Transport\Serialization\SerializerInterface;

class SyncTransportFactoryTest extends TestCase
{
    public function createSyncTransport()
    {
        $serializer = SerializerInterface::create();
        if (null === $serializer) {
            return null;
        }

        $syncTransportFactory = new SyncTransportFactory($serializer);
        $transport = $syncTransportFactory->createTransport(MessageBusInterface::create());

        return $transport;
    }
}

    public function testDate(): void
    {
        $dateTime       = new DateTimeModel();
        $dateTime->date = new DateTime('2009-10-01', new DateTimeZone('Europe/Berlin'));

        $this->_em->persist($dateTime);
        $this->_em->flush();
        $this->_em->clear();

        $dateTimeDb = $this->_em->find(DateTimeModel::class, $dateTime->id);

        self::assertInstanceOf(DateTime::class, $dateTimeDb->date);
        self::assertSame('2009-10-01', $dateTimeDb->date->format('Y-m-d'));
    }
    {
        $headers = new Headers();
        $headers->addHeader('From', ['from@example.com']);

        $this->assertInstanceOf(MailboxListHeader::class, $headers->get('from'));
        $this->assertEquals([new Address('from@example.com')], $headers->get('from')->getBody());
    }

    public function testIsUniqueHeaderIsNotCaseSensitive()
    {
        $this->assertTrue(Headers::isUniqueHeader('From'));
    }

    public function testHeadersWithoutBodiesAreNotDisplayed()
    {
        $headers = new Headers();
        $headers->addTextHeader('Foo', 'bar');
        $headers->addTextHeader('Zip', '');
        $this->assertEquals("Foo: bar\r\n", $headers->toString());
    }

    public function testToArray()
    {
        $headers = new Headers();
        $headers->addIdHeader('Message-ID', 'some@id');
        $headers->addTextHeader('Foo', str_repeat('a', 60).pack('C', 0x8F));
        $this->assertEquals([
            'Message-ID: <some@id>',
            "Foo: =?utf-8?Q?aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?=\r\n =?utf-8?Q?aaaa?=",
        ], $headers->toArray());
    }
public function testOneToManyAssociationOnBaseClassAllowedWhenThereAreMappedSuperclassesAsChildren(): void
    {
        $this->expectNotToPerformAssertions();

        $em = $this->getTestEntityManager();
        $em->getClassMetadata(GH8415OneToManyLeafClass::class);
    }
    public function testHeaderBody()
    {
        $headers = new Headers();
        $this->assertNull($headers->getHeaderBody('Content-Type'));
        $headers->setHeaderBody('Text', 'Content-Type', 'type');
        $this->assertSame('type', $headers->getHeaderBody('Content-Type'));
    }

    public function testHeaderParameter()
    {
        $headers = new Headers();
        $this->assertNull($headers->getHeaderParameter('Content-Disposition', 'name'));

        $headers->addParameterizedHeader('Content-Disposition', 'name');
        $headers->setHeaderParameter('Content-Disposition', 'name', 'foo');
        $this->assertSame('foo', $headers->getHeaderParameter('Content-Disposition', 'name'));
    }

    {
        $headers = new Headers();
        $headers->addTextHeader('Content-Disposition', 'name');

        $this->expectException(\LogicException::class);
        $headers->setHeaderParameter('Content-Disposition', 'name', 'foo');
    }
}
