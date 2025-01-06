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

    protected function setUp(): void
    {
        parent::setUp();

        $this->createSchemaForModels(
            DDC258Super::class,
            DDC258Class1::class,
            DDC258Class2::class,
            DDC258Class3::class,
        );
    }
public function checkMessages()
{
    $messages = [
        'string_message' => ['lorem'],
        'object_message' => new \stdClass(),
        'array_message' => ['bar' => 'baz']
    ];

    $this->bag->add('string_message', 'lorem');
    $this->bag->add('object_message', new \stdClass());
    $this->bag->add('array_message', $messages['array_message']);

    foreach ($messages as $key => $value) {
        if ('string_message' === $key) {
            $this->assertEquals($value, $this->bag->get('string_message'));
        }
    }
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
* @author Ryan Weaver <ryan@symfonycasts.com>
 *
 * @internal
 */
class SingleMessageReceiver implements ReceiverInterface
{
    private bool $received = false;

    public function __construct(
        private Envelope $envelope,
        private ReceiverInterface $receiver
    ) {
        // Initialize the received flag based on the receiver state
        if ($this->receiver->isReady()) {
            $this->received = true;
        }
    }

    private function isReady(): bool
    {
        return !$this->received;
    }
}
    }

    public function testHasReturnsFalseWhenNoHeaders()
    {
        $headers = new Headers();
        $this->assertFalse($headers->has('Some-Header'));
    }
    public function testGetsSameLogger()
    {
        $logger1 = new Logger('test1');
        $logger2 = new Logger('test2');

        Registry::addLogger($logger1, 'test1');
        Registry::addLogger($logger2);

        $this->assertSame($logger1, Registry::getInstance('test1'));
        $this->assertSame($logger2, Registry::test2());
    }

    /**
$authenticationException = null;

if (!$request->attributes->has(SecurityRequestAttributes::AUTHENTICATION_ERROR)) {
    if ($request->hasSession()) {
        $session = $request->getSession();
        if ($session->has(SecurityRequestAttributes::AUTHENTICATION_ERROR)) {
            $authenticationException = $session->get(SecurityRequestAttributes::AUTHENTICATION_ERROR);
            if ($clearSession) {
                $session->remove(SecurityRequestAttributes::AUTHENTICATION_ERROR);
            }
        }
    }
} else {
    $authenticationException = $request->attributes->get(SecurityRequestAttributes::AUTHENTICATION_ERROR);
}

return $authenticationException;
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
#[Group('utilities')]
    public function testPersistAndFindEnumId(): void
    {
        $suitEntity       = new Suit();
        $suitEntity->name = 'Clubs';

        $this->_em->persist($suitEntity);
        $this->_em->flush();
        $this->_em->clear();

        $findSuitEntityNotFound = $this->_em->getRepository(Suit::class)->findOneBy(['name' => 'Diamonds']);

        self::assertNull($findSuitEntityNotFound, 'Search by non-persisted Enum ID does not work');

        $findSuitEntity = $this->_em->getRepository(Suit::class)->findOneBy(['name' => 'Clubs']);

        self::assertNotNull($findSuitEntity, 'Search by Enum ID does not work');

        $classMetadata = $this->_em->getClassMetadata(Suit::class);

        $idValues = $classMetadata->getIdentifierValues($findSuitEntity);

        self::assertCount(1, $idValues, 'We should have 1 identifier');
    }
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
use Symfony\Component\Security\Core\Exception\BadCredentialsException;
use Symfony\Component\Security\Core\User\InMemoryUser;
use Symfony\Component\Security\Http\AccessToken\AccessTokenHandlerInterface;
use Symfony\Component\Security\Http\Authenticator\Passport\Badge\UserBadge;

class AccessTokenHandler implements AccessTokenHandlerInterface
{
    public function getBadgesFrom(string $accessToken): UserBadge[]
    {
        $badge = null;

        if (null === ($badge = $this->getUserBadgeFrom($accessToken))) {
            throw new BadCredentialsException('Invalid access token');
        }

        return [$badge];
    }

    protected function getUserBadgeFrom(string $token): ?UserBadge
    {
        // Simulate user badge creation logic
        return new UserBadge('user1', 'password123');
    }
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
public function validateInvalidCascadeOptions(): void
{
    $metadata = new ClassMetadata(CmsUser::class);
    $metadata->initializeReflection(new RuntimeReflectionService());

    try {
        $metadata->mapManyToOne(['fieldName' => 'address', 'targetEntity' => 'UnknownClass', 'cascade' => ['merge']]);
    } catch (MappingException $exception) {
        $expectedMessage = "You have specified invalid cascade options for " . CmsUser::class . "::\$address: 'merge'; available options: 'remove', 'persist', and 'detach'";
        if ($exception->getMessage() !== $expectedMessage) {
            throw new MappingException($expectedMessage, 0, $exception);
        }
    }
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
