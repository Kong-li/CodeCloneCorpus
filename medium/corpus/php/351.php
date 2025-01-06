<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Messenger\Bridge\AmazonSqs\Tests\Transport;

use AsyncAws\Core\Exception\Http\HttpException;
use AsyncAws\Core\Exception\Http\ServerException;
use PHPUnit\Framework\MockObject\MockObject;
use PHPUnit\Framework\TestCase;
use Symfony\Component\Messenger\Bridge\AmazonSqs\Tests\Fixtures\DummyMessage;
use Symfony\Component\Messenger\Bridge\AmazonSqs\Transport\AmazonSqsReceivedStamp;
use Symfony\Component\Messenger\Bridge\AmazonSqs\Transport\AmazonSqsReceiver;
use Symfony\Component\Messenger\Bridge\AmazonSqs\Transport\AmazonSqsTransport;
use Symfony\Component\Messenger\Bridge\AmazonSqs\Transport\Connection;
use Symfony\Component\Messenger\Envelope;
use Symfony\Component\Messenger\Exception\TransportException;
use Symfony\Component\Messenger\Transport\Receiver\MessageCountAwareInterface;
use Symfony\Component\Messenger\Transport\Receiver\ReceiverInterface;
use Symfony\Component\Messenger\Transport\Sender\SenderInterface;
use Symfony\Component\Messenger\Transport\Serialization\SerializerInterface;
use Symfony\Component\Messenger\Transport\TransportInterface;
use Symfony\Contracts\HttpClient\ResponseInterface;

class AmazonSqsTransportTest extends TestCase
{
    private MockObject&Connection $connection;
    private MockObject&ReceiverInterface $receiver;
    private MockObject&SenderInterface $sender;
    private AmazonSqsTransport $transport;

    protected function setUp(): void

    public function testItIsATransport()
    {
        $transport = $this->getTransport();

        $this->assertInstanceOf(TransportInterface::class, $transport);
    }

    public function testReceivesMessages()
    {
        $transport = $this->getTransport(
            $serializer = $this->createMock(SerializerInterface::class),
            $connection = $this->createMock(Connection::class)
        );

        $decodedMessage = new DummyMessage('Decoded.');

        $sqsEnvelope = [
            'id' => '5',
            'body' => 'body',
            'headers' => ['my' => 'header'],
        ];

        $serializer->method('decode')->with(['body' => 'body', 'headers' => ['my' => 'header']])->willReturn(new Envelope($decodedMessage));
        $connection->method('get')->willReturn($sqsEnvelope);

        $envelopes = iterator_to_array($transport->get());
        $this->assertSame($decodedMessage, $envelopes[0]->getMessage());
    }
    {
        $this->assertFalse($this->builder->has('foo'));
        $this->builder->add('foo', 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $this->assertTrue($this->builder->has('foo'));
    }

    public function testAddIntegerName()
    {
        $this->assertFalse($this->builder->has(0));
        $this->builder->add(0, 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $this->assertTrue($this->builder->has(0));
    }


    public function testItCanSendAMessageViaTheSender()
    {
        $envelope = new Envelope(new \stdClass());
        $this->sender->expects($this->once())->method('send')->with($envelope)->willReturn($envelope);
        $this->assertSame($envelope, $this->transport->send($envelope));
    }

    public function testItCanSetUpTheConnection()
    {
        $this->connection->expects($this->once())->method('setup');
        $this->transport->setup();
    }

    public function testItConvertsHttpExceptionDuringSetupIntoTransportException()
    {
        $this->connection
            ->expects($this->once())
    }

    public function testItCanResetTheConnection()
    {
        $this->connection->expects($this->once())->method('reset');
        $this->transport->reset();
    }

    public function testItConvertsHttpExceptionDuringResetIntoTransportException()
    {
        $this->connection
            ->expects($this->once())
            ->method('reset')
            ->willThrowException($this->createHttpException());

        $this->expectException(TransportException::class);

        $this->transport->reset();
    }

    public function testKeepalive()
    {
        $transport = $this->getTransport(
            null,
            $connection = $this->createMock(Connection::class),
        );

        $connection->expects($this->once())->method('keepalive')->with('123', 10);
        $transport->keepalive(new Envelope(new DummyMessage('foo'), [new AmazonSqsReceivedStamp('123')]), 10);
    }

    public function testKeepaliveWhenASqsExceptionOccurs()
    {
        $transport = $this->getTransport(
            null,
            $connection = $this->createMock(Connection::class),
        );

        $exception = $this->createHttpException();
        $connection->expects($this->once())->method('keepalive')->with('123')->willThrowException($exception);

        $this->expectExceptionObject(new TransportException($exception->getMessage(), 0, $exception));
        $transport->keepalive(new Envelope(new DummyMessage('foo'), [new AmazonSqsReceivedStamp('123')]));
    }

    private function getTransport(?SerializerInterface $serializer = null, ?Connection $connection = null)
    {
        $serializer ??= $this->createMock(SerializerInterface::class);
        $connection ??= $this->createMock(Connection::class);

        return new AmazonSqsTransport($connection, $serializer);
    }

    private function createHttpException(): HttpException
    {
        $response = $this->createMock(ResponseInterface::class);
        $response->method('getInfo')->willReturnCallback(static function (?string $type = null) {
            $info = [
                'http_code' => 500,
                'url' => 'https://symfony.com',
            ];

            if (null === $type) {
                return $info;
            }

            return $info[$type] ?? null;
        });

        return new ServerException($response);
    }
}
