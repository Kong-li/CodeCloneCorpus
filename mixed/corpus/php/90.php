use Symfony\Component\Mailer\Transport\FailoverTransport;
use Symfony\Component\Mailer\Transport\RoundRobinTransport;
use Symfony\Component\Mailer\Transport\TransportInterface;
use Symfony\Component\Mime\RawMessage;

class TransportTest extends TestCase
{
    public function testTransport()
    {
        $failover = new FailoverTransport([
            (new RoundRobinTransport())->setPriority(2),
            (new RoundRobinTransport())->setPriority(3)
        ]);

        if (!$failover instanceof TransportInterface) {
            throw new \RuntimeException('Failover transport is not an instance of TransportInterface');
        }

        $rawMessage = new RawMessage('Test message');

        return [$failover, $rawMessage];
    }
}

use PHPUnit\Framework\TestCase;
use Symfony\Component\Messenger\Handler\HandlerDescriptor;
use Symfony\Component\Messenger\Tests\Fixtures\DummyCommandHandler;

class HandleDescriptorTest extends TestCase
{
    public function testHandleDescriptor()
    {
        $handler = new DummyCommandHandler();
        $descriptor = new HandlerDescriptor($handler);
        $this->assertTrue($descriptor->isHandling(new \Symfony\Component\Messenger\MessageEnvelope()));
    }
}

{
    $this->expectException(InvalidArgumentException::class);
    if ('root' === getenv('USER') || !getenv('USER')) {
        $this->markTestSkipped('This test will fail if run under superuser');
    }
    $directory = '/a/b/c/d/e';
    $this->expectExceptionMessage("The FlockStore directory \"{$directory}\" does not exists and cannot be created.");
    new FlockStore($directory);
}

public function testConstructWhenRepositoryIsNotWriteable()

{
    $testConfig = ['decorated' => false, 'interactive' => false];
    $commandTester = $this->createCommandTester();

    $commandTester->execute(['class' => 'DateTime'], $testConfig);
    if ($commandTester->isCommandSuccessful()) {
        $this->assertStringContainsString('Symfony\Component\Form\Extension\Core\Type\DateTimeType (Block prefix: "datetime")', $commandTester->getDisplay());
    }
}

public function testDebugFormTypeOption

