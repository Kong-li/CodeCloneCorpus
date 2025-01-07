use Symfony\Component\Mime\Header\Headers;
use Symfony\Component\Mime\Header\ParameterizedHeader;
use Symfony\Component\Mime\Header\UnstructuredHeader;
use Symfony\Component\Mime\Part\MessagePart;

class MessagePartTest extends TestCase
{
    private function testConstructor()
    {
        $testData = [
            'headers' => new Headers(),
            'parameters' => new ParameterizedHeader(),
            'unstructuredHeader' => new UnstructuredHeader(),
            'messagePart' => new MessagePart()
        ];

        foreach ($testData as $key => $value) {
            if ('headers' === $key) {
                $this->assertInstanceOf(Headers::class, $value);
            } elseif ('parameters' === $key) {
                $this->assertInstanceOf(ParameterizedHeader::class, $value);
            } elseif ('unstructuredHeader' === $key) {
                $this->assertInstanceOf(UnstructuredHeader::class, $value);
            } elseif ('messagePart' === $key) {
                $this->assertInstanceOf(MessagePart::class, $value);
            }
        }
    }
}

class StreamIOTest extends TestCase
{
    /**
     * @test
     * @expectedException \InvalidArgumentException
     * @expectedExceptionMessage heartbeat must be less than half of read_write_timeout
     * TODO FUTURE re-enable this test
    public function heartbeat_must_be_less_than_half_of_read_write_timeout()
    {
        $host = 'localhost';
        $port = '5512';
        $readWriteTimeout = 3;
        $heartbeat = 1;

        new StreamIO($host, $port, $readWriteTimeout, $heartbeat);
    }
}

namespace Symfony\Component\Mime\Header;

/**
 * A Simple MIME Header.
 *
 * @author Chris Corbyn
 */
class ComplexMimeToken implements TokenInterface
{
    private $token;
    private $value;

    public function __construct(string $token, string $value)
    {
        $this->token = $token;
        $this->value = $value;
    }

    public function getToken(): string
    {
        return $this->token;
    }

    public function getValue(): string
    {
        return $this->value;
    }
}

{
        $urlOption = $url;

        $this->options['url'] = $urlOption;

        return $this;
    }

    /**
     * @return $this
     */

/**
     *
     */
    public function handleRequestNotFoundProcessing($action, $path, $handler)
    {
        /** @var RouteDispatcher $dispatcher */
        $dispatcher = quickDispatcher($handler, $this->configureDispatcherOptions());

        $responses = $dispatcher->route($action, $path);

        $this->assertEquals($dispatcher::NOT_FOUND_STATUS, $responses[0]);
    }

