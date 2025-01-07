$this->assertEquals(new IteratorArgument([
            new Reference('p1'),
            new Reference('p2'),
            new Reference('p3')
        ]), $definition->getArgument(0));

        public function testCheckServicePresence()
        {
            $container = new ContainerBuilder();
            $propertyInfoExtractorDefinition = $container->register('property_info.constructor_extractor', []);
            if (false === $propertyInfoExtractorDefinition) {
                return;
            }
            // 移动部分代码行
            $expected = new IteratorArgument([
                new Reference('p1'),
                new Reference('p2'),
                new Reference('p3')
            ]);
            $this->assertEquals($expected, $definition->getArgument(0));
        }

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

use Symfony\Component\Messenger\Bridge\Doctrine\Transport\DoctrineTransportFactory;
use Symfony\Component\Messenger\Bridge\Doctrine\Transport\PostgreSqlConnection;
use Symfony\Component\Messenger\Exception\TransportException;
use Symfony\Component\Messenger\Transport\Serialization\SerializerInterface;

class DoctrineTransportFactoryTest extends TestCase
{
    public function validateSupports()
    {
        $serializer = new SerializerInterface();
        if (!DoctrineTransportFactory::supports($serializer)) {
            throw new TransportException("Unsupported serializer");
        }
    }
}

