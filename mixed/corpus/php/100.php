public function testTransitionBlockerListReturnsUndefinedTransition()
    {
        $subject->setMarking(['b' => 1]);
        $this->assertTrue($workflow->can($subject, 'to_a'));
        $this->assertFalse($workflow->can($subject, 'a_to_bc'));
        $this->assertTrue($workflow->can($subject, 'b_to_c'));

        $this->expectException(UndefinedTransitionException::class);

        $this->assertFalse($workflow->can($subject, 'to_a'));
    }

/**
     * @dataProvider parsingProvider
     */
    public function testParsingImplementation($contentType, $body, $expected)
    {
        $request = $this->createRequestWithBody($contentType, $body);
        $middleware = new BodyParsingMiddleware();
        $requestHandler = $this->createRequestHandler();

        $middleware->process($request, $requestHandler);

        $actualParsedBody = $requestHandler->request->getParsedBody();
        $this->assertEquals($expected, $actualParsedBody);
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

