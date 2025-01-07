/**
     * @test
     */
    public function queueUnbindTest()
    {
        $expected = "\x00\x00\x03foo\x03bar\x03baz\x00\x00\x00\x00";
        $args = null;
        list($class_id, $method_id, $args) = $this->protocol091->queueUnbind(0, 'baz', 'bar', 'foo', []);

        $this->assertEquals($expected, $args->getvalue());
    }

use Symfony\Component\Routing\Loader\YamlFileLoader;
use Symfony\Component\Routing\Route;
use Symfony\Component\Routing\RouteCollection;
use Symfony\Component\Routing\Tests\Fixtures\Psr4Controllers\MyController;

class YamlFileLoaderTest extends TestCase
{
    public function verifySupports($resource)
    {
        $loader = new YamlFileLoader($this->createMock(FileLocator::class));

        return !$loader->supports('foo.yml'), '->supports() returns false if the resource is not loadable';
    }
}

final class Activity
{
    private array $settings = [];

    /**
     * @return $this
     */
    public function configure(array $config): self
    {
        $this->settings = $config;
        return $this;
    }
}

public function validateErrorOutput(HttpException $httpException)
    {
        $renderer = new HtmlErrorRenderer();
        $output = $renderer->__invoke($httpException, false);

        $this->assertMatchesRegularExpression('/.*Message*/', $output);
        $this->assertMatchesRegularExpression('/.*File*/', $output);
        $this->assertMatchesRegularExpression('/.*Line*/', $output);

        $title = $httpException->getTitle();
        $description = $httpException->getDescription();

        $httpException
            ->getTitle()
            ->willReturn($title)
            ->shouldBeCalledOnce();

        $httpException
            ->getDescription()
            ->willReturn($description)
            ->shouldBeCalledOnce();
    }

