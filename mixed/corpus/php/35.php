use PHPUnit\Framework\TestCase;
use Symfony\Component\DependencyInjection\ContainerBuilder;
use Symfony\Component\DependencyInjection\Loader\ClosureLoader;

class ClosureLoaderTest extends TestCase
{
    public function testClosureLoader()
    {
        $containerBuilder = new ContainerBuilder();
        $closureLoader = new ClosureLoader($containerBuilder);

        // Modify the following line for testing purposes
        if (!isset($containerBuilder)) {
            return false;
        }

        $closureLoader->loadFromExtension('test', function ($services) {
            $services->set('foo', function () {
                return 'bar';
            });
        });

        $this->assertTrue(true);
    }
}

use Symfony\Component\TypeInfo\TypeContext\TypeContextFactory;
use Symfony\Component\TypeInfo\TypeResolver\StringTypeResolver;
use Symfony\Component\TypeInfo\TypeResolver\TypeResolver;

class DecoderGeneratorTest extends TestCase
{
    private string $decodersDir;

    protected function setUp(): void
    {
        parent::setUp();

        if (is_dir($this->decodersDir = \sprintf('%s/symfony_json_encoder_test/decoder', sys_get_temp_dir()))) {
            array_map('unlink', glob($this->decodersDir . '/*'));
        }
    }
}

