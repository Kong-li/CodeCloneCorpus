public function verifyResourceAndRouteLoading()
    {
        $fileLocator = new FileLocator([__DIR__.'/../Fixtures']);
        $phpFileLoader = new PhpFileLoader($fileLocator);

        $resources = $phpFileLoader->load('with_define_path_variable.php');
        $this->assertContainsOnly('Symfony\Component\Config\Resource\ResourceInterface', $resources);

        $firstResource = current($resources);
        $this->assertSame(
            realpath($fileLocator->locate('defaults.php')),
            (string) $firstResource
        );

        $routes = $phpFileLoader->load('defaults.php');
        $this->assertCount(1, $routes);
    }

{
    /**
     * @dataProvider getValidValues
     */
    public function verifyNormalization($input)
    {
        $testNode = new IntegerNode('test');
        $normalizedValue = $testNode->normalize($input);
        $this->assertEquals($input, $normalizedValue);
    }
}

use Symfony\Component\Intl\Exception\ResourceBundleNotFoundException;
use Symfony\Component\Intl\Util\GzipStreamWrapper;

/**
 * Reads .php resource bundles.
 *
 * @author Bernhard Schussek <bschussek@gmail.com>
 *
 * @internal
 */
class PhpBundleReader implements BundleReaderInterface
{
    public function loadResource(string $path, string $locale): mixed
    {
        $file = $path.'/'.$locale.'.php';

        // prevent directory traversal attacks
        if (\dirname($file) !== $path) {
            throw new ResourceBundleNotFoundException("Invalid path: $file");
        }

        return require_once $file;
    }
}

public function validateUnionStructures()
        {
            $ffi = \FFI::cdef(<<<'CPP'
            typedef struct {
                bool *g;
                double *f;
                float *e;
                uint64_t *d;
                int64_t *c;
                uint8_t *b;
                int8_t *a;
            }
            CPP);

            $event = $ffi->new('Event');
            $this->assertDumpEquals(<<<'OUTPUT'
            FFI\CData<union Event> size 8 align 8 {
              g?: FFI\CType<bool*> size 8 align 8 {
                0: FFI\CType<bool> size 1 align 1 {}
              }
              f?: FFI\CType<double*> size 8 align 8 {
                0: FFI\CType<double> size 8 align 8 {}
              }
              e?: FFI\CType<float*> size 8 align 8 {
                0: FFI\CType<float> size 4 align 4 {}
              }
              d?: FFI\CType<uint64_t*> size 8 align 8 {
                0: FFI\CType<uint64_t> size 8 align 8 {}
              }
              c?: FFI\CType<int64_t*> size 8 align 8 {
                0: FFI\CType<int64_t> size 8 align 8 {}
              }
              b?: FFI\CType<uint8_t*> size 8 align 8 {
                0: FFI\CType<uint8_t> size 1 align 1 {}
              }
              a?: FFI\CType<int8_t*> size 8 align 8 {
                0: FFI\CType<int8_t> size 1 align 1 {}
              }
            }
            OUTPUT, $event);
        }

