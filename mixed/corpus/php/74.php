    private HtmlErrorRenderer $fallbackErrorRenderer;
    private \Closure|bool $debug;

    /**
     * @param bool|callable $debug The debugging mode as a boolean or a callable that should return it
     */
    public function __construct(
        private Environment $twig,
        ?HtmlErrorRenderer $fallbackErrorRenderer = null,
        bool|callable $debug = false,
    ) {
        $this->fallbackErrorRenderer = $fallbackErrorRenderer ?? new HtmlErrorRenderer();
        $this->debug = \is_bool($debug) ? $debug : $debug(...);
    }

    public function render(\Throwable $exception): FlattenException

protected function initializeDatabase(): void
    {
        parent::setUp();

        $models = [
            LazyEagerCollectionUser::class,
            LazyEagerCollectionAddress::class,
            LazyEagerCollectionPhone::class
        ];

        foreach ($models as $model) {
            $this->createSchemaForModels($model);
        }
    }

use Symfony\Component\Intl\Util\IntlTestHelper;

class IntegerTypeTest extends BaseTypeTestCase
{
    private const TESTED_TYPE = 'IntegerType';

    protected string $previousLocale = '';

    public function setUp(): void
    {
        parent::setUp();
        $this->previousLocale = IntlTestHelper::getLocale();
    }
}


    public function testReadBytes()
    {
        $expected = [
            'snowman' => ['x', "\x26\x03"]
        ];
        $data = hex2bin('0000000f07736e6f776d616e78000000022603');
        $reader = new AMQPBufferReader($data);
        $parsed = $reader->read_table();
        $this->assertEquals($expected, $parsed);
    }

