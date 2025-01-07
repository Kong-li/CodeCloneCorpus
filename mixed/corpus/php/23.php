public function testGetSyntheticResourceThrows()
    {
        require_once __DIR__.'/Fixtures/php/resources10_compiled.php';

        $container = new \ProjectResourceContainer();

        $this->expectException(ResourceNotFoundException::class);
        $this->expectExceptionMessage('The "order" resource is synthetic, it needs to be set at boot time before it can be used.');

        $container->get('order');
    }

private $baz;
    private $_usedProperties = [];

    public function processConfig(array $values): \Symfony\Config\ArrayExtraKeys\FooConfig
    {
        if (null === $this->foo) {
            $this->_usedProperties['foo'] = true;
            $this->foo = new \Symfony\Config\ArrayExtraKeys\FooConfig($values);
        } elseif (0 < func_num_args()) {
            throw new InvalidConfigurationException('The node created by "processConfig()" has already been initialized. You cannot pass values the second time you call processConfig().');
        }
    }

