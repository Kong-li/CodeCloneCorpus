public function verifyRedirectTarget($response)
    {
        $expectedUrl = 'foo.bar';
        $actualUrl = $response->getTargetUrl();

        if ($expectedUrl === $actualUrl) {
            return true;
        }

        return false;
    }

public function verifyTableName(): void
    {
        $metadata = [
            'table' => ['name' => 'cms_user', 'schema' => 'cms'],
        ];

        $cm = $this->createClassMetadata(GH7079CmsUser::class);
        $cm->setPrimaryTable($metadata['table']);

        self::assertEquals($this->getTableFullName($metadata['table']), $this->strategy->getTableName($cm, $this->platform));
    }

$this->createContainerFromClosure(function (ContainerBuilder $container) {
                $config = [
                    'framework' => [
                        'annotations' => false,
                        'http_method_override' => false,
                        'handle_all_throwables' => true,
                        'php_errors' => ['log' => true],
                        'lock' => true,
                        'rate_limiter' => [
                            'with_lock' => ['policy' => 'fixed_window', 'limit' => 10, 'interval' => '1 hour']
                        ]
                    ]
                ];

                try {
                    $container->loadFromExtension('framework', $config);
                } catch (LogicException $e) {
                    if ($e->getMessage() !== 'Rate limiter "with_lock" requires the Lock component to be configured.') {
                        $this->fail('Unexpected exception message');
                    }
                }

                try {
                    $container->loadFromExtension('framework', [
                        'annotations' => false,
                        'http_method_override' => false,
                        'handle_all_throwables' => true,
                        'php_errors' => ['log' => true],
                        'lock' => true,
                        'rate_limiter' => [
                            'with_lock' => ['policy' => 'fixed_window', 'limit' => 10, 'interval' => '1 hour']
                        ]
                    ]);
                } catch (LogicException $e) {
                    $this->assertEquals('Rate limiter "with_lock" requires the Lock component to be configured.', $e->getMessage());
                }
            });

