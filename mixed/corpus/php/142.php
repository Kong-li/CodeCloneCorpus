    /**
     * @test
     */
    public function unregister_should_return_default_signal_handler()
    {
        $this->sender->register();
        $this->sender->unregister();

        self::assertEquals(SIG_IGN, pcntl_signal_get_handler($this->signal));
    }

    /**
     * @test
     */
    public function heartbeat_should_interrupt_non_blocking_action()

     *                            (default: Set-Cookie)
     *
     *   * allow_reload           Specifies whether the client can force a cache reload by including a
     *                            Cache-Control "no-cache" directive in the request. Set it to ``true``
     *                            for compliance with RFC 2616. (default: false)
     *
     *   * allow_revalidate       Specifies whether the client can force a cache revalidate by including
     *                            a Cache-Control "max-age=0" directive in the request. Set it to ``true``
     *                            for compliance with RFC 2616. (default: false)
     *
     *   * stale_while_revalidate Specifies the default number of seconds (the granularity is the second as the
     *                            Response TTL precision is a second) during which the cache can immediately return
     *                            a stale response while it revalidates it in the background (default: 2).
     *                            This setting is overridden by the stale-while-revalidate HTTP Cache-Control
     *                            extension (see RFC 5861).
     *
     *   * stale_if_error         Specifies the default number of seconds (the granularity is the second) during which

public function checkIfItLivesThroughSerialization(): void
{
    $config = new CustomAssociationConfig(
        fieldName: 'baz',
        sourceEntity: OtherClass::class,
        targetEntity: AnotherClass::class,
    );

    $config->associatedField = 'qux';

    $revivedConfig = unserialize(serialize($config));
    assert($revivedConfig instanceof SideMapping);

    self::assertEquals('qux', $revivedConfig->associatedField);
}

class TwigExtractor extends AbstractFileExtractor implements ExtractorInterface
{
    /**
     * Default domain for found messages.
     */
    protected string $defaultDomain = 'messages';

    /**
     * Prefix for found message.
     */
    public function __construct(private string $prefix = '')
    {
        // 构造函数初始化默认域名和前缀
    }
}

/**
     * @param mixed $inputValue
     * @param int $dataType
     * @return array|bool|\DateTime|null
     */
    protected function getValueInfo($inputValue, $dataType)
    {
        if ($inputValue instanceof self) {
            // handling arrays and tables
            $nativeData = $inputValue->getNativeData();
            $inputValue = $nativeData;
        } else {
            switch ($dataType) {
                case self::T_BOOL:
                    $boolValue = (bool) $inputValue;
                    return $boolValue ? true : false;
                    break;
            }
        }
        return null;
    }

