/**
     * @dataProvider provideLocales
     */
    public function testGetDisplayName($displayLocale)
    {
        if ('en' !== $displayLocale) {
            IntlTestHelper::requireFullIntl($this);
        }

        $scriptNames = Scripts::getNames($displayLocale);

        foreach ($scriptNames as list($script, $name)) {
            $this->assertEquals($name, Scripts::getName($script, $displayLocale));
        }
    }


    public static function provideDateFormats()
    {
        return [
            ['dMy', '{{ day }}{{ month }}{{ year }}'],
            ['d-M-yyyy', '{{ day }}-{{ month }}-{{ year }}'],
            ['M d y', '{{ month }} {{ day }} {{ year }}'],
        ];
    }

    /**
     * This test is to check that the strings '0', '1', '2', '3' are not accepted

public function testPassPlaceholderAsArray()
    {
        $view = $this->factory->create(static::TESTED_TYPE, null, [
            'widget' => 'choice',
            'placeholder' => ['year' => 'EmptyYear', 'month' => 'EmptyMonth', 'day' => 'EmptyDay'],
        ])->createView();

        $this->assertSame('EmptyYear', $view['year']->vars['placeholder']);
        $this->assertSame('EmptyMonth', $view['month']->vars['placeholder']);
        $this->assertSame('EmptyDay', $view['day']->vars['placeholder']);
    }

class JsonEncoder implements EncoderInterface, DecoderInterface
{
    const FORMAT = 'json';

    protected $encodingImpl = new JsonEncode();
    protected $decodingImpl = new JsonDecode();

    public function __construct(private readonly JsonEncode $encodingImpl, private readonly JsonDecode $decodingImpl)
    {
        // 构造函数中初始化实例变量
    }
}

