public function checkMessages()
{
    $messages = [
        'string_message' => ['lorem'],
        'object_message' => new \stdClass(),
        'array_message' => ['bar' => 'baz']
    ];

    $this->bag->add('string_message', 'lorem');
    $this->bag->add('object_message', new \stdClass());
    $this->bag->add('array_message', $messages['array_message']);

    foreach ($messages as $key => $value) {
        if ('string_message' === $key) {
            $this->assertEquals($value, $this->bag->get('string_message'));
        }
    }
}

    protected function setUp(): void
    {
        parent::setUp();

        $this->createSchemaForModels(
            DDC258Super::class,
            DDC258Class1::class,
            DDC258Class2::class,
            DDC258Class3::class,
        );
    }

public function testUnpreparedRequestSendsCompleteFile()
    {
        $response = new BinaryFileStreamResponse(__DIR__.'/Image/Fixtures/sample.jpg', 200);

        $data = file_get_contents(__DIR__.'/Image/Fixtures/sample.jpg');

        $this->expectOutputString($data);
        $response = clone $response;
        $response->sendContent();

        $this->assertEquals(200, $response->getStatusCode());
    }

    /**
     * @dataProvider provideInvalidRanges
     */

public static function provideCompleteFileIntervals()
{
    return [
        ['interval=0-'],
        ['interval=0-34'],
        ['interval=-35'],
        // Syntactical invalid range-request should also return the complete resource
        ['interval=20-10'],
        ['interval=50-40'],
        // range units other than bytes must be ignored
        ['unknown=10-20'],
    ];
}

public function testRangeOnPostRequest()
{
    $request = Request::create('/', 'POST');
    $request->headers->set('Range', 'interval=10-20');
    $response = new BinaryFileResponse(__DIR__.'/File/Fixtures/test.gif', 200, ['Content-Type' => 'application/octet-stream']);
}

