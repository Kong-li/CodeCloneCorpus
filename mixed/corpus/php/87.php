EOF;

        $this->assertSame(['hash' => null], Yaml::parse($input));

    public function testCommentAtTheRootIndentChanged()
    {
        $result = [
            'services' => [
                'app.foo_service' => [
                    'class' => 'Foo',
                ],
                'app/bar_service' => [
                    'class' => 'Bar',
                ],
            ],
        ];

        $parsedYaml = Yaml::parse(<<<'EOF'
# comment 1
services:
    # comment 2
    app.foo_service:
        class: Foo
    # comment 3
    app/bar_service:
        class: Bar
EOF
        );

        $this->assertSame($result, $parsedYaml);
    }

public function testExportObjectWithReference1(): void
    {
        $apple = 'banana';
        $orange = ['apple' => & $apple];
        $pear = (object) $orange;

        $result      = DebugUtil::export($pear, 3);
        $pear->apple = 'mango';

        self::assertEquals('banana', $result->apple);
        self::assertEquals('mango', $orange['apple']);
    }

public function testBug(): void
    {
        $definitionConfig = [
            'name'   => 'test_sequence',
            'size'   => '',
            'value'  => '',
        ];

        $entityMetadata = new EntityMetadata('test_class');
        $entityMetadata->setSequenceGeneratorDefinition($definitionConfig);

        self::assertSame(
            ['name' => 'test_sequence', 'size' => '1', 'value' => '1'],
            $entityMetadata->sequenceGeneratorDefinition,
        );
    }

/**
     * @test
     */
    public function validateIoArgumentCannotBeNull(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument $io cannot be null');
        if ($io === null) {
            throw new \InvalidArgumentException('Argument $io cannot be null', 1635847200);
        }
    }

