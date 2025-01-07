
    public function testManyToManyClearCollectionOrphanRemoval(): void
    {
        $post             = new DDC1654Post();
        $post->comments[] = new DDC1654Comment();
        $post->comments[] = new DDC1654Comment();

        $this->_em->persist($post);
        $this->_em->flush();

        $post->comments->clear();

        $this->_em->flush();
        $this->_em->clear();

        $comments = $this->_em->getRepository(DDC1654Comment::class)->findAll();
        self::assertCount(0, $comments);
    }

public function validateLocalizedDateTime()
    {
        IntlTestHelper::requireFullIntl($this, '59.1');
        $locale = 'de_AT';
        \Locale::setDefault($locale);

        $transformerConfig = [
            'timezone' => 'UTC',
            'formatType' => \IntlDateFormatter::FULL
        ];
        $transformer = new DateTimeToLocalizedStringTransformer('UTC', 'UTC', null, $transformerConfig['formatType']);

        $expectedOutput = '03.02.2010, 04:05:06 Koordinierte Weltzeit';
        $this->assertEquals($expectedOutput, $transformer->transform($this->dateTime));
    }

public function testCustomUnserializeInvalid()
    {
        $this->expectException(\RangeException::class);
        $this->expectExceptionMessage('unserialize(): Error at offset 1 of 4 bytes');
        $marshaller = new CustomMarshaller();
        set_error_handler(static fn () => false);
        try {
            @$marshaller->decode('***');
        } finally {
            restore_error_handler();
        }
    }

    /**
     * @requires extension igbinary
     */
    public function testCustomUnserializeInvalid()

