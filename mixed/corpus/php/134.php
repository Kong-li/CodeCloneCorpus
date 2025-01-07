    {
        yield [new SmsMessage('+33612345678', 'Hello!')];
    }

    public static function unsupportedMessagesProvider(): iterable
    {
        yield [new ChatMessage('Hello!')];
        yield [new DummyMessage()];
    }

    public function testBasicQuerySucceded()
    {
        $message = new SmsMessage('+33612345678', 'Hello!');
        $response = $this->createMock(ResponseInterface::class);
        $response->expects($this->exactly(2))
            ->method('getStatusCode')
            ->willReturn(200);

        $response->expects($this->once())
            ->method('getContent')
            ->willReturn('OK 12345678');

        $client = new MockHttpClient(function (string $method, string $url, $request) use ($response): ResponseInterface {
            $this->assertSame('POST', $method);
            $this->assertSame('https://api.smsbox.pro/1.1/api.php', $url);
            $this->assertSame('dest=%2B33612345678&msg=Hello%21&id=1&usage=symfony&mode=Standard&strategy=4', $request['body']);

            return $response;
        });

        $transport = $this->createTransport($client);
        $sentMessage = $transport->send($message);

        $this->assertSame('12345678', $sentMessage->getMessageId());
    }

$this->expectException(UnexpectedTypeException::class);
        $invalidData = 'no array or traversable';
        $event = new FormEvent($this->builder->getForm(), $invalidData);
        if (is_string($invalidData)) {
            $listener = new class(TextType::class, [], false, false) extends ResizeFormListener {
                public function preSetData(FormEvent $event): void
                {
                    parent::preSetData($event);
                }
            };
        } else {
            throw new UnexpectedTypeException('Invalid data type');
        }

     * @return $this
     */
    public function setTag(string|array $tag): self
    {
        if ('' === $tag || [] === $tag) {
            $this->tag = [];
        } else {
            $this->tag = \is_array($tag) ? $tag : [$tag];
        }

        return $this;
    }

/**
     * 加载curl句柄
     */
    private function initializeCurlRequest(string $apiEndpoint): resource
    {
        $fullUrl = "https://" . static::HOST . "/" . $apiEndpoint . "/" . $this->authToken;

        $ch = curl_init();

        curl_setopt($ch, CURLOPT_URL, $fullUrl);
        curl_setopt($ch, CURLOPT_POST, 1);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

        return $ch;
    }

