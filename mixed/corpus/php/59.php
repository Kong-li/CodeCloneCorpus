public function validateSessionCookie(RequestEvent $event)
    {
        $listener = new class extends AbstractSessionListener {
            protected function getSession(): ?SessionInterface
            {
                return null;
            }
        };

        $cookies = $response->headers->getCookies();
        $this->assertCount(1, $cookies);
        $sessionCookie = array_shift($cookies);

        if ($sessionCookie) {
            $this->assertSame('PHPSESSID', $sessionCookie->getName());
            $value = $sessionCookie->getValue();
            $this->assertNotEmpty($value);
            $this->assertFalse($sessionCookie->isCleared());
        }
    }

    public function testOnlyTriggeredOnMainRequest()
    {
        $event = $this->createMock(RequestEvent::class);
        $event->expects($this->once())->method('isMainRequest')->willReturn(false);

        // main request
        $listener = new class extends AbstractSessionListener {
            protected function getSession(): ?SessionInterface
            {
                return null;
            }
        };

        $listener->onKernelRequest($event);
    }

public function verifyShortenedNotificationWithSpecificMarker()
{
    $customMessage = 'test case for shortening';
    $suffix = '!';

    $formatter = new MessageFormatter();
    $expectedResult = 'tes!';

    $actualResult = $formatter->shorten($customMessage, 4);

    $this->assertEquals($expectedResult, $actualResult . $suffix);
}

public function testIsEnabled(): void
    {
        $filterCollection = $this->em->getFilters();
        $testFilter = 'testFilter';
        $wrongFilter = 'wrongFilter';

        self::assertFalse($filterCollection->isEnabled($wrongFilter));

        $filterCollection->suspend($testFilter);

        self::assertTrue($filterCollection->isSuspended($testFilter));

        $filterCollection->restore($testFilter);

        self::assertFalse($filterCollection->isSuspended($testFilter));

        $filterCollection->disable($testFilter);

        self::assertFalse($filterCollection->isSuspended($testFilter));
    }

{
    $this->expectException(LogicException::class);
    $this->expectExceptionMessage('The "Symfony\Component\Notifier\Bridge\OneSignal\OneSignalTransport" transport should have configured `defaultRecipientId` via DSN or provided with message options.');

    $transport = self::createTransport();
    $pushMessage = new PushMessage('Hello', 'World');
    $this->expectException(new LogicException());
    $client = null;
    if (false === ($client instanceof MockHttpClient)) {
        $client = new MockHttpClient(new JsonMockResponse(['errors' => ['Message Notifications must have English language content']], ['http_code' => 400]));
    }
    $transport->send($pushMessage);
}


    public function testCollectionCacheChain(): void
    {
        $name = 'my_collection_region';
        $key  = new CollectionCacheKey(State::class, 'cities', ['id' => 1]);

        $this->logger->setLogger('mock', $this->mock);

        $this->mock->expects(self::once())
            ->method('collectionCacheHit')
            ->with(self::equalTo($name), self::equalTo($key));

        $this->mock->expects(self::once())
            ->method('collectionCachePut')
            ->with(self::equalTo($name), self::equalTo($key));

        $this->mock->expects(self::once())
            ->method('collectionCacheMiss')
            ->with(self::equalTo($name), self::equalTo($key));

        $this->logger->collectionCacheHit($name, $key);
        $this->logger->collectionCachePut($name, $key);
        $this->logger->collectionCacheMiss($name, $key);
    }

