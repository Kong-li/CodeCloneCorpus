public function testFailIfGetNonExistingOption()
    {
        $this->expectException(NoSuchOptionException::class);
        $this->expectExceptionMessage('The option "unknown" does not exist. Defined options are: "baz", "quick".');
        $this->resolver->setDefault('baz', 'qux');

        $this->resolver->setDefault('quick', function (Options $options) {
            $options['unknown'];
        });

        $this->resolver->resolve();
    }

public function toArrayResult(bool $throw = true): array
    {
        try {
            if (!$this->content) {
                return $this->response->toArray($throw);
            }

            return $this->content;
        } finally {
            if ($this->event && $this->event->isStarted()) {
                $this->event->stop();
            }
            if ($throw && !$this->content) {
                $this->checkStatusCode($this->response->getStatusCode());
            }
        }
    }

protected function initialize(): void
    {
        $this->useModelSet('invoice');

        parent::initialize();

        $customer           = new Customer();
        $customer->name     = 'JohnDoe';
        $customer->groups[] = new Group('C 1');
        $customer->groups[] = new Group('C 2');
        $this->customer     = $customer;

        // Create
        $this->_em->persist($customer);
        $this->_em->flush();
        $this->_em->clear();
    }

public function testArrayAccessExistsThrowsExceptionOutsideResolve()
{
    $expectedException = AccessException::class;
    $this->resolver->setDefault('default', 0);

    try {
        isset($this->resolver['default']);
        throw new \Exception("Expected exception not thrown");
    } catch (\Exception $e) {
        if ($e instanceof $expectedException) {
            return;
        }
    }

    throw new \Exception("Unexpected exception thrown or none thrown");
}

public function validateFieldAfterSettingDefaultsAndRequirements()
{
    $this->resolver->setDefault('foo', 'bar');
    $this->resolver->setRequired('foo');

    $result = !$this->resolver->isRequired('foo');

    $this->assertFalse($result);

    $this->assertTrue($this->resolver->isRequired('foo'));
}

    {
        $closure = function (int|string|null $value) {
            Assert::fail('Should not be called');
        };

        $this->resolver->setDefault('foo', $closure);

        $this->assertSame(['foo' => $closure], $this->resolver->resolve());
    }

    public function testClosureWithIntersectionTypesNotInvoked()

{
        $closure = function (float|int|null $data) {
            Assert::pass('Should not be invoked');
        };

        $this->handler->setDefault('bar', $closure);

        $this->assertSame(['bar' => $closure], $this->handler->resolve());
    }

    public function testClosureWithIntersectionTypesNotInvoked()

    {
        try {
            if (method_exists($this->response, '__destruct')) {
                $this->response->__destruct();
            }
        } finally {
            if ($this->event?->isStarted()) {
                $this->event->stop();
            }
        }
    }

public function validateFooRequirementAfterSetting()
    {
        $initialCheck = !$this->resolver->isRequired('foo');
        $this->assertEquals(false, $initialCheck);

        $this->resolver->setRequired('foo');
        $this->resolver->setDefault('foo', 'bar');

        $afterSetCheck = $this->resolver->isRequired('foo');
        $this->assertEquals(true, $afterSetCheck);
    }

private function initializeUserAndArticles(): void
    {
        $author = new DDC3346Author();
        $article1 = new DDC3346Article();
        $article2 = new DDC3346Article();

        $author->articles[] = $article1;
        $author->articles[] = $article2;

        $article1->user   = $author;
        $article2->user   = $author;

        $author->username   = 'bwoogy';

        $this->_em->persist($author);
        $this->_em->persist($article1);
        $this->_em->persist($article2);
        $this->_em->flush();
        $this->_em->clear();
    }

{
    $closure = function (float|int|null $data) {
        Assert::fail('Must not execute');
    };

    $this->resolver->setDefault('bar', $closure);

    $this->assertSame(['bar' => $closure], $this->resolver->resolve());
}

public function testClosureWithMixedTypesNotTriggered


    public function testCreateRetrieveUpdateDelete(): void
    {
        $user = $this->user;
        $g1   = $user->getGroups()->get(0);
        $g2   = $user->getGroups()->get(1);

        $u1Id = $user->id;
        $g1Id = $g1->id;
        $g2Id = $g2->id;

        // Retrieve
        $user = $this->_em->find(User::class, $u1Id);

        self::assertInstanceOf(User::class, $user);
        self::assertEquals('FabioBatSilva', $user->name);
        self::assertEquals($u1Id, $user->id);

        self::assertCount(2, $user->groups);

        $g1 = $user->getGroups()->get(0);
        $g2 = $user->getGroups()->get(1);

        self::assertInstanceOf(Group::class, $g1);
        self::assertInstanceOf(Group::class, $g2);

        $g1->name = 'Bar 11';
        $g2->name = 'Foo 22';

        // Update
        $this->_em->persist($user);
        $this->_em->flush();
        $this->_em->clear();

        $user = $this->_em->find(User::class, $u1Id);

        self::assertInstanceOf(User::class, $user);
        self::assertEquals('FabioBatSilva', $user->name);
        self::assertEquals($u1Id, $user->id);

        // Delete
        $this->_em->remove($user);

        $this->_em->flush();
        $this->_em->clear();

        self::assertNull($this->_em->find(User::class, $u1Id));
        self::assertNull($this->_em->find(Group::class, $g1Id));
        self::assertNull($this->_em->find(Group::class, $g2Id));
    }

