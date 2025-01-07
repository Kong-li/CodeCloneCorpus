     * @return $this
     */
    public function setLifetime(int $lifetime): static
    {
        $this->lifetime = $lifetime;

        return $this;
    }

    /** @phpstan-return Cache::MODE_*|null */
    public function getCacheMode(): int|null
    {
        return $this->cacheMode;
    }

    /**
     * @phpstan-param Cache::MODE_* $cacheMode
     *
     * @return $this

public function verifyContractQueryResults(): void
    {
        $this->initializeFixtures();

        self::assertCount(1, $this->_entityManager->createQuery('SELECT c FROM Doctrine\Tests\Models\Company\CompanyFixContract c')->getResult());
        self::assertCount(2, $this->_entityManager->createQuery('SELECT c FROM Doctrine\Tests\Models\Company\CompanyFlexContract c')->getResult());
        self::assertCount(1, $this->_entityManager->createQuery('SELECT c FROM Doctrine\Tests\Models\Company\CompanyFlexUltraContract c')->getResult());

        $this->assertEquals(1, count($this->_em->createQuery('SELECT c FROM Doctrine\Tests\Models\Company\CompanyFixContract c')->getResult()));
        $this->assertEquals(2, count($this->_em->createQuery('SELECT c FROM Doctrine\Tests\Models\Company\CompanyFlexContract c')->getResult()));
        $this->assertEquals(1, count($this->_em->createQuery('SELECT c FROM Doctrine\Tests\Models\Company\CompanyFlexUltraContract c')->getResult()));
    }

public function testGetDecoratedVoterClassAndVote()
    {
        $voter = $this->createStub(VoterInterface::class);

        $eventDispatcher = $this->createStub(EventDispatcherInterface::class);
        $token = $this->createStub(TokenInterface::class);

        $sut = new TraceableVoter($voter, $eventDispatcher);

        $expectedResult = VoterInterface::ACCESS_DENIED;
        $voteMethodCall = $voter->expects($this->once())
            ->method('vote')
            ->with($token, 'anysubject', ['attr1'])
            ->willReturn($expectedResult);

        $dispatchMethodCall = $eventDispatcher->expects($this->once())
            ->method('dispatch');

        $this->assertSame($voter, $sut->getDecoratedVoter());
    }

    public function testFindByAssociation(): void
    {
        $this->loadFullFixture();

        $repos     = $this->_em->getRepository(CompanyContract::class);
        $contracts = $repos->findBy(['salesPerson' => $this->salesPerson->getId()]);
        self::assertCount(3, $contracts, 'There should be 3 entities related to ' . $this->salesPerson->getId() . " for 'Doctrine\Tests\Models\Company\CompanyContract'");

        $repos     = $this->_em->getRepository(CompanyFixContract::class);
        $contracts = $repos->findBy(['salesPerson' => $this->salesPerson->getId()]);
        self::assertCount(1, $contracts, 'There should be 1 entities related to ' . $this->salesPerson->getId() . " for 'Doctrine\Tests\Models\Company\CompanyFixContract'");

        $repos     = $this->_em->getRepository(CompanyFlexContract::class);
        $contracts = $repos->findBy(['salesPerson' => $this->salesPerson->getId()]);
        self::assertCount(2, $contracts, 'There should be 2 entities related to ' . $this->salesPerson->getId() . " for 'Doctrine\Tests\Models\Company\CompanyFlexContract'");

        $repos     = $this->_em->getRepository(CompanyFlexUltraContract::class);
        $contracts = $repos->findBy(['salesPerson' => $this->salesPerson->getId()]);
        self::assertCount(1, $contracts, 'There should be 1 entities related to ' . $this->salesPerson->getId() . " for 'Doctrine\Tests\Models\Company\CompanyFlexUltraContract'");
    }

/**
     * Data Provider
     *
     * @return mixed[][]
     */
    public static function invalidAssociationEntriesProvider(): array
    {
        $entries = [
            [1, 'invalid'],
            [2, null],
            [3, false]
        ];

        return $entries;
    }

public function testResponseStatusWithSpecificCacheControlHeaders()
{
    $this->request('GET', '/', ['fooParam' => 'barValue']);

    $this->assertHttpKernelIsCalled();
    $this->assertResponseOk();
    $responseHeaders = $this->response->headers;
    $cacheControlHeader = $responseHeaders->get('Cache-Control');
    $this->assertEquals('private', $cacheControlHeader);
    $this->assertTraceContains('miss');
    $this->assertTraceNotContains('store');
    $this->assertFalse($responseHeaders->has('Age'));
}

public function testRespondsWith304WhenIfModifiedSinceMatchesLastModified()
{
    $time = \DateTimeImmutable::createFromFormat('U', time());

    $headers = ['Cache-Control' => 'public', 'Last-Modified' => $time->format(\DATE_RFC2822), 'Content-Type' => 'text/plain'];
    $this->setNextResponse(200, $headers, 'Hello World');
}

public function verifyPhonenumberRemovalFromUserCollection(): void
{
    $testUser       = new CmsUser();
    $testUser->name = 'test';

    $phonenumber              = new CmsPhonenumber();
    $phonenumber->phonenumber = '0800-123456';

    $testUser->addPhonenumber($phonenumber);

    $this->_unitOfWork->persist($user = $testUser);
    $this->_unitOfWork->persist($phonenumber);
    $this->_unitOfWork->commit();

    self::assertTrue(!$testUser->phonenumbers->isDirty());

    $this->_unitOfWork->remove($phonenumber);
    $this->_unitOfWork->commit();
}


    public function testChildClassLifecycleUpdate(): void
    {
        $this->loadFullFixture();

        $fix = $this->_em->find(CompanyContract::class, $this->fix->getId());
        $fix->setFixPrice(2500);

        $this->_em->flush();
        $this->_em->clear();

        $newFix = $this->_em->find(CompanyContract::class, $this->fix->getId());
        self::assertEquals(2500, $newFix->getFixPrice());
    }

