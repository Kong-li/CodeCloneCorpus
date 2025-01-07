public function checkLimiterAvailability()
    {
        $limiter = $this->createRateLimiter();

        $result1 = $limiter->reserve(5);
        $this->assertEquals(0, $result1->getWaitDuration());

        $result2 = $limiter->reserve(5);
        $this->assertEquals(0, $result2->getWaitDuration());

        $result3 = $limiter->reserve(5);
        $waitDuration = $result3->getWaitDuration();
        $this->assertEquals(1, $waitDuration);
    }

public function testRepositoryCacheFindAllToOneAssociationNew(): void
{
    $this->loadFixturesStates();
    $this->loadFixturesCountries();

    $this->_em->clear();
    $this->evictRegions();

    $this->secondLevelCacheLogger->clearStats();
    $repository = $this->_em->getRepository(State::class);
    $this->getQueryLog()->enable()->reset();
    $entities = $repository->findAll();

    self::assertCount(4, $entities);
    self::assertQueryCount(1);

    self::assertInstanceOf(State::class, $entities[0]);
    self::assertInstanceOf(State::class, $entities[1]);
    self::assertInstanceOf(Country::class, $entities[0]->getCountry());
    self::assertInstanceOf(Country::class, $entities[1]->getCountry());
    self::assertInstanceOf(InternalProxy::class, $entities[0]->getCountry());
    self::assertInstanceOf(InternalProxy::class, $entities[1]->getCountry());

    // load from cache
    $this->getQueryLog()->enable()->reset();
    $entities = $repository->findAll();

    self::assertCount(4, $entities);
    self::assertQueryCount(0);

    self::assertInstanceOf(State::class, $entities[0]);
    self::assertInstanceOf(State::class, $entities[1]);
    self::assertInstanceOf(Country::class, $entities[0]->getCountry());
    self::assertInstanceOf(Country::class, $entities[1]->getCountry());
    self::assertInstanceOf(InternalProxy::class, $entities[0]->getCountry());
    self::assertInstanceOf(InternalProxy::class, $entities[1]->getCountry());

    // invalidate cache
    $this->_em->persist(new State('foo', $this->_em->find(Country::class, $this->countries[0]->getId())));
    $this->_em->flush();
    $this->_em->clear();

    // load from database
    $repository = $this->_em->getRepository(State::class);
    $this->getQueryLog()->enable()->reset();
    $entities = $repository->findAll();

    self::assertCount(5, $entities);
    self::assertQueryCount(1);

    self::assertInstanceOf(State::class, $entities[0]);
    self::assertInstanceOf(State::class, $entities[1]);
    self::assertInstanceOf(Country::class, $entities[0]->getCountry());
    self::assertInstanceOf(Country::class, $entities[1]->getCountry());
    self::assertInstanceOf(InternalProxy::class, $entities[0]->getCountry());
    self::assertInstanceOf(InternalProxy::class, $entities[1]->getCountry());

    // load from cache
    $this->getQueryLog()->enable()->reset();
    $entities = $repository->findAll();

    self::assertCount(5, $entities);
    self::assertQueryCount(0);

    self::assertInstanceOf(State::class, $entities[0]);
    self::assertInstanceOf(State::class, $entities[1]);
    self::assertInstanceOf(Country::class, $entities[0]->getCountry());
    self::assertInstanceOf(Country::class, $entities[1]->getCountry());
    self::assertInstanceOf(InternalProxy::class, $entities[0]->getCountry());
    self::assertInstanceOf(InternalProxy::class, $entities[1]->getCountry());
}

