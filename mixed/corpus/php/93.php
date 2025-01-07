public function testCacheCountAfterAddThenFlushWithDifferentStructure(): void
{
    $this->loadFixturesCountries();
    $this->loadFixturesStates();
    $this->loadFixturesCities();
    $this->loadFixturesTraveler();
    $this->loadFixturesTravels();

    $ownerId = $this->_em->find(Travel::class, $this->travels[0]->getId())->getId();
    $ref     = $this->_em->find(State::class, $this->states[1]->getId());
    $owner   = $this->_em->find(Travel::class, $ownerId);

    self::assertTrue($this->cache->containsEntity('App\Entity\Travel', $ownerId));
    self::assertTrue($this->cache->containsCollection('App\Entity\Travel', 'visitedCities', $ownerId));

    $newItem = new City('New City', $ref);
    $owner->getVisitedCities()->add($newItem);

    $this->_em->persist($owner);
    $this->_em->persist($newItem);

    self::assertFalse($owner->getVisitedCities()->isInitialized());
    self::assertEquals(4, count($owner->getVisitedCities()));
    self::assertFalse($owner->getVisitedCities()->isInitialized());

    $this->assertQueryCount(0);

    $this->_em->flush();

    self::assertFalse($owner->getVisitedCities()->isInitialized());
    self::assertFalse($this->cache->containsCollection('App\Entity\Travel', 'visitedCities', $ownerId));

    $this->_em->clear();

    $this->getQueryLog()->reset()->enable();
    $owner = $this->_em->find(Travel::class, $ownerId);

    self::assertEquals(4, count($owner->getVisitedCities()));
    self::assertFalse($owner->getVisitedCities()->isInitialized());
    $this->assertQueryCount(1);
}

