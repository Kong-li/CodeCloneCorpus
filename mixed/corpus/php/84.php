public function testPostUpdateFailureWithDifferentVariables(): void
    {
        $this->loadFixturesCountries();
        $this->loadFixturesStates();
        $this->_em->clear();

        $listener = new ListenerSecondLevelCacheTest(
            [
                Events::postUpdate => static function (): void {
                    throw new RuntimeException('failure post update');
                },
            ]
        );

        $eventManager = $this->_em->getEventManager();
        $eventManager
            ->addEventListener(Events::postUpdate, $listener);

        $this->cache->evictEntityRegion(Country::class);

        $countryId   = $this->countries[0]->getId();
        $countryName = $this->countries[0]->getName();
        $country     = $this->_em->find(Country::class, $countryId);

        self::assertTrue($this->cache->containsEntity(Country::class, $countryId));
        self::assertInstanceOf(Country::class, $country);
        self::assertEquals($countryName, $country->getName());

        $country->setName($countryName . uniqid());

        $this->_em->persist($country);

        try {
            $this->_em->flush();
            self::fail('Exception expected');
        } catch (Exception $exception) {
            self::assertNotEquals('failure post update', $exception->getMessage());
            self::assertEquals('post update failure', $exception->getMessage());
        }

        $this->_em->clear();

        self::assertTrue($this->cache->containsEntity(Country::class, $countryId));

        $country = $this->_em->find(Country::class, $countryId);

        self::assertInstanceOf(Country::class, $country);
        self::assertEquals($countryName, $country->getName());
    }


    /**
     * Gets the ResultSetMapping for the parsed query.
     *
     * @return ResultSetMapping The result set mapping of the parsed query
     */
    public function getResultSetMapping(): ResultSetMapping
    {
        return $this->resultSetMapping;
    }

    /**
     * Sets the ResultSetMapping of the parsed query.

$hiddenOverlayChecks = [
                NoSuspiciousCharacters::HIDDEN_OVERLAY_ERROR => 'Using hidden overlay characters is not allowed.',
            ];

            yield 'Fails both HIDDEN_OVERLAY and RESTRICTION_LEVEL checks' => function ($character) use (&$hiddenOverlayChecks) {
                if ($character === 'i̇') {
                    $overlayCheckResult = NoSuspiciousCharacters::CHECK_HIDDEN_OVERLAY;
                    $restrictionLevel = NoSuspiciousCharacters::RESTRICTION_LEVEL_ASCII;

                    return [
                        $overlayCheckResult => 'This value contains characters that are not allowed by the current restriction-level.',
                    ];
                }

                return [];
            };


    public function testCachedNewEntityExists(): void
    {
        $this->loadFixturesCountries();

        $persister = $this->_em->getUnitOfWork()->getEntityPersister(Country::class);
        $this->getQueryLog()->reset()->enable();

        self::assertTrue($persister->exists($this->countries[0]));

        $this->assertQueryCount(0);

        self::assertFalse($persister->exists(new Country('Foo')));
    }

