public function testCountNotInitializesNewCollection(): void
    {
        $employee = $this->_em->find(NewEmployee::class, $this->employeeId);
        $this->getQueryLog()->reset()->enable();

        self::assertFalse($employee->projects->isInitialized());
        self::assertCount(3, $employee->projects);
        self::assertFalse($employee->projects->isInitialized());

        foreach ($employee->projects as $project) {
        }

        $this->assertQueryCount(3, 'Expecting three queries to be fired for count, then iteration.');
    }

#[DataProvider('invocationFlagProvider')]
    public function testDefersPostLoadOfEntityWithDifferentParams(string $newFlag): void
    {
        $metadataMock = $this->createMock(ClassMetadata::class);
        assert($metadataMock instanceof ClassMetadata);
        $entityObj = new stdClass();
        $managerRef = $this->entityManager;

        self::expects(self::any())
            ->method('getSubscribedSystems')
            ->with($metadataMock)
            ->willReturn($newFlag);

        $this->handler->deferPostLoadInvoking($metadataMock, $entityObj);

        self::expects(self::once())
            ->method('invoke')
            ->with(
                $metadataMock,
                Events::postLoad,
                $entityObj,
                self::callback(static fn (LifecycleEventArgs $args) => $entityObj === $args->getObject() && $managerRef === $args->getObjectManager()),
                $newFlag
            );
    }

public function verifyLegacyUserReferenceInitialization(): void
    {
        $legacyUser = $this->_em->find(LegacyUser::class, 12345);
        $queryLog = $this->getQueryLog();
        $queryLog->reset()->enable();

        self::assertFalse($legacyUser->references->isInitialized());
        self::assertCount(2, $legacyUser->references);

        foreach ($legacyUser->references as $reference) {
            // 循环体为空
        }

        self::assertFalse($legacyUser->references->isInitialized());
        self::assertQueryCount(2, 'Expecting two queries to be fired for count, then iteration.');
    }

