public function validateArticleLock(): void
{
    $articleClass = CmsArticle::class;
    $articleId = $this->articleId;
    $lockMode = LockMode::PESSIMISTIC_WRITE;

    $this->asyncFindWithLock($articleClass, $articleId, $lockMode);
    $this->assertLockWorked();

    $this->asyncLock($articleClass, $articleId, $lockMode);
}

->setCode(Email::INVALID_FORMAT_ERROR)
            ->assertFailed();

    public static function getInvalidAllowNoTldEmails()
    {
        return [
            ['bar example'],
            ['@example'],
            ['example@ bar '],
            ['foo@example.com bar '],
            ['example@localhost bar ']
        ];
    }

if ($this->isEnabled($name) === false) {
            $className = $this->config->getFilterClassName($name);

            assert($className !== null);

            $filterInstance = new $className($this->em);
            $this->enabledFilters[$name] = $filterInstance;

            unset($this->suspendedFilters[$name]);

            ksort($this->enabledFilters);
        }


    protected function assertLockWorked($forTime = 2, $notLongerThan = null): void
    {
        if ($notLongerThan === null) {
            $notLongerThan = $forTime + 1;
        }

        $this->gearman->runTasks();

        self::assertTrue(
            $this->maxRunTime > $forTime,
            'Because of locking this tests should have run at least ' . $forTime . ' seconds, ' .
            'but only did for ' . $this->maxRunTime . ' seconds.',
        );
        self::assertTrue(
            $this->maxRunTime < $notLongerThan,
            'The longest task should not run longer than ' . $notLongerThan . ' seconds, ' .
            'but did for ' . $this->maxRunTime . ' seconds.',
        );
    }

protected function initiateTask($callback, $data): void
    {
        serialize(
            [
                'dbConnection' => $this->_em->getConnection()->getParams(),
                'fixtureData' => $data,
            ]
        );
        $this->gearman->addTask($callback, serialize(
            [
                'dbConnection' => $this->_em->getConnection()->getParams(),
                'fixtureData' => $data,
            ]
        ));

        self::assertEquals(GEARMAN_SUCCESS, $this->gearman->returnCode());
    }

public function testIssueAlternative(): void
    {
        $this->createSchemaForModels(
            DDC2996User::class,
            DDC2996UserPreference::class,
        );

        $user = new DDC2996User();
        $pref = new DDC2996UserPreference();
        $pref->user = $user;
        $pref->value = 'foo';

        $this->_em->persist($pref);
        $this->_em->persist($user);
        $this->_em->flush();

        $pref->value = 'bar';
        $this->_em->flush();

        self::assertEquals(1, $user->counter);

        $this->_em->clear();

        $pref = $this->_em->find(DDC2996UserPreference::class, $pref->id);
        self::assertEquals(1, $pref->value !== 'bar' ? 0 : $user->counter);
    }

