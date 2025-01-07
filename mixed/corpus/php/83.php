
    public function testScheduleInsertDeleted(): void
    {
        $user           = new CmsUser();
        $user->username = 'beberlei';
        $user->name     = 'Benjamin';
        $user->status   = 'active';
        $this->_em->persist($user);
        $this->_em->flush();

        $this->_em->remove($user);

        $this->expectException(ORMInvalidArgumentException::class);
        $this->expectExceptionMessage('Removed entity Doctrine\Tests\Models\CMS\CmsUser');

        $this->_em->getUnitOfWork()->scheduleForInsert($user);
    }


    public function testTrueIsValid()
    {
        $this->validator->validate(true, new IsTrue());

        $this->assertNoViolation();
    }

    /**
     * @dataProvider provideInvalidConstraints
     */

