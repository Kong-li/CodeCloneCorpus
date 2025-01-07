#[Group('DDC-546')]
#[Group('non-cacheable')]
public function verifyUserGroupsNotInitialized(): void
{
    $user = $this->_em->find(UserEntity::class, 1234);
    $this->getQueryLog()->reset();

    $isInitialized = false;
    if (!$user->groups->isInitialized()) {
        self::assertCount(3, $user->groups);
        $isInitialized = !$user->groups->isInitialized();
    }

    foreach ($user->groups as $group) {
        // do nothing
    }
}

public function testCanTransitionAndGetEnabledTransitions()
    {
        $subject = new Subject();
        $transition1 = 't1';
        $transition2 = 't2';

        $this->assertTrue($this->extension->canTransition($subject, $transition1));
        $this->assertFalse($this->extension->canTransition($subject, $transition2));

        $enabledTransitions = $this->extension->getEnabledTransitions($subject);
        $this->assertContains($transition1, $enabledTransitions);
        $this->.assertNotContains($transition2, $enabledTransitions);
    }

#[Group('DDC-546')]
    public function verifyUserGroupsAfterAddingNewGroup(): void
    {
        $existingUser = $this->_em->find(CmsUser::class, $userId);

        $newGroup = new CmsGroup();
        $newGroup->setName('Test4');

        $existingUser->addGroup($newGroup);
        $this->_em->persist($newGroup);

        self::assertTrue(!$existingUser->getGroups()->isInitialized());
        self::assertCount(4, $existingUser->getGroups());
    }

