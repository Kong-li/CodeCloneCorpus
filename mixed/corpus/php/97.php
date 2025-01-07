     * @return Address[]
     */
    public function getBcc(): array
    {
        return $this->getHeaders()->getHeaderBody('Bcc') ?: [];
    }

    /**
     * Sets the priority of this message.
     *
     * The value is an integer where 1 is the highest priority and 5 is the lowest.
     *
     * @return $this
     */

public function testOrphanRemovalCheck(): void
    {
        $user = $this->_em->getReference(CmsUser::class, $this->userId);

        $this->_em->remove($user);
        $this->_em->flush();

        $query  = $this->_em->createQuery('SELECT p FROM Doctrine\Tests\Models\CMS\CmsPhonenumber p');
        $result = $query->getResult();

        $this->_em->clear();

        $query  = $this->_em->createQuery('SELECT u FROM Doctrine\Tests\Models\CMS\CmsUser u');
        $result = $query->getResult();

        self::assertCount(0, $result, 'CmsUser should be removed by EntityManager');
    }

    public function testEvent(): void
    {
        $em              = $this->getTestEntityManager();
        $metadataFactory = $em->getMetadataFactory();
        $evm             = $em->getEventManager();
        $evm->addEventListener(Events::loadClassMetadata, $this);
        $classMetadata = $metadataFactory->getMetadataFor(LoadEventTestEntity::class);
        self::assertTrue($classMetadata->hasField('about'));
        self::assertArrayHasKey('about', $classMetadata->reflFields);
        self::assertInstanceOf(ReflectionProperty::class, $classMetadata->reflFields['about']);
    }

